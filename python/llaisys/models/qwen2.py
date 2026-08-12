from typing import Sequence
from ..libllaisys import LIB_LLAISYS, DeviceType, DataType, LlaisysQwen2Meta

from ctypes import byref, c_int64, c_size_t, c_void_p
import json
from pathlib import Path


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        import safetensors
        import torch
        model_path = Path(model_path)
        with open(model_path / "config.json", encoding="utf-8") as config_file:
            config = json.load(config_file)

        torch_dtype = config.get("torch_dtype", "bfloat16")
        dtype = {
            "bfloat16": DataType.BF16,
            "float16": DataType.F16,
            "float32": DataType.F32,
        }[torch_dtype]
        self._end_token = int(config.get("eos_token_id", 151643))
        meta = LlaisysQwen2Meta(
            dtype=dtype,
            nlayer=config["num_hidden_layers"],
            hs=config["hidden_size"],
            nh=config["num_attention_heads"],
            nkvh=config["num_key_value_heads"],
            dh=config["hidden_size"] // config["num_attention_heads"],
            di=config["intermediate_size"],
            maxseq=config["max_position_embeddings"],
            voc=config["vocab_size"],
            epsilon=config["rms_norm_eps"],
            theta=config["rope_theta"],
            end_token=self._end_token,
        )
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(byref(meta), int(device), 0)
        if not self._model:
            raise RuntimeError("Failed to create Qwen2 model")

        for file in sorted(model_path.glob("*.safetensors")):
            with safetensors.safe_open(file, framework="pt", device="cpu") as tensors:
                for name in tensors.keys():
                    tensor = tensors.get_tensor(name).contiguous()
                    shape = (c_size_t * tensor.ndim)(*tensor.shape)
                    array_dtype = {
                        torch.bfloat16: DataType.BF16,
                        torch.float16: DataType.F16,
                        torch.float32: DataType.F32,
                    }[tensor.dtype]
                    LIB_LLAISYS.llaisysQwen2ModelLoad(
                        self._model,
                        name.encode(),
                        shape,
                        tensor.ndim,
                        int(array_dtype),
                        c_void_p(tensor.data_ptr()),
                    )
                    if name == "model.embed_tokens.weight" and config.get("tie_word_embeddings", False):
                        LIB_LLAISYS.llaisysQwen2ModelLoad(
                            self._model,
                            b"lm_head.weight",
                            shape,
                            tensor.ndim,
                            int(array_dtype),
                            c_void_p(tensor.data_ptr()),
                        )

    def __del__(self):
        if getattr(self, "_model", None):
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        if top_k != 1:
            raise ValueError("Qwen2 currently supports argmax sampling only (top_k=1)")
        max_new_tokens = 128 if max_new_tokens is None else max_new_tokens
        outputs = [int(token) for token in inputs]
        pending = outputs
        for _ in range(max_new_tokens):
            token_array = (c_int64 * len(pending))(*pending)
            next_token = int(
                LIB_LLAISYS.llaisysQwen2ModelInfer(self._model, token_array, len(pending))
            )
            outputs.append(next_token)
            if next_token == self._end_token:
                break
            pending = [next_token]
        return outputs
