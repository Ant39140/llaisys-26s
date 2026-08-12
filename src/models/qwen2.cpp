#include "qwen2.hpp"

#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"
#include "../utils.hpp"

#include <cmath>
#include <cstring>

namespace llaisys::models {
Qwen2::Qwen2(LlaisysQwen2Meta meta, llaisysDeviceType_t device, int device_id)
    : _meta(meta), _device(device), _device_id(device_id), _key_cache(meta.nlayer), _value_cache(meta.nlayer) {
}

void Qwen2::load(const std::string &name, const std::vector<size_t> &shape, llaisysDataType_t dtype, const void *data) {
    auto tensor = Tensor::create(shape, dtype, _device, _device_id);
    tensor->load(data);
    _weights[name] = std::move(tensor);
}

tensor_t Qwen2::weight(const std::string &name) const {
    const auto it = _weights.find(name);
    CHECK_ARGUMENT(it != _weights.end(), "Missing Qwen2 weight: " + name);
    return it->second;
}

tensor_t Qwen2::linear(tensor_t input, tensor_t matrix, tensor_t bias) const {
    auto output = Tensor::create({input->shape()[0], matrix->shape()[0]}, input->dtype(), _device, _device_id);
    ops::linear(output, input, matrix, bias);
    return output;
}

tensor_t Qwen2::rms_norm(tensor_t input, tensor_t scale) const {
    auto output = Tensor::create(input->shape(), input->dtype(), _device, _device_id);
    ops::rms_norm(output, input, scale, _meta.epsilon);
    return output;
}

int64_t Qwen2::infer(const int64_t *token_ids, size_t ntoken) {
    CHECK_ARGUMENT(ntoken > 0, "Qwen2 inference requires at least one token.");
    CHECK_ARGUMENT(_cache_len + ntoken <= _meta.maxseq, "Qwen2 context exceeds maximum sequence length.");

    auto ids = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, _device, _device_id);
    ids->load(token_ids);
    auto hidden = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device, _device_id);
    ops::embedding(hidden, ids, weight("model.embed_tokens.weight"));

    std::vector<int64_t> positions(ntoken);
    for (size_t i = 0; i < ntoken; ++i) positions[i] = static_cast<int64_t>(_cache_len + i);
    auto pos_ids = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, _device, _device_id);
    pos_ids->load(positions.data());

    for (size_t layer = 0; layer < _meta.nlayer; ++layer) {
        const std::string prefix = "model.layers." + std::to_string(layer) + ".";
        auto normed = rms_norm(hidden, weight(prefix + "input_layernorm.weight"));
        auto q = linear(normed, weight(prefix + "self_attn.q_proj.weight"), weight(prefix + "self_attn.q_proj.bias"))
                     ->view({ntoken, _meta.nh, _meta.dh});
        auto k_new = linear(normed, weight(prefix + "self_attn.k_proj.weight"), weight(prefix + "self_attn.k_proj.bias"))
                         ->view({ntoken, _meta.nkvh, _meta.dh});
        auto v_new = linear(normed, weight(prefix + "self_attn.v_proj.weight"), weight(prefix + "self_attn.v_proj.bias"))
                         ->view({ntoken, _meta.nkvh, _meta.dh});
        auto q_rot = Tensor::create(q->shape(), q->dtype(), _device, _device_id);
        auto k_rot = Tensor::create(k_new->shape(), k_new->dtype(), _device, _device_id);
        ops::rope(q_rot, q, pos_ids, _meta.theta);
        ops::rope(k_rot, k_new, pos_ids, _meta.theta);

        auto k_cache = Tensor::create({_cache_len + ntoken, _meta.nkvh, _meta.dh}, _meta.dtype, _device, _device_id);
        auto v_cache = Tensor::create({_cache_len + ntoken, _meta.nkvh, _meta.dh}, _meta.dtype, _device, _device_id);
        const size_t old_bytes = _cache_len * _meta.nkvh * _meta.dh * k_cache->elementSize();
        if (_cache_len > 0) {
            std::memcpy(k_cache->data(), _key_cache[layer]->data(), old_bytes);
            std::memcpy(v_cache->data(), _value_cache[layer]->data(), old_bytes);
        }
        std::memcpy(k_cache->data() + old_bytes, k_rot->data(), k_rot->numel() * k_rot->elementSize());
        std::memcpy(v_cache->data() + old_bytes, v_new->data(), v_new->numel() * v_new->elementSize());
        _key_cache[layer] = k_cache;
        _value_cache[layer] = v_cache;

        auto attn = Tensor::create({ntoken, _meta.nh, _meta.dh}, _meta.dtype, _device, _device_id);
        ops::self_attention(attn, q_rot, k_cache, v_cache, 1.0f / std::sqrt(static_cast<float>(_meta.dh)));
        auto projected = linear(attn->view({ntoken, _meta.hs}), weight(prefix + "self_attn.o_proj.weight"));
        auto residual = Tensor::create(hidden->shape(), hidden->dtype(), _device, _device_id);
        ops::add(residual, hidden, projected);

        normed = rms_norm(residual, weight(prefix + "post_attention_layernorm.weight"));
        auto gate = linear(normed, weight(prefix + "mlp.gate_proj.weight"));
        auto up = linear(normed, weight(prefix + "mlp.up_proj.weight"));
        auto activated = Tensor::create(gate->shape(), gate->dtype(), _device, _device_id);
        ops::swiglu(activated, gate, up);
        auto down = linear(activated, weight(prefix + "mlp.down_proj.weight"));
        hidden = Tensor::create(residual->shape(), residual->dtype(), _device, _device_id);
        ops::add(hidden, residual, down);
    }

    _cache_len += ntoken;
    auto last = hidden->slice(0, ntoken - 1, ntoken);
    auto normalized = rms_norm(last, weight("model.norm.weight"));
    auto logits = linear(normalized, weight("lm_head.weight"));
    auto max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, _device, _device_id);
    auto max_val = Tensor::create({1}, logits->dtype(), _device, _device_id);
    ops::argmax(max_idx, max_val, logits->view({_meta.voc}));
    return *reinterpret_cast<const int64_t *>(max_idx->data());
}
} // namespace llaisys::models
