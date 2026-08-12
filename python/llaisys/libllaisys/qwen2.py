import ctypes

from .llaisys_types import llaisysDataType_t, llaisysDeviceType_t


class LlaisysQwen2Meta(ctypes.Structure):
    _fields_ = [
        ("dtype", llaisysDataType_t),
        ("nlayer", ctypes.c_size_t),
        ("hs", ctypes.c_size_t),
        ("nh", ctypes.c_size_t),
        ("nkvh", ctypes.c_size_t),
        ("dh", ctypes.c_size_t),
        ("di", ctypes.c_size_t),
        ("maxseq", ctypes.c_size_t),
        ("voc", ctypes.c_size_t),
        ("epsilon", ctypes.c_float),
        ("theta", ctypes.c_float),
        ("end_token", ctypes.c_int64),
    ]


LlaisysQwen2Model = ctypes.c_void_p


def load_qwen2(lib):
    lib.llaisysQwen2ModelCreate.argtypes = [
        ctypes.POINTER(LlaisysQwen2Meta), llaisysDeviceType_t, ctypes.c_int
    ]
    lib.llaisysQwen2ModelCreate.restype = LlaisysQwen2Model
    lib.llaisysQwen2ModelDestroy.argtypes = [LlaisysQwen2Model]
    lib.llaisysQwen2ModelDestroy.restype = None
    lib.llaisysQwen2ModelLoad.argtypes = [
        LlaisysQwen2Model,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        llaisysDataType_t,
        ctypes.c_void_p,
    ]
    lib.llaisysQwen2ModelLoad.restype = None
    lib.llaisysQwen2ModelInfer.argtypes = [
        LlaisysQwen2Model, ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t
    ]
    lib.llaisysQwen2ModelInfer.restype = ctypes.c_int64
