#include "swiglu_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

namespace {
template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t numel) {
    for (size_t i = 0; i < numel; ++i) {
        const float g = llaisys::utils::cast<float>(gate[i]);
        const float u = llaisys::utils::cast<float>(up[i]);
        out[i] = llaisys::utils::cast<T>(u * g / (1.0f + std::exp(-g)));
    }
}
}

namespace llaisys::ops::cpu {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return swiglu_(reinterpret_cast<float *>(out->data()), reinterpret_cast<const float *>(gate->data()), reinterpret_cast<const float *>(up->data()), out->numel());
    case LLAISYS_DTYPE_F16:
        return swiglu_(reinterpret_cast<fp16_t *>(out->data()), reinterpret_cast<const fp16_t *>(gate->data()), reinterpret_cast<const fp16_t *>(up->data()), out->numel());
    case LLAISYS_DTYPE_BF16:
        return swiglu_(reinterpret_cast<bf16_t *>(out->data()), reinterpret_cast<const bf16_t *>(gate->data()), reinterpret_cast<const bf16_t *>(up->data()), out->numel());
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
}
