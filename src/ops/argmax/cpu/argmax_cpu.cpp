#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

namespace {
template <typename T>
void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    size_t best = 0;
    float best_value = llaisys::utils::cast<float>(vals[0]);
    for (size_t i = 1; i < numel; ++i) {
        const float value = llaisys::utils::cast<float>(vals[i]);
        if (value > best_value) {
            best = i;
            best_value = value;
        }
    }
    *max_idx = static_cast<int64_t>(best);
    *max_val = vals[best];
}
}

namespace llaisys::ops::cpu {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    auto *idx = reinterpret_cast<int64_t *>(max_idx->data());
    switch (vals->dtype()) {
    case LLAISYS_DTYPE_F32:
        return argmax_(idx, reinterpret_cast<float *>(max_val->data()), reinterpret_cast<const float *>(vals->data()), vals->numel());
    case LLAISYS_DTYPE_F16:
        return argmax_(idx, reinterpret_cast<fp16_t *>(max_val->data()), reinterpret_cast<const fp16_t *>(vals->data()), vals->numel());
    case LLAISYS_DTYPE_BF16:
        return argmax_(idx, reinterpret_cast<bf16_t *>(max_val->data()), reinterpret_cast<const bf16_t *>(vals->data()), vals->numel());
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(vals->dtype());
    }
}
}
