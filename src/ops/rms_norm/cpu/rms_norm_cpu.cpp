#include "rms_norm_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

namespace {
template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t rows, size_t cols, float eps) {
    for (size_t row = 0; row < rows; ++row) {
        float square_sum = 0.0f;
        for (size_t col = 0; col < cols; ++col) {
            const float value = llaisys::utils::cast<float>(in[row * cols + col]);
            square_sum += value * value;
        }
        const float inv_rms = 1.0f / std::sqrt(square_sum / static_cast<float>(cols) + eps);
        for (size_t col = 0; col < cols; ++col) {
            const float value = llaisys::utils::cast<float>(in[row * cols + col]);
            const float scale = llaisys::utils::cast<float>(weight[col]);
            out[row * cols + col] = llaisys::utils::cast<T>(value * inv_rms * scale);
        }
    }
}
}

namespace llaisys::ops::cpu {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    const size_t cols = in->shape().back(), rows = in->numel() / cols;
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out->data()), reinterpret_cast<const float *>(in->data()), reinterpret_cast<const float *>(weight->data()), rows, cols, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<fp16_t *>(out->data()), reinterpret_cast<const fp16_t *>(in->data()), reinterpret_cast<const fp16_t *>(weight->data()), rows, cols, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<bf16_t *>(out->data()), reinterpret_cast<const bf16_t *>(in->data()), reinterpret_cast<const bf16_t *>(weight->data()), rows, cols, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
}
