#include "linear_cpu.hpp"
#include "../../../utils.hpp"

namespace {
template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, size_t rows, size_t in_features, size_t out_features) {
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < out_features; ++col) {
            float sum = bias ? llaisys::utils::cast<float>(bias[col]) : 0.0f;
            for (size_t k = 0; k < in_features; ++k) {
                sum += llaisys::utils::cast<float>(in[row * in_features + k]) * llaisys::utils::cast<float>(weight[col * in_features + k]);
            }
            out[row * out_features + col] = llaisys::utils::cast<T>(sum);
        }
    }
}
}

namespace llaisys::ops::cpu {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    const size_t rows = in->shape()[0], in_features = in->shape()[1], out_features = weight->shape()[0];
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out->data()), reinterpret_cast<const float *>(in->data()), reinterpret_cast<const float *>(weight->data()), bias ? reinterpret_cast<const float *>(bias->data()) : nullptr, rows, in_features, out_features);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<fp16_t *>(out->data()), reinterpret_cast<const fp16_t *>(in->data()), reinterpret_cast<const fp16_t *>(weight->data()), bias ? reinterpret_cast<const fp16_t *>(bias->data()) : nullptr, rows, in_features, out_features);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<bf16_t *>(out->data()), reinterpret_cast<const bf16_t *>(in->data()), reinterpret_cast<const bf16_t *>(weight->data()), bias ? reinterpret_cast<const bf16_t *>(bias->data()) : nullptr, rows, in_features, out_features);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
}
