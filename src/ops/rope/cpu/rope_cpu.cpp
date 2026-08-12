#include "rope_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

namespace {
template <typename T>
void rope_(T *out, const T *in, const int64_t *positions, size_t seq_len, size_t heads, size_t head_dim, float theta) {
    const size_t half = head_dim / 2;
    for (size_t seq = 0; seq < seq_len; ++seq) {
        for (size_t head = 0; head < heads; ++head) {
            const size_t base = (seq * heads + head) * head_dim;
            for (size_t j = 0; j < half; ++j) {
                const float angle = static_cast<float>(positions[seq]) / std::pow(theta, 2.0f * static_cast<float>(j) / static_cast<float>(head_dim));
                const float a = llaisys::utils::cast<float>(in[base + j]);
                const float b = llaisys::utils::cast<float>(in[base + half + j]);
                const float sin_value = std::sin(angle), cos_value = std::cos(angle);
                out[base + j] = llaisys::utils::cast<T>(a * cos_value - b * sin_value);
                out[base + half + j] = llaisys::utils::cast<T>(b * cos_value + a * sin_value);
            }
        }
    }
}
}

namespace llaisys::ops::cpu {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    const auto *positions = reinterpret_cast<const int64_t *>(pos_ids->data());
    const size_t seq_len = in->shape()[0], heads = in->shape()[1], head_dim = in->shape()[2];
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out->data()), reinterpret_cast<const float *>(in->data()), positions, seq_len, heads, head_dim, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<fp16_t *>(out->data()), reinterpret_cast<const fp16_t *>(in->data()), positions, seq_len, heads, head_dim, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<bf16_t *>(out->data()), reinterpret_cast<const bf16_t *>(in->data()), positions, seq_len, heads, head_dim, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
}
