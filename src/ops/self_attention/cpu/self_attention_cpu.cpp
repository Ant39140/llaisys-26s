#include "self_attention_cpu.hpp"
#include "../../../utils.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace {
template <typename T>
void attention_(T *out, const T *q, const T *k, const T *v, size_t qlen, size_t kvlen, size_t heads, size_t kvheads, size_t qdim, size_t vdim, float scale) {
    std::vector<float> scores(kvlen);
    const size_t group = heads / kvheads;
    for (size_t qi = 0; qi < qlen; ++qi) {
        const size_t allowed = kvlen - qlen + qi + 1;
        for (size_t head = 0; head < heads; ++head) {
            const size_t kvhead = head / group;
            float maximum = -INFINITY;
            for (size_t kj = 0; kj < allowed; ++kj) {
                float score = 0.0f;
                for (size_t d = 0; d < qdim; ++d) {
                    score += llaisys::utils::cast<float>(q[(qi * heads + head) * qdim + d])
                           * llaisys::utils::cast<float>(k[(kj * kvheads + kvhead) * qdim + d]);
                }
                scores[kj] = score * scale;
                maximum = std::max(maximum, scores[kj]);
            }
            float denominator = 0.0f;
            for (size_t kj = 0; kj < allowed; ++kj) {
                scores[kj] = std::exp(scores[kj] - maximum);
                denominator += scores[kj];
            }
            for (size_t d = 0; d < vdim; ++d) {
                float value = 0.0f;
                for (size_t kj = 0; kj < allowed; ++kj) {
                    value += scores[kj] / denominator * llaisys::utils::cast<float>(v[(kj * kvheads + kvhead) * vdim + d]);
                }
                out[(qi * heads + head) * vdim + d] = llaisys::utils::cast<T>(value);
            }
        }
    }
}
}

namespace llaisys::ops::cpu {
void self_attention(tensor_t out, tensor_t q, tensor_t k, tensor_t v, float scale) {
    const size_t qlen = q->shape()[0], kvlen = k->shape()[0], heads = q->shape()[1], kvheads = k->shape()[1];
    const size_t qdim = q->shape()[2], vdim = v->shape()[2];
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return attention_(reinterpret_cast<float *>(out->data()), reinterpret_cast<const float *>(q->data()), reinterpret_cast<const float *>(k->data()), reinterpret_cast<const float *>(v->data()), qlen, kvlen, heads, kvheads, qdim, vdim, scale);
    case LLAISYS_DTYPE_F16:
        return attention_(reinterpret_cast<fp16_t *>(out->data()), reinterpret_cast<const fp16_t *>(q->data()), reinterpret_cast<const fp16_t *>(k->data()), reinterpret_cast<const fp16_t *>(v->data()), qlen, kvlen, heads, kvheads, qdim, vdim, scale);
    case LLAISYS_DTYPE_BF16:
        return attention_(reinterpret_cast<bf16_t *>(out->data()), reinterpret_cast<const bf16_t *>(q->data()), reinterpret_cast<const bf16_t *>(k->data()), reinterpret_cast<const bf16_t *>(v->data()), qlen, kvlen, heads, kvheads, qdim, vdim, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
}
