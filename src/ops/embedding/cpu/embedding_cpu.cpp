#include "embedding_cpu.hpp"
#include <cstring>

namespace llaisys::ops::cpu {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    const auto *indices = reinterpret_cast<const int64_t *>(index->data());
    const size_t row_bytes = weight->shape()[1] * weight->elementSize();
    for (size_t row = 0; row < index->numel(); ++row) {
        std::memcpy(out->data() + row * row_bytes, weight->data() + static_cast<size_t>(indices[row]) * row_bytes, row_bytes);
    }
}
}
