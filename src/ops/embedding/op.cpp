#include "op.hpp"
#include "../../utils.hpp"
#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_ARGUMENT(index->ndim() == 1 && index->dtype() == LLAISYS_DTYPE_I64, "Embedding indices must be 1D int64.");
    CHECK_ARGUMENT(weight->ndim() == 2 && out->shape() == std::vector<size_t>({index->numel(), weight->shape()[1]}), "Embedding shapes are incompatible.");
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(), "Embedding tensors must be contiguous.");
    if (out->deviceType() == LLAISYS_DEVICE_CPU) return cpu::embedding(out, index, weight);
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
