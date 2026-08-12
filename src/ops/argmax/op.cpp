#include "op.hpp"

#include "../../utils.hpp"
#include "cpu/argmax_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "../nvidia/ops.cuh"
#endif

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    CHECK_ARGUMENT(vals->ndim() == 1 && vals->numel() > 0, "Argmax input must be a non-empty 1D tensor.");
    CHECK_ARGUMENT(max_idx->numel() == 1 && max_val->numel() == 1, "Argmax outputs must contain one element.");
    CHECK_ARGUMENT(max_idx->dtype() == LLAISYS_DTYPE_I64, "Argmax index output must be int64.");
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    ASSERT(max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(), "Argmax tensors must be contiguous.");
    if (vals->deviceType() == LLAISYS_DEVICE_CPU) return cpu::argmax(max_idx, max_val, vals);
#ifdef ENABLE_NVIDIA_API
    if (vals->deviceType() == LLAISYS_DEVICE_NVIDIA) return nvidia::argmax(max_idx, max_val, vals);
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
