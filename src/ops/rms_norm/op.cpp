#include "op.hpp"
#include "../../utils.hpp"
#include "cpu/rms_norm_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "../nvidia/ops.cuh"
#endif

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    CHECK_ARGUMENT(in->ndim() >= 1 && weight->ndim() == 1 && weight->shape()[0] == in->shape().back(), "RMSNorm weight shape is incompatible.");
    CHECK_ARGUMENT(eps >= 0.0f, "RMSNorm epsilon must be non-negative.");
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "RMSNorm tensors must be contiguous.");
    if (out->deviceType() == LLAISYS_DEVICE_CPU) return cpu::rms_norm(out, in, weight, eps);
#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) return nvidia::rms_norm(out, in, weight, eps);
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
