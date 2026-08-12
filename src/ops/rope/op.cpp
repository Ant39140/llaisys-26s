#include "op.hpp"
#include "../../utils.hpp"
#include "cpu/rope_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "../nvidia/ops.cuh"
#endif

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_ARGUMENT(in->ndim() == 3 && in->shape()[2] % 2 == 0, "RoPE input must be 3D with even head dimension.");
    CHECK_ARGUMENT(pos_ids->ndim() == 1 && pos_ids->shape()[0] == in->shape()[0] && pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE positions must be matching 1D int64.");
    CHECK_ARGUMENT(theta > 0.0f, "RoPE theta must be positive.");
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(), "RoPE tensors must be contiguous.");
    if (out->deviceType() == LLAISYS_DEVICE_CPU) return cpu::rope(out, in, pos_ids, theta);
#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) return nvidia::rope(out, in, pos_ids, theta);
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
