#include "op.hpp"
#include "../../utils.hpp"
#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    CHECK_ARGUMENT(q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3 && attn_val->ndim() == 3, "Self-attention tensors must be 3D.");
    CHECK_ARGUMENT(k->shape()[0] == v->shape()[0] && k->shape()[1] == v->shape()[1], "Self-attention key/value shapes are incompatible.");
    CHECK_ARGUMENT(q->shape()[0] <= k->shape()[0] && q->shape()[1] % k->shape()[1] == 0 && q->shape()[2] == k->shape()[2], "Self-attention query/key shapes are incompatible.");
    CHECK_ARGUMENT(attn_val->shape() == std::vector<size_t>({q->shape()[0], q->shape()[1], v->shape()[2]}), "Self-attention output shape is incompatible.");
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(), "Self-attention tensors must be contiguous.");
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) return cpu::self_attention(attn_val, q, k, v, scale);
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
