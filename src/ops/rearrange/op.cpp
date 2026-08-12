#include "op.hpp"

#include "../../utils.hpp"
#include <cstring>

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    ASSERT(out->isContiguous(), "Rearrange output must be contiguous.");
    if (out->deviceType() != LLAISYS_DEVICE_CPU) {
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
    const size_t item_size = in->elementSize();
    for (size_t linear = 0; linear < in->numel(); ++linear) {
        size_t remaining = linear;
        ptrdiff_t source_index = 0;
        for (size_t dim = in->ndim(); dim-- > 0;) {
            source_index += static_cast<ptrdiff_t>(remaining % in->shape()[dim]) * in->strides()[dim];
            remaining /= in->shape()[dim];
        }
        std::memcpy(out->data() + linear * item_size, in->data() + source_index * item_size, item_size);
    }
}
} // namespace llaisys::ops
