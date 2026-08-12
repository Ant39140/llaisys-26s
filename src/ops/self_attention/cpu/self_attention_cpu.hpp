#pragma once
#include "../../../tensor/tensor.hpp"
namespace llaisys::ops::cpu {
void self_attention(tensor_t out, tensor_t q, tensor_t k, tensor_t v, float scale);
}
