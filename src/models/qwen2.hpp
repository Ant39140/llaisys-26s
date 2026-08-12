#pragma once

#include "../../include/llaisys/models/qwen2.h"
#include "../tensor/tensor.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace llaisys::models {
class Qwen2 {
public:
    Qwen2(LlaisysQwen2Meta meta, llaisysDeviceType_t device, int device_id);
    void load(const std::string &name, const std::vector<size_t> &shape, llaisysDataType_t dtype, const void *data);
    int64_t infer(const int64_t *token_ids, size_t ntoken);

private:
    tensor_t weight(const std::string &name) const;
    tensor_t linear(tensor_t input, tensor_t weight, tensor_t bias = nullptr) const;
    tensor_t rms_norm(tensor_t input, tensor_t weight) const;

    LlaisysQwen2Meta _meta;
    llaisysDeviceType_t _device;
    int _device_id;
    std::unordered_map<std::string, tensor_t> _weights;
    std::vector<tensor_t> _key_cache;
    std::vector<tensor_t> _value_cache;
    size_t _cache_len = 0;
};
}

struct LlaisysQwen2Model {
    llaisys::models::Qwen2 model;
};
