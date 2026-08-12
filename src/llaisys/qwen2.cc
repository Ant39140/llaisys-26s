#include "llaisys/models/qwen2.h"

#include "../models/qwen2.hpp"

#include <string>
#include <vector>

__C {
LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int device_id) {
    return new LlaisysQwen2Model{llaisys::models::Qwen2(*meta, device, device_id)};
}

void llaisysQwen2ModelDestroy(LlaisysQwen2Model *model) {
    delete model;
}

void llaisysQwen2ModelLoad(LlaisysQwen2Model *model, const char *name, const size_t *shape, size_t ndim, llaisysDataType_t dtype, const void *data) {
    model->model.load(name, std::vector<size_t>(shape, shape + ndim), dtype, data);
}

int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model *model, const int64_t *token_ids, size_t ntoken) {
    return model->model.infer(token_ids, ntoken);
}
}
