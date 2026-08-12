#ifndef LLAISYS_MODELS_QWEN2_H
#define LLAISYS_MODELS_QWEN2_H

#include "../tensor.h"

__C {
    struct LlaisysQwen2Meta {
        llaisysDataType_t dtype;
        size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
        float epsilon, theta;
        int64_t end_token;
    };

    struct LlaisysQwen2Model;

    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
        const struct LlaisysQwen2Meta *meta,
        llaisysDeviceType_t device,
        int device_id);

    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model);

    __export void llaisysQwen2ModelLoad(
        struct LlaisysQwen2Model *model,
        const char *name,
        const size_t *shape,
        size_t ndim,
        llaisysDataType_t dtype,
        const void *data);

    __export int64_t llaisysQwen2ModelInfer(
        struct LlaisysQwen2Model *model,
        const int64_t *token_ids,
        size_t ntoken);
}
#endif // LLAISYS_MODELS_QWEN2_H
