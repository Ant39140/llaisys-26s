#include "../runtime_api.hpp"

#include <cuda_runtime.h>
#include <stdexcept>

namespace llaisys::device::nvidia {
namespace runtime_api {
namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

cudaMemcpyKind copyKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H: return cudaMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D: return cudaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H: return cudaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D: return cudaMemcpyDeviceToDevice;
    default: throw std::invalid_argument("Invalid CUDA memcpy kind");
    }
}
} // namespace

int getDeviceCount() {
    int count = 0;
    check(cudaGetDeviceCount(&count));
    return count;
}

void setDevice(int device) { check(cudaSetDevice(device)); }
void deviceSynchronize() { check(cudaDeviceSynchronize()); }

llaisysStream_t createStream() {
    cudaStream_t stream;
    check(cudaStreamCreate(&stream));
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    if (stream) cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream));
}

void streamSynchronize(llaisysStream_t stream) {
    check(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    check(cudaMalloc(&ptr, size));
    return ptr;
}

void freeDevice(void *ptr) {
    if (ptr) cudaFree(ptr);
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    check(cudaMallocHost(&ptr, size));
    return ptr;
}

void freeHost(void *ptr) {
    if (ptr) cudaFreeHost(ptr);
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    check(cudaMemcpy(dst, src, size, copyKind(kind)));
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    check(cudaMemcpyAsync(dst, src, size, copyKind(kind), reinterpret_cast<cudaStream_t>(stream)));
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount, &setDevice, &deviceSynchronize,
    &createStream, &destroyStream, &streamSynchronize,
    &mallocDevice, &freeDevice, &mallocHost, &freeHost,
    &memcpySync, &memcpyAsync};
} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() { return &runtime_api::RUNTIME_API; }
} // namespace llaisys::device::nvidia
