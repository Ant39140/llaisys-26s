#include "ops.cuh"
#include "../../utils.hpp"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace llaisys::ops::nvidia {
namespace {
void check(cudaError_t status) { if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status)); }

template <typename T> __device__ float toFloat(T x) { return static_cast<float>(x); }
template <> __device__ float toFloat<__half>(__half x) { return __half2float(x); }
template <> __device__ float toFloat<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }
template <typename T> __device__ T fromFloat(float x) { return static_cast<T>(x); }
template <> __device__ __half fromFloat<__half>(float x) { return __float2half(x); }
template <> __device__ __nv_bfloat16 fromFloat<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

template <typename T> __global__ void addKernel(T *out, const T *a, const T *b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fromFloat<T>(toFloat(a[i]) + toFloat(b[i]));
}
template <typename T> __global__ void swigluKernel(T *out, const T *gate, const T *up, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float g = toFloat(gate[i]); out[i] = fromFloat<T>(toFloat(up[i]) * g / (1.0f + expf(-g))); }
}
template <typename T> __global__ void embeddingKernel(T *out, const int64_t *index, const T *weight, size_t cols, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = weight[static_cast<size_t>(index[i / cols]) * cols + i % cols];
}
template <typename T> __global__ void linearKernel(T *out, const T *in, const T *weight, const T *bias, size_t rows, size_t inf, size_t outf) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * outf) { size_t r=i/outf,c=i%outf; float s=bias?toFloat(bias[c]):0.0f; for(size_t k=0;k<inf;++k)s+=toFloat(in[r*inf+k])*toFloat(weight[c*inf+k]); out[i]=fromFloat<T>(s); }
}
template <typename T> __global__ void rmsKernel(T *out,const T *in,const T *weight,size_t rows,size_t cols,float eps){
    size_t r=blockIdx.x*blockDim.x+threadIdx.x; if(r<rows){float s=0;for(size_t c=0;c<cols;++c){float x=toFloat(in[r*cols+c]);s+=x*x;}float inv=rsqrtf(s/cols+eps);for(size_t c=0;c<cols;++c)out[r*cols+c]=fromFloat<T>(toFloat(in[r*cols+c])*inv*toFloat(weight[c]));}}
template <typename T> __global__ void ropeKernel(T*out,const T*in,const int64_t*pos,size_t n,size_t heads,size_t dim,float theta){size_t i=blockIdx.x*blockDim.x+threadIdx.x;size_t half=dim/2;if(i<n*heads*half){size_t j=i%half;size_t h=(i/half)%heads;size_t s=i/(half*heads);size_t base=(s*heads+h)*dim;float a=toFloat(in[base+j]),b=toFloat(in[base+half+j]),ang=pos[s]/powf(theta,2.0f*j/dim);float sn=sinf(ang),cs=cosf(ang);out[base+j]=fromFloat<T>(a*cs-b*sn);out[base+half+j]=fromFloat<T>(b*cs+a*sn);}}
template <typename T> __global__ void argmaxKernel(int64_t*idx,T*val,const T*x,size_t n){if(threadIdx.x==0&&blockIdx.x==0){size_t best=0;float m=toFloat(x[0]);for(size_t i=1;i<n;++i){float v=toFloat(x[i]);if(v>m){m=v;best=i;}}*idx=best;*val=x[best];}}
template <typename T> __global__ void attentionKernel(T*out,const T*q,const T*k,const T*v,size_t ql,size_t kl,size_t nh,size_t kh,size_t d,size_t dv,float scale){size_t z=blockIdx.x*blockDim.x+threadIdx.x;if(z<ql*nh*dv){size_t vd=z%dv;size_t h=(z/dv)%nh;size_t qi=z/(nh*dv);size_t kth=h/(nh/kh),allowed=kl-ql+qi+1;float mx=-INFINITY;for(size_t j=0;j<allowed;++j){float s=0;for(size_t x=0;x<d;++x)s+=toFloat(q[(qi*nh+h)*d+x])*toFloat(k[(j*kh+kth)*d+x]);mx=fmaxf(mx,s*scale);}float den=0,res=0;for(size_t j=0;j<allowed;++j){float s=0;for(size_t x=0;x<d;++x)s+=toFloat(q[(qi*nh+h)*d+x])*toFloat(k[(j*kh+kth)*d+x]);float e=expf(s*scale-mx);den+=e;res+=e*toFloat(v[(j*kh+kth)*dv+vd]);}out[z]=fromFloat<T>(res/den);}}

#define DISPATCH(DT, ...) do { switch (DT) { \
    case LLAISYS_DTYPE_F32: { using T = float; __VA_ARGS__; break; } \
    case LLAISYS_DTYPE_F16: { using T = __half; __VA_ARGS__; break; } \
    case LLAISYS_DTYPE_BF16: { using T = __nv_bfloat16; __VA_ARGS__; break; } \
    default: EXCEPTION_UNSUPPORTED_DATATYPE(DT); } check(cudaGetLastError()); } while (0)
constexpr int B=256; size_t blocks(size_t n){return (n+B-1)/B;}
}

void add(tensor_t o,tensor_t a,tensor_t b){DISPATCH(o->dtype(),addKernel<<<blocks(o->numel()),B>>>((T*)o->data(),(T*)a->data(),(T*)b->data(),o->numel()));}
void swiglu(tensor_t o,tensor_t g,tensor_t u){DISPATCH(o->dtype(),swigluKernel<<<blocks(o->numel()),B>>>((T*)o->data(),(T*)g->data(),(T*)u->data(),o->numel()));}
void embedding(tensor_t o,tensor_t i,tensor_t w){DISPATCH(o->dtype(),embeddingKernel<<<blocks(o->numel()),B>>>((T*)o->data(),(int64_t*)i->data(),(T*)w->data(),w->shape()[1],o->numel()));}
void linear(tensor_t o,tensor_t i,tensor_t w,tensor_t b){DISPATCH(o->dtype(),linearKernel<<<blocks(o->numel()),B>>>((T*)o->data(),(T*)i->data(),(T*)w->data(),b?(T*)b->data():nullptr,i->shape()[0],i->shape()[1],w->shape()[0]));}
void rms_norm(tensor_t o,tensor_t i,tensor_t w,float e){size_t c=i->shape().back(),r=i->numel()/c;DISPATCH(o->dtype(),rmsKernel<<<blocks(r),B>>>((T*)o->data(),(T*)i->data(),(T*)w->data(),r,c,e));}
void rope(tensor_t o,tensor_t i,tensor_t p,float t){size_t n=i->shape()[0],h=i->shape()[1],d=i->shape()[2];DISPATCH(o->dtype(),ropeKernel<<<blocks(n*h*d/2),B>>>((T*)o->data(),(T*)i->data(),(int64_t*)p->data(),n,h,d,t));}
void argmax(tensor_t i,tensor_t v,tensor_t x){DISPATCH(x->dtype(),argmaxKernel<<<1,1>>>((int64_t*)i->data(),(T*)v->data(),(T*)x->data(),x->numel()));}
void self_attention(tensor_t o,tensor_t q,tensor_t k,tensor_t v,float s){size_t n=o->numel();DISPATCH(o->dtype(),attentionKernel<<<blocks(n),B>>>((T*)o->data(),(T*)q->data(),(T*)k->data(),(T*)v->data(),q->shape()[0],k->shape()[0],q->shape()[1],k->shape()[1],q->shape()[2],v->shape()[2],s));}
#undef DISPATCH
}
