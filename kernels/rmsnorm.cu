// Naive RMSNorm CUDA kernel
//
//   y[i] = x[i] * rsqrt(mean(x^2, dim=-1) + eps) * weight[i]
//
// Layout: one thread block per row (one token's hidden vector).
// Each thread strides through `hidden_size / blockDim.x` elements.
// Sum-of-squares is accumulated in float32 regardless of input dtype.
// Reduction: warp-level via __shfl_xor_sync, then cross-warp via shared mem.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define BLOCK_SIZE 256
#define WARP_SIZE  32
#define MAX_WARPS  (BLOCK_SIZE / WARP_SIZE)

// ---------------------------------------------------------------------------
// dtype <-> float helpers
// ---------------------------------------------------------------------------

template <typename T> __device__ __forceinline__ float to_float(T v);
template <> __device__ __forceinline__ float to_float<float>(float v)               { return v; }
template <> __device__ __forceinline__ float to_float<__half>(__half v)             { return __half2float(v); }
template <> __device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v){ return __bfloat162float(v); }

template <typename T> __device__ __forceinline__ T from_float(float v);
template <> __device__ __forceinline__ float        from_float<float>(float v)        { return v; }
template <> __device__ __forceinline__ __half       from_float<__half>(float v)       { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v){ return __float2bfloat16(v); }

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

__device__ __forceinline__ float warp_reduce_sum(float v) {
    // Butterfly reduction across a warp.  __shfl_xor_sync is symmetric so every
    // lane in the warp ends up with the full sum (no broadcast needed).
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

// ---------------------------------------------------------------------------
// Kernel — one block per row
// ---------------------------------------------------------------------------

template <typename T>
__global__ void rmsnorm_kernel(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    T* __restrict__ y,
    int hidden_size,
    float eps)
{
    const int row     = blockIdx.x;
    const int tid     = threadIdx.x;
    const int lane    = tid & (WARP_SIZE - 1);
    const int warp_id = tid >> 5;                 // tid / 32
    const int n_warps = blockDim.x / WARP_SIZE;

    const T* x_row = x + (size_t)row * hidden_size;
    T*       y_row = y + (size_t)row * hidden_size;

    // 1) Per-thread partial sum-of-squares (accumulated in float).
    float sumsq = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        const float xv = to_float<T>(x_row[i]);
        sumsq += xv * xv;
    }

    // 2) Warp-level reduction.
    sumsq = warp_reduce_sum(sumsq);

    // 3) Cross-warp reduction via shared memory.  warp_sums[0] is reused at the
    //    end to broadcast the final scale to every thread in the block.
    __shared__ float warp_sums[MAX_WARPS];
    if (lane == 0) warp_sums[warp_id] = sumsq;
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane < n_warps) ? warp_sums[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) {
            warp_sums[0] = rsqrtf(v / static_cast<float>(hidden_size) + eps);
        }
    }
    __syncthreads();

    const float scale = warp_sums[0];

    // 4) Apply scale * weight.
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        const float xv = to_float<T>(x_row[i]);
        const float wv = to_float<T>(weight[i]);
        y_row[i] = from_float<T>(xv * scale * wv);
    }
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------

torch::Tensor rmsnorm_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    double eps)
{
    TORCH_CHECK(x.is_cuda(),       "rmsnorm_cuda: x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(),  "rmsnorm_cuda: weight must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(),       "rmsnorm_cuda: x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(),  "rmsnorm_cuda: weight must be contiguous");
    TORCH_CHECK(weight.dim() == 1, "rmsnorm_cuda: weight must be 1-D");
    TORCH_CHECK(x.dim() >= 1,      "rmsnorm_cuda: x must have at least 1 dim");
    TORCH_CHECK(x.scalar_type() == weight.scalar_type(),
                "rmsnorm_cuda: x and weight must share dtype");
    TORCH_CHECK(x.size(-1) == weight.size(0),
                "rmsnorm_cuda: x.size(-1) must equal weight.size(0)");

    auto y = torch::empty_like(x);

    const int hidden = static_cast<int>(x.size(-1));
    const int64_t n_rows = x.numel() / hidden;
    if (n_rows == 0) return y;

    TORCH_CHECK(n_rows <= (int64_t)std::numeric_limits<int>::max(),
                "rmsnorm_cuda: too many rows for a 1-D grid");

    const dim3 grid(static_cast<unsigned>(n_rows));
    const dim3 block(BLOCK_SIZE);
    auto stream = at::cuda::getCurrentCUDAStream();

    switch (x.scalar_type()) {
        case at::ScalarType::Float:
            rmsnorm_kernel<float><<<grid, block, 0, stream>>>(
                x.data_ptr<float>(),
                weight.data_ptr<float>(),
                y.data_ptr<float>(),
                hidden, static_cast<float>(eps));
            break;
        case at::ScalarType::Half:
            rmsnorm_kernel<__half><<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
                hidden, static_cast<float>(eps));
            break;
        case at::ScalarType::BFloat16:
            rmsnorm_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
                hidden, static_cast<float>(eps));
            break;
        default:
            TORCH_CHECK(false,
                "rmsnorm_cuda: unsupported dtype ", x.scalar_type(),
                " (expected float32/float16/bfloat16)");
    }

    AT_CUDA_CHECK(cudaGetLastError());
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_cuda", &rmsnorm_cuda,
          "Naive RMSNorm CUDA kernel",
          pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("eps") = 1e-6);
}
