#pragma once

// Shared CUDA utilities for RMSNorm-family kernels.
//
// The project currently builds each kernel as its own JIT extension, so this
// header keeps templated device code and the RMSNorm launcher in one place
// without introducing a separate link step.
//
// RMSNorm layout:
//   - one CUDA block handles one logical row, i.e. one token's hidden vector
//   - each thread walks the hidden dimension with a strided loop
//   - sum-of-squares is accumulated in float32 for fp32/fp16/bf16 inputs
//   - reduction is warp-level shuffle first, then cross-warp shared memory

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <limits>

#define RMSNORM_BLOCK_SIZE 256
#define RMSNORM_WARP_SIZE  32
#define RMSNORM_MAX_WARPS  (RMSNORM_BLOCK_SIZE / RMSNORM_WARP_SIZE)

// ---------------------------------------------------------------------------
// dtype <-> float helpers
// ---------------------------------------------------------------------------

template <typename T> __device__ __forceinline__ float to_float(T v);
template <> __device__ __forceinline__ float to_float<float>(float v) {
    return v;
}
template <> __device__ __forceinline__ float to_float<__half>(__half v) {
    return __half2float(v);
}
template <> __device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <typename T> __device__ __forceinline__ T from_float(float v);
template <> __device__ __forceinline__ float from_float<float>(float v) {
    return v;
}
template <> __device__ __forceinline__ __half from_float<__half>(float v) {
    return __float2half(v);
}
template <> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

__device__ __forceinline__ float warp_reduce_sum(float v) {
    // Butterfly reduction across a warp. __shfl_xor_sync is symmetric, so
    // every lane ends with the full warp sum and no separate broadcast is
    // needed before the cross-warp stage.
    //
    // 0xffffffffu is the full-warp participation mask: all 32 lanes take part.
    // At each step, lane i reads the register value from lane (i XOR offset)
    // and adds it locally. Offsets 16, 8, 4, 2, 1 cover the whole warp.
    #pragma unroll
    for (int offset = RMSNORM_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

// ---------------------------------------------------------------------------
// RMSNorm kernel - one block per row
// ---------------------------------------------------------------------------
//
// Computes:
//   y[row, i] = x[row, i] * rsqrt(mean(x[row, :]^2) + eps) * weight[i]
//
// This is intentionally a simple forward kernel. It is reused by
// rmsnorm.cu directly and by fused RMSNorm-family entry points as their
// normalization prologue.

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
    const int lane    = tid & (RMSNORM_WARP_SIZE - 1);
    const int warp_id = tid >> 5;  // tid / 32
    const int n_warps = blockDim.x / RMSNORM_WARP_SIZE;

    const T* x_row = x + (size_t)row * hidden_size;
    T*       y_row = y + (size_t)row * hidden_size;

    // 1) Each thread accumulates a partial sum-of-squares over a strided slice
    //    of the hidden dimension. The accumulator stays float32 even when the
    //    input/output tensor is fp16 or bf16.
    float sumsq = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        const float xv = to_float<T>(x_row[i]);
        sumsq += xv * xv;
    }

    // 2) Reduce within each warp. After this call every lane in a warp holds
    //    that warp's partial sum.
    sumsq = warp_reduce_sum(sumsq);

    // 3) Store one partial sum per warp, then let warp 0 reduce those partials.
    //    warp_sums[0] is reused afterward to broadcast the final RMS scale.
    __shared__ float warp_sums[RMSNORM_MAX_WARPS];
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

    // 4) Apply the row scale and learned RMSNorm weight, then cast back to the
    //    original dtype for storage.
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        const float xv = to_float<T>(x_row[i]);
        const float wv = to_float<T>(weight[i]);
        y_row[i] = from_float<T>(xv * scale * wv);
    }
}

// ---------------------------------------------------------------------------
// Host-side validation and launcher helpers
// ---------------------------------------------------------------------------

inline void check_rmsnorm_inputs(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const char* op_name)
{
    // Keep validation close to the shared launcher so all RMSNorm-family entry
    // points fail with the same messages for the same bad inputs.
    TORCH_CHECK(x.is_cuda(),      op_name, ": x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), op_name, ": weight must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(),      op_name, ": x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), op_name, ": weight must be contiguous");
    TORCH_CHECK(weight.dim() == 1, op_name, ": weight must be 1-D");
    TORCH_CHECK(x.dim() >= 1,     op_name, ": x must have at least 1 dim");
    TORCH_CHECK(x.scalar_type() == weight.scalar_type(),
                op_name, ": x and weight must share dtype");
    TORCH_CHECK(x.size(-1) > 0,
                op_name, ": x.size(-1) must be positive");
    TORCH_CHECK(x.size(-1) == weight.size(0),
                op_name, ": x.size(-1) must equal weight.size(0)");
}

inline void launch_rmsnorm_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    torch::Tensor& y,
    double eps,
    const char* op_name)
{
    const int hidden = static_cast<int>(x.size(-1));
    const int64_t n_rows = x.numel() / hidden;

    // Empty leading dimensions are valid: e.g. shape (0, hidden). In that case
    // return the already-allocated output without launching an empty grid.
    if (n_rows == 0) return;

    TORCH_CHECK(n_rows <= (int64_t)std::numeric_limits<int>::max(),
                op_name, ": too many rows for a 1-D grid");

    const dim3 grid(static_cast<unsigned>(n_rows));
    const dim3 block(RMSNORM_BLOCK_SIZE);
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
                op_name, ": unsupported dtype ", x.scalar_type(),
                " (expected float32/float16/bfloat16)");
    }

    AT_CUDA_CHECK(cudaGetLastError());
}

inline torch::Tensor rmsnorm_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    double eps,
    const char* op_name)
{
    check_rmsnorm_inputs(x, weight, op_name);

    auto y = torch::empty_like(x);
    launch_rmsnorm_cuda(x, weight, y, eps, op_name);
    return y;
}
