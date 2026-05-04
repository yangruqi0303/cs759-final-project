#pragma once

// Shared scalar-FMA tiled Linear helper for fusion experiments.
//
// This helper is intentionally simpler than cuBLAS/CUTLASS. It computes:
//
//   output = input_2d @ weight.T
//
// using a 16x16 output tile and a 32-wide K tile in shared memory. It exists so
// the materialized and prologue-fused RMSNormLinear experiments can use the
// same GEMM structure when comparing the cost of materializing `normed`.

#include "rmsnorm_common.cuh"

#define TILED_LINEAR_BLOCK_M 16
#define TILED_LINEAR_BLOCK_N 16
#define TILED_LINEAR_TILE_K  32

template <typename T>
__global__ void linear_tiled_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    T* __restrict__ output,
    int n_rows,
    int in_features,
    int out_features)
{
    __shared__ float a_tile[TILED_LINEAR_BLOCK_M][TILED_LINEAR_TILE_K];
    __shared__ float b_tile[TILED_LINEAR_TILE_K][TILED_LINEAR_BLOCK_N];

    const int local_row = threadIdx.y;
    const int local_col = threadIdx.x;
    const int row = blockIdx.y * TILED_LINEAR_BLOCK_M + local_row;
    const int col = blockIdx.x * TILED_LINEAR_BLOCK_N + local_col;
    const int tid = local_row * TILED_LINEAR_BLOCK_N + local_col;
    const bool valid_output = (row < n_rows && col < out_features);

    float acc = 0.0f;

    for (int k0 = 0; k0 < in_features; k0 += TILED_LINEAR_TILE_K) {
        for (int idx = tid;
             idx < TILED_LINEAR_BLOCK_M * TILED_LINEAR_TILE_K;
             idx += TILED_LINEAR_BLOCK_M * TILED_LINEAR_BLOCK_N) {
            const int tile_row = idx / TILED_LINEAR_TILE_K;
            const int tile_k = idx % TILED_LINEAR_TILE_K;
            const int global_row = blockIdx.y * TILED_LINEAR_BLOCK_M + tile_row;
            const int global_k = k0 + tile_k;

            float v = 0.0f;
            if (global_row < n_rows && global_k < in_features) {
                v = to_float<T>(
                    input[(size_t)global_row * in_features + global_k]);
            }
            a_tile[tile_row][tile_k] = v;
        }

        // Weight follows torch.nn.Linear layout: (out_features, in_features).
        for (int idx = tid;
             idx < TILED_LINEAR_TILE_K * TILED_LINEAR_BLOCK_N;
             idx += TILED_LINEAR_BLOCK_M * TILED_LINEAR_BLOCK_N) {
            const int tile_k = idx / TILED_LINEAR_BLOCK_N;
            const int tile_col = idx % TILED_LINEAR_BLOCK_N;
            const int global_k = k0 + tile_k;
            const int global_col = blockIdx.x * TILED_LINEAR_BLOCK_N + tile_col;

            float v = 0.0f;
            if (global_col < out_features && global_k < in_features) {
                v = to_float<T>(
                    weight[(size_t)global_col * in_features + global_k]);
            }
            b_tile[tile_k][tile_col] = v;
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TILED_LINEAR_TILE_K; ++kk) {
            acc += a_tile[local_row][kk] * b_tile[kk][local_col];
        }

        __syncthreads();
    }

    if (valid_output) {
        output[(size_t)row * out_features + col] = from_float<T>(acc);
    }
}

inline void check_tiled_linear_inputs(
    const torch::Tensor& input_2d,
    const torch::Tensor& weight,
    const char* op_name)
{
    TORCH_CHECK(input_2d.is_cuda(), op_name, ": input_2d must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(),   op_name, ": weight must be a CUDA tensor");
    TORCH_CHECK(input_2d.is_contiguous(),
                op_name, ": input_2d must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), op_name, ": weight must be contiguous");
    TORCH_CHECK(input_2d.dim() == 2, op_name, ": input_2d must be 2-D");
    TORCH_CHECK(weight.dim() == 2,   op_name, ": weight must be 2-D");
    TORCH_CHECK(input_2d.scalar_type() == weight.scalar_type(),
                op_name, ": input_2d and weight must share dtype");
    TORCH_CHECK(input_2d.size(1) == weight.size(1),
                op_name, ": input_2d.size(1) must equal weight.size(1)");
    TORCH_CHECK(input_2d.size(0) <= (int64_t)std::numeric_limits<int>::max(),
                op_name, ": too many rows for a 1-D grid");
    TORCH_CHECK(input_2d.size(1) <= (int64_t)std::numeric_limits<int>::max(),
                op_name, ": input size too large for a 1-D grid");
    TORCH_CHECK(weight.size(0) <= (int64_t)std::numeric_limits<int>::max(),
                op_name, ": output size too large for a 1-D grid");
}

template <typename T>
void launch_linear_tiled(
    const torch::Tensor& input_2d,
    const torch::Tensor& weight,
    torch::Tensor& output_2d)
{
    const int n_rows = static_cast<int>(input_2d.size(0));
    const int in_features = static_cast<int>(input_2d.size(1));
    const int out_features = static_cast<int>(weight.size(0));
    if (n_rows == 0 || out_features == 0) return;

    const dim3 block(TILED_LINEAR_BLOCK_N, TILED_LINEAR_BLOCK_M);
    const dim3 grid(
        (out_features + TILED_LINEAR_BLOCK_N - 1) / TILED_LINEAR_BLOCK_N,
        (n_rows + TILED_LINEAR_BLOCK_M - 1) / TILED_LINEAR_BLOCK_M);
    auto stream = at::cuda::getCurrentCUDAStream();

    linear_tiled_kernel<T><<<grid, block, 0, stream>>>(
        reinterpret_cast<const T*>(input_2d.data_ptr()),
        reinterpret_cast<const T*>(weight.data_ptr()),
        reinterpret_cast<T*>(output_2d.data_ptr()),
        n_rows,
        in_features,
        out_features);
    AT_CUDA_CHECK(cudaGetLastError());
}

inline torch::Tensor linear_tiled_cuda(
    const torch::Tensor& input_2d,
    const torch::Tensor& weight,
    const char* op_name)
{
    check_tiled_linear_inputs(input_2d, weight, op_name);

    auto output_2d = torch::empty(
        {input_2d.size(0), weight.size(0)},
        input_2d.options());
    if (input_2d.size(0) == 0 || weight.size(0) == 0) {
        return output_2d;
    }

    switch (input_2d.scalar_type()) {
        case at::ScalarType::Float:
            launch_linear_tiled<float>(input_2d, weight, output_2d);
            break;
        case at::ScalarType::Half:
            launch_linear_tiled<__half>(input_2d, weight, output_2d);
            break;
        case at::ScalarType::BFloat16:
            launch_linear_tiled<__nv_bfloat16>(input_2d, weight, output_2d);
            break;
        default:
            TORCH_CHECK(false,
                op_name, ": unsupported dtype ", input_2d.scalar_type(),
                " (expected float32/float16/bfloat16)");
    }

    return output_2d;
}
