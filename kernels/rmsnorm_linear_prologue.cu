// Prologue-fused RMSNorm + tiled Linear CUDA entry point.
//
// This version avoids materializing the full normalized tensor. It first
// computes one RMS scale per row, then uses a custom GEMM-style kernel whose
// input prologue applies:
//
//   x[row, k] * scale[row] * gamma[k]
//
// directly while accumulating the Linear projection. This is a true prologue
// fusion experiment, but it is intentionally simple and does not try to beat
// cuBLAS's highly optimized GEMM kernels.

#include "rmsnorm_common.cuh"
#include "tiled_linear_common.cuh"

#include <vector>

// ---------------------------------------------------------------------------
// Scale kernel - one block per row
// ---------------------------------------------------------------------------

template <typename T>
__global__ void rmsnorm_scale_kernel(
    const T* __restrict__ x,
    float* __restrict__ scale,
    int hidden_size,
    float eps)
{
    const int row     = blockIdx.x;
    const int tid     = threadIdx.x;
    const int lane    = tid & (RMSNORM_WARP_SIZE - 1);
    const int warp_id = tid >> 5;
    const int n_warps = blockDim.x / RMSNORM_WARP_SIZE;

    const T* x_row = x + (size_t)row * hidden_size;

    float sumsq = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        const float xv = to_float<T>(x_row[i]);
        sumsq += xv * xv;
    }

    sumsq = warp_reduce_sum(sumsq);

    __shared__ float warp_sums[RMSNORM_MAX_WARPS];
    if (lane == 0) warp_sums[warp_id] = sumsq;
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane < n_warps) ? warp_sums[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) {
            scale[row] = rsqrtf(v / static_cast<float>(hidden_size) + eps);
        }
    }
}

// ---------------------------------------------------------------------------
// Prologue-fused Linear kernel
// ---------------------------------------------------------------------------
//
// Each thread block computes a 16x16 output tile. For each K tile, the block
// cooperatively loads:
//
//   A_tile = x[row, k] * scale[row] * gamma[k]
//   B_tile = weight[col, k]
//
// into shared memory, then reuses those values for the local output tile. This
// is still a scalar-FMA GEMM, but it avoids the worst global-memory rereads of
// the naive one-thread-per-output implementation.

template <typename T>
__global__ void rmsnorm_linear_prologue_kernel(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ gamma,
    const float* __restrict__ scale,
    T* __restrict__ y,
    int n_rows,
    int hidden_size,
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

    for (int k0 = 0; k0 < hidden_size; k0 += TILED_LINEAR_TILE_K) {
        // Load A tile. There are 16*32 elements, so each of the 256 threads
        // loads two elements in the common case.
        for (int idx = tid;
             idx < TILED_LINEAR_BLOCK_M * TILED_LINEAR_TILE_K;
             idx += TILED_LINEAR_BLOCK_M * TILED_LINEAR_BLOCK_N) {
            const int tile_row = idx / TILED_LINEAR_TILE_K;
            const int tile_k = idx % TILED_LINEAR_TILE_K;
            const int global_row = blockIdx.y * TILED_LINEAR_BLOCK_M + tile_row;
            const int global_k = k0 + tile_k;

            float v = 0.0f;
            if (global_row < n_rows && global_k < hidden_size) {
                const float xv = to_float<T>(
                    x[(size_t)global_row * hidden_size + global_k]);
                const float gv = to_float<T>(gamma[global_k]);
                v = xv * scale[global_row] * gv;
            }
            a_tile[tile_row][tile_k] = v;
        }

        // Load B tile from row-major Linear weight: (out_features, hidden).
        for (int idx = tid;
             idx < TILED_LINEAR_TILE_K * TILED_LINEAR_BLOCK_N;
             idx += TILED_LINEAR_BLOCK_M * TILED_LINEAR_BLOCK_N) {
            const int tile_k = idx / TILED_LINEAR_BLOCK_N;
            const int tile_col = idx % TILED_LINEAR_BLOCK_N;
            const int global_k = k0 + tile_k;
            const int global_col = blockIdx.x * TILED_LINEAR_BLOCK_N + tile_col;

            float v = 0.0f;
            if (global_col < out_features && global_k < hidden_size) {
                v = to_float<T>(
                    weight[(size_t)global_col * hidden_size + global_k]);
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
        y[(size_t)row * out_features + col] = from_float<T>(acc);
    }
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

inline void check_rmsnorm_linear_prologue_inputs(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& gamma)
{
    TORCH_CHECK(x.is_cuda(),      "rmsnorm_linear_prologue_cuda: x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "rmsnorm_linear_prologue_cuda: weight must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(),  "rmsnorm_linear_prologue_cuda: gamma must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(),      "rmsnorm_linear_prologue_cuda: x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "rmsnorm_linear_prologue_cuda: weight must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(),  "rmsnorm_linear_prologue_cuda: gamma must be contiguous");
    TORCH_CHECK(x.dim() >= 1,       "rmsnorm_linear_prologue_cuda: x must have at least 1 dim");
    TORCH_CHECK(weight.dim() == 2,  "rmsnorm_linear_prologue_cuda: weight must be 2-D");
    TORCH_CHECK(gamma.dim() == 1,   "rmsnorm_linear_prologue_cuda: gamma must be 1-D");
    TORCH_CHECK(x.scalar_type() == weight.scalar_type() &&
                x.scalar_type() == gamma.scalar_type(),
                "rmsnorm_linear_prologue_cuda: x, weight, and gamma must share dtype");

    const int64_t hidden = x.size(-1);
    TORCH_CHECK(hidden > 0,
                "rmsnorm_linear_prologue_cuda: x.size(-1) must be positive");
    TORCH_CHECK(hidden == weight.size(1),
                "rmsnorm_linear_prologue_cuda: x.size(-1) must equal weight.size(1)");
    TORCH_CHECK(hidden == gamma.size(0),
                "rmsnorm_linear_prologue_cuda: x.size(-1) must equal gamma.size(0)");
    TORCH_CHECK(hidden <= (int64_t)std::numeric_limits<int>::max(),
                "rmsnorm_linear_prologue_cuda: hidden size too large for a 1-D grid");
    TORCH_CHECK(weight.size(0) <= (int64_t)std::numeric_limits<int>::max(),
                "rmsnorm_linear_prologue_cuda: output size too large for a 1-D grid");
}

template <typename T>
void launch_rmsnorm_linear_prologue(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& gamma,
    torch::Tensor& y,
    torch::Tensor& scale,
    double eps)
{
    const int hidden = static_cast<int>(x.size(-1));
    const int out_features = static_cast<int>(weight.size(0));
    const int64_t n_rows_i64 = x.numel() / hidden;
    if (n_rows_i64 == 0 || out_features == 0) return;

    TORCH_CHECK(n_rows_i64 <= (int64_t)std::numeric_limits<int>::max(),
                "rmsnorm_linear_prologue_cuda: too many rows for a 1-D grid");
    const int n_rows = static_cast<int>(n_rows_i64);

    auto stream = at::cuda::getCurrentCUDAStream();
    rmsnorm_scale_kernel<T><<<n_rows, RMSNORM_BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<const T*>(x.data_ptr()),
        scale.data_ptr<float>(),
        hidden,
        static_cast<float>(eps));
    AT_CUDA_CHECK(cudaGetLastError());

    const dim3 block(TILED_LINEAR_BLOCK_N, TILED_LINEAR_BLOCK_M);
    const dim3 grid(
        (out_features + TILED_LINEAR_BLOCK_N - 1) / TILED_LINEAR_BLOCK_N,
        (n_rows + TILED_LINEAR_BLOCK_M - 1) / TILED_LINEAR_BLOCK_M);
    rmsnorm_linear_prologue_kernel<T><<<grid, block, 0, stream>>>(
        reinterpret_cast<const T*>(x.data_ptr()),
        reinterpret_cast<const T*>(weight.data_ptr()),
        reinterpret_cast<const T*>(gamma.data_ptr()),
        scale.data_ptr<float>(),
        reinterpret_cast<T*>(y.data_ptr()),
        n_rows,
        hidden,
        out_features);
    AT_CUDA_CHECK(cudaGetLastError());
}

torch::Tensor rmsnorm_linear_prologue_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& gamma,
    double eps)
{
    check_rmsnorm_linear_prologue_inputs(x, weight, gamma);

    std::vector<int64_t> out_shape(x.sizes().begin(), x.sizes().end());
    out_shape.back() = weight.size(0);
    auto y = torch::empty(out_shape, x.options());

    const int64_t n_rows = x.numel() / x.size(-1);
    auto scale = torch::empty({n_rows}, x.options().dtype(torch::kFloat32));
    if (n_rows == 0 || weight.size(0) == 0) return y;

    switch (x.scalar_type()) {
        case at::ScalarType::Float:
            launch_rmsnorm_linear_prologue<float>(x, weight, gamma, y, scale, eps);
            break;
        case at::ScalarType::Half:
            launch_rmsnorm_linear_prologue<__half>(x, weight, gamma, y, scale, eps);
            break;
        case at::ScalarType::BFloat16:
            launch_rmsnorm_linear_prologue<__nv_bfloat16>(
                x, weight, gamma, y, scale, eps);
            break;
        default:
            TORCH_CHECK(false,
                "rmsnorm_linear_prologue_cuda: unsupported dtype ", x.scalar_type(),
                " (expected float32/float16/bfloat16)");
    }

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_linear_prologue_cuda", &rmsnorm_linear_prologue_cuda,
          "RMSNormLinear prologue: scale kernel plus prologue-fused tiled GEMM",
          pybind11::arg("x"),
          pybind11::arg("weight"),
          pybind11::arg("gamma"),
          pybind11::arg("eps") = 1e-6);
}
