// Prologue-fused RMSNorm + register-tiled Linear (v2).
//
// Why a v2:
//   The v1 prologue kernel demonstrated kernel-level fusion correctness but
//   used a 1-thread-per-output toy GEMM whose own inefficiency dominated
//   runtime, so the saving from not materializing `normed` was hidden in the
//   noise. This v2 raises the GEMM's arithmetic intensity (each thread now
//   computes a 4x4 register sub-tile and uses vectorized loads), so the cost
//   saved by elision becomes a measurable fraction of total runtime.
//
// Two phases:
//   1. rmsnorm_scale_v2_kernel: one block per row, computes a single
//      `scale = rsqrt(mean(x^2) + eps)` per row and writes it to a small
//      (rows,) fp32 buffer. Identical math to the v1 prologue's scale phase.
//   2. rmsnorm_linear_prologue_v2_kernel: the fused GEMM. Three additions on
//      top of linear_tiled_v2_kernel:
//        - gamma_tile loaded into shared memory once per K-tile so the per-A
//          read of gamma stays on-chip
//        - scale_tile loaded into shared memory once per block (BLOCK_M
//          entries) so the per-A read of scale stays on-chip
//        - the A-tile load applies x * scale * gamma on the fly, so `normed`
//          is never materialized to global memory

#include "rmsnorm_common.cuh"
#include "tiled_linear_common.cuh"

#include <vector>

// ---------------------------------------------------------------------------
// Phase 1: per-row scale
// ---------------------------------------------------------------------------

template <typename T>
__global__ void rmsnorm_scale_v2_kernel(
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
// Phase 2: prologue-fused register-tiled GEMM
// ---------------------------------------------------------------------------

template <typename T>
__global__ void rmsnorm_linear_prologue_v2_kernel(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ gamma,
    const float* __restrict__ scale,
    T* __restrict__ y,
    int n_rows,
    int hidden_size,
    int out_features)
{
    constexpr int BM  = TILED_LINEAR_V2_BLOCK_M;
    constexpr int BN  = TILED_LINEAR_V2_BLOCK_N;
    constexpr int TM  = TILED_LINEAR_V2_THREAD_M;
    constexpr int TN  = TILED_LINEAR_V2_THREAD_N;
    constexpr int TK  = TILED_LINEAR_V2_TILE_K;
    constexpr int VEC = TILED_LINEAR_V2_VEC;

    __shared__ float a_tile[BM][TK + 1];
    __shared__ float b_tile[TK][BN + VEC];
    __shared__ float gamma_tile[TK];   // refreshed once per K-tile
    __shared__ float scale_tile[BM];   // loaded once per block

    const int local_y = threadIdx.y;
    const int local_x = threadIdx.x;
    const int tid = local_y * TILED_LINEAR_V2_THREADS_X + local_x;
    const int block_row0 = blockIdx.y * BM;
    const int block_col0 = blockIdx.x * BN;

    const int a_tile_row = tid / (TK / VEC);
    const int a_tile_k0  = (tid % (TK / VEC)) * VEC;
    const int b_tile_col = tid / (TK / VEC);
    const int b_tile_k0  = (tid % (TK / VEC)) * VEC;

    // scale_tile: BM = 64 entries, the first 64 threads each load one.
    // (The block has 256 threads; the rest do nothing for this load.)
    if (tid < BM) {
        const int g_row = block_row0 + tid;
        scale_tile[tid] = (g_row < n_rows) ? scale[g_row] : 0.0f;
    }

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
    }

    // Make scale_tile visible to all threads before any A-load uses it.
    __syncthreads();

    for (int k0 = 0; k0 < hidden_size; k0 += TK) {
        // ---- Refresh gamma_tile (TK = 16 entries, first 16 threads). ----
        if (tid < TK) {
            const int g_k = k0 + tid;
            gamma_tile[tid] = (g_k < hidden_size)
                                ? to_float<T>(gamma[g_k])
                                : 0.0f;
        }

        // ---- Stage raw x into thread-local registers. ----
        const int x_g_row = block_row0 + a_tile_row;
        const int x_g_k = k0 + a_tile_k0;
        float xvals[VEC] = {0.f, 0.f, 0.f, 0.f};
        if (x_g_row < n_rows && x_g_k + VEC <= hidden_size) {
            load4_to_float<T>(x + (size_t)x_g_row * hidden_size + x_g_k, xvals);
        } else if (x_g_row < n_rows) {
            #pragma unroll
            for (int e = 0; e < VEC; ++e) {
                const int g_k_e = x_g_k + e;
                if (g_k_e < hidden_size) {
                    xvals[e] = to_float<T>(
                        x[(size_t)x_g_row * hidden_size + g_k_e]);
                }
            }
        }

        // ---- Stage raw weight into thread-local registers. ----
        const int w_g_col = block_col0 + b_tile_col;
        const int w_g_k = k0 + b_tile_k0;
        float wvals[VEC] = {0.f, 0.f, 0.f, 0.f};
        if (w_g_col < out_features && w_g_k + VEC <= hidden_size) {
            load4_to_float<T>(
                weight + (size_t)w_g_col * hidden_size + w_g_k, wvals);
        } else if (w_g_col < out_features) {
            #pragma unroll
            for (int e = 0; e < VEC; ++e) {
                const int g_k_e = w_g_k + e;
                if (g_k_e < hidden_size) {
                    wvals[e] = to_float<T>(
                        weight[(size_t)w_g_col * hidden_size + g_k_e]);
                }
            }
        }

        // gamma_tile must be fully written before any thread reads it from
        // a row that another thread loaded. scale_tile is already valid
        // from the pre-loop sync, so this single barrier covers both.
        __syncthreads();

        // ---- Apply prologue and write A-tile. ----
        const float sv = scale_tile[a_tile_row];
        #pragma unroll
        for (int e = 0; e < VEC; ++e) {
            const float gv = gamma_tile[a_tile_k0 + e];
            a_tile[a_tile_row][a_tile_k0 + e] = xvals[e] * sv * gv;
        }

        // ---- Write B-tile transposed. ----
        #pragma unroll
        for (int e = 0; e < VEC; ++e) {
            b_tile[b_tile_k0 + e][b_tile_col] = wvals[e];
        }

        __syncthreads();

        // ---- Inner FMAs: TM x TN x TK = 256 FMAs per thread per K-tile. ----
        #pragma unroll
        for (int kk = 0; kk < TK; ++kk) {
            float a_reg[TM];
            float b_reg[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                a_reg[i] = a_tile[local_y * TM + i][kk];
            #pragma unroll
            for (int j = 0; j < TN; ++j)
                b_reg[j] = b_tile[kk][local_x * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += a_reg[i] * b_reg[j];
            }
        }

        __syncthreads();
    }

    // ---- Write the TM x TN register sub-tile back to global memory. ----
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int g_row = block_row0 + local_y * TM + i;
        if (g_row >= n_rows) continue;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int g_col = block_col0 + local_x * TN + j;
            if (g_col < out_features) {
                y[(size_t)g_row * out_features + g_col]
                    = from_float<T>(acc[i][j]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

inline void check_rmsnorm_linear_prologue_v2_inputs(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& gamma)
{
    TORCH_CHECK(x.is_cuda(),
                "rmsnorm_linear_prologue_v2_cuda: x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(),
                "rmsnorm_linear_prologue_v2_cuda: weight must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(),
                "rmsnorm_linear_prologue_v2_cuda: gamma must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(),
                "rmsnorm_linear_prologue_v2_cuda: x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(),
                "rmsnorm_linear_prologue_v2_cuda: weight must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(),
                "rmsnorm_linear_prologue_v2_cuda: gamma must be contiguous");
    TORCH_CHECK(x.dim() >= 1,
                "rmsnorm_linear_prologue_v2_cuda: x must have at least 1 dim");
    TORCH_CHECK(weight.dim() == 2,
                "rmsnorm_linear_prologue_v2_cuda: weight must be 2-D");
    TORCH_CHECK(gamma.dim() == 1,
                "rmsnorm_linear_prologue_v2_cuda: gamma must be 1-D");
    TORCH_CHECK(x.scalar_type() == weight.scalar_type() &&
                x.scalar_type() == gamma.scalar_type(),
                "rmsnorm_linear_prologue_v2_cuda: x, weight, and gamma must share dtype");

    const int64_t hidden = x.size(-1);
    TORCH_CHECK(hidden > 0,
                "rmsnorm_linear_prologue_v2_cuda: x.size(-1) must be positive");
    TORCH_CHECK(hidden == weight.size(1),
                "rmsnorm_linear_prologue_v2_cuda: x.size(-1) must equal weight.size(1)");
    TORCH_CHECK(hidden == gamma.size(0),
                "rmsnorm_linear_prologue_v2_cuda: x.size(-1) must equal gamma.size(0)");
    TORCH_CHECK(hidden <= (int64_t)std::numeric_limits<int>::max(),
                "rmsnorm_linear_prologue_v2_cuda: hidden size too large for a 1-D grid");
    TORCH_CHECK(weight.size(0) <= (int64_t)std::numeric_limits<int>::max(),
                "rmsnorm_linear_prologue_v2_cuda: output size too large for a 1-D grid");
}

template <typename T>
void launch_rmsnorm_linear_prologue_v2(
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
                "rmsnorm_linear_prologue_v2_cuda: too many rows for a 1-D grid");
    const int n_rows = static_cast<int>(n_rows_i64);

    auto stream = at::cuda::getCurrentCUDAStream();

    rmsnorm_scale_v2_kernel<T><<<n_rows, RMSNORM_BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<const T*>(x.data_ptr()),
        scale.data_ptr<float>(),
        hidden,
        static_cast<float>(eps));
    AT_CUDA_CHECK(cudaGetLastError());

    const dim3 block(TILED_LINEAR_V2_THREADS_X, TILED_LINEAR_V2_THREADS_Y);
    const dim3 grid(
        (out_features + TILED_LINEAR_V2_BLOCK_N - 1) / TILED_LINEAR_V2_BLOCK_N,
        (n_rows + TILED_LINEAR_V2_BLOCK_M - 1) / TILED_LINEAR_V2_BLOCK_M);
    rmsnorm_linear_prologue_v2_kernel<T><<<grid, block, 0, stream>>>(
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

torch::Tensor rmsnorm_linear_prologue_v2_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& gamma,
    double eps)
{
    check_rmsnorm_linear_prologue_v2_inputs(x, weight, gamma);

    std::vector<int64_t> out_shape(x.sizes().begin(), x.sizes().end());
    out_shape.back() = weight.size(0);
    auto y = torch::empty(out_shape, x.options());

    const int64_t n_rows = x.numel() / x.size(-1);
    auto scale = torch::empty({n_rows}, x.options().dtype(torch::kFloat32));
    if (n_rows == 0 || weight.size(0) == 0) return y;

    switch (x.scalar_type()) {
        case at::ScalarType::Float:
            launch_rmsnorm_linear_prologue_v2<float>(
                x, weight, gamma, y, scale, eps);
            break;
        case at::ScalarType::Half:
            launch_rmsnorm_linear_prologue_v2<__half>(
                x, weight, gamma, y, scale, eps);
            break;
        case at::ScalarType::BFloat16:
            launch_rmsnorm_linear_prologue_v2<__nv_bfloat16>(
                x, weight, gamma, y, scale, eps);
            break;
        default:
            TORCH_CHECK(false,
                "rmsnorm_linear_prologue_v2_cuda: unsupported dtype ",
                x.scalar_type(),
                " (expected float32/float16/bfloat16)");
    }

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_linear_prologue_v2_cuda", &rmsnorm_linear_prologue_v2_cuda,
          "RMSNormLinear prologue v2: scale kernel + register-tiled fused GEMM",
          pybind11::arg("x"),
          pybind11::arg("weight"),
          pybind11::arg("gamma"),
          pybind11::arg("eps") = 1e-6);
}
