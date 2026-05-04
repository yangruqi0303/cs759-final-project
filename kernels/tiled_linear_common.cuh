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

// ===========================================================================
// V2: register-tiled GEMM with 4-element vector loads.
//
// Three differences vs the v1 kernel above:
//   1. Each thread now computes a TM x TN = 4 x 4 register sub-tile of the
//      output instead of one element. This raises the inner-loop arithmetic
//      ratio from ~1 FMA per shared-mem load to ~4, which is the whole point
//      of register tiling on top of shared-memory tiling.
//   2. Global loads are 4-wide vectorized: float4 for fp32, packed 4xhalf /
//      4xbf16 (= 8-byte int2) for low precision. This cuts the number of LDG
//      instructions issued by 4x.
//   3. Shared-memory tiles are padded by +1 / +VEC respectively so the most
//      common bank-conflict patterns from the inner loop are broken.
//
// Block layout: 16x16 = 256 threads (same as v1) compute a 64x64 output tile
// per block. Inner K tile is 16 wide.
// ===========================================================================

#define TILED_LINEAR_V2_BLOCK_M  64
#define TILED_LINEAR_V2_BLOCK_N  64
#define TILED_LINEAR_V2_THREAD_M  4
#define TILED_LINEAR_V2_THREAD_N  4
#define TILED_LINEAR_V2_TILE_K   16
#define TILED_LINEAR_V2_VEC       4
#define TILED_LINEAR_V2_THREADS_X (TILED_LINEAR_V2_BLOCK_N / TILED_LINEAR_V2_THREAD_N)
#define TILED_LINEAR_V2_THREADS_Y (TILED_LINEAR_V2_BLOCK_M / TILED_LINEAR_V2_THREAD_M)
#define TILED_LINEAR_V2_THREADS   (TILED_LINEAR_V2_THREADS_X * TILED_LINEAR_V2_THREADS_Y)

// 4-element global load that always returns float values, regardless of T.
// The fp16 / bf16 specializations issue a single 8-byte load (LDG.E.64) and
// then unpack via the half-pair intrinsics; the fp32 specialization issues a
// 16-byte load (LDG.E.128). The fallback is a 4x scalar loop in case T is
// some other floating type that the project hasn't enabled yet.
template <typename T>
__device__ __forceinline__ void load4_to_float(const T* p, float out[4]) {
    out[0] = to_float<T>(p[0]); out[1] = to_float<T>(p[1]);
    out[2] = to_float<T>(p[2]); out[3] = to_float<T>(p[3]);
}

template <>
__device__ __forceinline__ void load4_to_float<float>(const float* p, float out[4]) {
    const float4 v = *reinterpret_cast<const float4*>(p);
    out[0] = v.x; out[1] = v.y; out[2] = v.z; out[3] = v.w;
}

template <>
__device__ __forceinline__ void load4_to_float<__half>(const __half* p, float out[4]) {
    const int2 raw = *reinterpret_cast<const int2*>(p);
    const __half2 lo = *reinterpret_cast<const __half2*>(&raw.x);
    const __half2 hi = *reinterpret_cast<const __half2*>(&raw.y);
    out[0] = __low2float(lo); out[1] = __high2float(lo);
    out[2] = __low2float(hi); out[3] = __high2float(hi);
}

template <>
__device__ __forceinline__ void load4_to_float<__nv_bfloat16>(
    const __nv_bfloat16* p, float out[4])
{
    const int2 raw = *reinterpret_cast<const int2*>(p);
    const __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(&raw.x);
    const __nv_bfloat162 hi = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
    out[0] = __low2float(lo); out[1] = __high2float(lo);
    out[2] = __low2float(hi); out[3] = __high2float(hi);
}

template <typename T>
__global__ void linear_tiled_v2_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    T* __restrict__ output,
    int n_rows,
    int in_features,
    int out_features)
{
    constexpr int BM  = TILED_LINEAR_V2_BLOCK_M;
    constexpr int BN  = TILED_LINEAR_V2_BLOCK_N;
    constexpr int TM  = TILED_LINEAR_V2_THREAD_M;
    constexpr int TN  = TILED_LINEAR_V2_THREAD_N;
    constexpr int TK  = TILED_LINEAR_V2_TILE_K;
    constexpr int VEC = TILED_LINEAR_V2_VEC;

    // a_tile: +1 padding on the K dim breaks the bank conflict that occurs
    //         when threads in the same warp read the same column at different
    //         strides of 16 floats (= 64 bytes = 16 banks).
    // b_tile: stored as [TK][BN], i.e. row-major across BN columns, because
    //         the inner FMA loop reads b_tile[kk][col] with consecutive col
    //         per thread. +VEC padding on the N dim breaks the conflict that
    //         the transposed write pattern would otherwise create.
    __shared__ float a_tile[BM][TK + 1];
    __shared__ float b_tile[TK][BN + VEC];

    const int local_y = threadIdx.y;
    const int local_x = threadIdx.x;
    const int tid = local_y * TILED_LINEAR_V2_THREADS_X + local_x;
    const int block_row0 = blockIdx.y * BM;
    const int block_col0 = blockIdx.x * BN;

    // For tile loads: every one of the 256 threads loads exactly one VEC
    // (4 elements) into either a_tile or b_tile, so total elements are
    // BM*TK = BN*TK = 64*16 = 1024 = 256 * VEC.
    const int a_tile_row = tid / (TK / VEC);
    const int a_tile_k0  = (tid % (TK / VEC)) * VEC;
    const int b_tile_col = tid / (TK / VEC);
    const int b_tile_k0  = (tid % (TK / VEC)) * VEC;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
    }

    for (int k0 = 0; k0 < in_features; k0 += TK) {
        // ---- Load A tile (input is row-major (n_rows, in_features)). ----
        {
            const int g_row = block_row0 + a_tile_row;
            const int g_k = k0 + a_tile_k0;
            float vals[VEC] = {0.f, 0.f, 0.f, 0.f};
            if (g_row < n_rows && g_k + VEC <= in_features) {
                load4_to_float<T>(
                    input + (size_t)g_row * in_features + g_k, vals);
            } else if (g_row < n_rows) {
                #pragma unroll
                for (int e = 0; e < VEC; ++e) {
                    const int g_k_e = g_k + e;
                    if (g_k_e < in_features) {
                        vals[e] = to_float<T>(
                            input[(size_t)g_row * in_features + g_k_e]);
                    }
                }
            }
            #pragma unroll
            for (int e = 0; e < VEC; ++e)
                a_tile[a_tile_row][a_tile_k0 + e] = vals[e];
        }

        // ---- Load B tile (weight is row-major (out_features, in_features)).
        //      We load along K within one row of weight (coalesced), then
        //      write transposed into b_tile[k][col] so the inner FMA loop
        //      reads consecutive col per thread.
        {
            const int g_col = block_col0 + b_tile_col;
            const int g_k = k0 + b_tile_k0;
            float vals[VEC] = {0.f, 0.f, 0.f, 0.f};
            if (g_col < out_features && g_k + VEC <= in_features) {
                load4_to_float<T>(
                    weight + (size_t)g_col * in_features + g_k, vals);
            } else if (g_col < out_features) {
                #pragma unroll
                for (int e = 0; e < VEC; ++e) {
                    const int g_k_e = g_k + e;
                    if (g_k_e < in_features) {
                        vals[e] = to_float<T>(
                            weight[(size_t)g_col * in_features + g_k_e]);
                    }
                }
            }
            #pragma unroll
            for (int e = 0; e < VEC; ++e)
                b_tile[b_tile_k0 + e][b_tile_col] = vals[e];
        }

        __syncthreads();

        // ---- Inner FMAs over the K tile, accumulating into registers. ----
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
                output[(size_t)g_row * out_features + g_col]
                    = from_float<T>(acc[i][j]);
            }
        }
    }
}

template <typename T>
void launch_linear_tiled_v2(
    const torch::Tensor& input_2d,
    const torch::Tensor& weight,
    torch::Tensor& output_2d)
{
    const int n_rows = static_cast<int>(input_2d.size(0));
    const int in_features = static_cast<int>(input_2d.size(1));
    const int out_features = static_cast<int>(weight.size(0));
    if (n_rows == 0 || out_features == 0) return;

    const dim3 block(TILED_LINEAR_V2_THREADS_X, TILED_LINEAR_V2_THREADS_Y);
    const dim3 grid(
        (out_features + TILED_LINEAR_V2_BLOCK_N - 1) / TILED_LINEAR_V2_BLOCK_N,
        (n_rows + TILED_LINEAR_V2_BLOCK_M - 1) / TILED_LINEAR_V2_BLOCK_M);
    auto stream = at::cuda::getCurrentCUDAStream();

    linear_tiled_v2_kernel<T><<<grid, block, 0, stream>>>(
        reinterpret_cast<const T*>(input_2d.data_ptr()),
        reinterpret_cast<const T*>(weight.data_ptr()),
        reinterpret_cast<T*>(output_2d.data_ptr()),
        n_rows,
        in_features,
        out_features);
    AT_CUDA_CHECK(cudaGetLastError());
}

inline torch::Tensor linear_tiled_v2_cuda(
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
            launch_linear_tiled_v2<float>(input_2d, weight, output_2d);
            break;
        case at::ScalarType::Half:
            launch_linear_tiled_v2<__half>(input_2d, weight, output_2d);
            break;
        case at::ScalarType::BFloat16:
            launch_linear_tiled_v2<__nv_bfloat16>(input_2d, weight, output_2d);
            break;
        default:
            TORCH_CHECK(false,
                op_name, ": unsupported dtype ", input_2d.scalar_type(),
                " (expected float32/float16/bfloat16)");
    }

    return output_2d;
}
