#pragma once

// Shared cuBLAS helper for Linear layers in RMSNorm-family fused kernels.
//
// Interface convention:
//   input_2d: (rows, in_features), contiguous CUDA tensor
//   weight:   (out_features, in_features), contiguous CUDA tensor
//   output:   (rows, out_features), contiguous CUDA tensor
//
// This matches torch.nn.Linear's weight layout while keeping GEMM on the
// explicit cuBLAS path instead of routing through ATen mm/matmul.

#include <torch/extension.h>
#include <ATen/cuda/CUDABlas.h>

#include <cublas_v2.h>

#include <limits>

inline cudaDataType_t cublas_linear_data_type(
    at::ScalarType dtype,
    const char* op_name)
{
    switch (dtype) {
        case at::ScalarType::Float:
            return CUDA_R_32F;
        case at::ScalarType::Half:
            return CUDA_R_16F;
        case at::ScalarType::BFloat16:
            return CUDA_R_16BF;
        default:
            TORCH_CHECK(false,
                op_name, ": unsupported dtype ", dtype,
                " (expected float32/float16/bfloat16)");
    }
}

inline cublasGemmAlgo_t cublas_linear_gemm_algo(at::ScalarType dtype) {
    // Tensor-core algorithms are appropriate for fp16/bf16 inputs. For fp32,
    // keep the default SIMT path so correctness stays close to the PyTorch
    // reference tolerance.
    if (dtype == at::ScalarType::Half || dtype == at::ScalarType::BFloat16) {
        return CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    }
    return CUBLAS_GEMM_DEFAULT;
}

inline torch::Tensor linear_cublas_cuda(
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

    const int64_t n_rows = input_2d.size(0);
    const int64_t in_features = input_2d.size(1);
    const int64_t out_features = weight.size(0);

    TORCH_CHECK(n_rows <= (int64_t)std::numeric_limits<int>::max(),
                op_name, ": too many rows for cuBLAS GEMM");
    TORCH_CHECK(in_features <= (int64_t)std::numeric_limits<int>::max(),
                op_name, ": input size too large for cuBLAS GEMM");
    TORCH_CHECK(out_features <= (int64_t)std::numeric_limits<int>::max(),
                op_name, ": output size too large for cuBLAS GEMM");

    auto output_2d = torch::empty({n_rows, out_features}, input_2d.options());
    if (n_rows == 0 || out_features == 0) {
        return output_2d;
    }

    const int m = static_cast<int>(out_features);
    const int n = static_cast<int>(n_rows);
    const int k = static_cast<int>(in_features);
    const cudaDataType_t dtype =
        cublas_linear_data_type(input_2d.scalar_type(), op_name);
    const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    const cublasGemmAlgo_t algo =
        cublas_linear_gemm_algo(input_2d.scalar_type());
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    at::cuda::blas::PointerModeGuard pointer_mode(
        handle, CUBLAS_POINTER_MODE_HOST);

    // PyTorch tensors are row-major, while cuBLAS expects column-major inputs.
    // Rather than transposing data, reinterpret the same memory:
    //
    //   output_row = input_row @ weight_row.T       // (rows, out_features)
    //
    // is equivalent in memory to column-major:
    //
    //   output_col = weight_row @ input_row.T       // (out_features, rows)
    //
    // The cuBLAS call below computes that column-major form directly into the
    // row-major output buffer.
    TORCH_CUDABLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        m,
        n,
        k,
        &alpha,
        weight.data_ptr(),
        dtype,
        k,
        input_2d.data_ptr(),
        dtype,
        k,
        &beta,
        output_2d.data_ptr(),
        dtype,
        m,
        compute_type,
        algo));

    return output_2d;
}
