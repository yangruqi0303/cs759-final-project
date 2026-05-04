// Fused RMSNorm + Linear CUDA entry point (kernel v1).
//
// This version keeps GEMM on the cuBLAS path: first run the shared RMSNorm
// CUDA kernel, then launch cuBLAS GEMM for the linear projection.
// It is an API-level fusion target for benchmarking against PyTorch's
// RMSNormLinear module, not a custom GEMM prologue fusion.

#include "cublas_linear_common.cuh"
#include "rmsnorm_common.cuh"

#include <vector>

torch::Tensor rmsnorm_linear_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& gamma,
    double eps)
{
    // This entry point follows torch.nn.Linear's storage convention:
    // weight has shape (out_features, hidden_size), so the GEMM uses weight.T.
    TORCH_CHECK(x.is_cuda(),      "rmsnorm_linear_cuda: x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "rmsnorm_linear_cuda: weight must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(),  "rmsnorm_linear_cuda: gamma must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(),      "rmsnorm_linear_cuda: x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "rmsnorm_linear_cuda: weight must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(),  "rmsnorm_linear_cuda: gamma must be contiguous");
    TORCH_CHECK(x.dim() >= 1,       "rmsnorm_linear_cuda: x must have at least 1 dim");
    TORCH_CHECK(weight.dim() == 2,  "rmsnorm_linear_cuda: weight must be 2-D");
    TORCH_CHECK(gamma.dim() == 1,   "rmsnorm_linear_cuda: gamma must be 1-D");
    TORCH_CHECK(x.scalar_type() == weight.scalar_type() &&
                x.scalar_type() == gamma.scalar_type(),
                "rmsnorm_linear_cuda: x, weight, and gamma must share dtype");

    const int64_t hidden = x.size(-1);
    const int64_t out_features = weight.size(0);
    TORCH_CHECK(hidden > 0,
                "rmsnorm_linear_cuda: x.size(-1) must be positive");
    TORCH_CHECK(hidden == weight.size(1),
                "rmsnorm_linear_cuda: x.size(-1) must equal weight.size(1)");
    TORCH_CHECK(hidden == gamma.size(0),
                "rmsnorm_linear_cuda: x.size(-1) must equal gamma.size(0)");

    // Run the shared RMSNorm kernel first. This produces a materialized
    // normalized tensor; a future custom GEMM prologue could avoid this
    // intermediate global-memory round trip.
    const torch::Tensor normed =
        rmsnorm_forward_cuda(x, gamma, eps, "rmsnorm_linear_cuda");

    // Collapse all leading dimensions into a row count, then launch cuBLAS for
    // the GEMM: (rows, hidden) x (hidden, out_features).
    const int64_t n_rows = x.numel() / hidden;
    torch::Tensor normed_2d = normed.reshape({n_rows, hidden});
    torch::Tensor out_2d =
        linear_cublas_cuda(normed_2d, weight, "rmsnorm_linear_cuda");

    // Restore the original leading dimensions and replace the hidden dimension
    // with the linear projection size.
    std::vector<int64_t> out_shape(x.sizes().begin(), x.sizes().end());
    out_shape.back() = out_features;
    return out_2d.reshape(out_shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_linear_cuda", &rmsnorm_linear_cuda,
          "RMSNorm followed by Linear using CUDA RMSNorm and cuBLAS GEMM",
          pybind11::arg("x"),
          pybind11::arg("weight"),
          pybind11::arg("gamma"),
          pybind11::arg("eps") = 1e-6);
}
