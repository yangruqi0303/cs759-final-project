// Fused RMSNorm + MLP CUDA entry point (kernel v2).
//
// This version implements the full forward path:
//
//   Linear2(GELU(Linear1(RMSNorm(x))))
//
// RMSNorm uses the shared custom CUDA kernel. Both Linear projections use the
// explicit cuBLAS helper. GELU uses ATen's CUDA implementation with the tanh
// approximation so the behavior matches baseline.RMSNormMLP exactly.

#include "cublas_linear_common.cuh"
#include "rmsnorm_common.cuh"

#include <ATen/ops/gelu.h>

#include <vector>

torch::Tensor rmsnorm_mlp_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight1,
    const torch::Tensor& weight2,
    const torch::Tensor& gamma,
    double eps)
{
    // Weight layout follows torch.nn.Linear:
    //   weight1: (intermediate_size, hidden_size)
    //   weight2: (hidden_size, intermediate_size)
    TORCH_CHECK(x.is_cuda(),       "rmsnorm_mlp_cuda: x must be a CUDA tensor");
    TORCH_CHECK(weight1.is_cuda(), "rmsnorm_mlp_cuda: weight1 must be a CUDA tensor");
    TORCH_CHECK(weight2.is_cuda(), "rmsnorm_mlp_cuda: weight2 must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(),   "rmsnorm_mlp_cuda: gamma must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(),       "rmsnorm_mlp_cuda: x must be contiguous");
    TORCH_CHECK(weight1.is_contiguous(), "rmsnorm_mlp_cuda: weight1 must be contiguous");
    TORCH_CHECK(weight2.is_contiguous(), "rmsnorm_mlp_cuda: weight2 must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(),   "rmsnorm_mlp_cuda: gamma must be contiguous");
    TORCH_CHECK(x.dim() >= 1,        "rmsnorm_mlp_cuda: x must have at least 1 dim");
    TORCH_CHECK(weight1.dim() == 2,  "rmsnorm_mlp_cuda: weight1 must be 2-D");
    TORCH_CHECK(weight2.dim() == 2,  "rmsnorm_mlp_cuda: weight2 must be 2-D");
    TORCH_CHECK(gamma.dim() == 1,    "rmsnorm_mlp_cuda: gamma must be 1-D");
    TORCH_CHECK(x.scalar_type() == weight1.scalar_type() &&
                x.scalar_type() == weight2.scalar_type() &&
                x.scalar_type() == gamma.scalar_type(),
                "rmsnorm_mlp_cuda: x, weight1, weight2, and gamma must share dtype");

    const int64_t hidden = x.size(-1);
    const int64_t intermediate = weight1.size(0);
    TORCH_CHECK(hidden > 0,
                "rmsnorm_mlp_cuda: x.size(-1) must be positive");
    TORCH_CHECK(hidden == weight1.size(1),
                "rmsnorm_mlp_cuda: x.size(-1) must equal weight1.size(1)");
    TORCH_CHECK(hidden == weight2.size(0),
                "rmsnorm_mlp_cuda: x.size(-1) must equal weight2.size(0)");
    TORCH_CHECK(hidden == gamma.size(0),
                "rmsnorm_mlp_cuda: x.size(-1) must equal gamma.size(0)");
    TORCH_CHECK(intermediate == weight2.size(1),
                "rmsnorm_mlp_cuda: weight1.size(0) must equal weight2.size(1)");

    // 1) Shared RMSNorm CUDA prologue.
    const torch::Tensor normed =
        rmsnorm_forward_cuda(x, gamma, eps, "rmsnorm_mlp_cuda");

    // 2) First projection: (rows, hidden) x (hidden, intermediate).
    const int64_t n_rows = x.numel() / hidden;
    torch::Tensor normed_2d = normed.reshape({n_rows, hidden});
    torch::Tensor h1 =
        linear_cublas_cuda(normed_2d, weight1, "rmsnorm_mlp_cuda");

    // 3) Tanh-approximated GELU, matching baseline.RMSNormMLP.
    torch::Tensor h2 = at::gelu(h1, "tanh");

    // 4) Second projection: (rows, intermediate) x (intermediate, hidden).
    torch::Tensor out_2d =
        linear_cublas_cuda(h2, weight2, "rmsnorm_mlp_cuda");

    // RMSNormMLP projects back to the original hidden size, so the output shape
    // is exactly the input shape.
    return out_2d.reshape(x.sizes());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_mlp_cuda", &rmsnorm_mlp_cuda,
          "RMSNorm + Linear + GELU(tanh) + Linear using CUDA RMSNorm and cuBLAS GEMMs",
          pybind11::arg("x"),
          pybind11::arg("weight1"),
          pybind11::arg("weight2"),
          pybind11::arg("gamma"),
          pybind11::arg("eps") = 1e-6);
}
