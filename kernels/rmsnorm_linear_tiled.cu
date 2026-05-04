// RMSNorm + Linear with materialized RMSNorm and tiled custom GEMM.
//
// This variant is for comparing against the prologue-fused tiled kernel. It
// intentionally uses the same tiled GEMM helper as RMSNormLinear v2, but first
// writes the complete normalized tensor to global memory.

#include "rmsnorm_common.cuh"
#include "tiled_linear_common.cuh"

#include <vector>

torch::Tensor rmsnorm_linear_tiled_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& gamma,
    double eps)
{
    TORCH_CHECK(x.is_cuda(),      "rmsnorm_linear_tiled_cuda: x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "rmsnorm_linear_tiled_cuda: weight must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(),  "rmsnorm_linear_tiled_cuda: gamma must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(),      "rmsnorm_linear_tiled_cuda: x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "rmsnorm_linear_tiled_cuda: weight must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(),  "rmsnorm_linear_tiled_cuda: gamma must be contiguous");
    TORCH_CHECK(x.dim() >= 1,       "rmsnorm_linear_tiled_cuda: x must have at least 1 dim");
    TORCH_CHECK(weight.dim() == 2,  "rmsnorm_linear_tiled_cuda: weight must be 2-D");
    TORCH_CHECK(gamma.dim() == 1,   "rmsnorm_linear_tiled_cuda: gamma must be 1-D");
    TORCH_CHECK(x.scalar_type() == weight.scalar_type() &&
                x.scalar_type() == gamma.scalar_type(),
                "rmsnorm_linear_tiled_cuda: x, weight, and gamma must share dtype");

    const int64_t hidden = x.size(-1);
    const int64_t out_features = weight.size(0);
    TORCH_CHECK(hidden > 0,
                "rmsnorm_linear_tiled_cuda: x.size(-1) must be positive");
    TORCH_CHECK(hidden == weight.size(1),
                "rmsnorm_linear_tiled_cuda: x.size(-1) must equal weight.size(1)");
    TORCH_CHECK(hidden == gamma.size(0),
                "rmsnorm_linear_tiled_cuda: x.size(-1) must equal gamma.size(0)");

    const torch::Tensor normed =
        rmsnorm_forward_cuda(x, gamma, eps, "rmsnorm_linear_tiled_cuda");

    const int64_t n_rows = x.numel() / hidden;
    torch::Tensor normed_2d = normed.reshape({n_rows, hidden});
    torch::Tensor out_2d =
        linear_tiled_cuda(normed_2d, weight, "rmsnorm_linear_tiled_cuda");

    std::vector<int64_t> out_shape(x.sizes().begin(), x.sizes().end());
    out_shape.back() = out_features;
    return out_2d.reshape(out_shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_linear_tiled_cuda", &rmsnorm_linear_tiled_cuda,
          "RMSNormLinear tiled: materialized RMSNorm plus tiled custom GEMM",
          pybind11::arg("x"),
          pybind11::arg("weight"),
          pybind11::arg("gamma"),
          pybind11::arg("eps") = 1e-6);
}
