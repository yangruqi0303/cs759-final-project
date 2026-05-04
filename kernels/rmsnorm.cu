// Naive RMSNorm CUDA kernel
//
//   y[i] = x[i] * rsqrt(mean(x^2, dim=-1) + eps) * weight[i]
//
// The shared CUDA implementation lives in rmsnorm_common.cuh so later fused kernels
// can reuse the dtype helpers, warp reduction, and launcher.

#include "rmsnorm_common.cuh"

torch::Tensor rmsnorm_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    double eps)
{
    // Standalone Phase 2 entry point. The implementation is shared with later
    // fused kernels through rmsnorm_common.cuh.
    return rmsnorm_forward_cuda(x, weight, eps, "rmsnorm_cuda");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_cuda", &rmsnorm_cuda,
          "Naive RMSNorm CUDA kernel",
          pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("eps") = 1e-6);
}
