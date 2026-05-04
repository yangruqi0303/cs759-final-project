// RMSNorm + Linear with materialized RMSNorm and naive custom GEMM.
//
// This variant is intentionally simple: it first writes the normalized tensor
// to global memory, then launches one CUDA thread per output element. There is
// no shared-memory tiling, so it is mainly a baseline for showing how much the
// tiled and prologue-fused experiments improve the custom GEMM path.

#include "rmsnorm_common.cuh"

#include <vector>

template <typename T>
__global__ void linear_naive_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    T* __restrict__ output,
    int n_rows,
    int in_features,
    int out_features)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows || col >= out_features) return;

    float acc = 0.0f;
    for (int k = 0; k < in_features; ++k) {
        const float a = to_float<T>(input[(size_t)row * in_features + k]);
        // torch.nn.Linear stores weight as (out_features, in_features).
        const float b = to_float<T>(weight[(size_t)col * in_features + k]);
        acc += a * b;
    }

    output[(size_t)row * out_features + col] = from_float<T>(acc);
}

inline void check_rmsnorm_linear_naive_inputs(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& gamma)
{
    TORCH_CHECK(x.is_cuda(),      "rmsnorm_linear_naive_cuda: x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "rmsnorm_linear_naive_cuda: weight must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(),  "rmsnorm_linear_naive_cuda: gamma must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(),      "rmsnorm_linear_naive_cuda: x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "rmsnorm_linear_naive_cuda: weight must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(),  "rmsnorm_linear_naive_cuda: gamma must be contiguous");
    TORCH_CHECK(x.dim() >= 1,       "rmsnorm_linear_naive_cuda: x must have at least 1 dim");
    TORCH_CHECK(weight.dim() == 2,  "rmsnorm_linear_naive_cuda: weight must be 2-D");
    TORCH_CHECK(gamma.dim() == 1,   "rmsnorm_linear_naive_cuda: gamma must be 1-D");
    TORCH_CHECK(x.scalar_type() == weight.scalar_type() &&
                x.scalar_type() == gamma.scalar_type(),
                "rmsnorm_linear_naive_cuda: x, weight, and gamma must share dtype");

    const int64_t hidden = x.size(-1);
    TORCH_CHECK(hidden > 0,
                "rmsnorm_linear_naive_cuda: x.size(-1) must be positive");
    TORCH_CHECK(hidden == weight.size(1),
                "rmsnorm_linear_naive_cuda: x.size(-1) must equal weight.size(1)");
    TORCH_CHECK(hidden == gamma.size(0),
                "rmsnorm_linear_naive_cuda: x.size(-1) must equal gamma.size(0)");
    TORCH_CHECK(x.numel() / hidden <= (int64_t)std::numeric_limits<int>::max(),
                "rmsnorm_linear_naive_cuda: too many rows for a 1-D grid");
    TORCH_CHECK(hidden <= (int64_t)std::numeric_limits<int>::max(),
                "rmsnorm_linear_naive_cuda: hidden size too large for a 1-D grid");
    TORCH_CHECK(weight.size(0) <= (int64_t)std::numeric_limits<int>::max(),
                "rmsnorm_linear_naive_cuda: output size too large for a 1-D grid");
}

template <typename T>
void launch_linear_naive(
    const torch::Tensor& input_2d,
    const torch::Tensor& weight,
    torch::Tensor& output_2d)
{
    const int n_rows = static_cast<int>(input_2d.size(0));
    const int in_features = static_cast<int>(input_2d.size(1));
    const int out_features = static_cast<int>(weight.size(0));
    if (n_rows == 0 || out_features == 0) return;

    const dim3 block(16, 16);
    const dim3 grid(
        (out_features + block.x - 1) / block.x,
        (n_rows + block.y - 1) / block.y);
    auto stream = at::cuda::getCurrentCUDAStream();

    linear_naive_kernel<T><<<grid, block, 0, stream>>>(
        reinterpret_cast<const T*>(input_2d.data_ptr()),
        reinterpret_cast<const T*>(weight.data_ptr()),
        reinterpret_cast<T*>(output_2d.data_ptr()),
        n_rows,
        in_features,
        out_features);
    AT_CUDA_CHECK(cudaGetLastError());
}

torch::Tensor rmsnorm_linear_naive_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& gamma,
    double eps)
{
    check_rmsnorm_linear_naive_inputs(x, weight, gamma);

    const int64_t hidden = x.size(-1);
    const int64_t out_features = weight.size(0);
    const torch::Tensor normed =
        rmsnorm_forward_cuda(x, gamma, eps, "rmsnorm_linear_naive_cuda");

    const int64_t n_rows = x.numel() / hidden;
    torch::Tensor normed_2d = normed.reshape({n_rows, hidden});
    torch::Tensor out_2d = torch::empty({n_rows, out_features}, x.options());

    switch (x.scalar_type()) {
        case at::ScalarType::Float:
            launch_linear_naive<float>(normed_2d, weight, out_2d);
            break;
        case at::ScalarType::Half:
            launch_linear_naive<__half>(normed_2d, weight, out_2d);
            break;
        case at::ScalarType::BFloat16:
            launch_linear_naive<__nv_bfloat16>(normed_2d, weight, out_2d);
            break;
        default:
            TORCH_CHECK(false,
                "rmsnorm_linear_naive_cuda: unsupported dtype ", x.scalar_type(),
                " (expected float32/float16/bfloat16)");
    }

    std::vector<int64_t> out_shape(x.sizes().begin(), x.sizes().end());
    out_shape.back() = out_features;
    return out_2d.reshape(out_shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_linear_naive_cuda", &rmsnorm_linear_naive_cuda,
          "RMSNormLinear naive: materialized RMSNorm plus untiled custom GEMM",
          pybind11::arg("x"),
          pybind11::arg("weight"),
          pybind11::arg("gamma"),
          pybind11::arg("eps") = 1e-6);
}
