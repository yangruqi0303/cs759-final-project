"""JIT-compiled CUDA kernels for the cs759 final project.

Importing this package compiles CUDA sources on first use via
`torch.utils.cpp_extension.load` and exposes the entry points as plain
Python callables.

Usage:
    from kernels import (
        rmsnorm_cuda,
        rmsnorm_linear_cuda,
        rmsnorm_linear_naive_cuda,
        rmsnorm_linear_tiled_cuda,
        rmsnorm_linear_prologue_cuda,
        rmsnorm_mlp_cuda,
    )
    y = rmsnorm_cuda(x, weight, eps=1e-6)
    z = rmsnorm_linear_cuda(x, linear_weight, gamma, eps=1e-6)
    z_naive = rmsnorm_linear_naive_cuda(x, linear_weight, gamma, eps=1e-6)
    z_tiled = rmsnorm_linear_tiled_cuda(x, linear_weight, gamma, eps=1e-6)
    z_prologue = rmsnorm_linear_prologue_cuda(x, linear_weight, gamma, eps=1e-6)
    h = rmsnorm_mlp_cuda(x, linear1_weight, linear2_weight, gamma, eps=1e-6)
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_THIS_DIR = Path(__file__).resolve().parent

# PyTorch 2.11 / CUDA 13 ships its CUDA headers (cusparse.h, etc.) inside
# the `nvidia.cu13` wheel rather than /usr/local/cuda, so we need to add
# that include path explicitly for ATen's headers to compile.
def _nvidia_cuda_include() -> list[str]:
    try:
        import nvidia.cu13 as _cu13  # type: ignore
    except Exception:
        return []
    # `nvidia.cu13` is a namespace package, so __file__ may be None;
    # its __path__ list is the reliable source.
    for base in getattr(_cu13, "__path__", []):
        inc = Path(base) / "include"
        if inc.is_dir():
            return [str(inc)]
    return []


_extra_includes = _nvidia_cuda_include()
# `-isystem` lowers priority below the toolkit's own headers so nvcc 13.1
# keeps using the matching 13.1 cccl/cuda_runtime, and we only fall through
# to the wheel for the genuinely missing `cusparse.h`.
_iflags = []
for p in _extra_includes:
    _iflags += ["-isystem", p]

_extra_cflags = ["-O3", "-std=c++17"] + _iflags
_extra_cuda_cflags = [
    "-O3", "-std=c++17",
    "-gencode=arch=compute_89,code=sm_89",   # for 4060 Ti
    "-gencode=arch=compute_120,code=sm_120", # for 5070 Ti
    "--expt-relaxed-constexpr",
] + _iflags

# Keep each CUDA entry point as a separate JIT extension. That lets each .cu
# file own a small PYBIND11_MODULE while sharing templated CUDA code through
# rmsnorm_common.cuh.
_rmsnorm_ext = load(
    name="rmsnorm_cuda_ext",
    sources=[str(_THIS_DIR / "rmsnorm.cu")],
    extra_cflags=_extra_cflags,
    extra_cuda_cflags=_extra_cuda_cflags,
    verbose=True,
)

_rmsnorm_linear_ext = load(
    name="rmsnorm_linear_cuda_ext",
    sources=[str(_THIS_DIR / "rmsnorm_linear.cu")],
    extra_cflags=_extra_cflags,
    extra_cuda_cflags=_extra_cuda_cflags,
    extra_ldflags=["-lcublas"],
    verbose=True,
)

_rmsnorm_linear_naive_ext = load(
    name="rmsnorm_linear_naive_cuda_ext",
    sources=[str(_THIS_DIR / "rmsnorm_linear_naive.cu")],
    extra_cflags=_extra_cflags,
    extra_cuda_cflags=_extra_cuda_cflags,
    verbose=True,
)

_rmsnorm_linear_tiled_ext = load(
    name="rmsnorm_linear_tiled_cuda_ext",
    sources=[str(_THIS_DIR / "rmsnorm_linear_tiled.cu")],
    extra_cflags=_extra_cflags,
    extra_cuda_cflags=_extra_cuda_cflags,
    verbose=True,
)

_rmsnorm_linear_prologue_ext = load(
    name="rmsnorm_linear_prologue_cuda_ext",
    sources=[str(_THIS_DIR / "rmsnorm_linear_prologue.cu")],
    extra_cflags=_extra_cflags,
    extra_cuda_cflags=_extra_cuda_cflags,
    verbose=True,
)

_rmsnorm_mlp_ext = load(
    name="rmsnorm_mlp_cuda_ext",
    sources=[str(_THIS_DIR / "rmsnorm_mlp.cu")],
    extra_cflags=_extra_cflags,
    extra_cuda_cflags=_extra_cuda_cflags,
    extra_ldflags=["-lcublas"],
    verbose=True,
)


def rmsnorm_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply RMSNorm to the last dim of `x`.

    Args:
        x: contiguous CUDA tensor, shape ``(*, hidden_size)``,
           dtype fp32 / fp16 / bf16.
        weight: contiguous 1-D CUDA tensor of shape ``(hidden_size,)``,
           same dtype as ``x``.
        eps: numerical-stability epsilon.

    Returns:
        Tensor of the same shape and dtype as ``x``.
    """
    return _rmsnorm_ext.rmsnorm_cuda(x, weight, eps)


def rmsnorm_linear_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply RMSNorm followed by a cuBLAS linear projection.

    Args:
        x: contiguous CUDA tensor, shape ``(*, hidden_size)``,
           dtype fp32 / fp16 / bf16.
        weight: contiguous 2-D CUDA tensor of shape
           ``(out_features, hidden_size)``, same dtype as ``x``.
        gamma: contiguous 1-D CUDA tensor of shape ``(hidden_size,)``,
           same dtype as ``x``.
        eps: numerical-stability epsilon.

    Returns:
        Tensor of shape ``(*, out_features)`` and the same dtype as ``x``.
    """
    return _rmsnorm_linear_ext.rmsnorm_linear_cuda(x, weight, gamma, eps)


def rmsnorm_linear_naive_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply RMSNormLinear with materialized RMSNorm and untiled custom GEMM.

    This version is intentionally simple and slow: one CUDA thread computes one
    output element by reading the full hidden dimension from global memory. It
    is useful as the lowest custom-GEMM baseline in the variants benchmark.

    Args:
        x: contiguous CUDA tensor, shape ``(*, hidden_size)``,
           dtype fp32 / fp16 / bf16.
        weight: contiguous 2-D CUDA tensor of shape
           ``(out_features, hidden_size)``, same dtype as ``x``.
        gamma: contiguous 1-D CUDA tensor of shape ``(hidden_size,)``,
           same dtype as ``x``.
        eps: numerical-stability epsilon.

    Returns:
        Tensor of shape ``(*, out_features)`` and the same dtype as ``x``.
    """
    return _rmsnorm_linear_naive_ext.rmsnorm_linear_naive_cuda(
        x, weight, gamma, eps)


def rmsnorm_linear_tiled_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply RMSNormLinear with materialized RMSNorm and tiled custom GEMM.

    This version first writes the normalized tensor to global memory, then uses
    the shared scalar-FMA tiled GEMM helper. It is intended for comparison
    against the prologue-fused tiled version.

    Args:
        x: contiguous CUDA tensor, shape ``(*, hidden_size)``,
           dtype fp32 / fp16 / bf16.
        weight: contiguous 2-D CUDA tensor of shape
           ``(out_features, hidden_size)``, same dtype as ``x``.
        gamma: contiguous 1-D CUDA tensor of shape ``(hidden_size,)``,
           same dtype as ``x``.
        eps: numerical-stability epsilon.

    Returns:
        Tensor of shape ``(*, out_features)`` and the same dtype as ``x``.
    """
    return _rmsnorm_linear_tiled_ext.rmsnorm_linear_tiled_cuda(
        x, weight, gamma, eps)


def rmsnorm_linear_prologue_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply RMSNormLinear with a prologue-fused custom GEMM.

    This version computes one RMS scale per row, then applies
    ``x * scale * gamma`` inside the custom GEMM loop instead of materializing
    the full normalized tensor.

    Args:
        x: contiguous CUDA tensor, shape ``(*, hidden_size)``,
           dtype fp32 / fp16 / bf16.
        weight: contiguous 2-D CUDA tensor of shape
           ``(out_features, hidden_size)``, same dtype as ``x``.
        gamma: contiguous 1-D CUDA tensor of shape ``(hidden_size,)``,
           same dtype as ``x``.
        eps: numerical-stability epsilon.

    Returns:
        Tensor of shape ``(*, out_features)`` and the same dtype as ``x``.
    """
    return _rmsnorm_linear_prologue_ext.rmsnorm_linear_prologue_cuda(
        x, weight, gamma, eps)


def rmsnorm_mlp_cuda(
    x: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply RMSNorm followed by a two-layer GELU MLP.

    Args:
        x: contiguous CUDA tensor, shape ``(*, hidden_size)``,
           dtype fp32 / fp16 / bf16.
        weight1: contiguous 2-D CUDA tensor of shape
           ``(intermediate_size, hidden_size)``, same dtype as ``x``.
        weight2: contiguous 2-D CUDA tensor of shape
           ``(hidden_size, intermediate_size)``, same dtype as ``x``.
        gamma: contiguous 1-D CUDA tensor of shape ``(hidden_size,)``,
           same dtype as ``x``.
        eps: numerical-stability epsilon.

    Returns:
        Tensor of the same shape and dtype as ``x``.
    """
    return _rmsnorm_mlp_ext.rmsnorm_mlp_cuda(x, weight1, weight2, gamma, eps)


__all__ = [
    "rmsnorm_cuda",
    "rmsnorm_linear_cuda",
    "rmsnorm_linear_naive_cuda",
    "rmsnorm_linear_tiled_cuda",
    "rmsnorm_linear_prologue_cuda",
    "rmsnorm_mlp_cuda",
]
