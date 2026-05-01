"""JIT-compiled CUDA kernels for the cs759 final project.

Importing this package compiles `rmsnorm.cu` on first use via
`torch.utils.cpp_extension.load` and exposes the entry points as plain
Python callables.

Usage:
    from kernels import rmsnorm_cuda
    y = rmsnorm_cuda(x, weight, eps=1e-6)
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
    "-O3",
    "-std=c++17",
    "-arch=sm_120",
    "--expt-relaxed-constexpr",
] + _iflags

_ext = load(
    name="rmsnorm_cuda_ext",
    sources=[str(_THIS_DIR / "rmsnorm.cu")],
    extra_cflags=_extra_cflags,
    extra_cuda_cflags=_extra_cuda_cflags,
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
    return _ext.rmsnorm_cuda(x, weight, eps)


__all__ = ["rmsnorm_cuda"]
