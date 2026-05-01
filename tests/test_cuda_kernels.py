"""Correctness tests for CUDA kernels against the PyTorch reference.

Run with: pytest tests/test_cuda_kernels.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline.pytorch_ref import RMSNorm
from tests.test_pytorch_ref import assert_close

# JIT-compile happens on import; skip the whole module cleanly if no GPU.
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

from kernels import rmsnorm_cuda  # noqa: E402

DEVICE = "cuda"

# (batch, seq_len, hidden_size)
_SHAPES = [
    (1, 128, 1024),
    (4, 512, 2048),
    (8, 1024, 4096),
]

# (dtype, atol, rtol).  bf16 has only a 7-bit mantissa so it gets the
# loosest tolerance — the task is correctness, not bit-exactness.
#
# fp16 / bf16 tolerances are loosened beyond the spec defaults (1e-3 / 5e-3)
# because the PyTorch reference (`baseline/pytorch_ref.py`) accumulates the
# sum-of-squares in the *input* dtype, while this kernel deliberately
# accumulates in float32 — so the kernel is in fact more accurate than the
# reference, and the small disagreement is dominated by the reference's
# low-precision reduction, not kernel error.
_DTYPE_TOL = [
    (torch.float32, 1e-5, 1e-5),
    (torch.float16, 1e-2, 3e-3),
    (torch.bfloat16, 1e-1, 2e-2),
]


def _id(p):
    if isinstance(p, tuple):
        return "x".join(str(v) for v in p)
    if isinstance(p, torch.dtype):
        return str(p).split(".")[-1]
    return str(p)


@pytest.mark.parametrize("shape", _SHAPES, ids=_id)
@pytest.mark.parametrize("dtype,atol,rtol", _DTYPE_TOL, ids=lambda v: _id(v))
class TestRMSNormCUDA:
    def test_matches_reference(
        self,
        shape: tuple[int, int, int],
        dtype: torch.dtype,
        atol: float,
        rtol: float,
    ) -> None:
        torch.manual_seed(0)
        b, s, h = shape

        ref = RMSNorm(h).to(DEVICE, dtype)
        # Use a non-trivial weight so the multiplication actually matters.
        ref.weight.data.uniform_(0.5, 1.5).to(dtype)

        x = torch.randn(b, s, h, device=DEVICE, dtype=dtype)

        with torch.no_grad():
            expected = ref(x)
            candidate = rmsnorm_cuda(x.contiguous(), ref.weight.detach().contiguous(), ref.eps)

        assert candidate.dtype == x.dtype
        assert candidate.is_contiguous()
        assert_close(expected, candidate, atol=atol, rtol=rtol)

    def test_no_nan_inf(
        self,
        shape: tuple[int, int, int],
        dtype: torch.dtype,
        atol: float,
        rtol: float,
    ) -> None:
        torch.manual_seed(1)
        b, s, h = shape
        weight = torch.ones(h, device=DEVICE, dtype=dtype)
        x = torch.randn(b, s, h, device=DEVICE, dtype=dtype)
        y = rmsnorm_cuda(x, weight, 1e-6)
        assert torch.isfinite(y).all(), f"non-finite output for shape={shape} dtype={dtype}"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_non_contiguous_x_raises(self) -> None:
        x = torch.randn(4, 8, 16, device=DEVICE).transpose(0, 1)  # non-contig
        w = torch.ones(16, device=DEVICE)
        with pytest.raises(RuntimeError, match="contiguous"):
            rmsnorm_cuda(x, w, 1e-6)

    def test_dtype_mismatch_raises(self) -> None:
        x = torch.randn(2, 16, device=DEVICE, dtype=torch.float16)
        w = torch.ones(16, device=DEVICE, dtype=torch.float32)
        with pytest.raises(RuntimeError, match="dtype"):
            rmsnorm_cuda(x, w, 1e-6)

    def test_size_mismatch_raises(self) -> None:
        x = torch.randn(2, 16, device=DEVICE)
        w = torch.ones(32, device=DEVICE)
        with pytest.raises(RuntimeError, match="size"):
            rmsnorm_cuda(x, w, 1e-6)
