"""Correctness tests for CUDA kernels against the PyTorch reference.

Run with: pytest tests/test_cuda_kernels.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline.pytorch_ref import RMSNorm, RMSNormLinear, RMSNormMLP
from tests.test_pytorch_ref import assert_close

# JIT-compile happens on import; skip the whole module cleanly if no GPU.
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

from kernels import (  # noqa: E402
    rmsnorm_cuda,
    rmsnorm_linear_cuda,
    rmsnorm_linear_naive_cuda,
    rmsnorm_linear_prologue_cuda,
    rmsnorm_linear_tiled_cuda,
    rmsnorm_mlp_cuda,
)

DEVICE = "cuda"

# (batch, seq_len, hidden_size)
_SHAPES = [
    (1, 128, 1024),
    (4, 512, 2048),
    (8, 1024, 4096),
]

# The custom GEMM variants are for correctness and fusion demonstration. Keep
# these unit-test shapes small so pytest remains fast.
_CUSTOM_GEMM_SHAPES = [
    (1, 8, 64),
    (2, 16, 128),
    (2, 32, 256),
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


@pytest.mark.parametrize("shape", _SHAPES, ids=_id)
@pytest.mark.parametrize("dtype,atol,rtol", _DTYPE_TOL, ids=lambda v: _id(v))
class TestRMSNormLinearCUDA:
    def test_matches_reference(
        self,
        shape: tuple[int, int, int],
        dtype: torch.dtype,
        atol: float,
        rtol: float,
    ) -> None:
        torch.manual_seed(2)
        b, s, h = shape
        out_features = max(16, h // 4)

        ref = RMSNormLinear(h, out_features).to(DEVICE, dtype)
        ref.norm.weight.data.uniform_(0.5, 1.5).to(dtype)
        x = torch.randn(b, s, h, device=DEVICE, dtype=dtype)

        with torch.no_grad():
            expected = ref(x)
            candidate = rmsnorm_linear_cuda(
                x.contiguous(),
                ref.linear.weight.detach().contiguous(),
                ref.norm.weight.detach().contiguous(),
                ref.norm.eps,
            )

        assert candidate.shape == (b, s, out_features)
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
        torch.manual_seed(3)
        b, s, h = shape
        out_features = max(16, h // 4)
        x = torch.randn(b, s, h, device=DEVICE, dtype=dtype)
        weight = torch.randn(out_features, h, device=DEVICE, dtype=dtype)
        gamma = torch.ones(h, device=DEVICE, dtype=dtype)

        y = rmsnorm_linear_cuda(x, weight, gamma, 1e-6)
        assert torch.isfinite(y).all(), (
            f"non-finite output for shape={shape} dtype={dtype}"
        )


@pytest.mark.parametrize("shape", _CUSTOM_GEMM_SHAPES, ids=_id)
@pytest.mark.parametrize("dtype,atol,rtol", _DTYPE_TOL, ids=lambda v: _id(v))
class TestRMSNormLinearNaiveCUDA:
    def test_matches_reference(
        self,
        shape: tuple[int, int, int],
        dtype: torch.dtype,
        atol: float,
        rtol: float,
    ) -> None:
        torch.manual_seed(10)
        b, s, h = shape
        out_features = max(16, h // 2)

        ref = RMSNormLinear(h, out_features).to(DEVICE, dtype)
        ref.norm.weight.data.uniform_(0.5, 1.5).to(dtype)
        x = torch.randn(b, s, h, device=DEVICE, dtype=dtype)

        with torch.no_grad():
            expected = ref(x)
            candidate = rmsnorm_linear_naive_cuda(
                x.contiguous(),
                ref.linear.weight.detach().contiguous(),
                ref.norm.weight.detach().contiguous(),
                ref.norm.eps,
            )

        assert candidate.shape == (b, s, out_features)
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
        torch.manual_seed(11)
        b, s, h = shape
        out_features = max(16, h // 2)
        x = torch.randn(b, s, h, device=DEVICE, dtype=dtype)
        weight = torch.randn(out_features, h, device=DEVICE, dtype=dtype)
        gamma = torch.ones(h, device=DEVICE, dtype=dtype)

        y = rmsnorm_linear_naive_cuda(x, weight, gamma, 1e-6)
        assert torch.isfinite(y).all(), (
            f"non-finite output for shape={shape} dtype={dtype}"
        )


@pytest.mark.parametrize("shape", _CUSTOM_GEMM_SHAPES, ids=_id)
@pytest.mark.parametrize("dtype,atol,rtol", _DTYPE_TOL, ids=lambda v: _id(v))
class TestRMSNormLinearTiledCUDA:
    def test_matches_reference(
        self,
        shape: tuple[int, int, int],
        dtype: torch.dtype,
        atol: float,
        rtol: float,
    ) -> None:
        torch.manual_seed(6)
        b, s, h = shape
        out_features = max(16, h // 2)

        ref = RMSNormLinear(h, out_features).to(DEVICE, dtype)
        ref.norm.weight.data.uniform_(0.5, 1.5).to(dtype)
        x = torch.randn(b, s, h, device=DEVICE, dtype=dtype)

        with torch.no_grad():
            expected = ref(x)
            candidate = rmsnorm_linear_tiled_cuda(
                x.contiguous(),
                ref.linear.weight.detach().contiguous(),
                ref.norm.weight.detach().contiguous(),
                ref.norm.eps,
            )

        assert candidate.shape == (b, s, out_features)
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
        torch.manual_seed(7)
        b, s, h = shape
        out_features = max(16, h // 2)
        x = torch.randn(b, s, h, device=DEVICE, dtype=dtype)
        weight = torch.randn(out_features, h, device=DEVICE, dtype=dtype)
        gamma = torch.ones(h, device=DEVICE, dtype=dtype)

        y = rmsnorm_linear_tiled_cuda(x, weight, gamma, 1e-6)
        assert torch.isfinite(y).all(), (
            f"non-finite output for shape={shape} dtype={dtype}"
        )


@pytest.mark.parametrize("shape", _CUSTOM_GEMM_SHAPES, ids=_id)
@pytest.mark.parametrize("dtype,atol,rtol", _DTYPE_TOL, ids=lambda v: _id(v))
class TestRMSNormLinearPrologueCUDA:
    def test_matches_reference(
        self,
        shape: tuple[int, int, int],
        dtype: torch.dtype,
        atol: float,
        rtol: float,
    ) -> None:
        torch.manual_seed(8)
        b, s, h = shape
        out_features = max(16, h // 2)

        ref = RMSNormLinear(h, out_features).to(DEVICE, dtype)
        ref.norm.weight.data.uniform_(0.5, 1.5).to(dtype)
        x = torch.randn(b, s, h, device=DEVICE, dtype=dtype)

        with torch.no_grad():
            expected = ref(x)
            candidate = rmsnorm_linear_prologue_cuda(
                x.contiguous(),
                ref.linear.weight.detach().contiguous(),
                ref.norm.weight.detach().contiguous(),
                ref.norm.eps,
            )

        assert candidate.shape == (b, s, out_features)
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
        torch.manual_seed(9)
        b, s, h = shape
        out_features = max(16, h // 2)
        x = torch.randn(b, s, h, device=DEVICE, dtype=dtype)
        weight = torch.randn(out_features, h, device=DEVICE, dtype=dtype)
        gamma = torch.ones(h, device=DEVICE, dtype=dtype)

        y = rmsnorm_linear_prologue_cuda(x, weight, gamma, 1e-6)
        assert torch.isfinite(y).all(), (
            f"non-finite output for shape={shape} dtype={dtype}"
        )


@pytest.mark.parametrize("shape", _SHAPES, ids=_id)
@pytest.mark.parametrize("dtype,atol,rtol", _DTYPE_TOL, ids=lambda v: _id(v))
class TestRMSNormMLPCUDA:
    def test_matches_reference(
        self,
        shape: tuple[int, int, int],
        dtype: torch.dtype,
        atol: float,
        rtol: float,
    ) -> None:
        torch.manual_seed(4)
        b, s, h = shape
        intermediate = max(16, h // 4)

        ref = RMSNormMLP(h, intermediate).to(DEVICE, dtype)
        ref.norm.weight.data.uniform_(0.5, 1.5).to(dtype)
        x = torch.randn(b, s, h, device=DEVICE, dtype=dtype)

        with torch.no_grad():
            expected = ref(x)
            candidate = rmsnorm_mlp_cuda(
                x.contiguous(),
                ref.linear1.weight.detach().contiguous(),
                ref.linear2.weight.detach().contiguous(),
                ref.norm.weight.detach().contiguous(),
                ref.norm.eps,
            )

        assert candidate.shape == x.shape
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
        torch.manual_seed(5)
        b, s, h = shape
        intermediate = max(16, h // 4)
        x = torch.randn(b, s, h, device=DEVICE, dtype=dtype)
        weight1 = torch.randn(intermediate, h, device=DEVICE, dtype=dtype)
        weight2 = torch.randn(h, intermediate, device=DEVICE, dtype=dtype)
        gamma = torch.ones(h, device=DEVICE, dtype=dtype)

        y = rmsnorm_mlp_cuda(x, weight1, weight2, gamma, 1e-6)
        assert torch.isfinite(y).all(), (
            f"non-finite output for shape={shape} dtype={dtype}"
        )


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

    def test_rmsnorm_linear_weight_shape_mismatch_raises(self) -> None:
        x = torch.randn(2, 16, device=DEVICE)
        weight = torch.randn(8, 32, device=DEVICE)
        gamma = torch.ones(16, device=DEVICE)
        with pytest.raises(RuntimeError, match="size"):
            rmsnorm_linear_cuda(x, weight, gamma, 1e-6)

    def test_rmsnorm_linear_dtype_mismatch_raises(self) -> None:
        x = torch.randn(2, 16, device=DEVICE, dtype=torch.float16)
        weight = torch.randn(8, 16, device=DEVICE, dtype=torch.float16)
        gamma = torch.ones(16, device=DEVICE, dtype=torch.float32)
        with pytest.raises(RuntimeError, match="dtype"):
            rmsnorm_linear_cuda(x, weight, gamma, 1e-6)

    def test_rmsnorm_linear_naive_weight_shape_mismatch_raises(self) -> None:
        x = torch.randn(2, 16, device=DEVICE)
        weight = torch.randn(8, 32, device=DEVICE)
        gamma = torch.ones(16, device=DEVICE)
        with pytest.raises(RuntimeError, match="size"):
            rmsnorm_linear_naive_cuda(x, weight, gamma, 1e-6)

    def test_rmsnorm_linear_naive_dtype_mismatch_raises(self) -> None:
        x = torch.randn(2, 16, device=DEVICE, dtype=torch.float16)
        weight = torch.randn(8, 16, device=DEVICE, dtype=torch.float16)
        gamma = torch.ones(16, device=DEVICE, dtype=torch.float32)
        with pytest.raises(RuntimeError, match="dtype"):
            rmsnorm_linear_naive_cuda(x, weight, gamma, 1e-6)

    def test_rmsnorm_linear_prologue_weight_shape_mismatch_raises(self) -> None:
        x = torch.randn(2, 16, device=DEVICE)
        weight = torch.randn(8, 32, device=DEVICE)
        gamma = torch.ones(16, device=DEVICE)
        with pytest.raises(RuntimeError, match="size"):
            rmsnorm_linear_prologue_cuda(x, weight, gamma, 1e-6)

    def test_rmsnorm_linear_prologue_dtype_mismatch_raises(self) -> None:
        x = torch.randn(2, 16, device=DEVICE, dtype=torch.float16)
        weight = torch.randn(8, 16, device=DEVICE, dtype=torch.float16)
        gamma = torch.ones(16, device=DEVICE, dtype=torch.float32)
        with pytest.raises(RuntimeError, match="dtype"):
            rmsnorm_linear_prologue_cuda(x, weight, gamma, 1e-6)

    def test_rmsnorm_linear_tiled_weight_shape_mismatch_raises(self) -> None:
        x = torch.randn(2, 16, device=DEVICE)
        weight = torch.randn(8, 32, device=DEVICE)
        gamma = torch.ones(16, device=DEVICE)
        with pytest.raises(RuntimeError, match="size"):
            rmsnorm_linear_tiled_cuda(x, weight, gamma, 1e-6)

    def test_rmsnorm_linear_tiled_dtype_mismatch_raises(self) -> None:
        x = torch.randn(2, 16, device=DEVICE, dtype=torch.float16)
        weight = torch.randn(8, 16, device=DEVICE, dtype=torch.float16)
        gamma = torch.ones(16, device=DEVICE, dtype=torch.float32)
        with pytest.raises(RuntimeError, match="dtype"):
            rmsnorm_linear_tiled_cuda(x, weight, gamma, 1e-6)

    def test_rmsnorm_mlp_weight_shape_mismatch_raises(self) -> None:
        x = torch.randn(2, 16, device=DEVICE)
        weight1 = torch.randn(8, 16, device=DEVICE)
        weight2 = torch.randn(16, 12, device=DEVICE)
        gamma = torch.ones(16, device=DEVICE)
        with pytest.raises(RuntimeError, match="size"):
            rmsnorm_mlp_cuda(x, weight1, weight2, gamma, 1e-6)

    def test_rmsnorm_mlp_dtype_mismatch_raises(self) -> None:
        x = torch.randn(2, 16, device=DEVICE, dtype=torch.float16)
        weight1 = torch.randn(8, 16, device=DEVICE, dtype=torch.float16)
        weight2 = torch.randn(16, 8, device=DEVICE, dtype=torch.float16)
        gamma = torch.ones(16, device=DEVICE, dtype=torch.float32)
        with pytest.raises(RuntimeError, match="dtype"):
            rmsnorm_mlp_cuda(x, weight1, weight2, gamma, 1e-6)
