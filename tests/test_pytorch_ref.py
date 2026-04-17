"""Correctness tests for PyTorch reference implementations.

Run with: pytest tests/test_pytorch_ref.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline.configs import BENCH_CONFIGS, BenchConfig
from baseline.pytorch_ref import RMSNorm, RMSNormLinear, RMSNormMLP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# A small config used by most unit tests.
_UNIT = BenchConfig(name="unit", batch=2, seq_len=64, hidden_size=128, intermediate_size=512)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_close(
    ref: torch.Tensor,
    candidate: torch.Tensor,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> None:
    """Assert two tensors match within tolerance.

    Designed so CUDA kernel outputs can be plugged in later.
    """
    assert ref.shape == candidate.shape, f"Shape mismatch: {ref.shape} vs {candidate.shape}"
    torch.testing.assert_close(ref, candidate, atol=atol, rtol=rtol)


def _make_input(cfg: BenchConfig, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.randn(cfg.batch, cfg.seq_len, cfg.hidden_size, device=DEVICE, dtype=dtype)


# ---------------------------------------------------------------------------
# RMSNorm tests
# ---------------------------------------------------------------------------

class TestRMSNorm:
    def test_output_shape(self) -> None:
        m = RMSNorm(_UNIT.hidden_size).to(DEVICE)
        x = _make_input(_UNIT)
        assert m(x).shape == x.shape

    def test_manual_formula(self) -> None:
        """Re-derive RMSNorm from raw ops and compare."""
        torch.manual_seed(42)
        m = RMSNorm(_UNIT.hidden_size).to(DEVICE)
        x = _make_input(_UNIT)

        expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + m.eps) * m.weight
        assert_close(expected, m(x))

    def test_gradient_flows(self) -> None:
        m = RMSNorm(_UNIT.hidden_size).to(DEVICE)
        x = _make_input(_UNIT).requires_grad_(True)
        loss = m(x).sum()
        loss.backward()
        assert x.grad is not None
        assert m.weight.grad is not None

    def test_golden_regression(self) -> None:
        """Fixed-seed output must be bit-stable across runs."""
        torch.manual_seed(123)
        m = RMSNorm(32).to(DEVICE)
        torch.manual_seed(456)
        x = torch.randn(1, 4, 32, device=DEVICE)
        y1 = m(x).detach().clone()

        torch.manual_seed(123)
        m2 = RMSNorm(32).to(DEVICE)
        torch.manual_seed(456)
        x2 = torch.randn(1, 4, 32, device=DEVICE)
        y2 = m2(x2)

        assert_close(y1, y2, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# RMSNormLinear tests
# ---------------------------------------------------------------------------

class TestRMSNormLinear:
    def test_output_shape(self) -> None:
        m = RMSNormLinear(_UNIT.hidden_size, _UNIT.intermediate_size).to(DEVICE)
        x = _make_input(_UNIT)
        out = m(x)
        assert out.shape == (
            _UNIT.batch,
            _UNIT.seq_len,
            _UNIT.intermediate_size,
        )

    def test_matches_decomposed(self) -> None:
        """Output must match manually chained RMSNorm -> Linear."""
        torch.manual_seed(99)
        m = RMSNormLinear(_UNIT.hidden_size, _UNIT.intermediate_size).to(DEVICE)
        x = _make_input(_UNIT)

        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + m.norm.eps) * m.norm.weight
        expected = torch.nn.functional.linear(normed, m.linear.weight, m.linear.bias)
        assert_close(expected, m(x), atol=1e-5, rtol=1e-4)

    def test_gradient_flows(self) -> None:
        m = RMSNormLinear(_UNIT.hidden_size, _UNIT.intermediate_size).to(DEVICE)
        x = _make_input(_UNIT).requires_grad_(True)
        loss = m(x).sum()
        loss.backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# RMSNormMLP tests
# ---------------------------------------------------------------------------

class TestRMSNormMLP:
    def test_output_shape(self) -> None:
        m = RMSNormMLP(_UNIT.hidden_size, _UNIT.intermediate_size).to(DEVICE)
        x = _make_input(_UNIT)
        out = m(x)
        assert out.shape == x.shape

    def test_matches_decomposed(self) -> None:
        """Output must match manually chained RMSNorm -> Linear1 -> GELU -> Linear2."""
        torch.manual_seed(77)
        m = RMSNormMLP(_UNIT.hidden_size, _UNIT.intermediate_size).to(DEVICE)
        x = _make_input(_UNIT)

        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + m.norm.eps) * m.norm.weight
        h = torch.nn.functional.gelu(
            torch.nn.functional.linear(normed, m.linear1.weight, m.linear1.bias),
            approximate="tanh",
        )
        expected = torch.nn.functional.linear(h, m.linear2.weight, m.linear2.bias)
        assert_close(expected, m(x), atol=1e-5, rtol=1e-4)

    def test_gradient_flows(self) -> None:
        m = RMSNormMLP(_UNIT.hidden_size, _UNIT.intermediate_size).to(DEVICE)
        x = _make_input(_UNIT).requires_grad_(True)
        loss = m(x).sum()
        loss.backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# Dtype sweep — no NaN/Inf on largest config
# ---------------------------------------------------------------------------

_LARGEST = max(BENCH_CONFIGS, key=lambda c: c.total_tokens * c.hidden_size)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
class TestDtypeStability:
    def _check_finite(self, t: torch.Tensor) -> None:
        assert torch.isfinite(t).all(), "Output contains NaN or Inf"

    def test_rmsnorm(self, dtype: torch.dtype) -> None:
        m = RMSNorm(_LARGEST.hidden_size).to(DEVICE, dtype)
        x = _make_input(_LARGEST, dtype)
        self._check_finite(m(x))

    def test_rmsnorm_linear(self, dtype: torch.dtype) -> None:
        m = RMSNormLinear(_LARGEST.hidden_size, _LARGEST.intermediate_size).to(DEVICE, dtype)
        x = _make_input(_LARGEST, dtype)
        self._check_finite(m(x))

    def test_rmsnorm_mlp(self, dtype: torch.dtype) -> None:
        m = RMSNormMLP(_LARGEST.hidden_size, _LARGEST.intermediate_size).to(DEVICE, dtype)
        x = _make_input(_LARGEST, dtype)
        self._check_finite(m(x))
