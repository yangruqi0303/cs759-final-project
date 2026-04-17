"""PyTorch reference implementations for RMSNorm and fused MLP variants.

These are the "obviously correct" baselines that CUDA kernels will be
validated against.  No optimisation tricks — just readable PyTorch.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalisation (no bias).

    y = x * rsqrt(mean(x^2, dim=-1) + eps) * weight

    Args:
        hidden_size: Size of the last dimension to normalise over.
        eps: Small constant for numerical stability.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RMSNormLinear(nn.Module):
    """RMSNorm followed by a linear projection: Linear(RMSNorm(x)).

    Fusion target for CUDA kernel v1.

    Args:
        hidden_size: Input (and normalisation) dimension.
        out_features: Output dimension of the linear layer.
        eps: RMSNorm epsilon.
        bias: Whether the linear layer has a bias term.
    """

    def __init__(
        self,
        hidden_size: int,
        out_features: int,
        eps: float = 1e-6,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=eps)
        self.linear = nn.Linear(hidden_size, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


class RMSNormMLP(nn.Module):
    """RMSNorm + two-layer MLP with GELU: Linear2(GELU(Linear1(RMSNorm(x)))).

    Uses the tanh-approximated GELU (standard in Llama / Qwen).
    Fusion target for CUDA kernel v2.

    Args:
        hidden_size: Input (and normalisation) dimension.
        intermediate_size: Hidden dimension of the MLP.
        eps: RMSNorm epsilon.
        bias: Whether linear layers have bias terms.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        eps: float = 1e-6,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=eps)
        self.linear1 = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.act = nn.GELU(approximate="tanh")
        self.linear2 = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.linear1(self.norm(x)))
        return self.linear2(h)
