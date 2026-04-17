"""Benchmark shape configurations for Transformer RMSNorm + MLP experiments.

Each config represents a realistic (batch, seq_len, hidden_size, intermediate_size)
workload.  Configs are organised so total tokens (batch * seq_len) spans ~512-8192.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchConfig:
    """Single benchmark shape configuration."""

    name: str
    batch: int
    seq_len: int
    hidden_size: int
    intermediate_size: int

    @property
    def total_tokens(self) -> int:
        return self.batch * self.seq_len


# ---------------------------------------------------------------------------
# Model-size tiers
# ---------------------------------------------------------------------------
_SMALL = dict(hidden_size=1024, intermediate_size=4096)
_MEDIUM = dict(hidden_size=2048, intermediate_size=8192)
_LARGE = dict(hidden_size=4096, intermediate_size=11008)  # Qwen2 / Llama-7B

# ---------------------------------------------------------------------------
# Token-count sweeps per model size
# ---------------------------------------------------------------------------
_TOKEN_SHAPES: list[tuple[int, int]] = [
    (1, 512),
    (1, 2048),
    (4, 512),
    (4, 2048),
    (8, 1024),
]

BENCH_CONFIGS: list[BenchConfig] = []

for tier_name, tier_kwargs in [("small", _SMALL), ("medium", _MEDIUM), ("large", _LARGE)]:
    for batch, seq_len in _TOKEN_SHAPES:
        cfg = BenchConfig(
            name=f"{tier_name}_b{batch}_s{seq_len}",
            batch=batch,
            seq_len=seq_len,
            **tier_kwargs,
        )
        BENCH_CONFIGS.append(cfg)
