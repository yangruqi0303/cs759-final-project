"""Shared helpers for plot scripts.

Dependencies: matplotlib + numpy + standard library only (no pandas / seaborn).
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(REPO, "results")
FIGURES = os.path.join(HERE, "figures")
os.makedirs(FIGURES, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "legend.frameon": False,
    "legend.fontsize": 9,
})

DTYPES = ("float32", "float16", "bfloat16")
DTYPE_LABEL = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}
DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}

# Tier inference from hidden size.
TIERS = [(1024, "small (H=1024, I=4096)"),
         (2048, "medium (H=2048, I=8192)"),
         (4096, "large (H=4096, I=11008)")]

# Colors: PyTorch baseline = neutral gray. CUDA family in distinct hues.
C_PT = "#7f7f7f"
C_RMSNORM = "#1f77b4"
C_LINEAR = "#2ca02c"
C_MLP = "#d62728"
C_SPLIT = "#9467bd"

VARIANT_ORDER = [
    "naive_rmsnorm_linear",
    "tiled_rmsnorm_linear",
    "prologue_rmsnorm_linear",
    "tiled_rmsnorm_linear_v2",
    "prologue_rmsnorm_linear_v2",
    "fused_rmsnorm_linear",  # cuBLAS default
]
VARIANT_LABEL = {
    "naive_rmsnorm_linear":        "naive custom GEMM  (no tiling)",
    "tiled_rmsnorm_linear":        "custom GEMM v1",
    "prologue_rmsnorm_linear":     "custom GEMM v1  ·  w/ prologue",
    "tiled_rmsnorm_linear_v2":     "custom GEMM v2",
    "prologue_rmsnorm_linear_v2":  "custom GEMM v2  ·  w/ prologue",
    "fused_rmsnorm_linear":        "cuBLAS  (Tensor-Core reference)",
}
# What "v1" and "v2" mean for the custom GEMMs:
#   v1 = shared-memory tiled, scalar FMA, 1 thread = 1 output element
#        (tiled_linear_common.cuh: BLOCK_M=BLOCK_N=16, TILE_K=32)
#   v2 = v1 + register sub-tile per thread (TM=TN=4) + 4-elem vec loads
#        + cached gamma in shared mem + cached scale in registers
#        (tiled_linear_common.cuh: BLOCK_M=BLOCK_N=64, TILE_K=16)
# Neither uses Tensor Cores; cuBLAS does.
# Sequential colour ramp: orange = naive baseline,  blue gradient = GEMM v1→v2
# with no-prologue / with-prologue interleaved,  green = cuBLAS reference.
VARIANT_COLOR = {
    "naive_rmsnorm_linear":        "#fdae61",
    "tiled_rmsnorm_linear":        "#abd9e9",
    "prologue_rmsnorm_linear":     "#74add1",
    "tiled_rmsnorm_linear_v2":     "#4575b4",
    "prologue_rmsnorm_linear_v2":  "#313695",
    "fused_rmsnorm_linear":        "#2ca02c",
}


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------
@dataclass
class Row:
    kernel: str  # "" for PyTorch baseline
    module: str
    batch: int
    seq_len: int
    hidden: int
    intermediate: int
    median_ms: float
    p10_ms: float
    p90_ms: float
    min_ms: float
    dtype: str
    config: str = ""

    @property
    def total_tokens(self) -> int:
        return self.batch * self.seq_len

    @property
    def shape_label(self) -> str:
        return f"B{self.batch}S{self.seq_len}"

    @property
    def tier(self) -> str:
        for h, name in TIERS:
            if self.hidden == h:
                return name
        return f"hidden={self.hidden}"

    @property
    def tier_short(self) -> str:
        return self.tier.split(" ")[0]

    @property
    def shape_key(self) -> tuple:
        return (self.batch, self.seq_len, self.hidden, self.intermediate)


def _load(path: str) -> list[Row]:
    rows: list[Row] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(Row(
                kernel=r.get("kernel", ""),
                module=r["module"],
                batch=int(r["batch"]),
                seq_len=int(r["seq_len"]),
                hidden=int(r["hidden"]),
                intermediate=int(r["intermediate"]),
                median_ms=float(r["median_ms"]),
                p10_ms=float(r["p10_ms"]),
                p90_ms=float(r["p90_ms"]),
                min_ms=float(r["min_ms"]),
                dtype=r["dtype"],
                config=r.get("config", ""),
            ))
    return rows


def load_pytorch() -> list[Row]:
    return _load(os.path.join(RESULTS, "pytorch_baseline.csv"))


def load_cuda() -> list[Row]:
    return _load(os.path.join(RESULTS, "cuda_kernels.csv"))


def load_split() -> list[Row]:
    return _load(os.path.join(RESULTS, "rmsnorm_linear_split.csv"))


def load_variants() -> list[Row]:
    return _load(os.path.join(RESULTS, "rmsnorm_linear_variants.csv"))


# ---------------------------------------------------------------------------
# Indexing / lookup
# ---------------------------------------------------------------------------
def by_shape(rows: Iterable[Row], dtype: str = None,
             kernel: str = None, module: str = None) -> dict[tuple, Row]:
    """Index by (batch, seq, hidden, intermediate)."""
    out: dict[tuple, Row] = {}
    for r in rows:
        if dtype is not None and r.dtype != dtype:
            continue
        if kernel is not None and r.kernel != kernel:
            continue
        if module is not None and r.module != module:
            continue
        out[r.shape_key] = r
    return out


def shape_order_for_tier(tier_hidden: int) -> list[tuple]:
    """Return shape keys in canonical order (matches baseline/configs.py)."""
    token_shapes = [(1, 128), (1, 256), (1, 512), (1, 2048),
                    (4, 512), (4, 2048), (8, 1024)]
    intermediate = {1024: 4096, 2048: 8192, 4096: 11008}[tier_hidden]
    return [(b, s, tier_hidden, intermediate) for b, s in token_shapes]


def all_shape_keys() -> list[tuple]:
    keys = []
    for h, _ in TIERS:
        keys.extend(shape_order_for_tier(h))
    return keys


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def geomean(xs: Iterable[float]) -> float:
    xs = [x for x in xs if x > 0 and math.isfinite(x)]
    if not xs:
        return float("nan")
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def rmsnorm_bytes(batch: int, seq: int, hidden: int, dtype: str) -> int:
    """Approximate bytes touched: read x + write y (both N*H*sizeof) + small gamma."""
    n = batch * seq
    sz = DTYPE_BYTES[dtype]
    return 2 * n * hidden * sz + hidden * sz


def gbps(bytes_: int, ms: float) -> float:
    if ms <= 0:
        return float("nan")
    return bytes_ / (ms * 1e-3) / 1e9


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------
def save(fig, name: str) -> str:
    path = os.path.join(FIGURES, name)
    fig.savefig(path)
    print(f"  → {os.path.relpath(path, REPO)}")
    return path
