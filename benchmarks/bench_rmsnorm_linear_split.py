"""Benchmark for the *split* RMSNorm+Linear path: CUDA RMSNorm followed by
PyTorch (cuBLAS) Linear, with no cross-op fusion.

Why this exists:

The other two CSVs already on disk give us the two endpoints —
- `results/pytorch_baseline.csv`  : PT RMSNorm + PT Linear (`RMSNormLinear` row)
- `results/cuda_kernels.csv`      : `fused_rmsnorm_linear` (everything fused)

This bench fills in the middle row: the RMSNorm runs on the custom CUDA kernel,
its output is materialized to global memory, and the Linear is executed by
`torch.nn.functional.linear` (i.e. cuBLAS GEMM via PyTorch). Comparing the
three rows attributes speedup to its source:

    PT RMSN+PT Linear  →  CUDA RMSN + PT Linear : gain from the RMSNorm kernel
    CUDA RMSN + PT Linear  →  fused_rmsnorm_linear : gain from the fusion

CSV format mirrors `cuda_kernels.csv` (leading `kernel` column) so all three
files can be concatenated for plotting.

Usage:
    python benchmarks/bench_rmsnorm_linear_split.py --dtype fp32
    python benchmarks/bench_rmsnorm_linear_split.py --all-dtypes
"""

from __future__ import annotations

import argparse
import csv
import gc
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline.configs import BENCH_CONFIGS, BenchConfig
from kernels import rmsnorm_cuda

WARMUP = 10
TIMED = 200
KERNEL_NAME = "cuda_rmsnorm+pt_linear"

DTYPE_MAP: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _time_fn(fn: callable, warmup: int, iters: int) -> list[float]:
    """Time *fn()* using CUDA events.  Returns per-iteration ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def _percentile(data: list[float], p: float) -> float:
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _split_forward(
    x: torch.Tensor,
    linear_weight: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """CUDA RMSNorm followed by PyTorch (cuBLAS) Linear, two separate launches."""
    h = rmsnorm_cuda(x, gamma, eps)
    return F.linear(h, linear_weight)


def bench_one(cfg: BenchConfig, dtype: torch.dtype) -> dict[str, object]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    x = torch.randn(cfg.batch, cfg.seq_len, cfg.hidden_size, device="cuda", dtype=dtype)
    # Mirror the layout used by `bench_cuda.py::bench_rmsnorm_linear` so the
    # GEMM shape (and weight memory layout) is identical to the fused case.
    linear_weight = torch.randn(
        cfg.intermediate_size,
        cfg.hidden_size,
        device="cuda",
        dtype=dtype,
    )
    gamma = torch.ones(cfg.hidden_size, device="cuda", dtype=dtype)
    eps = 1e-6

    with torch.no_grad():
        times = _time_fn(
            lambda: _split_forward(x, linear_weight, gamma, eps),
            warmup=WARMUP,
            iters=TIMED,
        )

    median = statistics.median(times)
    p10 = _percentile(times, 10)
    p90 = _percentile(times, 90)
    min_t = min(times)

    return {
        "kernel": KERNEL_NAME,
        "module": "RMSNormLinear",
        "batch": cfg.batch,
        "seq_len": cfg.seq_len,
        "hidden": cfg.hidden_size,
        "intermediate": cfg.intermediate_size,
        "median_ms": round(median, 4),
        "p10_ms": round(p10, 4),
        "p90_ms": round(p90, 4),
        "min_ms": round(min_t, 4),
        "n_iters": TIMED,
        "dtype": str(dtype).split(".")[-1],
    }


FIELDNAMES = [
    "kernel", "module", "batch", "seq_len", "hidden", "intermediate",
    "median_ms", "p10_ms", "p90_ms", "min_ms", "n_iters", "dtype",
]

HEADER_FMT = (
    "{kernel:<24} {module:<14} {batch:>5} {seq:>5} {hidden:>6} {inter:>6} "
    "{median:>10} {p10:>10} {p90:>10} {min:>10} {dtype:>8}"
)


def _run_dtype(dtype: torch.dtype) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []

    print(f"\n=== dtype: {dtype} ===")
    print(HEADER_FMT.format(
        kernel="kernel", module="module", batch="batch", seq="seq",
        hidden="hidden", inter="inter", median="median_ms",
        p10="p10_ms", p90="p90_ms", min="min_ms", dtype="dtype",
    ))
    print("-" * 130)

    for cfg in BENCH_CONFIGS:
        row = bench_one(cfg, dtype)
        results.append(row)
        print(HEADER_FMT.format(
            kernel=row["kernel"], module=row["module"], batch=row["batch"],
            seq=row["seq_len"], hidden=row["hidden"], inter=row["intermediate"],
            median=f"{row['median_ms']:.4f}", p10=f"{row['p10_ms']:.4f}",
            p90=f"{row['p90_ms']:.4f}", min=f"{row['min_ms']:.4f}",
            dtype=row["dtype"],
        ))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark CUDA RMSNorm + PyTorch Linear (split, un-fused)")
    parser.add_argument(
        "--dtype",
        choices=list(DTYPE_MAP.keys()),
        default=None,
        help="Data type (default: fp32)",
    )
    parser.add_argument(
        "--all-dtypes",
        action="store_true",
        help="Run fp32, fp16, and bf16 sequentially",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Benchmarks require a GPU.", file=sys.stderr)
        sys.exit(1)

    if args.all_dtypes:
        dtypes = [torch.float32, torch.float16, torch.bfloat16]
    elif args.dtype:
        dtypes = [DTYPE_MAP[args.dtype]]
    else:
        dtypes = [torch.float32]

    all_results: list[dict[str, object]] = []
    for dtype in dtypes:
        all_results.extend(_run_dtype(dtype))

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "rmsnorm_linear_split.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults written to {csv_path}")


if __name__ == "__main__":
    main()
