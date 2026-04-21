"""Benchmark harness for PyTorch reference implementations.

Uses CUDA events for accurate GPU timing.  Outputs a printed table and a CSV
to results/pytorch_baseline.csv.

Usage:
    python benchmarks/bench_pytorch.py --dtype fp32
    python benchmarks/bench_pytorch.py --dtype fp16
    python benchmarks/bench_pytorch.py --all-dtypes
"""

from __future__ import annotations

import argparse
import csv
import gc
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline.configs import BENCH_CONFIGS, BenchConfig
from baseline.pytorch_ref import RMSNorm, RMSNormLinear, RMSNormMLP

WARMUP = 10
TIMED = 200

DTYPE_MAP: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _time_fn(fn: callable, warmup: int, iters: int) -> list[float]:
    """Time *fn()* using CUDA events.  Returns list of per-iteration ms."""
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


def _build_module(
    module_name: str, cfg: BenchConfig, dtype: torch.dtype,
) -> torch.nn.Module:
    if module_name == "RMSNorm":
        m = RMSNorm(cfg.hidden_size)
    elif module_name == "RMSNormLinear":
        m = RMSNormLinear(cfg.hidden_size, cfg.intermediate_size)
    elif module_name == "RMSNormMLP":
        m = RMSNormMLP(cfg.hidden_size, cfg.intermediate_size)
    else:
        raise ValueError(f"Unknown module: {module_name}")
    return m.to("cuda", dtype).eval()


def _percentile(data: list[float], p: float) -> float:
    """Simple linear-interpolation percentile."""
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def bench_one(
    module_name: str, cfg: BenchConfig, dtype: torch.dtype,
) -> dict[str, object]:
    """Benchmark a single (module, config, dtype) combination."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    m = _build_module(module_name, cfg, dtype)
    x = torch.randn(cfg.batch, cfg.seq_len, cfg.hidden_size, device="cuda", dtype=dtype)

    with torch.no_grad():
        times = _time_fn(lambda: m(x), warmup=WARMUP, iters=TIMED)

    median = statistics.median(times)
    p10 = _percentile(times, 10)
    p90 = _percentile(times, 90)
    min_t = min(times)

    return {
        "module": module_name,
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
    "module", "batch", "seq_len", "hidden", "intermediate",
    "median_ms", "p10_ms", "p90_ms", "min_ms", "n_iters", "dtype",
]

HEADER_FMT = (
    "{module:<16} {batch:>5} {seq:>5} {hidden:>6} {inter:>6} "
    "{median:>10} {p10:>10} {p90:>10} {min:>10} {dtype:>8}"
)


def _run_dtype(dtype: torch.dtype, module_names: list[str]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []

    print(f"\n=== dtype: {dtype} ===")
    print(HEADER_FMT.format(
        module="module", batch="batch", seq="seq", hidden="hidden",
        inter="inter", median="median_ms", p10="p10_ms", p90="p90_ms",
        min="min_ms", dtype="dtype",
    ))
    print("-" * 100)

    for cfg in BENCH_CONFIGS:
        for name in module_names:
            row = bench_one(name, cfg, dtype)
            results.append(row)
            print(HEADER_FMT.format(
                module=row["module"], batch=row["batch"], seq=row["seq_len"],
                hidden=row["hidden"], inter=row["intermediate"],
                median=f"{row['median_ms']:.4f}", p10=f"{row['p10_ms']:.4f}",
                p90=f"{row['p90_ms']:.4f}", min=f"{row['min_ms']:.4f}",
                dtype=row["dtype"],
            ))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch reference models")
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

    module_names = ["RMSNorm", "RMSNormLinear", "RMSNormMLP"]

    all_results: list[dict[str, object]] = []
    for dtype in dtypes:
        all_results.extend(_run_dtype(dtype, module_names))

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "pytorch_baseline.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults written to {csv_path}")


if __name__ == "__main__":
    main()
