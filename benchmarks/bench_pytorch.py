"""Benchmark harness for PyTorch reference implementations.

Uses CUDA events for accurate GPU timing.  Outputs a printed table and a CSV
to results/pytorch_baseline.csv.

Usage:
    python benchmarks/bench_pytorch.py --dtype fp32
    python benchmarks/bench_pytorch.py --dtype fp16
    python benchmarks/bench_pytorch.py --dtype bf16
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline.configs import BENCH_CONFIGS, BenchConfig
from baseline.pytorch_ref import RMSNorm, RMSNormLinear, RMSNormMLP

WARMUP = 10
TIMED = 100

DTYPE_MAP: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _time_fn(fn: callable, warmup: int, iters: int) -> list[float]:
    """Time *fn()* using CUDA events.  Returns list of per-iteration ms."""
    for _ in range(warmup):
        fn()

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


def bench_one(
    module_name: str, cfg: BenchConfig, dtype: torch.dtype,
) -> dict[str, object]:
    """Benchmark a single (module, config, dtype) combination."""
    m = _build_module(module_name, cfg, dtype)
    x = torch.randn(cfg.batch, cfg.seq_len, cfg.hidden_size, device="cuda", dtype=dtype)

    with torch.no_grad():
        times = _time_fn(lambda: m(x), warmup=WARMUP, iters=TIMED)

    median = statistics.median(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0.0

    return {
        "module": module_name,
        "batch": cfg.batch,
        "seq_len": cfg.seq_len,
        "hidden": cfg.hidden_size,
        "intermediate": cfg.intermediate_size,
        "median_ms": round(median, 4),
        "stdev_ms": round(stdev, 4),
        "dtype": str(dtype).split(".")[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch reference models")
    parser.add_argument(
        "--dtype",
        choices=list(DTYPE_MAP.keys()),
        default="fp32",
        help="Data type (default: fp32)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Benchmarks require a GPU.", file=sys.stderr)
        sys.exit(1)

    dtype = DTYPE_MAP[args.dtype]
    module_names = ["RMSNorm", "RMSNormLinear", "RMSNormMLP"]

    results: list[dict[str, object]] = []

    header = f"{'module':<16} {'batch':>5} {'seq':>5} {'hidden':>6} {'inter':>6} {'median_ms':>10} {'stdev_ms':>10}"
    print(header)
    print("-" * len(header))

    for cfg in BENCH_CONFIGS:
        for name in module_names:
            row = bench_one(name, cfg, dtype)
            results.append(row)
            print(
                f"{row['module']:<16} {row['batch']:>5} {row['seq_len']:>5} "
                f"{row['hidden']:>6} {row['intermediate']:>6} "
                f"{row['median_ms']:>10.4f} {row['stdev_ms']:>10.4f}"
            )

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "pytorch_baseline.csv"

    fieldnames = ["module", "batch", "seq_len", "hidden", "intermediate", "median_ms", "stdev_ms", "dtype"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to {csv_path}")


if __name__ == "__main__":
    main()
