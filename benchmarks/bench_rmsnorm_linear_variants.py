"""Small-to-medium benchmark for RMSNormLinear implementation variants.

This benchmark is intentionally separate from `bench_cuda.py`: the tiled custom
GEMM variants are useful for studying prologue fusion, but they are much slower
than cuBLAS on the full project sweep. These configs are larger than unit-test
shapes so the custom GEMM kernels do measurable work, but still smaller than
the full 21-config sweep.

The configs cover three regimes:
    1. balanced GEMM cases, which show naive-vs-tiled GEMM differences;
    2. prologue-focused cases, where hidden/tokens are large and out_features
       is intentionally small so materializing RMSNorm becomes more visible;
    3. L2-overflow prologue cases, where rows*hidden*sizeof(T) exceeds the L2
       cache (~32 MB on sm_120) so the materialized `normed` cannot stay in
       L2 and the prologue's saved write+read shows up in DRAM traffic.

Usage:
    python benchmarks/bench_rmsnorm_linear_variants.py --dtype fp32
    python benchmarks/bench_rmsnorm_linear_variants.py --all-dtypes
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

from baseline.configs import BenchConfig
from kernels import (
    rmsnorm_linear_cuda,
    rmsnorm_linear_naive_cuda,
    rmsnorm_linear_prologue_cuda,
    rmsnorm_linear_tiled_cuda,
    rmsnorm_linear_prologue_v2_cuda,
    rmsnorm_linear_tiled_v2_cuda,
    
)

WARMUP = 10
TIMED = 50

DTYPE_MAP: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

VARIANT_CONFIGS = [
    # Keep one launch-dominated case for comparison with the earlier tiny sweep.
    BenchConfig("launch_b1_s16_h128_o256",
                batch=1, seq_len=16, hidden_size=128, intermediate_size=256),
    # Balanced Linear shapes: out_features scales with hidden_size, so GEMM
    # dominates. These are mainly for naive-vs-tiled custom GEMM comparison.
    BenchConfig("balanced_b1_s128_h512_o1024",
                batch=1, seq_len=128, hidden_size=512, intermediate_size=1024),
    BenchConfig("balanced_b1_s256_h1024_o2048",
                batch=1, seq_len=256, hidden_size=1024, intermediate_size=2048),
    # Prologue-focused shapes: large normalized input, small projection. These
    # make the extra global-memory write/read in materialized RMSNorm easier to
    # see against the same tiled GEMM structure.
    BenchConfig("prologue_b1_s512_h1024_o64",
                batch=1, seq_len=512, hidden_size=1024, intermediate_size=64),
    BenchConfig("prologue_b4_s512_h1024_o64",
                batch=4, seq_len=512, hidden_size=1024, intermediate_size=64),
    BenchConfig("prologue_b1_s512_h2048_o128",
                batch=1, seq_len=512, hidden_size=2048, intermediate_size=128),
    BenchConfig("prologue_b1_s512_h4096_o128",
                batch=1, seq_len=512, hidden_size=4096, intermediate_size=128),
    BenchConfig("prologue_b2_s512_h4096_o128",
                batch=2, seq_len=512, hidden_size=4096, intermediate_size=128),
    # L2-overflow prologue cases. The "prologue_*" configs above all leave the
    # materialized `normed` small enough to fit in L2 (~32 MB on sm_120,
    # ~36 MB on sm_89), so the tiled variant's extra normed write/read is
    # absorbed by L2 and prologue's saving never reaches DRAM. These cases
    # push rows*hidden*sizeof(T) past L2 to expose the saving in real DRAM
    # traffic. Same hidden/out as `prologue_*` so only rows is differential.
    #   - b8_s512_h4096_o128 : rows=4096, normed = 64 MB fp32 / 32 MB low-prec
    #   - b4_s2048_h4096_o128: rows=8192, normed = 128 MB fp32 / 64 MB low-prec
    BenchConfig("prologue_l2_b8_s512_h4096_o128",
                batch=8, seq_len=512, hidden_size=4096, intermediate_size=128),
    BenchConfig("prologue_l2_b4_s2048_h4096_o128",
                batch=4, seq_len=2048, hidden_size=4096, intermediate_size=128),
]


def _time_fn(fn: callable, warmup: int, iters: int) -> list[float]:
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


def bench_one(
    kernel_name: str,
    fn: callable,
    cfg: BenchConfig,
    dtype: torch.dtype,
    iters: int,
) -> dict[str, object]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    x = torch.randn(cfg.batch, cfg.seq_len, cfg.hidden_size, device="cuda", dtype=dtype)
    weight = torch.randn(
        cfg.intermediate_size,
        cfg.hidden_size,
        device="cuda",
        dtype=dtype,
    )
    gamma = torch.ones(cfg.hidden_size, device="cuda", dtype=dtype)
    eps = 1e-6

    with torch.no_grad():
        times = _time_fn(
            lambda: fn(x, weight, gamma, eps),
            warmup=WARMUP,
            iters=iters,
        )

    median = statistics.median(times)
    p10 = _percentile(times, 10)
    p90 = _percentile(times, 90)
    min_t = min(times)

    return {
        "config": cfg.name,
        "kernel": kernel_name,
        "module": "RMSNormLinear",
        "batch": cfg.batch,
        "seq_len": cfg.seq_len,
        "hidden": cfg.hidden_size,
        "intermediate": cfg.intermediate_size,
        "median_ms": round(median, 4),
        "p10_ms": round(p10, 4),
        "p90_ms": round(p90, 4),
        "min_ms": round(min_t, 4),
        "n_iters": iters,
        "dtype": str(dtype).split(".")[-1],
    }


FIELDNAMES = [
    "config", "kernel", "module", "batch", "seq_len", "hidden", "intermediate",
    "median_ms", "p10_ms", "p90_ms", "min_ms", "n_iters", "dtype",
]

HEADER_FMT = (
    "{config:<31} {kernel:<28} {batch:>5} {seq:>5} {hidden:>6} {inter:>6} "
    "{median:>10} {p10:>10} {p90:>10} {min:>10} {dtype:>8}"
)


def _run_dtype(dtype: torch.dtype, iters: int) -> list[dict[str, object]]:
    variants = [
        ("fused_rmsnorm_linear", rmsnorm_linear_cuda),
        ("naive_rmsnorm_linear", rmsnorm_linear_naive_cuda),
        ("tiled_rmsnorm_linear", rmsnorm_linear_tiled_cuda),
        ("prologue_rmsnorm_linear", rmsnorm_linear_prologue_cuda),
        ("tiled_rmsnorm_linear_v2", rmsnorm_linear_tiled_v2_cuda),
        ("prologue_rmsnorm_linear_v2", rmsnorm_linear_prologue_v2_cuda),
    ]
    results: list[dict[str, object]] = []

    print(f"\n=== dtype: {dtype} ===")
    print(HEADER_FMT.format(
        config="config", kernel="kernel", batch="batch", seq="seq",
        hidden="hidden", inter="inter", median="median_ms", p10="p10_ms",
        p90="p90_ms", min="min_ms", dtype="dtype",
    ))
    print("-" * 155)

    for cfg in VARIANT_CONFIGS:
        for kernel_name, fn in variants:
            row = bench_one(kernel_name, fn, cfg, dtype, iters)
            results.append(row)
            print(HEADER_FMT.format(
                config=row["config"], kernel=row["kernel"], batch=row["batch"],
                seq=row["seq_len"], hidden=row["hidden"],
                inter=row["intermediate"], median=f"{row['median_ms']:.4f}",
                p10=f"{row['p10_ms']:.4f}", p90=f"{row['p90_ms']:.4f}",
                min=f"{row['min_ms']:.4f}", dtype=row["dtype"],
            ))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark RMSNormLinear CUDA implementation variants")
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
    parser.add_argument(
        "--iters",
        type=int,
        default=TIMED,
        help=f"Timed iterations per case (default: {TIMED})",
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
        all_results.extend(_run_dtype(dtype, args.iters))

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "rmsnorm_linear_variants.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults written to {csv_path}")


if __name__ == "__main__":
    main()
