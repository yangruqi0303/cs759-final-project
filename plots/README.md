# plots/ — figures for the RMSNorm fused-kernel report

Standalone matplotlib scripts that render every figure used in the report
straight from `results/*.csv`. No pandas, no seaborn — only `matplotlib`,
`numpy` and the standard library.

## Run

```bash
# from repo root
pip install matplotlib numpy
python plots/run_all.py
```

Renders all PNGs into `plots/figures/`. To regenerate just one:

```bash
python plots/fig03_speedup_heatmap.py
```

## What each figure says

| File | One-line takeaway |
|---|---|
| `fig01_speedup_overview.png` | CUDA-vs-PyTorch speedup across all 21 shapes × 3 dtypes for each deliverable kernel; geomean line marks "typical" win. |
| `fig02_latency_vs_tokens.png` | Log-log latency-vs-tokens curves per (model size × module); shows how the PT/CUDA gap closes as work grows. |
| `fig03_speedup_heatmap.png` | One-page diverging-color matrix to scan where the wins/losses live. |
| `fig04_rmsnorm_bandwidth.png` | RMSNorm achieved bandwidth (GB/s) vs bytes touched on RTX 4060 Ti 16 GB. Yellow band marks working-sets that fit in L2 (32 MB) — those points are cache-hot and can exceed DRAM peak; only points to the right of the band are true DRAM-bound and comparable to the 288 GB/s peak line. |
| `fig05_fusion_decomposition.png` | Honesty plot: `cuda_rmsnorm + pytorch_linear` ≈ `fused_rmsnorm_linear`, so the speedup over PyTorch is the RMSNorm overhead, **not** real RMSNorm-into-GEMM fusion. |
| `fig06_variants_ladder.png` | Optimization ladder for the custom GEMM variants: naive → tiled → prologue v1 → tiled v2 → prologue v2, plus cuBLAS reference. |
| `fig07_variants_vs_cublas.png` | Each variant's speedup relative to cuBLAS default (1.0× = parity). Highlights the (small) shapes where prologue v2 actually beats cuBLAS. |

## Source data

All figures read from `../results/`:

- `pytorch_baseline.csv` — PyTorch reference, 21 configs × 3 dtypes
- `cuda_kernels.csv` — `naive_rmsnorm`, `fused_rmsnorm_linear`,
  `fused_rmsnorm_mlp` on the same 21 × 3 sweep
- `rmsnorm_linear_split.csv` — control: `cuda_rmsnorm + pytorch_linear`
- `rmsnorm_linear_variants.csv` — 9 custom shapes × 6 GEMM variants × 3 dtypes

`common.py` holds CSV loading, shape ordering, geomean / bandwidth helpers,
and the shared color palette.
