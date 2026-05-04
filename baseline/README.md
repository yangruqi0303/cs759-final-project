# PyTorch Baseline — RMSNorm + MLP Pipeline

Reference implementations and benchmarking harness for the CS759 fused-kernel project.

## Install

See the top-level [README](../README.md) for the verified conda
environment (`environment.yml`) and the manual install fallback.
A working `pytorch-dev` env with PyTorch 2.11+cu130 and nvcc 13.0 is
assumed below.

## Run tests

```bash
pytest tests/ -v
```

## Run benchmarks


```bash
python benchmarks/bench_pytorch.py --all-dtype
```

Results are printed to stdout and saved to `results/pytorch_baseline.csv`.

For a certain `dtype`, e.g. `fp32`, run 

```bash
python benchmarks/bench_pytorch.py --dtype fp32
```

## CUDA kernels

The naive `RMSNorm` CUDA kernel lives in `kernels/rmsnorm.cu` and is JIT-compiled
on first import via `torch.utils.cpp_extension.load` (no setuptools build).
Compilation requires `nvcc` and a Blackwell GPU (`-arch=sm_120`).

Run the CUDA correctness tests:

```bash
pytest tests/test_cuda_kernels.py -v
```

The first run will compile the kernel (a few seconds); subsequent runs reuse
the cached `.so` under `~/.cache/torch_extensions/`.

Run the CUDA benchmark:

```bash
python benchmarks/bench_cuda.py --dtype fp32
python benchmarks/bench_cuda.py --all-dtypes
```

Output is written to `results/cuda_kernels.csv` with the same columns as
`pytorch_baseline.csv` plus a leading `kernel` column (value
`naive_rmsnorm`).

## CSV columns

| Column | Description |
|---|---|
| `kernel` | (CUDA CSV only) implementation tag, e.g. `naive_rmsnorm` |
| `module` | Module name (`RMSNorm`, `RMSNormLinear`, `RMSNormMLP`) |
| `batch` | Batch dimension |
| `seq_len` | Sequence-length dimension |
| `hidden` | Hidden size (normalisation dimension) |
| `intermediate` | MLP intermediate size (unused for `RMSNorm`) |
| `median_ms` | Median GPU time across all timed iterations (ms) |
| `p10_ms` / `p90_ms` | 10th / 90th-percentile GPU time (ms) |
| `min_ms` | Best single iteration (ms) |
| `n_iters` | Number of timed iterations (post-warmup) |
| `dtype` | Data type (`float32`, `float16`, `bfloat16`) |
