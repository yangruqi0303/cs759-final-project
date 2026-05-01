# PyTorch Baseline — RMSNorm + MLP Pipeline

Reference implementations and benchmarking harness for the CS759 fused-kernel project.

## Install

```bash
pip install torch pytest
```

PyTorch 2.x with CUDA support is required for benchmarks.

## Run tests

```bash
pytest tests/ -v
```

## Run benchmarks

```bash
python benchmarks/bench_pytorch.py --dtype fp32
python benchmarks/bench_pytorch.py --dtype fp16
python benchmarks/bench_pytorch.py --dtype bf16
```

Results are printed to stdout and saved to `results/pytorch_baseline.csv`.

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
| `module` | Module name (`RMSNorm`, `RMSNormLinear`, `RMSNormMLP`) |
| `batch` | Batch dimension |
| `seq_len` | Sequence-length dimension |
| `hidden` | Hidden size (normalisation dimension) |
| `intermediate` | MLP intermediate size (unused for `RMSNorm`) |
| `median_ms` | Median GPU time over 100 iterations (ms) |
| `stdev_ms` | Standard deviation of GPU time (ms) |
| `dtype` | Data type (`float32`, `float16`, `bfloat16`) |
