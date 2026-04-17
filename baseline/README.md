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
