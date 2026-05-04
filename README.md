# cs759-final-project

Fused CUDA kernels for RMSNorm and MLP layers in Transformer inference.
Final project for **ME/ECE/CS 759 — Spring 2026**.

The repo holds three things in parallel:

1. A PyTorch reference implementation (`baseline/`) used as the correctness
   oracle and performance floor.
2. CUDA kernels (`kernels/`) that progressively fuse more of the RMSNorm +
   MLP block. Each kernel is JIT-compiled on first import via
   `torch.utils.cpp_extension.load` — no setuptools / CMake build step.
3. Shared correctness tests (`tests/`) and a benchmark harness
   (`benchmarks/`) that produce CSVs in `results/` for direct comparison
   between PyTorch and every CUDA kernel.

For per-step write-ups see [`log_baseline.md`](log_baseline.md) (Step 1 —
PyTorch baseline) and [`log_naive_rmsnorm.md`](log_naive_rmsnorm.md)
(Step 2 — naive RMSNorm CUDA kernel). The full directory layout is in
[`file_tree.md`](file_tree.md).

---

## Hardware

The kernels target NVIDIA discrete GPUs and have been verified on:

| GPU | Arch | Compute capability | Notes |
|---|---|---|---|
| RTX 4060 Ti | Ada Lovelace | `sm_89` | primary dev box (WSL2) |
| RTX 5070 Ti | Blackwell | `sm_120` | second dev box |

Both archs are baked into the JIT compile flags in
[`kernels/__init__.py`](kernels/__init__.py), so the cached `.so` runs on
either machine without recompilation. To target a different GPU, add the
matching `-gencode=arch=compute_XX,code=sm_XX` line.

A Linux host (or WSL2) with a recent NVIDIA driver is assumed. The
`pytorch-dev` env has been exercised on **WSL2 Ubuntu** with Windows
driver **591.86**.

---

## Software environment

The verified stack is:

| Component | Version | Source |
|---|---|---|
| Python | 3.14.4 | conda (`defaults`) |
| PyTorch | 2.11.0+cu130 | pip (`download.pytorch.org/whl/cu130`) |
| CUDA toolkit (nvcc) | 13.0.88 | conda (`nvidia::cuda-nvcc`) |
| cuDNN | 9.10 (bundled with torch wheel) | pip |
| Host compiler | gcc/g++ 14.3 | conda (`gcc_linux-64`, `gxx_linux-64`) |
| ninja | 1.13.0 | pip |
| pytest | 9.0.3 | pip |

A few non-obvious points worth knowing before you start swapping versions:

- **PyTorch 2.11 is the first release with a `cu130` wheel.** That fixes
  the toolkit major version at 13.x. nvcc 12.x will not link against the
  ATen libs in this wheel.
- **The `cu130` torch wheel does not ship `cusparse.h`** in the locations
  ATen's headers expect. Instead it brings in the `nvidia-cusparse` /
  `nvidia-cublas` / etc. wheels, which install headers under the
  `nvidia.cu13` namespace package. The JIT loader in
  [`kernels/__init__.py`](kernels/__init__.py) detects this directory and
  appends it via `-isystem` (lower priority than nvcc's own toolkit
  headers, so the 13.0 vs 13.1 minor-version check inside `cccl` does
  not trip — see the comments in that file).
- **The conda `cuda-nvcc` package does not include the host C++
  compiler.** `gxx_linux-64=14` is required separately so nvcc has a
  `-ccbin`. Mixing in the system `g++` instead is fragile because of
  glibc / sysroot differences against the conda Python.

---

## Setup

### Option A — one-shot conda env (recommended)

```bash
conda env create -f environment.yml
conda activate pytorch-dev
```

This recreates the exact env above. First activation will not yet have
compiled any CUDA extension — that happens lazily on first import.

### Option B — manual

```bash
conda create -n pytorch-dev -c nvidia -c defaults \
    python=3.14 pip \
    gcc_linux-64=14 gxx_linux-64=14 sysroot_linux-64=2.28 \
    nvidia::cuda-nvcc=13.0 nvidia::cuda-cudart-dev=13.0 \
    nvidia::libcublas-dev=13.0 nvidia::cuda-cccl=13.0
conda activate pytorch-dev
pip install --extra-index-url https://download.pytorch.org/whl/cu130 \
    torch==2.11.0 torchvision==0.26.0
pip install ninja pytest
```

### Sanity check

After activation, confirm the toolchain wires up correctly:

```bash
nvcc --version            # → "Cuda compilation tools, release 13.0, V13.0.88"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# → 2.11.0+cu130 13.0 True
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
# → e.g. NVIDIA GeForce RTX 4060 Ti (8, 9)
```

---

## Verification — reproduce the published results

All commands below assume `conda activate pytorch-dev` and the repo root
as the working directory.

### Run the test suite

```bash
pytest tests/ -v
```

Expected: **40 passed** (19 PyTorch reference + 21 CUDA kernel).
The first invocation triggers a one-time JIT compile of `kernels/rmsnorm.cu`
into `~/.cache/torch_extensions/pyXYZ_cu130/rmsnorm_cuda_ext/`. Subsequent
runs reuse the cached `.so`.

### Run the benchmarks

PyTorch baseline (writes `results/pytorch_baseline.csv`):

```bash
python benchmarks/bench_pytorch.py --all-dtypes
```

CUDA kernel sweep (writes `results/cuda_kernels.csv` with the same column
schema plus a leading `kernel` column):

```bash
python benchmarks/bench_cuda.py --all-dtypes
```

Each sweep is 21 configs × 3 dtypes = 63 rows; benchmarks use CUDA events
with 10 warmup + 200 timed iterations and report
median / p10 / p90 / min in ms.

To force a clean rebuild of the kernel (e.g. after editing `.cu`):

```bash
rm -rf ~/.cache/torch_extensions/py*_cu*/rmsnorm_cuda_ext
```

---

## Repo layout

```
cs759-final-project/
├── baseline/         PyTorch reference modules + benchmark configs
├── kernels/          CUDA .cu sources + JIT loader
├── tests/            pytest suites for PyTorch ref and CUDA kernels
├── benchmarks/       CUDA-event timing harness, writes results/*.csv
├── results/          Benchmark CSVs (committed)
├── environment.yml   Conda env spec (this file)
├── log_*.md          Per-step write-ups
└── file_tree.md      Annotated tree
```

See [`file_tree.md`](file_tree.md) for the full annotated layout.
