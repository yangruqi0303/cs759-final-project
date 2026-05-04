# Step 3/4: RMSNormLinear Variants + RMSNormMLP — 完成记录

## 任务概述

Step 3 的目标是实现 `RMSNorm + Linear`，并在同一个测试/benchmark 体系里比较多个实现层级：

1. 默认 `fused_rmsnorm_linear`：RMSNorm CUDA kernel + cuBLAS GEMM
2. `naive_rmsnorm_linear`：materialized RMSNorm + naive one-thread-per-output GEMM
3. `tiled_rmsnorm_linear`：materialized RMSNorm + shared-memory tiled GEMM
4. `prologue_rmsnorm_linear`：RMS scale kernel + prologue-fused tiled GEMM

Step 4 的目标是实现一个可运行、可测试、可 benchmark 的 `RMSNorm + MLP` 版本：

```
Linear2(GELU(Linear1(RMSNorm(x))))
```

本项目当前 Step 4 是 operator-level fusion：RMSNorm 使用自定义 CUDA kernel，两个 Linear 使用 cuBLAS，GELU 使用 PyTorch/ATen 的 tanh approximation。它不是 fully fused MLP kernel，但作为 Phase 4 v1 已经能进入完整 benchmark sweep。

**硬件 / 工具链**：NVIDIA RTX 5070 Ti（Blackwell, sm_120, 16 GB），PyTorch 2.11.0 + CUDA 13.0 runtime，nvcc 13.1.

---

## 完成的具体工作

### 1. 公共 CUDA helper

#### `kernels/rmsnorm_common.cuh`

把 Step 2 中 RMSNorm 相关的公共内容抽出成 header：

- `to_float<T>` / `from_float<T>` dtype 转换 helper
- `warp_reduce_sum` butterfly reduction
- `rmsnorm_kernel`
- `rmsnorm_forward_cuda(...)`

这样后续 `rmsnorm_linear`、`rmsnorm_mlp`、custom GEMM variants 都能复用完全一致的 RMSNorm 实现，避免多个 `.cu` 文件里复制同一段 reduction 逻辑。

#### `kernels/cublas_linear_common.cuh`

封装 cuBLAS Linear：

```
output = input_2d @ weight.T
```

其中 `weight` 按 PyTorch `nn.Linear.weight` 约定存储为 `(out_features, in_features)`。内部使用 `cublasGemmEx`，并处理 fp32 / fp16 / bf16 三种 dtype 的 dispatch。

#### `kernels/tiled_linear_common.cuh`

为实验版本提供一个简单的 scalar-FMA tiled GEMM helper：

| 参数 | 当前值 |
|---|---:|
| output tile M | 16 |
| output tile N | 16 |
| K tile | 32 |
| block threads | 16 × 16 = 256 |

这个 helper 不是为了打败 cuBLAS，而是为了让 `materialized RMSNorm + tiled GEMM` 和 `prologue fusion + tiled GEMM` 使用同一个 GEMM 结构，从而更公平地观察 prologue fusion 是否能省掉 normalized tensor 的 global-memory round trip。

---

## Step 3: RMSNormLinear 多版本

### 1. 默认版本：`kernels/rmsnorm_linear.cu`

Python 入口：

```python
rmsnorm_linear_cuda(x, weight, gamma, eps=1e-6)
```

实现流程：

1. 调用 `rmsnorm_forward_cuda(x, gamma, eps, ...)` 得到 materialized normalized tensor
2. reshape 成 `(rows, hidden)`
3. 调用 `linear_cublas_cuda(normed_2d, weight, ...)`
4. reshape 回 `(*, out_features)`

这是主 benchmark 里的默认 `fused_rmsnorm_linear`。虽然它仍然 materialize normalized tensor，但它减少了 PyTorch reference 里 RMSNorm 分解成多个 ATen op 的 overhead，并把 GEMM 保持在 cuBLAS 快路径上。

### 2. Naive GEMM 对照：`kernels/rmsnorm_linear_naive.cu`

流程：

1. materialized RMSNorm
2. naive custom GEMM

GEMM kernel 安排：

| 设计 | 内容 |
|---|---|
| grid | 2-D output tile grid |
| block | 16 × 16 threads |
| 每个线程 | 计算一个 `output[row, col]` |
| K 维访问 | 直接从 global memory 逐个读 `input[row, k]` 和 `weight[col, k]` |
| shared memory | 无 |

这个版本故意朴素，用来证明 shared-memory tiling 的必要性。

### 3. Tiled GEMM 对照：`kernels/rmsnorm_linear_tiled.cu`

流程：

1. materialized RMSNorm
2. 调用 `linear_tiled_cuda(...)`

这个版本仍然把 normalized tensor 写回 global memory，但 GEMM 部分通过 shared memory tile 复用输入和权重数据。它和 prologue 版本的 GEMM tile 形状一致，是观察 prologue fusion 的直接对照组。

### 4. Prologue fusion 实验：`kernels/rmsnorm_linear_prologue.cu`

流程：

1. `rmsnorm_scale_kernel`：每行算一个 `scale[row] = rsqrt(mean(x[row]^2) + eps)`
2. `rmsnorm_linear_prologue_kernel`：在 GEMM 读取 A tile 时直接计算

```cuda
A_tile = x[row, k] * scale[row] * gamma[k]
```

也就是说，它不再 materialize 完整的 normalized tensor，而是把 RMSNorm 的乘法部分融合进 GEMM prologue。这个版本更接近“真正的 fusion”概念，但仍然是 scalar-FMA tiled GEMM，不使用 Tensor Core。

---

## Step 4: RMSNormMLP v1

### `kernels/rmsnorm_mlp.cu`

Python 入口：

```python
rmsnorm_mlp_cuda(x, weight1, weight2, gamma, eps=1e-6)
```

实现流程：

1. `normed = rmsnorm_forward_cuda(x, gamma, eps, ...)`
2. `h1 = linear_cublas_cuda(normed_2d, weight1, ...)`
3. `h1 = at::gelu(h1, "tanh")`
4. `out = linear_cublas_cuda(h1, weight2, ...)`
5. reshape 回输入 shape

约定：

- `weight1`: `(intermediate_size, hidden_size)`
- `weight2`: `(hidden_size, intermediate_size)`
- GELU 固定使用 `approximate="tanh"`，与 PyTorch baseline 一致
- fp32 / fp16 / bf16 三种 dtype 走同一接口

这个版本没有实现 fully fused MLP，也没有把 GELU 融进 GEMM epilogue。它的价值是把 Step 2/3 的 RMSNorm + cuBLAS Linear 流水线扩展到完整 MLP block，并进入主 benchmark。

---

## Python 接口与文件结构

### `kernels/__init__.py`

当前导出：

```python
__all__ = [
    "rmsnorm_cuda",
    "rmsnorm_linear_cuda",
    "rmsnorm_linear_naive_cuda",
    "rmsnorm_linear_tiled_cuda",
    "rmsnorm_linear_prologue_cuda",
    "rmsnorm_mlp_cuda",
]
```

主路径：

| Python API | CUDA source | 用途 |
|---|---|---|
| `rmsnorm_linear_cuda` | `rmsnorm_linear.cu` | 默认 RMSNormLinear，cuBLAS GEMM |
| `rmsnorm_linear_naive_cuda` | `rmsnorm_linear_naive.cu` | naive GEMM 对照 |
| `rmsnorm_linear_tiled_cuda` | `rmsnorm_linear_tiled.cu` | materialized RMSNorm + tiled GEMM |
| `rmsnorm_linear_prologue_cuda` | `rmsnorm_linear_prologue.cu` | prologue-fused tiled GEMM |
| `rmsnorm_mlp_cuda` | `rmsnorm_mlp.cu` | RMSNorm + Linear + GELU + Linear |

---

## 测试与 benchmark

### Correctness tests

`tests/test_cuda_kernels.py` 已覆盖：

- `RMSNorm`
- 默认 `RMSNormLinear`
- `RMSNormLinearNaive`
- `RMSNormLinearTiled`
- `RMSNormLinearPrologue`
- `RMSNormMLP`
- input validation：non-contiguous / dtype mismatch / shape mismatch

所有 CUDA 输出都对照 `baseline/pytorch_ref.py` 里的 reference module，并复用 Step 1 的 `assert_close`。

### 主 benchmark：`benchmarks/bench_cuda.py`

主 sweep 只保留 project deliverable 的三个 kernel：

| kernel column | module |
|---|---|
| `naive_rmsnorm` | `RMSNorm` |
| `fused_rmsnorm_linear` | `RMSNormLinear` |
| `fused_rmsnorm_mlp` | `RMSNormMLP` |

输出：

```
results/cuda_kernels.csv
```

列顺序：

```
kernel,module,batch,seq_len,hidden,intermediate,
median_ms,p10_ms,p90_ms,min_ms,n_iters,dtype
```

### RMSNormLinear variants benchmark

为了不污染主 benchmark，新建：

```
benchmarks/bench_rmsnorm_linear_variants.py
```

输出：

```
results/rmsnorm_linear_variants.csv
```

CSV 比主表多一列 `config`，因为 variants benchmark 不是完整 21-config sweep，而是专门设计了两类 shape：

1. `balanced_*`：out_features 随 hidden 增长，主要观察 naive vs tiled GEMM
2. `prologue_*`：hidden/tokens 大、out_features 小，尽量放大 materialized RMSNorm 的 global-memory round trip 成本

---

## 结果观察

### 1. 默认 RMSNormLinear 相对 PyTorch baseline

数据来自：

- `results/pytorch_baseline.csv`
- `results/cuda_kernels.csv`

按 21 个 config 统计 median speedup：

| module | dtype | median speedup | min | max |
|---|---:|---:|---:|---:|
| RMSNormLinear | fp32 | 1.07× | 1.02× | 1.47× |
| RMSNormLinear | fp16 | 1.10× | 1.03× | 2.08× |
| RMSNormLinear | bf16 | 1.10× | 1.04× | 2.20× |

代表性 case：

| shape | dtype | PyTorch | CUDA default | speedup |
|---|---:|---:|---:|---:|
| B=1, S=128, H=1024, O=4096 | fp32 | 0.1946 ms | 0.1760 ms | 1.11× |
| B=8, S=1024, H=4096, O=11008 | fp32 | 68.9095 ms | 66.8850 ms | 1.03× |
| B=1, S=128, H=1024, O=4096 | fp16 | 0.1188 ms | 0.0570 ms | 2.08× |
| B=8, S=1024, H=4096, O=11008 | bf16 | 17.9389 ms | 16.5967 ms | 1.08× |

结论：默认 RMSNormLinear v1 有稳定但不夸张的收益。主要收益来自替换 PyTorch RMSNorm 的多个 ATen op / kernel launch，而 GEMM 本身仍然由 cuBLAS 主导，所以大 shape 上 speedup 会收敛到接近 1×。

### 2. RMSNormLinear variants：naive vs tiled

数据来自：

```
results/rmsnorm_linear_variants.csv
```

在有意义的 balanced/prologue configs 上，`naive_rmsnorm_linear` 明显慢于 `tiled_rmsnorm_linear`：

| config | dtype | naive | tiled | speedup |
|---|---:|---:|---:|---:|
| balanced_b1_s128_h512_o1024 | fp32 | 0.4646 ms | 0.1054 ms | 4.41× |
| balanced_b1_s256_h1024_o2048 | fp32 | 3.5117 ms | 0.7074 ms | 4.96× |
| balanced_b1_s256_h1024_o2048 | fp16 | 3.7862 ms | 0.7122 ms | 5.32× |
| prologue_b2_s512_h4096_o128 | bf16 | 3.8911 ms | 0.7892 ms | 4.93× |

结论：shared-memory tiling 的价值非常清楚。即使这个 tiled GEMM 仍是 scalar-FMA、没有 Tensor Core，它也能通过 tile 复用把 naive GEMM 拉开 3×-5×。

### 3. RMSNormLinear variants：tiled vs prologue fusion

prologue fusion 的结果不稳定，不能写成“一定更快”。

有些 case prologue 更快：

| config | dtype | tiled | prologue | speedup |
|---|---:|---:|---:|---:|
| prologue_b1_s512_h2048_o128 | fp32 | 0.2589 ms | 0.2181 ms | 1.19× |
| balanced_b1_s128_h512_o1024 | fp16 | 0.1454 ms | 0.1154 ms | 1.26× |
| prologue_b1_s512_h1024_o64 | bf16 | 0.0774 ms | 0.0655 ms | 1.18× |

但很多 case 基本持平或更慢：

| config | dtype | tiled | prologue | tiled/prologue |
|---|---:|---:|---:|---:|
| prologue_b1_s512_h1024_o64 | fp32 | 0.0655 ms | 0.0764 ms | 0.86× |
| balanced_b1_s256_h1024_o2048 | fp32 | 0.7074 ms | 0.7438 ms | 0.95× |
| prologue_b2_s512_h4096_o128 | fp16 | 0.7975 ms | 0.8148 ms | 0.98× |
| prologue_b2_s512_h4096_o128 | bf16 | 0.7892 ms | 0.8177 ms | 0.97× |

结论：当前 prologue-fused tiled GEMM 没有稳定超过 materialized tiled GEMM。省掉 normalized tensor 的 global-memory write/read 是正确方向，但当前实现还要额外 launch 一个 scale kernel，而且 GEMM 本身仍然不是 Tensor Core 级别优化，prologue 的理论收益经常被这些成本抵消。

### 4. cuBLAS 默认版 vs custom GEMM variants

cuBLAS 仍然是 practical baseline。

在 fp16/bf16 上尤其明显：

| config | dtype | cuBLAS default | tiled | prologue |
|---|---:|---:|---:|---:|
| prologue_b2_s512_h4096_o128 | fp16 | 0.0580 ms | 0.7975 ms | 0.8148 ms |
| prologue_b2_s512_h4096_o128 | bf16 | 0.0584 ms | 0.7892 ms | 0.8177 ms |

原因：cuBLAS 会使用高度优化的 GEMM 路径（通常包括 Tensor Core），而本项目 custom GEMM variants 都是教学/实验性质的 scalar-FMA GEMM。它们适合说明 tiling/prologue 的结构差异，不适合作为最终最快实现。

### 5. RMSNormMLP 相对 PyTorch baseline

按 21 个 config 统计 median speedup：

| module | dtype | median speedup | min | max |
|---|---:|---:|---:|---:|
| RMSNormMLP | fp32 | 1.04× | 0.97× | 1.27× |
| RMSNormMLP | fp16 | 1.05× | 1.01× | 1.61× |
| RMSNormMLP | bf16 | 1.05× | 1.02× | 1.89× |

代表性 case：

| shape | dtype | PyTorch | CUDA MLP | speedup |
|---|---:|---:|---:|---:|
| B=1, S=128, H=1024, I=4096 | fp32 | 0.3308 ms | 0.2692 ms | 1.23× |
| B=8, S=1024, H=4096, I=11008 | fp32 | 136.5875 ms | 133.7444 ms | 1.02× |
| B=1, S=128, H=1024, I=4096 | fp16 | 0.1580 ms | 0.0983 ms | 1.61× |
| B=8, S=1024, H=4096, I=11008 | bf16 | 35.7846 ms | 34.4166 ms | 1.04× |

结论：MLP v1 也有稳定但有限的收益。小 shape 更容易受 RMSNorm launch/op overhead 影响，所以 speedup 更明显；大 shape 被两个 GEMM 主导，cuBLAS 和 PyTorch 内部 GEMM 都很强，整体 speedup 接近 1×。

---

## 关键设计决策

### 1. 默认 RMSNormLinear 保持 cuBLAS，不用自写 GEMM

自写 GEMM 很难超过 cuBLAS，特别是 fp16/bf16。默认 deliverable 应该是最稳、最有工程意义的版本：RMSNorm 自定义 kernel + cuBLAS Linear。

custom GEMM variants 单独放进 `bench_rmsnorm_linear_variants.py`，作为实验/报告素材，而不是主 benchmark 的默认结果。

### 2. Prologue fusion 使用“同 GEMM 结构”对照

如果拿 prologue custom GEMM 去和 cuBLAS 比，会混入太多 GEMM 实现差异，无法单独判断 prologue fusion。于是专门做：

| 对照组 | RMSNorm | GEMM |
|---|---|---|
| `tiled_rmsnorm_linear` | materialized | tiled scalar-FMA |
| `prologue_rmsnorm_linear` | scale + prologue | tiled scalar-FMA |

两者 GEMM tile 结构一致，差异集中在 normalized tensor 是否 materialize。

### 3. Variants benchmark 不放进主 sweep

主 `bench_cuda.py` 保持 project deliverable 的简洁表：

```
naive_rmsnorm
fused_rmsnorm_linear
fused_rmsnorm_mlp
```

variants benchmark 单独输出 `results/rmsnorm_linear_variants.csv`。这样 final report 可以同时写：

- 主线 deliverable 的 PyTorch vs CUDA 对照
- 实验性 GEMM/prologue fusion 的消融分析

### 4. Step 4 暂不做 fully fused MLP

fully fused MLP 至少需要处理：

- RMSNorm prologue 融进 GEMM
- Linear1 GEMM
- GELU epilogue fusion
- Linear2 GEMM
- 中间 activation 的存储/复用
- Tensor Core / CUTLASS epilogue-prologue 定制

这个复杂度明显高于当前项目主线。当前 Step 4 v1 先完成 correctness + benchmark + cuBLAS path，是合理的工程切分。后续可以把 fully fused MLP 写成 future work。

---

## 最终结论

Step 3/4 的结果可以支持以下 final report 说法：

1. **RMSNormLinear cuBLAS v1 是实用默认实现**：相对 PyTorch baseline 有稳定但有限的 speedup，median 大约 1.07×-1.10×。
2. **shared-memory tiling 明显优于 naive custom GEMM**：在 variants benchmark 上通常 3×-5×。
3. **当前 prologue fusion 没有稳定 speedup**：少数 case 能到约 1.18×-1.26×，但很多 case 持平或更慢，主要受额外 scale kernel launch 和 scalar-FMA GEMM 限制。
4. **cuBLAS 仍然远强于自写 scalar GEMM**：尤其 fp16/bf16 上，Tensor Core 路径让 cuBLAS default 明显领先 custom variants。
5. **RMSNormMLP v1 可用但不是 fully fused**：相对 PyTorch 有小幅稳定收益，median 大约 1.04×-1.05×；更大的提升需要 CUTLASS/Tensor Core 级别的 epilogue/prologue fusion。
