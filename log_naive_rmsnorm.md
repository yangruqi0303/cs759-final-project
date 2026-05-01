# Step 2: Naive RMSNorm CUDA Kernel — 完成记录

## 任务概述

为 `baseline/pytorch_ref.py` 中的 `RMSNorm` 模块写一份 CUDA 实现，作为整个项目第一个真正的 GPU kernel。这一步不做任何融合（不接 Linear，不接 MLP）、不做向量化（无 float4），目标是把"warp shuffle reduction"这套 ME759 标准技法落地，并把 kernel 接进 Step 1 已经搭好的测试 + benchmark 流水线，让 PyTorch baseline 和 CUDA kernel 能在同一张表上对比。

**硬件 / 工具链**：NVIDIA RTX 5070 Ti（Blackwell, sm_120, 16 GB），PyTorch 2.11.0 + CUDA 13.0 runtime，nvcc 13.1.

## 完成的具体工作

### 1. CUDA kernel (`kernels/rmsnorm.cu`)

数学定义和 PyTorch reference 一致：

```
y[i] = x[i] * rsqrt(mean(x², dim=-1) + eps) * weight[i]
```

布局：

| 维度 | 安排 |
|---|---|
| Grid | `numel(x) / hidden_size` 个 block，每个 block 处理一个 token 的 hidden 向量 |
| Block | 256 threads（8 warps × 32 lanes），硬编码 |
| 单线程工作量 | strided loop：thread `t` 处理 `i = t, t+blockDim.x, t+2*blockDim.x, ...` 直到 `hidden_size` |
| 累加器 dtype | float32（无论输入是 fp32 / fp16 / bf16），数值稳定性的标准做法 |

reduction 两段式：
1. **warp 内**：`__shfl_xor_sync` 蝴蝶式求和（butterfly），32 个 lane 一轮内全部拿到 warp 总和
2. **warp 间**：每个 warp 的 lane 0 把局部和写进 `__shared__ float warp_sums[8]`，再让 warp 0 用一次相同的 `warp_reduce_sum` 归约出全 block 总和

最终 `warp_sums[0]` 复用为 scale 广播槽：lane 0 写 `rsqrtf(sum/H + eps)`，全 block `__syncthreads()` 后每个线程读这一个 float，再做 `y = x * scale * weight` 的写回。

dtype 模板化：`float / __half / __nv_bfloat16` 三种特化，通过 `to_float<T>` / `from_float<T>` 设备端 helper 收敛。

### 2. PyTorch binding + JIT 加载 (`kernels/__init__.py`)

- 用 `torch.utils.cpp_extension.load`，**不写 setuptools / CMake**，第一次 import `kernels` 时由 ninja 调 nvcc 编译成 `.so` 缓存到 `~/.cache/torch_extensions/`
- 编译参数：`-O3 -std=c++17 -arch=sm_120 --expt-relaxed-constexpr`
- 暴露的 Python 接口：

```python
def rmsnorm_cuda(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor
```

- 入口校验：CUDA / 连续 / dtype 一致 / 形状匹配；任何不满足都从 `TORCH_CHECK` 抛 `RuntimeError`，错误消息明确指向哪个条件失败

### 3. 正确性测试 (`tests/test_cuda_kernels.py`)

21 个 pytest 用例，复用 `tests/test_pytorch_ref.py` 暴露的 `assert_close`：

- **3 shape × 3 dtype 的 reference 比对**：`(1, 128, 1024) / (4, 512, 2048) / (8, 1024, 4096)` × `fp32 / fp16 / bf16`，每个 case 用一个 `weight ~ U(0.5, 1.5)` 的非平凡权重，断言 CUDA 输出与 PyTorch `RMSNorm(...)` 在 tolerance 内一致
- **同样 3×3 的 finite check**：所有输出 `torch.isfinite().all()`
- **3 个输入校验测试**：non-contiguous x / dtype mismatch / hidden size mismatch 都必须抛 `RuntimeError` 且消息包含相关关键字

### 4. CUDA benchmark (`benchmarks/bench_cuda.py`)

完全镜像 `bench_pytorch.py`：

- 复用 `baseline/configs.BENCH_CONFIGS`（21 组配置）
- 同样的 CUDA event 计时（10 warmup + 200 timed），同样的 `_time_fn` / `_percentile` 形态
- 仅 benchmark `RMSNorm`（融合版本是后续任务）
- `--dtype {fp32,fp16,bf16}` 单独跑，`--all-dtypes` 一次跑完
- 输出 `results/cuda_kernels.csv`，列与 `pytorch_baseline.csv` 完全一致，**首列额外加一个 `kernel`**，本步固定值为 `naive_rmsnorm`

### 5. README 更新 (`baseline/README.md`)

加了 "CUDA kernels" 一节，给出 JIT 编译说明、运行测试的命令、运行 benchmark 的命令。

---

## 架构设计

```
project_759/
├── kernels/                ← 本步新增
│   ├── __init__.py         ← JIT load + Python 入口 rmsnorm_cuda
│   └── rmsnorm.cu          ← kernel + pybind binding (单文件)
├── tests/
│   └── test_cuda_kernels.py  ← 21 个 pytest，复用 assert_close
├── benchmarks/
│   └── bench_cuda.py       ← 21 configs × 3 dtypes = 63 行 CSV
└── results/
    └── cuda_kernels.csv    ← 与 pytorch_baseline.csv 同列 + 首列 `kernel`
```

后续 fused kernel 的接入路径已在本步打通：
1. 新增 `kernels/<name>.cu` → 在 `kernels/__init__.py` 里再 `load(...)` 一份
2. 测试：在 `tests/test_cuda_kernels.py` 里复用 `assert_close`，对照 `RMSNormLinear` / `RMSNormMLP`
3. benchmark：把 `bench_cuda.py` 的 `KERNEL_NAME` 换成新名字，CSV 自动 append 同结构

---

## 接口规范

### `rmsnorm_cuda(x, weight, eps)`

```python
def rmsnorm_cuda(
    x: torch.Tensor,            # (*, hidden_size), CUDA, contiguous, fp32/fp16/bf16
    weight: torch.Tensor,        # (hidden_size,), 同 dtype 同 device 同 contiguous
    eps: float = 1e-6,
) -> torch.Tensor                # 同 shape 同 dtype 的新 tensor
```

### CSV 列（`results/cuda_kernels.csv`）

```
kernel, module, batch, seq_len, hidden, intermediate,
median_ms, p10_ms, p90_ms, min_ms, n_iters, dtype
```

第一列是这一步唯一相对 PyTorch baseline CSV 多出来的列。后续所有 CUDA kernel 的 CSV 都按此格式追加，画图脚本一次 `pd.read_csv` 就能把 baseline + 多版本 kernel 摞起来对比。

---

## 验收标准 & 结果

| # | 标准 | 状态 |
|---|---|---|
| 1 | `pytest tests/` 零失败（PyTorch 19 + CUDA 21 = 40） | ✅ 40 passed |
| 2 | `bench_cuda.py --all-dtypes` 端到端跑通并写 CSV | ✅ 63 行 |
| 3 | 干净环境下 JIT 编译成功（清空 `~/.cache/torch_extensions/...`） | ✅ |
| 4 | fp32/fp16/bf16 输出与 PyTorch reference 在 tolerance 内一致 | ✅（见下方说明） |
| 5 | CSV 列顺序与 `pytorch_baseline.csv` 一致（多一个首列 `kernel`） | ✅ |

### tolerance 说明

任务规范里给的默认容差是：

| dtype | atol | rtol |
|---|---|---|
| fp32 | 1e-5 | 1e-5 |
| fp16 | 1e-3 | 1e-3 |
| bf16 | 5e-3 | 5e-3 |

实测 fp32 完全在范围内；fp16/bf16 各自在大 hidden 上略微超出（fp16 max abs diff ~7.8e-3，bf16 max abs diff ~6.25e-2）。**原因不是 kernel 算错了，而是 PyTorch reference 的 `x.pow(2).mean(-1)` 是用输入 dtype 累加的，本 kernel 按规范要求用 fp32 累加，所以 kernel 反而比 reference 更精确**。

按任务规范"小幅松动并加注释，不要去削 kernel 来追 reference"的指引，把 fp16 容差改成 `(atol=1e-2, rtol=3e-3)`、bf16 改成 `(atol=1e-1, rtol=2e-2)`，并在测试文件 [tests/test_cuda_kernels.py](tests/test_cuda_kernels.py) 的 `_DTYPE_TOL` 处写明原因。

### 性能对照（fp32, RMSNorm only, median ms）

任务要求至少看以下三组 shape：

| shape (B, S, H) | PyTorch baseline | naive CUDA | speedup |
|---|---:|---:|---:|
| (1, 128, 1024)   | 0.0846 | 0.0045 | **18.8×** |
| (4, 512, 2048)   | 0.0918 | 0.0186 | **4.9×**  |
| (8, 1024, 4096)  | 0.1048 | 0.3442 | **0.30×**（更慢） |

小 token 数大幅领先 PyTorch 的 `aten::rms_norm`（PyTorch 在该 shape 是 launch-bound，naive kernel 直接吃满该制约）；大 token / 大 hidden 上反过来被 PyTorch 拉开 3×，因为没做向量化加载、strided per-thread loop 在带宽利用率上吃亏。这正是后续 float4 / fused 版本要解决的问题——本步刻意保留这个差距作为对照基线。

---

## 关键设计决策

### 1. 用 `__shfl_xor_sync`（butterfly）而不是 `__shfl_down_sync`（tree）

```cuda
__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}
```

两者归约出同一个数值结果，但 xor 是对称的——归约结束时 warp 内 32 个 lane 全部拿到一致的总和，**不需要再做一次 broadcast**。`shfl_down` 只有 lane 0 持有结果，cross-warp 写共享内存时还得多一次条件判断或者额外 shuffle，xor 在第二阶段（warp 0 归约 warp_sums）天然吻合。

### 2. fp32 累加器，无视输入 dtype

```cuda
float sumsq = 0.0f;
for (int i = tid; i < hidden_size; i += blockDim.x) {
    const float xv = to_float<T>(x_row[i]);
    sumsq += xv * xv;
}
```

hidden_size 可以到 4096 量级，bf16/fp16 就地累加平方和会很快丢精度（bf16 的 mantissa 只有 7 bit）。生产 kernel（vLLM/Apex/HuggingFace 的 fused\_rmsnorm 等）都是 fp32 累加，这里跟齐。代价是寄存器多一点点，但 256 threads × 1 float = 1 KB 量级，对 occupancy 没影响。

### 3. `warp_sums[0]` 复用作为 scale 广播槽

```cuda
__shared__ float warp_sums[MAX_WARPS];      // 8 floats
// ... 第一阶段把每个 warp 的部分和写进 warp_sums[warp_id]
if (warp_id == 0) {
    float v = (lane < n_warps) ? warp_sums[lane] : 0.0f;
    v = warp_reduce_sum(v);
    if (lane == 0) {
        warp_sums[0] = rsqrtf(v / float(hidden_size) + eps);
    }
}
__syncthreads();
const float scale = warp_sums[0];
```

避免另开一个 shared float 变量。结构紧凑，逻辑也容易读：写回阶段所有线程读同一个 shared 槽，是天然的 broadcast，不需要再来一轮 shuffle。

### 4. 单文件 kernel + binding，不拆 `binding.cpp`

`kernels/rmsnorm.cu` 末尾直接 `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)`，pybind11 + Torch 的 JIT 会一并编译。本步只有一个 kernel，拆出 `binding.cpp` 没收益，反而要多一次 `#include`、多一次 nvcc/clang 边界讨论。后续 kernel 多了之后再拆才有性价比。

### 5. JIT 在 PyTorch 2.11+cu130 环境下的 cusparse.h 坑

PyTorch 2.11 + CUDA 13.0 把 `cusparse.h` 等头文件单独装在 `nvidia.cu13` 这个 namespace package 里（`/usr/local/lib/.../nvidia/cu13/include/`），**不在** `/usr/local/cuda` 下。直接 JIT 会因为 `ATen/cuda/CUDAContextLight.h` 找不到 `cusparse.h` 而失败。

解决：在 `kernels/__init__.py` 里探测 `nvidia.cu13.__path__`，把它的 `include/` 用 `-isystem` 加到 nvcc 命令尾部。**注意必须用 `-isystem` 而不是 `-I`**，否则 wheel 里附带的 13.0 版本 `cuda_runtime_api.h`（`CUDART_VERSION=13000`）会优先于 nvcc 13.1 自带的 13.1 版本，cccl 的 toolkit 版本检查会立刻报错 "compiler and toolkit headers are incompatible"。`-isystem` 把这个目录排到搜索路径最低优先级，只有真正缺失的 cusparse.h 才会从那里 fallback。

```python
for p in _extra_includes:
    _iflags += ["-isystem", p]
```

这段排坑写在 `kernels/__init__.py` 的注释里。后续 fused kernel 复用同一个 JIT 入口就不会再撞。

### 6. CSV 首列加 `kernel` 而不是把 `module` 当 kernel 标识

`module="RMSNorm"` 表达的是"被实现的数学模块"，多个 kernel 版本（naive / vectorized / fused）都可以实现同一个 `module`。第一列 `kernel="naive_rmsnorm"` 标识"哪一版 CUDA 实现"。两列正交，画图时按 `kernel` 分系列、按 `module` 分小图，是最干净的拼接方式。
