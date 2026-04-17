# Step 1: PyTorch Baseline — 完成记录

## 任务概述

为 CS759 final project "Fused CUDA Kernels for RMSNorm and MLP in Transformer Inference" 搭建 PyTorch 参考实现 + 正确性测试 + 性能基准。这是整个项目的"标尺"——后续所有 CUDA kernel 的正确性和性能都以此为对照。

## 完成的具体工作

### 1. 三个参考模块 (`baseline/pytorch_ref.py`)

| 模块 | 数学表达 | 对应 CUDA 融合目标 |
|---|---|---|
| `RMSNorm` | `y = x * rsqrt(mean(x², dim=-1) + eps) * weight` | standalone kernel |
| `RMSNormLinear` | `Linear(RMSNorm(x))` | kernel v1 |
| `RMSNormMLP` | `Linear2(GELU(Linear1(RMSNorm(x))))` | kernel v2 |

所有模块均为标准 `torch.nn.Module`，无任何优化技巧，保证"显然正确"。

### 2. 基准配置 (`baseline/configs.py`)

用 `BenchConfig` dataclass 定义了 15 组配置，覆盖三个模型规模：

- **Small**: hidden=1024, intermediate=4096
- **Medium**: hidden=2048, intermediate=8192
- **Large (Qwen2/Llama-7B)**: hidden=4096, intermediate=11008

每个规模下 5 种 (batch, seq_len) 组合，total tokens 从 512 到 8192。

### 3. 正确性测试 (`tests/test_pytorch_ref.py`)

19 个 pytest 用例，覆盖：
- 输出 shape 验证
- 梯度回传（autograd sanity）
- Golden regression（固定 seed，输出跨 run 位级一致）
- 手写公式对照（RMSNorm 逐步计算 vs 模块输出，atol=1e-6）
- 三种 dtype（fp32/fp16/bf16）在最大 config 上无 NaN/Inf

### 4. 性能基准 (`benchmarks/bench_pytorch.py`)

- 10 次 warmup + 100 次计时，报告 median 和 stdev
- 支持 `--dtype {fp32, fp16, bf16}` 参数
- 输出：stdout 表格 + `results/pytorch_baseline.csv`

---

## 架构设计

```
baseline/
  configs.py        ← 所有 shape 配置，dataclass 可迭代
  pytorch_ref.py    ← 三个 nn.Module，CUDA kernel 的对照实现
  __init__.py

tests/
  test_pytorch_ref.py  ← 正确性 harness，可复用的 assert_close

benchmarks/
  bench_pytorch.py     ← CUDA event 计时，CSV 输出

results/
  pytorch_baseline.csv ← 结构化性能数据
```

后续 CUDA kernel 的接入方式：
1. kernel 输出 tensor → 调用 `tests/test_pytorch_ref.py` 中的 `assert_close(ref, candidate)` 验证正确性
2. kernel 封装成 callable → 传入 `benchmarks/bench_pytorch.py` 的 `_time_fn()` 计时
3. 结果追加到 CSV → 和 baseline 同表对比

---

## 接口规范

### `assert_close(ref, candidate, atol, rtol)`

```python
def assert_close(
    ref: torch.Tensor,
    candidate: torch.Tensor,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> None:
```

- 先检查 shape 一致，再调用 `torch.testing.assert_close`
- CUDA kernel 测试直接导入使用：`from tests.test_pytorch_ref import assert_close`
- fp16/bf16 kernel 可以放宽 tolerance：`atol=1e-3, rtol=1e-2`

### `_time_fn(fn, warmup, iters) -> list[float]`

```python
def _time_fn(fn: callable, warmup: int, iters: int) -> list[float]:
```

- 返回每次迭代的毫秒数列表
- 使用 CUDA events 计时（见下方设计决策）

### `BenchConfig` dataclass

```python
@dataclass(frozen=True)
class BenchConfig:
    name: str
    batch: int
    seq_len: int
    hidden_size: int
    intermediate_size: int
```

所有测试和 benchmark 都遍历 `BENCH_CONFIGS` 列表。新增配置只需往列表里加。

### CSV 输出格式

列：`module, batch, seq_len, hidden, intermediate, median_ms, stdev_ms, dtype`

后续 CUDA kernel benchmark 产出同格式 CSV，可以直接合并画图。

---

## 验收标准 & 结果

| # | 标准 | 状态 |
|---|---|---|
| 1 | `pytest tests/` 零失败 | ✅ 19/19 passed |
| 2 | `bench_pytorch.py` 端到端运行并产出 CSV | ✅ |
| 3 | RMSNorm 输出与手写公式在 fp32 精度内一致 | ✅ `test_manual_formula` |
| 4 | fp32/fp16/bf16 在最大 config 上无 NaN/Inf | ✅ `TestDtypeStability` |
| 5 | 文件结构干净，无废代码 | ✅ |

---

## 关键设计决策

### 1. GELU 必须用 `approximate='tanh'`

```python
self.act = nn.GELU(approximate="tanh")
```

tanh 近似 GELU 和精确 GELU 数值上不同。Llama、Qwen 等现代 LLM 统一使用 tanh 近似版本。CUDA kernel 里会 hard-code tanh 近似的数学公式：

```
GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
```

baseline 必须和 kernel 用同一个公式，否则正确性测试会因为数值差异误报失败。这个选择在 Step 1 锁定，后续不可更改。

### 2. 必须用 CUDA Events 计时，不能用 `time.time()`

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
torch.cuda.synchronize()
start.record()
fn()
end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end)
```

PyTorch 的 GPU 操作是异步的——`fn()` 返回时 GPU 可能还没执行完。如果用 `time.time()`，测到的是 CPU 端 launch kernel 的时间，不是 GPU 实际执行时间。对于小 kernel（如 RMSNorm ~0.08ms），`time.time()` 的数字会完全失真。

`torch.cuda.Event` 在 GPU timeline 上打点，`elapsed_time()` 返回两个事件之间的 GPU 实际耗时。前后各加一次 `synchronize()` 确保：
- 前一个 `synchronize()`：排空之前的 GPU 队列，让计时干净
- 后一个 `synchronize()`：等 end event 完成后再读时间

### 3. 正确性 harness 设计为可复用

`assert_close(ref, candidate, atol, rtol)` 的接口刻意设计得通用：

- 不绑定任何特定模块，只要求两个 tensor + tolerance
- CUDA kernel 测试流程：用 baseline 模块算 `ref`，用 kernel 算 `candidate`，调同一个函数
- fp32 kernel 用默认 tolerance，fp16/bf16 可以放宽

测试结构也可扩展——后续为 CUDA kernel 新增 test class 时，可以复用 `_make_input()`、`_UNIT` config 等辅助函数，不需要从头写 harness。

### 4. CSV 结构化输出

benchmark 同时输出 stdout 表格（人看）和 CSV 文件（程序用）。CSV 的好处：

- 后续 CUDA kernel benchmark 产出同格式 CSV，可以 `cat` 合并
- 画图时直接 `pandas.read_csv()` 或用 matplotlib 读
- 写 report 时数据可追溯，不需要从 terminal log 里手动抄数字

列名设计覆盖了所有维度（module/shape/dtype），一个 CSV 就能支撑所有对比分析。
