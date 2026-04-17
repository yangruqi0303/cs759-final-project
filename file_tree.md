# File Tree — cs759-final-project

```
cs759-final-project/
│
├── baseline/                        # PyTorch 参考实现
│   ├── __init__.py                  # 包标记
│   ├── configs.py                   # BenchConfig dataclass + 15 组基准配置
│   │                                #   3 模型规模 (small/medium/large)
│   │                                #   × 5 token 形状 (batch × seq_len)
│   ├── pytorch_ref.py               # 三个 nn.Module 参考实现
│   │   ├── RMSNorm                  #   y = x * rsqrt(mean(x²) + eps) * weight
│   │   ├── RMSNormLinear            #   Linear(RMSNorm(x))          → CUDA kernel v1
│   │   └── RMSNormMLP              #   Linear2(GELU(Linear1(RMSNorm(x)))) → kernel v2
│   └── README.md                    # 安装、测试、benchmark 运行说明
│
├── tests/                           # 正确性测试
│   └── test_pytorch_ref.py          # 19 个 pytest 用例
│       ├── TestRMSNorm              #   shape / 手写公式对照 / 梯度 / golden regression
│       ├── TestRMSNormLinear        #   shape / 分解对照 / 梯度
│       ├── TestRMSNormMLP           #   shape / 分解对照 / 梯度
│       ├── TestDtypeStability       #   fp32/fp16/bf16 无 NaN/Inf
│       └── assert_close()           #   可复用的 tolerance 比较函数 (供 CUDA kernel 测试)
│
├── benchmarks/                      # 性能基准
│   └── bench_pytorch.py             # CUDA event 计时, --dtype 参数, CSV 输出
│       ├── _time_fn()               #   通用 GPU 计时函数 (warmup + timed iterations)
│       └── bench_one()              #   单配置 benchmark, 返回 dict → CSV row
│
├── results/                         # 输出数据
│   └── pytorch_baseline.csv         # 45 行 (15 configs × 3 modules), fp32
│
├── log_baseline.md                  # Step 1 完成记录：任务/架构/接口/设计决策
└── file_tree.md                     # 本文件
```

## 后续扩展点

- `kernels/` — CUDA/C++ kernel 源码 + pybind11 绑定
- `tests/test_cuda_kernels.py` — 导入 `assert_close`，对照 `pytorch_ref` 验证 kernel 正确性
- `benchmarks/bench_cuda.py` — 复用 `_time_fn()` 和 `BenchConfig`，产出同格式 CSV
- `results/` — 各 kernel 版本的 CSV，用于画图和 report
