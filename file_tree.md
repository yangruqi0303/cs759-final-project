# File Tree — cs759-final-project

```
cs759-final-project/
│
├── baseline/                        # PyTorch 参考实现
│   ├── __init__.py                  # 包标记
│   ├── configs.py                   # BenchConfig dataclass + 21 组基准配置
│   │                                #   3 模型规模 (small/medium/large)
│   │                                #   × 7 token 形状 (batch × seq_len)
│   ├── pytorch_ref.py               # 三个 nn.Module 参考实现
│   │   ├── RMSNorm                  #   y = x * rsqrt(mean(x²) + eps) * weight  → CUDA kernel (Step 2 ✅)
│   │   ├── RMSNormLinear            #   Linear(RMSNorm(x))                       → CUDA kernel v1 (后续)
│   │   └── RMSNormMLP               #   Linear2(GELU(Linear1(RMSNorm(x))))       → CUDA kernel v2 (后续)
│   └── README.md                    # 安装、PyTorch 测试/bench、CUDA 测试/bench 运行说明
│
├── kernels/                         # CUDA kernels (Step 2 新增)
│   ├── __init__.py                  # JIT load (torch.utils.cpp_extension.load)
│   │                                #   含 nvidia.cu13 wheel 头路径探测 (cusparse.h)
│   │                                #   暴露: rmsnorm_cuda(x, weight, eps)
│   └── rmsnorm.cu                   # naive RMSNorm kernel + pybind binding (单文件)
│       ├── warp_reduce_sum()        #   __shfl_xor_sync butterfly, fp32
│       ├── rmsnorm_kernel<T>()      #   1 block / row, 256 threads, fp32 累加
│       └── rmsnorm_cuda()           #   host launcher, dispatch fp32/fp16/bf16
│
├── tests/                           # 正确性测试
│   ├── test_pytorch_ref.py          # 19 个 pytest 用例
│   │   ├── TestRMSNorm              #   shape / 手写公式对照 / 梯度 / golden regression
│   │   ├── TestRMSNormLinear        #   shape / 分解对照 / 梯度
│   │   ├── TestRMSNormMLP           #   shape / 分解对照 / 梯度
│   │   ├── TestDtypeStability       #   fp32/fp16/bf16 无 NaN/Inf
│   │   └── assert_close()           #   可复用的 tolerance 比较函数 (供 CUDA kernel 测试)
│   └── test_cuda_kernels.py         # 21 个 pytest 用例 (Step 2 新增)
│       ├── TestRMSNormCUDA          #   3 shape × 3 dtype 对照 PyTorch ref + finite check
│       └── TestInputValidation      #   non-contig / dtype mismatch / size mismatch 必抛 RuntimeError
│
├── benchmarks/                      # 性能基准
│   ├── bench_pytorch.py             # PyTorch 三模块 benchmark, --dtype/--all-dtypes
│   │   ├── _time_fn()               #   CUDA event 计时 (warmup + timed iterations)
│   │   └── bench_one()              #   单 (module, config, dtype) → CSV row
│   └── bench_cuda.py                # CUDA RMSNorm benchmark (Step 2 新增)
│                                    #   完全镜像 bench_pytorch.py 的计时 / CSV 形态
│                                    #   首列 kernel="naive_rmsnorm"
│
├── results/                         # 输出数据
│   ├── pytorch_baseline.csv         # 63 行 (21 configs × 3 modules), fp32 baseline
│   └── cuda_kernels.csv             # 63 行 (21 configs × 3 dtypes), naive_rmsnorm (Step 2 新增)
│
├── log_baseline.md                  # Step 1 完成记录：PyTorch 参考实现 + 测试 + benchmark
├── log_naive_rmsnorm.md             # Step 2 完成记录：naive CUDA RMSNorm kernel + JIT 接入
├── .gitignore                       # __pycache__ / *.so / .pytest_cache 等
└── file_tree.md                     # 本文件
```

## 已完成

- **Step 1** — `log_baseline.md`：PyTorch 参考实现 (`RMSNorm`, `RMSNormLinear`, `RMSNormMLP`) + 19 pytest + bench_pytorch
- **Step 2** — `log_naive_rmsnorm.md`：naive CUDA `RMSNorm` kernel + JIT + 21 pytest + bench_cuda
  - kernel: 1 block / row, `__shfl_xor_sync` warp reduction + 共享内存 cross-warp，fp32 累加器
  - 验收：`pytest tests/` 40/40 通过；`bench_cuda.py --all-dtypes` 输出 63 行 CSV
  - 性能：小 token 数对 PyTorch baseline ~5–19× 加速；大 token / 大 hidden 上 ~3× 慢于 PyTorch（无 float4 vectorization）

## 后续扩展点

- `kernels/rmsnorm_vec.cu` — float4 / vectorized 版 RMSNorm（先把 naive 在大 shape 上的 3× 差距补回来）
- `kernels/rmsnorm_linear.cu` — 融合 RMSNorm + Linear（对照 `RMSNormLinear`）
- `kernels/rmsnorm_mlp.cu` — 融合 RMSNorm + 双 Linear + GELU(tanh)（对照 `RMSNormMLP`）
- 上述每个 kernel 在 `bench_cuda.py` 里加一个 `KERNEL_NAME`，CSV 自动 append；画图脚本按 `kernel` 列分系列即可
```
