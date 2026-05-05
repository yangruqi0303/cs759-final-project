# File Tree — cs759-final-project

```
cs759-final-project/
│
├── README.md                        # 项目总览：硬件/软件栈、conda 安装、运行测试与 benchmark
├── environment.yml                  # conda 一键环境（与 README Option A 对应）
├── .gitignore
├── file_tree.md                     # 本文件
│
├── baseline/                        # PyTorch 参考实现（正确性与性能基线）
│   ├── __init__.py
│   ├── configs.py                   # BenchConfig + 多组基准配置
│   ├── pytorch_ref.py               # RMSNorm / RMSNormLinear / RMSNormMLP
│   └── README.md                    # 安装与 bench / 测试说明
│
├── kernels/                         # CUDA kernels（JIT: torch.utils.cpp_extension.load）
│   ├── __init__.py                  # 多扩展分别 load；nvidia.cu13 头路径；-lcublas 链到 linear/mlp
│   ├── rmsnorm.cu                   # naive RMSNorm + pybind
│   ├── rmsnorm_common.cuh           # RMSNorm 共享规约 / 类型逻辑（被多 .cu 包含）
│   ├── rmsnorm_linear.cu            # RMSNorm + cuBLAS Linear（对照 ref）
│   ├── rmsnorm_linear_naive.cu      # 全手写 naive GEMM baseline
│   ├── rmsnorm_linear_tiled.cu      # 物化 RMSNorm + 标量 FMA 分块 GEMM
│   ├── rmsnorm_linear_tiled_v2.cu   # 物化 + v2 寄存器分块 / 向量化 GEMM
│   ├── rmsnorm_linear_prologue.cu   # prologue 融合：scale×gamma 进 GEMM，不物化整行 norm
│   ├── rmsnorm_linear_prologue_v2.cu
│   ├── rmsnorm_mlp.cu               # RMSNorm + 双层 Linear + GELU(tanh)
│   ├── cublas_linear_common.cuh     # cuBLAS 侧公共封装
│   └── tiled_linear_common.cuh      # 分块 GEMM 共享逻辑
│
├── tests/
│   ├── test_pytorch_ref.py          # PyTorch 参考：shape / 分解 / 梯度 / dtype
│   └── test_cuda_kernels.py         # 全 CUDA 入口：RMSNorm、RMSNormLinear 各变体、RMSNormMLP 等
│
├── benchmarks/
│   ├── bench_pytorch.py             # PyTorch 三模块 benchmark → CSV
│   ├── bench_cuda.py                # 多 kernel 列（RMSNorm + linear/mlp 等），镜像计时与 CSV 形态
│   ├── bench_rmsnorm_linear_split.py    # split / 分解类 benchmark
│   └── bench_rmsnorm_linear_variants.py # naive / tiled / prologue / v2 等变体对比
│
├── results/                         # 主开发机导出的 benchmark CSV
│   ├── pytorch_baseline.csv
│   ├── cuda_kernels.csv
│   ├── rmsnorm_linear_split.csv
│   └── rmsnorm_linear_variants.csv
│
├── results.5070Ti/                  # RTX 5070 Ti 上跑出的对照结果
│   ├── pytorch_baseline.csv
│   └── cuda_kernels.csv
│
├── plots/                           # 论文式图表：读 results CSV，写 figures/
│   ├── README.md
│   ├── common.py                    # 读 CSV、样式、路径等公共函数
│   ├── run_all.py                   # 一键生成全部图
│   ├── fig01_speedup_overview.py
│   ├── fig02_latency_vs_tokens.py
│   ├── fig03_speedup_heatmap.py
│   ├── fig04_rmsnorm_bandwidth.py
│   ├── fig05_fusion_decomposition.py
│   ├── fig06_variants_ladder.py
│   ├── fig07_variants_vs_cublas.py
│   └── figures/                     # 导出的 PNG
│       ├── fig01_speedup_overview.png
│       ├── fig02_latency_vs_tokens.png
│       ├── fig03_speedup_heatmap.png
│       ├── fig04_rmsnorm_bandwidth.png
│       ├── fig05_fusion_decomposition.png
│       ├── fig06_variants_ladder.png
│       └── fig07_variants_vs_cublas.png
│
├── log_baseline.md                  # Step 1：PyTorch 参考 + 测试 + bench
├── log_naive_rmsnorm.md             # Step 2：naive CUDA RMSNorm
├── log_rmsnorm_linear_mlp.md        # RMSNormLinear 变体与 RMSNormMLP 融合实验记录
│
└── .claude/worktrees/suspicious-chatelet-a4d2db   # git submodule（Claude 工作树，可选检出）
```

## 已完成（概要）

- **Step 1** — `log_baseline.md`：`baseline/` 三模块 + `test_pytorch_ref.py` + `bench_pytorch.py`
- **Step 2** — `log_naive_rmsnorm.md`：`rmsnorm.cu` + JIT + `bench_cuda.py` 中 RMSNorm 路径
- **扩展** — `kernels/` 中 RMSNormLinear 多实现（cuBLAS、naive、tiled、prologue 及 v2）、`rmsnorm_mlp.cu`；`bench_rmsnorm_linear_*.py`；`results/` 与 `results.5070Ti/`；`plots/` 全套脚本与 `figures/`；`log_rmsnorm_linear_mlp.md` 记录融合与变体实验

## 后续扩展点

- `kernels/rmsnorm_vec.cu`（或等价向量化）— 大 shape 上补齐 naive RMSNorm 相对 PyTorch 的带宽差距
- 更深 MLP / 其他激活或布局的融合与 profile 驱动调参
- 新 kernel 时在 `bench_cuda.py` 增加 `KERNEL_NAME` 行格式，CSV 与 `plots/common.py` 约定列名即可接图
