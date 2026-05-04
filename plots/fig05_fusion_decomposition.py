"""Figure 5 — RMSNormLinear "fusion gain" decomposition.

For each shape config, three bars:
    PyTorch RMSNormLinear         (baseline)
    cuda_rmsnorm + pt_linear      (only RMSNorm replaced with CUDA)
    fused_rmsnorm_linear          (RMSNorm CUDA + cuBLAS linear)

The point: bars 2 and 3 are essentially identical, so the speedup over
PyTorch comes from the RMSNorm part, not from any real RMSNorm-Linear fusion.
This is the honesty plot the project log argues for.

Reads:
    results/pytorch_baseline.csv
    results/cuda_kernels.csv
    results/rmsnorm_linear_split.csv
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import common as C


def main():
    pt = C.load_pytorch()
    cu = C.load_cuda()
    sp = C.load_split()

    fig, axes = plt.subplots(1, len(C.DTYPES), figsize=(15, 5.5),
                              sharey=False)

    shape_keys = C.all_shape_keys()
    xs = np.arange(len(shape_keys))
    width = 0.27

    for ci, dtype in enumerate(C.DTYPES):
        ax = axes[ci]
        pt_idx = C.by_shape(pt, dtype=dtype, module="RMSNormLinear")
        sp_idx = C.by_shape(sp, dtype=dtype, kernel="cuda_rmsnorm+pt_linear",
                             module="RMSNormLinear")
        cu_idx = C.by_shape(cu, dtype=dtype, kernel="fused_rmsnorm_linear",
                             module="RMSNormLinear")

        pt_y = np.array([pt_idx[k].median_ms if k in pt_idx else np.nan
                          for k in shape_keys])
        sp_y = np.array([sp_idx[k].median_ms if k in sp_idx else np.nan
                          for k in shape_keys])
        cu_y = np.array([cu_idx[k].median_ms if k in cu_idx else np.nan
                          for k in shape_keys])

        ax.bar(xs - width, pt_y, width, color=C.C_PT,
               label="PyTorch RMSNormLinear", edgecolor="black", linewidth=0.3)
        ax.bar(xs, sp_y, width, color=C.C_SPLIT,
               label="CUDA RMSNorm + PyTorch Linear", edgecolor="black", linewidth=0.3)
        ax.bar(xs + width, cu_y, width, color=C.C_LINEAR,
               label="fused_rmsnorm_linear (CUDA + cuBLAS)",
               edgecolor="black", linewidth=0.3)

        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"H={k[2]}\nB{k[0]}S{k[1]}" for k in shape_keys],
                            rotation=70, ha="right", fontsize=6.5)
        ax.set_title(C.DTYPE_LABEL[dtype], fontweight="bold")
        if ci == 0:
            ax.set_ylabel("median latency (ms, log scale)")
        if ci == 1:
            ax.legend(loc="upper left", fontsize=8)

        # Tier separators
        for x_b in [6.5, 13.5]:
            ax.axvline(x_b, color="black", lw=0.4, alpha=0.4)

    fig.suptitle(
        "RMSNormLinear: where does the speedup come from?\n"
        "Replacing only RMSNorm (purple) is already as fast as the \"fused\" "
        "kernel (green) — the win is RMSNorm overhead, not RMSNorm-into-GEMM fusion.",
        fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    C.save(fig, "fig05_fusion_decomposition.png")


if __name__ == "__main__":
    main()
