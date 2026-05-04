"""Figure 1 — Speedup overview.

3 modules (rows) × 3 dtypes (cols). Each panel: 21 vertical bars (one per shape
config, ordered small→medium→large). Bar height = CUDA / PyTorch speedup.
A horizontal dashed line marks geomean speedup; horizontal solid line at 1.0
marks parity with PyTorch.

Reads:
    results/pytorch_baseline.csv
    results/cuda_kernels.csv
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import common as C


MODULE_TO_KERNEL = {
    "RMSNorm": "naive_rmsnorm",
    "RMSNormLinear": "fused_rmsnorm_linear",
    "RMSNormMLP": "fused_rmsnorm_mlp",
}
MODULE_COLOR = {
    "RMSNorm": C.C_RMSNORM,
    "RMSNormLinear": C.C_LINEAR,
    "RMSNormMLP": C.C_MLP,
}


def main():
    pt = C.load_pytorch()
    cu = C.load_cuda()

    modules = list(MODULE_TO_KERNEL)
    fig, axes = plt.subplots(len(modules), len(C.DTYPES),
                              figsize=(13, 8.5), sharex=True)

    shape_keys = C.all_shape_keys()
    xs = np.arange(len(shape_keys))
    xticklabels = []
    tier_boundaries = []  # for vertical separators
    last_h = None
    for i, key in enumerate(shape_keys):
        b, s, h, _ = key
        xticklabels.append(f"B{b}S{s}")
        if last_h is not None and h != last_h:
            tier_boundaries.append(i - 0.5)
        last_h = h

    for ri, module in enumerate(modules):
        kernel = MODULE_TO_KERNEL[module]
        for ci, dtype in enumerate(C.DTYPES):
            ax = axes[ri, ci]
            pt_idx = C.by_shape(pt, dtype=dtype, module=module)
            cu_idx = C.by_shape(cu, dtype=dtype, kernel=kernel, module=module)

            speedups = []
            for k in shape_keys:
                if k in pt_idx and k in cu_idx:
                    speedups.append(pt_idx[k].median_ms / cu_idx[k].median_ms)
                else:
                    speedups.append(float("nan"))
            speedups = np.array(speedups)

            colors = [MODULE_COLOR[module] if s_ >= 1 else "#bbbbbb"
                      for s_ in speedups]
            ax.bar(xs, speedups, color=colors, edgecolor="black",
                   linewidth=0.4)

            ax.axhline(1.0, color="black", lw=0.8)
            gm = C.geomean(speedups)
            ax.axhline(gm, color="red", lw=0.9, ls="--",
                       label=f"geomean = {gm:.2f}×")

            for x_b in tier_boundaries:
                ax.axvline(x_b, color="black", lw=0.4, alpha=0.4)

            ax.set_yscale("log")
            ax.set_ylim(0.4, max(20.0, np.nanmax(speedups) * 1.15))
            ax.set_yticks([0.5, 1, 2, 5, 10, 20])
            ax.set_yticklabels(["0.5×", "1×", "2×", "5×", "10×", "20×"])

            if ri == 0:
                ax.set_title(C.DTYPE_LABEL[dtype], fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{module}\nspeedup over PyTorch")
            ax.legend(loc="upper right")

    # Tier annotations on bottom row
    tier_centers = [3, 10, 17]  # 7 shapes per tier
    for ax in axes[-1]:
        ax.set_xticks(xs)
        ax.set_xticklabels(xticklabels, rotation=70, ha="right", fontsize=7)
        ymin = ax.get_ylim()[0]
        for ci_, name in zip(tier_centers,
                              ["small\nH=1024", "medium\nH=2048", "large\nH=4096"]):
            ax.annotate(name, xy=(ci_, ymin), xytext=(0, -32),
                        textcoords="offset points",
                        ha="center", va="top", fontsize=8, fontweight="bold")

    fig.suptitle(
        "CUDA kernel speedup over PyTorch baseline\n"
        "21 shape configurations × 3 dtypes; bars below 1.0 = slower than PyTorch",
        fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    C.save(fig, "fig01_speedup_overview.png")


if __name__ == "__main__":
    main()
