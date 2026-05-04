"""Figure 2 — Latency vs total tokens, log-log.

Rows = 3 modules (RMSNorm / RMSNormLinear / RMSNormMLP).
Cols = 3 model size tiers (small / medium / large).
For each panel: x = total tokens (batch * seq_len), y = median latency (ms),
PyTorch and CUDA shown for all three dtypes (different markers/colors).

The reader can see how the gap PT vs CUDA scales with token count, and how
the gap closes (or opens) as the model grows.
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

DTYPE_COLOR = {"float32": "#1f77b4", "float16": "#2ca02c", "bfloat16": "#d62728"}


def main():
    pt = C.load_pytorch()
    cu = C.load_cuda()

    modules = list(MODULE_TO_KERNEL)
    fig, axes = plt.subplots(len(modules), len(C.TIERS),
                              figsize=(12.5, 8.5), sharex="col")

    for ri, module in enumerate(modules):
        kernel = MODULE_TO_KERNEL[module]
        for ci, (h, tier_name) in enumerate(C.TIERS):
            ax = axes[ri, ci]
            shape_keys = C.shape_order_for_tier(h)

            # Sort by total tokens for nice lines.
            sorted_keys = sorted(shape_keys, key=lambda k: k[0] * k[1])
            tokens = np.array([k[0] * k[1] for k in sorted_keys])

            for dtype in C.DTYPES:
                pt_idx = C.by_shape(pt, dtype=dtype, module=module)
                cu_idx = C.by_shape(cu, dtype=dtype, kernel=kernel, module=module)
                pt_y = np.array([pt_idx[k].median_ms if k in pt_idx else np.nan
                                  for k in sorted_keys])
                cu_y = np.array([cu_idx[k].median_ms if k in cu_idx else np.nan
                                  for k in sorted_keys])

                color = DTYPE_COLOR[dtype]
                ax.plot(tokens, pt_y, color=color, lw=1.5, ls="--",
                        marker="o", markersize=5, mfc="white",
                        label=f"PyTorch {C.DTYPE_LABEL[dtype]}")
                ax.plot(tokens, cu_y, color=color, lw=1.8, ls="-",
                        marker="s", markersize=5,
                        label=f"CUDA {C.DTYPE_LABEL[dtype]}")

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks([128, 512, 1024, 2048, 4096, 8192])
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

            if ri == 0:
                ax.set_title(tier_name, fontsize=10, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{module}\nmedian latency (ms)")
            if ri == len(modules) - 1:
                ax.set_xlabel("total tokens (batch × seq_len)")

            if ri == 0 and ci == len(C.TIERS) - 1:
                ax.legend(loc="lower right", ncol=2, fontsize=7,
                          columnspacing=0.6, handletextpad=0.3)

    fig.suptitle(
        "Latency scaling: PyTorch (dashed, ○) vs CUDA (solid, ■)\n"
        "log-log; the vertical gap between matched colors is the per-config speedup",
        fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    C.save(fig, "fig02_latency_vs_tokens.png")


if __name__ == "__main__":
    main()
