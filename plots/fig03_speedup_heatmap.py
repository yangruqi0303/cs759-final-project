"""Figure 3 — Speedup heatmap.

Rows = 21 shape configs (small / medium / large blocks).
Cols = 9 = 3 modules × 3 dtypes.
Cell = log2(speedup of CUDA over PyTorch). Diverging colormap centered at 0
(parity); blue = CUDA wins, red = CUDA loses.

Annotated with speedup value (×) inside each cell.
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


def main():
    pt = C.load_pytorch()
    cu = C.load_cuda()

    shape_keys = C.all_shape_keys()
    row_labels = []
    for k in shape_keys:
        b, s, h, _ = k
        row_labels.append(f"H={h}  B{b}S{s}")

    modules = list(MODULE_TO_KERNEL)
    col_groups = [(m, d) for m in modules for d in C.DTYPES]
    col_labels = [f"{m}\n{C.DTYPE_LABEL[d]}" for m, d in col_groups]

    M = np.full((len(shape_keys), len(col_groups)), np.nan)
    for ci, (module, dtype) in enumerate(col_groups):
        kernel = MODULE_TO_KERNEL[module]
        pt_idx = C.by_shape(pt, dtype=dtype, module=module)
        cu_idx = C.by_shape(cu, dtype=dtype, kernel=kernel, module=module)
        for ri, k in enumerate(shape_keys):
            if k in pt_idx and k in cu_idx:
                M[ri, ci] = pt_idx[k].median_ms / cu_idx[k].median_ms

    log_M = np.log2(M)

    fig, ax = plt.subplots(figsize=(11.5, 9.5))

    vmax = float(np.nanmax(np.abs(log_M)))
    vmax = max(vmax, 1.0)
    im = ax.imshow(log_M, cmap="RdBu", vmin=-vmax, vmax=vmax,
                    aspect="auto")

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=0, fontsize=8)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8, family="monospace")

    # Tier separators
    ax.axhline(6.5, color="black", lw=1.0)
    ax.axhline(13.5, color="black", lw=1.0)
    # Module separators
    ax.axvline(2.5, color="black", lw=1.0)
    ax.axvline(5.5, color="black", lw=1.0)
    ax.set_xticks(np.arange(len(col_labels) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(row_labels) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", lw=0.4)
    ax.tick_params(which="minor", length=0)
    ax.grid(False)

    # Annotate cell values.
    for ri in range(len(row_labels)):
        for ci in range(len(col_labels)):
            v = M[ri, ci]
            if np.isnan(v):
                continue
            color = "white" if abs(log_M[ri, ci]) > vmax * 0.55 else "black"
            ax.text(ci, ri, f"{v:.2f}×", ha="center", va="center",
                    fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(r"$\log_2$(speedup)   blue = CUDA wins")

    ax.set_title("Speedup heatmap: CUDA / PyTorch median latency\n"
                 "rows grouped by model size; columns grouped by module",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    C.save(fig, "fig03_speedup_heatmap.png")


if __name__ == "__main__":
    main()
