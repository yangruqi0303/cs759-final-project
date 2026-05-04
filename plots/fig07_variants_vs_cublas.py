"""Figure 7 — Variants speedup relative to cuBLAS default.

Each variant divided by cuBLAS-default time (so "1.0" = parity with cuBLAS).
A bar above 1.0 means the variant beat cuBLAS for that shape.

This isolates the dramatic finding from log_rmsnorm_linear_mlp.md:
prologue_v2 sometimes beats cuBLAS on prologue-friendly shapes (small
out_features) — the rest of the time cuBLAS wins by 5-15× thanks to Tensor
Core paths.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import common as C


def _size(b, s, h, o):
    return b * s * h * o


# Variants to plot (exclude the cuBLAS default itself, which is the reference).
VARIANTS = [
    "naive_rmsnorm_linear",
    "tiled_rmsnorm_linear",
    "prologue_rmsnorm_linear",
    "tiled_rmsnorm_linear_v2",
    "prologue_rmsnorm_linear_v2",
]


def main():
    rows = C.load_variants()

    seen = []
    config_shape = {}
    for r in rows:
        if r.config not in config_shape:
            seen.append(r.config)
            config_shape[r.config] = (r.batch, r.seq_len, r.hidden, r.intermediate)
    configs = sorted(seen, key=lambda c: _size(*config_shape[c]))

    fig, axes = plt.subplots(len(C.DTYPES), 1, figsize=(13.5, 11),
                              sharex=True)

    n_var = len(VARIANTS)
    width = 0.85 / n_var
    xs = np.arange(len(configs))

    for ri, dtype in enumerate(C.DTYPES):
        ax = axes[ri]

        # cuBLAS reference per config
        cublas_t = {}
        for r in rows:
            if r.dtype == dtype and r.kernel == "fused_rmsnorm_linear":
                cublas_t[r.config] = r.median_ms

        for vi, var in enumerate(VARIANTS):
            ys = np.full(len(configs), np.nan)
            for r in rows:
                if r.dtype != dtype or r.kernel != var:
                    continue
                if r.config not in cublas_t:
                    continue
                idx = configs.index(r.config)
                ys[idx] = cublas_t[r.config] / r.median_ms  # speedup vs cuBLAS
            offset = (vi - n_var / 2 + 0.5) * width
            ax.bar(xs + offset, ys, width,
                   color=C.VARIANT_COLOR[var],
                   edgecolor="black", linewidth=0.4,
                   label=C.VARIANT_LABEL[var])

        ax.axhline(1.0, color="black", lw=1.0)
        ax.set_yscale("log")
        ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1, 2])
        ax.set_yticklabels(["0.05×", "0.1×", "0.2×", "0.5×", "1×", "2×"])
        ax.set_ylabel(f"{C.DTYPE_LABEL[dtype]}\nspeedup vs cuBLAS")

    axes[-1].set_xticks(xs)
    labels = []
    for c in configs:
        b, s, h, o = config_shape[c]
        labels.append(f"{c}\n(B{b}·S{s}·H{h}·O{o})")
    axes[-1].set_xticklabels(labels, rotation=30, ha="right", fontsize=7)

    fig.suptitle(
        "Custom GEMM variants vs cuBLAS default (1.0× = parity)\n"
        "bars above the black line beat cuBLAS — most cells lose, "
        "but prologue v2 wins on prologue-friendly small-output shapes.",
        fontsize=12, fontweight="bold")

    # Reserve right margin for the one-per-row legend so it never overlaps
    # any bar.
    fig.tight_layout(rect=(0, 0, 0.78, 0.95))
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels,
               loc="center left",
               bbox_to_anchor=(0.785, 0.55),
               ncol=1, fontsize=10,
               handlelength=1.6, handletextpad=0.6,
               borderaxespad=0, frameon=False)
    C.save(fig, "fig07_variants_vs_cublas.png")


if __name__ == "__main__":
    main()
