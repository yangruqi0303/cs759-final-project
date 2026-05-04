"""Figure 6 — RMSNormLinear GEMM variants ablation ladder.

For each variants config (rmsnorm_linear_variants.csv), six bars:
    naive GEMM
    tiled GEMM
    prologue v1 (fused RMSNorm prologue, scalar tiled GEMM)
    tiled v2 (register tiling + vec load + smem gamma)
    prologue v2
    cuBLAS (default fused_rmsnorm_linear)

Faceted by dtype. The point: shows the optimization "ladder" — naive → tiled
is a ~5× win, prologue v1 over tiled v1 is mostly noise, but v2 versions are
significantly better. Sometimes prologue_v2 even beats cuBLAS.

Reads:
    results/rmsnorm_linear_variants.csv
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import common as C


# Order configs by problem size (rough — total tokens × hidden × out).
def _size(b, s, h, o):
    return b * s * h * o


def main():
    rows = C.load_variants()

    # Discover unique configs (ordered by appearance in csv).
    seen = []
    config_shape = {}
    for r in rows:
        if r.config not in config_shape:
            seen.append(r.config)
            config_shape[r.config] = (r.batch, r.seq_len, r.hidden, r.intermediate)
    configs = sorted(seen, key=lambda c: _size(*config_shape[c]))

    fig, axes = plt.subplots(len(C.DTYPES), 1, figsize=(13.5, 11),
                              sharex=True)

    n_var = len(C.VARIANT_ORDER)
    width = 0.85 / n_var
    xs = np.arange(len(configs))

    for ri, dtype in enumerate(C.DTYPES):
        ax = axes[ri]

        # Build per-variant arrays.
        for vi, var in enumerate(C.VARIANT_ORDER):
            ys = np.full(len(configs), np.nan)
            for r in rows:
                if r.dtype != dtype or r.kernel != var:
                    continue
                if r.config not in configs:
                    continue
                ys[configs.index(r.config)] = r.median_ms
            offset = (vi - n_var / 2 + 0.5) * width
            ax.bar(xs + offset, ys, width,
                   color=C.VARIANT_COLOR[var],
                   edgecolor="black", linewidth=0.4,
                   label=C.VARIANT_LABEL[var])

        ax.set_yscale("log")
        ax.set_ylabel(f"{C.DTYPE_LABEL[dtype]}\nmedian latency (ms, log)")
        ax.grid(axis="x", alpha=0)

    axes[-1].set_xticks(xs)
    labels = []
    for c in configs:
        b, s, h, o = config_shape[c]
        labels.append(f"{c}\n(B{b}·S{s}·H{h}·O{o})")
    axes[-1].set_xticklabels(labels, rotation=30, ha="right", fontsize=7)

    fig.suptitle(
        "RMSNormLinear GEMM variants: optimization ladder\n"
        "configs ordered by problem size; lower bar = faster",
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
    C.save(fig, "fig06_variants_ladder.png")


if __name__ == "__main__":
    main()
