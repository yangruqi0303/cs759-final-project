"""Figure 4 — Achieved memory bandwidth for RMSNorm (RTX 4060 Ti 16 GB).

RMSNorm is a streaming, bandwidth-bound op:
    bytes touched ≈ read x  +  write y  +  read gamma
                 ≈ 2 * N * H * sizeof(T)   (gamma is hidden-only)

We plot achieved BW = bytes / time vs total bytes touched, for PyTorch and
CUDA, faceted by dtype.

Two reference annotations are drawn:
  * a horizontal dashed line at the GDDR6 peak BW of the RTX 4060 Ti
    (288 GB/s — same on the 8 GB and 16 GB SKU);
  * a shaded vertical band marking the working-set range that fits in the
    AD106 L2 cache (32 MB). Inside that band the benchmark loop reuses data
    from L2 across timed iterations, so the measured "BW" is *effective*
    bandwidth (cache-hot) and can legitimately exceed the DRAM peak.
    Only the points to the right of the shaded band are DRAM-bound and can
    be compared against the peak line.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import common as C


# RTX 4060 Ti (AD106). Same for 8 GB and 16 GB variants.
PEAK_BW_GBPS = 288.0     # GDDR6, 128-bit bus, 18 Gbps
L2_CACHE_MB = 32.0       # AD106 L2 size


def main():
    pt = C.load_pytorch()
    cu = C.load_cuda()

    fig, axes = plt.subplots(1, len(C.DTYPES), figsize=(13, 4.6), sharey=True)

    shape_keys = C.all_shape_keys()

    for ci, dtype in enumerate(C.DTYPES):
        ax = axes[ci]
        pt_idx = C.by_shape(pt, dtype=dtype, module="RMSNorm")
        cu_idx = C.by_shape(cu, dtype=dtype, kernel="naive_rmsnorm",
                             module="RMSNorm")

        bytes_arr = []
        pt_bw = []
        cu_bw = []
        tier_color = []
        tier_palette = {1024: "#1f77b4", 2048: "#ff7f0e", 4096: "#2ca02c"}

        for k in shape_keys:
            b, s, h, _ = k
            bytes_ = C.rmsnorm_bytes(b, s, h, dtype)
            bytes_arr.append(bytes_)
            pt_bw.append(C.gbps(bytes_, pt_idx[k].median_ms) if k in pt_idx else np.nan)
            cu_bw.append(C.gbps(bytes_, cu_idx[k].median_ms) if k in cu_idx else np.nan)
            tier_color.append(tier_palette[h])

        bytes_arr = np.array(bytes_arr)
        pt_bw = np.array(pt_bw)
        cu_bw = np.array(cu_bw)

        # --- L2-resident shaded band ---
        # Set x-limits first so the band covers the full plotted range below L2.
        x_mb = bytes_arr / 1e6
        x_lo = max(x_mb.min() * 0.5, 0.1)
        x_hi = x_mb.max() * 2.0
        ax.set_xlim(x_lo, x_hi)
        ax.axvspan(x_lo, L2_CACHE_MB, color="#fff2cc", alpha=0.55, zorder=0)
        ax.axvline(L2_CACHE_MB, color="#cc9900", lw=0.8, ls="--", zorder=1)
        ax.text(L2_CACHE_MB, ax.get_ylim()[1] if False else 0,  # filled later
                "", fontsize=7, color="#7f6000")

        # --- DRAM peak reference line ---
        ax.axhline(PEAK_BW_GBPS, ls=":", color="black", lw=1.2)

        ax.scatter(x_mb, pt_bw, c=tier_color, marker="o",
                   s=55, edgecolors="black", linewidths=0.5, alpha=0.7,
                   zorder=3)
        ax.scatter(x_mb, cu_bw, c=tier_color, marker="*",
                   s=130, edgecolors="black", linewidths=0.5, zorder=3)

        ax.set_xscale("log")
        ax.set_xlabel("bytes touched per call (MB)")
        ax.set_title(C.DTYPE_LABEL[dtype], fontweight="bold")
        if ci == 0:
            ax.set_ylabel("achieved bandwidth (GB/s)")

        # Annotate the two reference lines (placed once per panel; cheap text).
        ymax = max(np.nanmax(cu_bw), PEAK_BW_GBPS) * 1.12
        ax.set_ylim(0, ymax)
        ax.text(x_lo * 1.1, PEAK_BW_GBPS,
                " RTX 4060 Ti DRAM peak = 288 GB/s",
                fontsize=7.5, color="black", va="bottom", ha="left")
        ax.text(L2_CACHE_MB * 0.95, ymax * 0.97,
                "L2 (32 MB) →",
                fontsize=7.5, color="#7f6000", va="top", ha="right")
        ax.text(L2_CACHE_MB * 1.07, ymax * 0.97,
                "DRAM-bound region",
                fontsize=7.5, color="#444444", va="top", ha="left", style="italic")

    # --- Combined legend in first panel ---
    from matplotlib.lines import Line2D
    tier_palette = {1024: "#1f77b4", 2048: "#ff7f0e", 4096: "#2ca02c"}
    legend_handles = [
        Line2D([0], [0], marker="s", color="w", mfc=tier_palette[h],
               markersize=8, label=f"H={h}")
        for h in [1024, 2048, 4096]
    ] + [
        Line2D([0], [0], marker="o", color="w", mfc="gray",
               markeredgecolor="black", markersize=8, label="PyTorch"),
        Line2D([0], [0], marker="*", color="w", mfc="gray",
               markeredgecolor="black", markersize=12, label="CUDA"),
    ]
    axes[0].legend(handles=legend_handles, loc="upper left",
                   fontsize=7.5, ncol=2,
                   columnspacing=0.6, handletextpad=0.3)

    fig.suptitle(
        "RMSNorm achieved bandwidth on RTX 4060 Ti 16 GB\n"
        "yellow band = working set fits in L2 (32 MB) → cache-hot, can exceed "
        "DRAM peak;  white region = true DRAM-bound regime",
        fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    C.save(fig, "fig04_rmsnorm_bandwidth.png")


if __name__ == "__main__":
    main()
