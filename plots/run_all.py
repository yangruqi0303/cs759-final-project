"""Render every figure under plots/figures/."""
from __future__ import annotations

import importlib
import sys
import time

MODULES = [
    "fig01_speedup_overview",
    "fig02_latency_vs_tokens",
    "fig03_speedup_heatmap",
    "fig04_rmsnorm_bandwidth",
    "fig05_fusion_decomposition",
    "fig06_variants_ladder",
    "fig07_variants_vs_cublas",
]


def main():
    sys.path.insert(0, ".")  # so `import common` works when launched from repo root
    sys.path.insert(0, __import__("os").path.dirname(__file__))
    for name in MODULES:
        t0 = time.time()
        print(f"[plots] {name} ...", flush=True)
        m = importlib.import_module(name)
        m.main()
        print(f"        done in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
