"""Plot the TARGET-CONVERGED refined Stage 10 results; never interpolate or hide data."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "stage10" / "pool_wide14_tol1e4"
LABELS = {"b2":"Fixed-ToMA + FP", "b3":"FPA-UPA + FP",
          "b6":"MA + FP", "proposed":"Proposed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = args.output / "stage10_tradeoff.csv"
    if not source.exists():
        raise FileNotFoundError(f"Run Stage 10 --phase all first: {source}")
    with source.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows or any(r.get("converged", "").lower() != "true" for r in rows):
        raise RuntimeError("Plot requires converged target-optimized results only")
    fig, ax = plt.subplots(figsize=(8.5, 5.7))
    for scheme, label in LABELS.items():
        data = sorted((r for r in rows if r["scheme"] == scheme),
                      key=lambda r:float(r["omega_comm"]))
        if not data: continue
        x = np.array([float(r["sum_rate_comm"]) for r in data])
        y = np.array([float(r["sum_rate_sensing"]) for r in data])
        ax.plot(x,y, marker="o", markersize=4.2, linewidth=2.3 if scheme=="proposed" else 1.5,
                label=label, alpha=.95)
    ax.set_xlabel("Sum communication rate (bit/s/Hz)")
    ax.set_ylabel("Sum sensing rate (bit/s/Hz)")
    ax.grid(alpha=.22,linewidth=.6)
    ax.legend(frameon=False, ncol=2,fontsize=9,loc="best")
    fig.tight_layout()
    figures = args.output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for suffix in ("png","pdf"):
        path = figures / f"communication_sensing_tradeoff.{suffix}"
        fig.savefig(path,dpi=350,bbox_inches="tight")
        print("Saved:",path)
    plt.close(fig)


if __name__ == "__main__":
    main()
