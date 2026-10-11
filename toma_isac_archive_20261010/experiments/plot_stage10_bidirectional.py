"""Plot converged Stage 10 bidirectional cold+warm candidates (separate outputs)."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "results" / "stage10" / "bidirectional_warmstart"
LABELS = {
    "b2": "Fixed-ToMA + FP",
    "b3": "FPA-UPA + FP",
    "b6": "MA + FP",
    "proposed": "FP-aware RCG-ToMA + FP (Proposed)",
}


def main():
    csv_path = DIR / "stage10_tradeoff.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}; run stage10 bidirectional simulation first.")
    with csv_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("No converged candidates. Figure not generated.")
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    for s, label in LABELS.items():
        group = sorted((r for r in rows if r["scheme"] == s and r["converged"].lower() == "true"),
                       key=lambda r: float(r["omega_comm"]))
        if not group:
            continue
        x = np.array([float(r["sum_rate_comm"]) for r in group])
        y = np.array([float(r["sum_rate_sensing"]) for r in group])
        ax.plot(x, y, "-o", linewidth=2.7 if s == "proposed" else 1.9,
                markersize=6.5, label=label)
    failures = DIR / "stage10_failures.csv"
    if failures.exists():
        with failures.open(encoding="utf-8", newline="") as f:
            missing = list(csv.DictReader(f))
        if missing:
            ax.set_title(f"Partial data: {len(missing)} unconverged scheme/weight cases", fontsize=11)
    ax.set_xlabel("Sum Communication Rate (bit/s/Hz)")
    ax.set_ylabel("Sum Sensing Rate (bit/s/Hz)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=9)
    fig.tight_layout()
    plot_dir = DIR / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = plot_dir / f"communication_sensing_tradeoff_bidirectional.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
