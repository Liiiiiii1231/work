"""绘制 Stage 10 参考非均匀权重 weighted-sum trade-off。"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_stage10_tradeoff import SCHEME_LABELS, SCHEME_ORDER

CSV_PATH = (
    ROOT
    / "results"
    / "stage10_final_weights"
    / "stage10_final_weights.csv"
)
FIG_DIR = ROOT / "results" / "stage10_final_weights" / "figures"
PNG_PATH = FIG_DIR / "communication_sensing_tradeoff_final_weights.png"
PDF_PATH = FIG_DIR / "communication_sensing_tradeoff_final_weights.pdf"


def load_rows() -> list[dict[str, str]]:
    with CSV_PATH.open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main() -> None:
    rows = load_rows()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.5, 7.0))

    for scheme in SCHEME_ORDER:
        scheme_rows = sorted(
            [row for row in rows if row["scheme"] == scheme],
            key=lambda row: float(row["omega_comm"]),
        )

        if not scheme_rows:
            continue

        rc = np.asarray(
            [float(row["sum_rate_comm"]) for row in scheme_rows],
            dtype=float,
        )
        rs = np.asarray(
            [float(row["sum_rate_sensing"]) for row in scheme_rows],
            dtype=float,
        )
        omega = np.asarray(
            [float(row["omega_comm"]) for row in scheme_rows],
            dtype=float,
        )

        ax.plot(rc, rs, marker="o", linewidth=2.0, label=SCHEME_LABELS[scheme])

        # 只标注三个代表性通信权重，避免图过于拥挤。
        for target in (0.0, 0.50, 1.0):
            if np.any(np.isclose(omega, target)):
                idx = int(np.argmin(np.abs(omega - target)))
                ax.annotate(
                    rf"$\omega_C={omega[idx]:.2f}$",
                    (rc[idx], rs[idx]),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=9,
                )

    ax.set_xlabel("Sum Communication Rate (bit/s/Hz)")
    ax.set_ylabel("Sum Sensing Rate (bit/s/Hz)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(PNG_PATH, dpi=220, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")

    print(f"PNG saved to: {PNG_PATH}")
    print(f"PDF saved to: {PDF_PATH}")


if __name__ == "__main__":
    main()
