"""绘制 Stage 10 Communication-Sensing trade-off 曲线。"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "stage10" / "cold_wide14_tol1e4" / "stage10_tradeoff.csv"
FIGURE_DIR = ROOT / "results" / "stage10" / "cold_wide14_tol1e4" / "figures"

SCHEME_ORDER = ("b2", "b3", "b6", "proposed")
SCHEME_LABELS = {
    "b2": "Fixed-ToMA + FP",
    "b3": "FPA-UPA + FP",
    "b6": "MA + FP",
    "proposed": "FP-aware RCG-ToMA + FP (Proposed)",
}


def load_rows() -> list[dict[str, str]]:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CSV_PATH}. Run run_stage10_tradeoff.py first."
        )

    with CSV_PATH.open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def plot_tradeoff() -> None:
    rows = load_rows()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise RuntimeError("No converged Stage 10 points. Figure was NOT generated.")

    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    plotted = 0
    for scheme in SCHEME_ORDER:
        scheme_rows = [
            row for row in rows
            if row["scheme"] == scheme
            and row.get("converged", "True").lower() == "true"
        ]
        scheme_rows.sort(key=lambda row: float(row["omega_comm"]))
        if not scheme_rows:
            print(f"[SKIP] {SCHEME_LABELS[scheme]}: no converged data points")
            continue

        x = np.asarray(
            [float(row["sum_rate_comm"]) for row in scheme_rows],
            dtype=float,
        )
        y = np.asarray(
            [float(row["sum_rate_sensing"]) for row in scheme_rows],
            dtype=float,
        )
        omega = np.asarray(
            [float(row["omega_comm"]) for row in scheme_rows],
            dtype=float,
        )

        linewidth = 2.8 if scheme == "proposed" else 1.9
        markersize = 7.0 if scheme == "proposed" else 6.0

        plotted += len(x)
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=linewidth,
            markersize=markersize,
            label=SCHEME_LABELS[scheme],
        )

        if scheme == "proposed":
            for target in (0.1, 0.5, 0.9):
                index = int(np.argmin(np.abs(omega - target)))
                if np.isclose(omega[index], target, atol=1.0e-12):
                    ax.annotate(
                        rf"$\omega_C={target:.1f}$",
                        (x[index], y[index]),
                        xytext=(7, 7),
                        textcoords="offset points",
                        fontsize=9,
                    )

    if plotted == 0:
        plt.close(fig)
        raise RuntimeError("No converged Stage 10 points. Figure was NOT generated.")

    # Flag partial plots and keep invalid solutions OUT of the graph.
    failures_path = ROOT / "results" / "stage10" / "cold_wide14_tol1e4" / "stage10_failures.csv"
    if failures_path.exists():
        with failures_path.open("r", encoding="utf-8") as file:
            failures = list(csv.DictReader(file))
        if failures:
            ax.set_title("Partial result: converged points only", fontsize=11)
            print(f"[WARN] {len(failures)} failed scheme/weight cases omitted")

    ax.set_xlabel("Sum Communication Rate (bit/s/Hz)", fontsize=12)
    ax.set_ylabel("Sum Sensing Rate (bit/s/Hz)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        fontsize=9,
    )

    fig.tight_layout()

    png_path = FIGURE_DIR / "communication_sensing_tradeoff.png"
    pdf_path = FIGURE_DIR / "communication_sensing_tradeoff.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure saved to: {png_path}")
    print(f"Figure saved to: {pdf_path}")


if __name__ == "__main__":
    plot_tradeoff()
