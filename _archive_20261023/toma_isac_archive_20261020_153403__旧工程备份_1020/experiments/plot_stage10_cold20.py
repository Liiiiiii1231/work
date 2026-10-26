r"""Stage 10: publication-ready plot for the 20-weight *independent cold-start* run.

Plotting only: reads the existing 80-row CSV. No solver calls, no data filtering,
no smoothing, no interpolation, and no across-weight warm start.

Run from the project root:
    python -u .\experiments\plot_stage10_cold20.py

Output (original filenames, plus SVG):
    results/stage10/cold_wide20_dl114_tol1e4/figures/
        communication_sensing_tradeoff.png
        communication_sensing_tradeoff.pdf
        communication_sensing_tradeoff.svg

NOTE: All neighboring sampled-weight results are connected using solid lines.
Connecting lines are visual guides; intermediate locations are not evaluated points.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Also works from Windows CMD without opening a GUI window.
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "results" / "stage10" / "cold_wide20_dl114_tol1e4"
DEFAULT_CSV = RUN_DIR / "stage10_tradeoff.csv"
DEFAULT_OUTPUT = RUN_DIR / "figures"

SCHEME_ORDER = ("b2", "b3", "b6", "proposed")
STYLE = {
    "b2": {
        "label": "Fixed-ToMA + FP", "color": "#2678B9", "marker": "o", "lw": 1.8,
    },
    "b3": {
        "label": "FPA-UPA + FP", "color": "#E87917", "marker": "s", "lw": 1.8,
    },
    "b6": {
        "label": "MA + FP", "color": "#26933B", "marker": "^", "lw": 1.9,
    },
    "proposed": {
        "label": "FP-aware RCG-ToMA + FP (Proposed)",
        "color": "#D62828", "marker": "D", "lw": 2.5,
    },
}
WEIGHTS = (
    0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.35,
    0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9,
)

def load_data(csv_path: Path) -> dict[str, list[dict[str, float]]]:
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Missing Stage 10 cold-start CSV: {csv_path}\n"
            "Please use the completed 20-point independent cold-start results."
        )
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 80:
        raise ValueError(f"Expected 80 rows (20 weights x 4 schemes); got {len(rows)}")

    grouped: dict[str, list[dict[str, float]]] = {k: [] for k in SCHEME_ORDER}
    for i, row in enumerate(rows, start=2):
        scheme = row.get("scheme", "")
        if scheme not in grouped:
            raise ValueError(f"Unknown scheme {scheme!r} in CSV line {i}")
        if row.get("converged", "true").strip().lower() not in {"true", "1", "yes"}:
            raise ValueError(f"Non-converged result in line {i}; do not publish a partial graph")
        item: dict[str, float] = {}
        for key in ("omega_comm", "sum_rate_comm", "sum_rate_sensing"):
            try:
                item[key] = float(row[key])
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(f"Bad {key} in CSV line {i}") from e
            if not math.isfinite(item[key]):
                raise ValueError(f"Non-finite {key} in CSV line {i}")
        grouped[scheme].append(item)

    expected = {round(w, 9) for w in WEIGHTS}
    for scheme, group in grouped.items():
        group.sort(key=lambda t: t["omega_comm"])
        actual = [round(t["omega_comm"], 9) for t in group]
        if len(actual) != 20 or len(set(actual)) != 20 or set(actual) != expected:
            raise ValueError(f"Incomplete or duplicate weights for {scheme}: {actual}")
    return grouped


def render(data: dict[str, list[dict[str, float]]], output: Path, *, label_weights: bool = True) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    all_x = [p["sum_rate_comm"] for g in data.values() for p in g]
    all_y = [p["sum_rate_sensing"] for g in data.values() for p in g]
    print(f"[CHECK] Plotting {sum(map(len, data.values()))} raw converged CSV points; no interpolation.")

    legend_handles = []
    for scheme in SCHEME_ORDER:
        style = STYLE[scheme]
        group = data[scheme]
        x = np.array([p["sum_rate_comm"] for p in group])
        y = np.array([p["sum_rate_sensing"] for p in group])
        w = np.array([p["omega_comm"] for p in group])

        # Connect neighboring sampled weights with solid lines only.
        # Lines are visual guides; no interpolated points are added.
        ax.plot(
            x, y,
            color=style["color"],
            linewidth=style["lw"],
            linestyle="-",
            alpha=0.96,
            zorder=3,
            solid_capstyle="round",
        )
        ax.scatter(
            x, y, s=36 if scheme == "proposed" else 31,
            marker=style["marker"], color=style["color"],
            edgecolors="white", linewidths=0.5, zorder=5,
        )
        legend_handles.append(Line2D(
            [0], [0], linestyle="-", color=style["color"],
            linewidth=style["lw"], marker=style["marker"],
            markersize=5.8, markeredgecolor="white", markeredgewidth=0.5,
            label=style["label"],
        ))

        # In particular, avoid the crowded (x~96, y~0) omega labels.
        if scheme == "proposed" and label_weights:
            j = int(np.where(np.isclose(w, 0.1, atol=1e-12))[0][0])
            ax.annotate(
                r"$\omega_C=0.1$", xy=(x[j], y[j]),
                xytext=(10, 14), textcoords="offset points",
                fontsize=9, color="#333333",
                ha="left", va="bottom",
                arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.75},
                zorder=6,
            )

    ax.set_xlabel("Sum Communication Rate (bit/s/Hz)", labelpad=7)
    ax.set_ylabel("Sum Sensing Rate (bit/s/Hz)", labelpad=7)
    ax.set_xlim(min(-2.0, min(all_x)-1.0), max(all_x)+3.0)
    ax.set_ylim(min(-1.0, min(all_y)-0.6), max(all_y)+2.2)
    ax.grid(True, color="#89939D", alpha=0.23, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_linewidth(0.8)

    # Two-column legend above the plot, in the same order as the historical figure.
    ax.legend(
        handles=legend_handles, ncol=2, loc="lower center",
        bbox_to_anchor=(0.5, 1.012), frameon=True, fancybox=False,
        edgecolor="#CCCCCC", facecolor="white", borderpad=0.55,
        columnspacing=1.8, handlelength=2.2, labelspacing=0.55,
    )
    fig.subplots_adjust(left=0.115, right=0.98, bottom=0.14, top=0.79)
    output.mkdir(parents=True, exist_ok=True)

    for ext in ("png", "pdf", "svg"):
        out = output / f"communication_sensing_tradeoff.{ext}"
        fig.savefig(out, dpi=400 if ext == "png" else None, bbox_inches="tight", pad_inches=0.10)
        print(f"[SAVED] {out}")
    print("[NOTE] All connecting segments are solid; all 80 raw points retained.")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper plot for Stage 10 cold20 only (no optimization).")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Override the CSV input path")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Override figure directory")
    parser.add_argument("--no-weight-label", action="store_true", help="Remove omega_C=0.1 annotation")
    args = parser.parse_args()
    data = load_data(args.csv)
    render(data, args.output_dir, label_weights=not args.no_weight_label)


if __name__ == "__main__":
    main()
