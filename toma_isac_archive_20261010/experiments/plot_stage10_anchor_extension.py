"""Plot original vs anchor-refined target-converged states, without interpolation."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "results" / "stage10" / "transition_050_060_tol1e4" / "anchor_extension"


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=DEFAULT)
    args = p.parse_args()
    out = args.output.resolve()
    rows = read_csv(out / "selected_after_anchors.csv")
    audits = read_csv(out / "anchor_comparison.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.3))
    for ax, scheme, title, color in zip(axes, ("proposed", "b6"),
                                        ("Proposed", "MA + FP (B6)"),
                                        ("tab:red", "tab:green")):
        vals = sorted([r for r in rows if r["scheme"] == scheme],
                      key=lambda r: float(r["omega_comm"]))
        info = sorted([r for r in audits if r["scheme"] == scheme],
                      key=lambda r: float(r["omega_comm"]))
        weights = [float(r["omega_comm"]) for r in info]
        ax.plot(weights, [float(r["old_target_best_wsr"]) for r in info],
                "o--", color="0.45", label="Previous target-converged", ms=4)
        ax.plot(weights, [float(r["anchor_direct_feasible_wsr"]) for r in info],
                "^:", color="tab:blue", label="Anchor direct reweight (feasible only)", ms=4)
        ax.plot(weights, [float(r["selected_target_converged_wsr"]) for r in info],
                "o-", color=color, label="Best target-converged after anchors", ms=5)
        ax.set(title=title, xlabel=r"Communication weight $\omega_C$", ylabel="Weighted sum rate")
        ax.grid(alpha=.24)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Stage 10 anchor extension: WSR quality diagnostic")
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    for fmt in ("png", "pdf"):
        f = out / f"anchor_wsr_comparison.{fmt}"
        fig.savefig(f, dpi=250, bbox_inches="tight")
        print("Saved:", f)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.3))
    for ax, scheme, title, color in zip(axes, ("proposed", "b6"),
                                        ("Proposed", "MA + FP (B6)"),
                                        ("tab:red", "tab:green")):
        vals = sorted([r for r in rows if r["scheme"] == scheme],
                      key=lambda r: float(r["omega_comm"]))
        ax.plot([float(r["sum_rate_comm"]) for r in vals],
                [float(r["sum_rate_sensing"]) for r in vals],
                "o--", color=color, lw=1.2, label="Selected points (ordered by weight)")
        for r in vals:
            ax.annotate(f'{float(r["omega_comm"]):.2f}',
                        (float(r["sum_rate_comm"]), float(r["sum_rate_sensing"])),
                        xytext=(4, 5), textcoords="offset points", fontsize=8)
        ax.set(title=title, xlabel="Communication sum rate (bit/s/Hz)",
               ylabel="Sensing sum rate (bit/s/Hz)")
        ax.grid(alpha=.24)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Selected target-converged solutions (not a smoothed Pareto curve)")
    fig.tight_layout()
    f = out / "anchor_selected_rates.png"
    fig.savefig(f, dpi=250, bbox_inches="tight")
    print("Saved:", f)
    plt.close(fig)


if __name__ == "__main__":
    main()
