"""Diagnostic visualization of ALL converged candidates vs selected WSR winners.

Do not interpret line connections as a continuous Pareto boundary.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "stage10" / "transition_050_060_tol1e4"
WEIGHTS = (0.50,0.52,0.54,0.55,0.56,0.58,0.60)


def read_csv(path):
    if not path.exists():
        return []
    with path.open("r",newline="",encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    out = args.output.resolve()
    attempts = read_csv(out / "all_converged_candidates.csv")
    selected = read_csv(out / "stage10_tradeoff.csv")
    if not attempts:
        raise FileNotFoundError("Run transition probe --phase cold first")
    fig, axs = plt.subplots(1, 2, figsize=(12,5.2), sharex=False, sharey=False)
    for ax, (scheme, title, col) in zip(axs, (("proposed","Proposed", "tab:red"),
                                                  ("b6", "MA + FP (B6)", "tab:green"))):
        cold = [a for a in attempts if a["scheme"]==scheme and a["method"]=="cold"]
        warm = [a for a in attempts if a["scheme"]==scheme and a["method"]!="cold"]
        final = sorted((s for s in selected if s["scheme"]==scheme),key=lambda x:float(x["omega_comm"]))
        if cold:
            ax.scatter([float(a["sum_rate_comm"]) for a in cold],
                       [float(a["sum_rate_sensing"]) for a in cold],
                       s=20,c="0.53",alpha=0.58,label="All converged cold candidates", zorder=2)
        if warm:
            ax.scatter([float(a["sum_rate_comm"]) for a in warm],
                       [float(a["sum_rate_sensing"]) for a in warm],
                       s=32, marker="x", color="tab:blue",alpha=.75,label="Converged cross-weight refinements",zorder=3)
        if final:
            x=[float(s["sum_rate_comm"]) for s in final]
            y=[float(s["sum_rate_sensing"]) for s in final]
            ax.plot(x,y,linestyle="--",linewidth=1.2,color=col,alpha=.78,zorder=4)
            ax.scatter(x,y,s=48,color=col,edgecolor="white",linewidth=.6,zorder=5,label="Selected max-WSR result")
            for r in final:
                ax.annotate(f'{float(r["omega_comm"]):.2f}',
                            (float(r["sum_rate_comm"]),float(r["sum_rate_sensing"])),
                            xytext=(4,5),textcoords="offset points",fontsize=7.5)
        ax.set_title(title)
        ax.set_xlabel("Communication sum rate (bit/s/Hz)")
        ax.set_ylabel("Sensing sum rate (bit/s/Hz)")
        ax.grid(alpha=.22)
        ax.legend(loc="best",fontsize=7.5,frameon=False)
    fig.suptitle("Stage 10 transition-zone diagnostic: candidates vs WSR-selected solutions")
    fig.tight_layout()
    out.mkdir(parents=True,exist_ok=True)
    for ext in ("png","pdf"):
        name=out/f"transition_candidates.{ext}"
        fig.savefig(name,dpi=300,bbox_inches="tight")
        print("Saved:",name)
    plt.close(fig)

    matrix=read_csv(out/"all_reweighted_scores.csv")
    if matrix:
        fig,axs=plt.subplots(1,2,figsize=(12,4.4))
        for ax,scheme in zip(axs,("proposed","b6")):
            vals=[v for v in matrix if v["scheme"]==scheme]
            origins=sorted(set(float(r["origin_omega"]) for r in vals))
            for origin in origins:
                d=sorted((r for r in vals if abs(float(r["origin_omega"])-origin)<1e-12),
                         key=lambda r: float(r["target_omega"]))
                if d:
                    # These are best-at-origin reweighted values, not optimized for the x-weight.
                    bytarget={float(r["target_omega"]):max(float(q["target_reweighted_wsr"])
                              for q in d if float(q["target_omega"])==float(r["target_omega"])) for r in d}
                    ax.plot(sorted(bytarget),[bytarget[w] for w in sorted(bytarget)],alpha=.6,
                            linewidth=1.15,label=f"origin {origin:.2f}")
            ax.set_title(f"{scheme}: reweighted feasible candidates")
            ax.set_xlabel("Target communication weight")
            ax.set_ylabel("Re-evaluated WSR (not target convergence)")
            ax.grid(alpha=.2)
        axs[-1].legend(loc="center left",bbox_to_anchor=(1,0.5),fontsize=7,frameon=False)
        fig.tight_layout()
        name=out/"cross_weight_candidate_scores.png"
        fig.savefig(name,dpi=300,bbox_inches="tight")
        print("Saved:",name)
        plt.close(fig)


if __name__ == "__main__":
    main()
