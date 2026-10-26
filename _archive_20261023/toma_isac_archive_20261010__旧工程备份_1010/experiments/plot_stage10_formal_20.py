"""Plot Stage 10 twenty-weight formal WSR solutions; never interpolate missing solutions."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "stage10" / "formal_20point_dl114_multiround"
LABELS = {"b2":"Fixed-ToMA + FP", "b3":"FPA-UPA + FP", "b6":"MA + FP", "proposed":"Proposed"}
COLORS = {"b2":"tab:blue", "b3":"tab:orange", "b6":"tab:green", "proposed":"tab:red"}


def draw(rows, output, *, connect_all=False):
    fig,ax = plt.subplots(figsize=(9,5.8))
    xs = np.array([float(r["sum_rate_comm"]) for r in rows])
    ys = np.array([float(r["sum_rate_sensing"]) for r in rows])
    scale_x = max(float(np.ptp(xs)), 1.0)
    scale_y = max(float(np.ptp(ys)), 1.0)
    for scheme,label in LABELS.items():
        matches = sorted([r for r in rows if r["scheme"] == scheme],
                         key=lambda r:float(r["omega_comm"]))
        if not matches:
            continue
        x = np.array([float(r["sum_rate_comm"]) for r in matches])
        y = np.array([float(r["sum_rate_sensing"]) for r in matches])
        color = COLORS[scheme]
        ax.scatter(x,y,s=30 if scheme=="proposed" else 23,color=color,
                   label=label,zorder=4)
        for i in range(1,len(x)):
            d = np.hypot((x[i]-x[i-1])/scale_x,(y[i]-y[i-1])/scale_y)
            if connect_all or d <= 0.28:
                ax.plot(x[i-1:i+1],y[i-1:i+1],color=color,
                        lw=2.0 if scheme=="proposed" else 1.5,alpha=.88)
    ax.set_xlabel("Sum communication rate (bit/s/Hz)")
    ax.set_ylabel("Sum sensing rate (bit/s/Hz)")
    ax.grid(alpha=.24,linewidth=.6)
    ax.legend(frameon=False,ncol=2,loc="upper center",bbox_to_anchor=(.5,1.18),fontsize=9)
    fig.tight_layout()
    figures=output/"figures"
    figures.mkdir(parents=True,exist_ok=True)
    for ext in ("png","pdf"):
        path = figures/f"communication_sensing_tradeoff.{ext}"
        fig.savefig(path,dpi=350,bbox_inches="tight")
        print("Saved",path)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    p.add_argument("--connect-all",action="store_true",help="Diagnostic only: include long mode-jump segments")
    args=p.parse_args()
    source=args.output/"stage10_tradeoff.csv"
    with source.open(encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    if not rows or not all(r.get("converged","").lower()=="true" for r in rows):
        raise RuntimeError("Only target-converged solutions may enter the formal plot")
    from collections import Counter
    counts = Counter(row["scheme"] for row in rows)
    if len(rows) != 80 or any(counts.get(scheme) != 20 for scheme in LABELS):
        raise RuntimeError(f"Expected 80 rows (20 per scheme), got {len(rows)}: {dict(counts)}")
    for scheme in LABELS:
        w = [float(row["omega_comm"]) for row in rows if row["scheme"] == scheme]
        if len(set(w)) != 20:
            raise RuntimeError(f"Duplicate or missing weights for {scheme}")
    draw(rows,args.output,connect_all=args.connect_all)


if __name__=="__main__":
    main()
