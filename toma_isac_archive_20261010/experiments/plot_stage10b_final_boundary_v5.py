"""Stage 10B-v5：把临界区间精查结果并入 v4 候选池并重新提取最终边界。

依赖结果：
    results/stage10b/epsilon_lexicographic_upa_adaptive.csv
    results/stage10b/stage10b_local_refine_v4.csv
    results/stage10b/stage10b_threshold_refine_v5.csv
    results/stage10a/stage10a_tradeoff.csv

运行：
    python experiments/plot_stage10b_final_boundary_v5.py

输出：
    results/stage10b/stage10b_final_candidates_v5.csv
    results/stage10b/stage10b_final_boundary_v5.csv
    results/stage10b/stage10b_final_boundary_v5.png
    results/stage10b/stage10b_final_boundary_v5.pdf
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import plot_stage10b_final_boundary_v4 as v4plot  # noqa: E402


RESULT_DIR = ROOT / "results" / "stage10b"
V5_CSV = RESULT_DIR / "stage10b_threshold_refine_v5.csv"

CANDIDATE_CSV = (
    RESULT_DIR / "stage10b_final_candidates_v5.csv"
)
BOUNDARY_CSV = (
    RESULT_DIR / "stage10b_final_boundary_v5.csv"
)
FIG_PNG = RESULT_DIR / "stage10b_final_boundary_v5.png"
FIG_PDF = RESULT_DIR / "stage10b_final_boundary_v5.pdf"


def load_v5() -> list[v4plot.Candidate]:
    if not V5_CSV.exists():
        raise FileNotFoundError(V5_CSV)

    out: list[v4plot.Candidate] = []

    with V5_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        for row in csv.DictReader(file):
            converged = (
                v4plot.to_bool(
                    row["stage1_converged"]
                )
                and v4plot.to_bool(
                    row["stage2_converged"]
                )
            )

            out.append(
                v4plot.Candidate(
                    source="epsilon_threshold_v5",
                    source_parameter=float(
                        row["gamma_comm"]
                    ),
                    sum_rate_comm=float(
                        row["sum_rate_comm"]
                    ),
                    sum_rate_sensing=float(
                        row["sum_rate_sensing"]
                    ),
                    converged=converged,
                    detail=(
                        f"start="
                        f"{row['selected_step1_source']}"
                    ),
                )
            )

    return out


def save_candidates(
    candidates: list[v4plot.Candidate],
) -> None:
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "source",
        "source_parameter",
        "sum_rate_comm",
        "sum_rate_sensing",
        "converged",
        "is_pareto",
        "detail",
    ]

    with CANDIDATE_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )
        writer.writeheader()

        for item in sorted(
            candidates,
            key=lambda x: (
                x.sum_rate_comm,
                -x.sum_rate_sensing,
            ),
        ):
            writer.writerow(
                {
                    "source": item.source,
                    "source_parameter": (
                        item.source_parameter
                    ),
                    "sum_rate_comm": (
                        item.sum_rate_comm
                    ),
                    "sum_rate_sensing": (
                        item.sum_rate_sensing
                    ),
                    "converged": item.converged,
                    "is_pareto": item.is_pareto,
                    "detail": item.detail,
                }
            )


def save_boundary(
    boundary: list[v4plot.Candidate],
) -> None:
    with BOUNDARY_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "boundary_index",
                "source",
                "source_parameter",
                "sum_rate_comm",
                "sum_rate_sensing",
                "converged",
            ],
        )
        writer.writeheader()

        for index, item in enumerate(
            boundary,
            start=1,
        ):
            writer.writerow(
                {
                    "boundary_index": index,
                    "source": item.source,
                    "source_parameter": (
                        item.source_parameter
                    ),
                    "sum_rate_comm": (
                        item.sum_rate_comm
                    ),
                    "sum_rate_sensing": (
                        item.sum_rate_sensing
                    ),
                    "converged": item.converged,
                }
            )


def plot(
    adaptive,
    local,
    threshold,
    weighted,
    boundary,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.9))

    ax.scatter(
        [x.sum_rate_comm for x in adaptive],
        [x.sum_rate_sensing for x in adaptive],
        marker="o",
        s=30,
        alpha=0.18,
        label="Adaptive epsilon candidates",
    )

    ax.scatter(
        [x.sum_rate_comm for x in local],
        [x.sum_rate_sensing for x in local],
        marker="s",
        s=38,
        alpha=0.35,
        label="v4 local-refine candidates",
    )

    ax.scatter(
        [x.sum_rate_comm for x in threshold],
        [x.sum_rate_sensing for x in threshold],
        marker="^",
        s=58,
        alpha=0.75,
        label="v5 threshold-refine candidates",
    )

    ax.scatter(
        [x.sum_rate_comm for x in weighted],
        [x.sum_rate_sensing for x in weighted],
        marker="x",
        s=55,
        alpha=0.60,
        label="Stage 10A weighted-sum candidates",
    )

    ax.plot(
        [x.sum_rate_comm for x in boundary],
        [x.sum_rate_sensing for x in boundary],
        marker="D",
        linewidth=2.5,
        markersize=6,
        label="Final nondominated achievable boundary",
    )

    ax.set_xlabel(
        "Sum Communication Rate (bit/s/Hz)"
    )
    ax.set_ylabel(
        "Sum Sensing Rate (bit/s/Hz)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIG_PNG, dpi=300)
    fig.savefig(FIG_PDF)
    plt.close(fig)

    print(f"Figure saved to: {FIG_PNG}")
    print(f"PDF saved to:    {FIG_PDF}")


def main() -> None:
    adaptive = v4plot.load_adaptive()
    local = v4plot.load_local()
    weighted = v4plot.load_weighted()
    threshold = load_v5()

    pool = [
        item
        for item in (
            adaptive
            + local
            + threshold
            + weighted
        )
        if item.converged
    ]

    pool = v4plot.collapse_duplicates(pool)
    boundary = v4plot.pareto_filter(pool)

    if len(boundary) < 2:
        raise RuntimeError(
            "Too few nondominated points."
        )

    save_candidates(pool)
    save_boundary(boundary)
    plot(
        adaptive=adaptive,
        local=local,
        threshold=threshold,
        weighted=weighted,
        boundary=boundary,
    )

    print("\n=== Stage 10B-v5 final boundary ===")
    print(
        f"threshold-refine candidates: "
        f"{len(threshold)}"
    )
    print(
        f"final nondominated points: "
        f"{len(boundary)}"
    )

    for index, item in enumerate(
        boundary,
        start=1,
    ):
        print(
            f"  {index:02d}: "
            f"R_C={item.sum_rate_comm:.6f}, "
            f"R_S={item.sum_rate_sensing:.6f}, "
            f"source={item.source}, "
            f"parameter={item.source_parameter:.6f}"
        )

    print(
        "\nIf the high-sensing branch ends within "
        "Gamma_C=25.11~25.20 and the low-sensing branch "
        "remains separated after three-start refinement, "
        "freeze the discontinuous transition as a genuine "
        "numerical operating-mode switch of the nonconvex problem."
    )


if __name__ == "__main__":
    main()
