"""Stage 10B-v4：合并 adaptive epsilon、局部精查、weighted-sum 候选并提取最终边界。

运行：
    python experiments/plot_stage10b_final_boundary_v4.py

输入：
    results/stage10b/epsilon_lexicographic_upa_adaptive.csv
    results/stage10b/stage10b_local_refine_v4.csv
    results/stage10a/stage10a_tradeoff.csv

输出：
    results/stage10b/stage10b_final_candidates_v4.csv
    results/stage10b/stage10b_final_boundary_v4.csv
    results/stage10b/stage10b_final_boundary_v4.png
    results/stage10b/stage10b_final_boundary_v4.pdf

注意：
这里得到的是“当前已知可行候选集上的 nondominated achievable boundary”，
不是全局 Pareto frontier 的证明。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

RESULT_DIR = ROOT / "results" / "stage10b"

ADAPTIVE_CSV = (
    RESULT_DIR
    / "epsilon_lexicographic_upa_adaptive.csv"
)
LOCAL_CSV = RESULT_DIR / "stage10b_local_refine_v4.csv"
WEIGHTED_CSV = (
    ROOT
    / "results"
    / "stage10a"
    / "stage10a_tradeoff.csv"
)

CANDIDATE_CSV = (
    RESULT_DIR / "stage10b_final_candidates_v4.csv"
)
BOUNDARY_CSV = (
    RESULT_DIR / "stage10b_final_boundary_v4.csv"
)
FIG_PNG = RESULT_DIR / "stage10b_final_boundary_v4.png"
FIG_PDF = RESULT_DIR / "stage10b_final_boundary_v4.pdf"

PARETO_TOL = 1.0e-4
DUPLICATE_TOL = 1.0e-6


@dataclass
class Candidate:
    source: str
    source_parameter: float
    sum_rate_comm: float
    sum_rate_sensing: float
    converged: bool
    detail: str = ""
    is_pareto: bool = False


def to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {
        "1", "true", "yes", "y", "t"
    }


def load_adaptive() -> list[Candidate]:
    if not ADAPTIVE_CSV.exists():
        raise FileNotFoundError(ADAPTIVE_CSV)

    out: list[Candidate] = []

    with ADAPTIVE_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        for row in csv.DictReader(file):
            converged = (
                to_bool(row["stage1_converged"])
                and to_bool(row["stage2_converged"])
            )
            out.append(
                Candidate(
                    source="epsilon_adaptive",
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
                )
            )

    return out


def load_local() -> list[Candidate]:
    if not LOCAL_CSV.exists():
        raise FileNotFoundError(LOCAL_CSV)

    out: list[Candidate] = []

    with LOCAL_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        for row in csv.DictReader(file):
            converged = (
                to_bool(row["stage1_converged"])
                and to_bool(row["stage2_converged"])
            )
            out.append(
                Candidate(
                    source="epsilon_local_v4",
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
                        f"group={row['group']}, "
                        f"start={row['selected_step1_source']}"
                    ),
                )
            )

    return out


def load_weighted() -> list[Candidate]:
    if not WEIGHTED_CSV.exists():
        raise FileNotFoundError(WEIGHTED_CSV)

    out: list[Candidate] = []

    with WEIGHTED_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        for row in csv.DictReader(file):
            if str(row["scheme"]).strip() != "b3":
                continue

            out.append(
                Candidate(
                    source="weighted_sum",
                    source_parameter=float(
                        row["omega_comm"]
                    ),
                    sum_rate_comm=float(
                        row["sum_rate_comm"]
                    ),
                    sum_rate_sensing=float(
                        row["sum_rate_sensing"]
                    ),
                    converged=to_bool(
                        row["converged"]
                    ),
                )
            )

    if not out:
        raise RuntimeError(
            "No Stage 10A B3 candidates were found."
        )

    return out


def near_duplicate(
    a: Candidate,
    b: Candidate,
) -> bool:
    return (
        abs(
            a.sum_rate_comm - b.sum_rate_comm
        ) <= DUPLICATE_TOL
        and abs(
            a.sum_rate_sensing - b.sum_rate_sensing
        ) <= DUPLICATE_TOL
    )


def collapse_duplicates(
    candidates: list[Candidate],
) -> list[Candidate]:
    kept: list[Candidate] = []

    for candidate in sorted(
        candidates,
        key=lambda item: (
            item.sum_rate_comm,
            -item.sum_rate_sensing,
        ),
    ):
        duplicate_index = None

        for index, existing in enumerate(kept):
            if near_duplicate(
                candidate,
                existing,
            ):
                duplicate_index = index
                break

        if duplicate_index is None:
            kept.append(candidate)
            continue

        existing = kept[duplicate_index]
        if (
            candidate.converged
            and not existing.converged
        ):
            kept[duplicate_index] = candidate

    return kept


def dominates(
    a: Candidate,
    b: Candidate,
) -> bool:
    not_worse = (
        a.sum_rate_comm
        >= b.sum_rate_comm - PARETO_TOL
        and a.sum_rate_sensing
        >= b.sum_rate_sensing - PARETO_TOL
    )

    strictly_better = (
        a.sum_rate_comm
        > b.sum_rate_comm + PARETO_TOL
        or a.sum_rate_sensing
        > b.sum_rate_sensing + PARETO_TOL
    )

    return not_worse and strictly_better


def pareto_filter(
    candidates: list[Candidate],
) -> list[Candidate]:
    for candidate in candidates:
        candidate.is_pareto = False

    boundary: list[Candidate] = []

    for index, candidate in enumerate(candidates):
        if any(
            other_index != index
            and dominates(other, candidate)
            for other_index, other
            in enumerate(candidates)
        ):
            continue

        candidate.is_pareto = True
        boundary.append(candidate)

    boundary.sort(
        key=lambda item: (
            item.sum_rate_comm,
            -item.sum_rate_sensing,
        )
    )
    return boundary


def save_candidates(
    candidates: list[Candidate],
) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

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
    boundary: list[Candidate],
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
    adaptive: list[Candidate],
    local: list[Candidate],
    weighted: list[Candidate],
    boundary: list[Candidate],
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.8))

    ax.scatter(
        [x.sum_rate_comm for x in adaptive],
        [x.sum_rate_sensing for x in adaptive],
        marker="o",
        s=35,
        alpha=0.28,
        label="Adaptive epsilon candidates",
    )

    ax.scatter(
        [x.sum_rate_comm for x in local],
        [x.sum_rate_sensing for x in local],
        marker="s",
        s=45,
        alpha=0.65,
        label="Local three-start refine candidates",
    )

    ax.scatter(
        [x.sum_rate_comm for x in weighted],
        [x.sum_rate_sensing for x in weighted],
        marker="x",
        s=55,
        alpha=0.65,
        label="Stage 10A weighted-sum candidates",
    )

    ax.plot(
        [x.sum_rate_comm for x in boundary],
        [x.sum_rate_sensing for x in boundary],
        marker="D",
        linewidth=2.4,
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
    adaptive = load_adaptive()
    local = load_local()
    weighted = load_weighted()

    # 正式 boundary 默认只允许已收敛候选参与。
    pool = [
        item
        for item in (
            adaptive + local + weighted
        )
        if item.converged
    ]

    pool = collapse_duplicates(pool)
    boundary = pareto_filter(pool)

    if len(boundary) < 2:
        raise RuntimeError(
            "Too few nondominated points."
        )

    save_candidates(pool)
    save_boundary(boundary)
    plot(
        adaptive=adaptive,
        local=local,
        weighted=weighted,
        boundary=boundary,
    )

    print("\n=== Stage 10B-v4 final boundary ===")
    print(
        f"adaptive candidates: {len(adaptive)}"
    )
    print(
        f"local-refine candidates: {len(local)}"
    )
    print(
        f"weighted-sum candidates: {len(weighted)}"
    )
    print(
        f"final nondominated points: {len(boundary)}"
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
        "\nIf the two original cliffs remain after this local "
        "three-start refinement, keep them as genuine "
        "nonconvex operating-mode transitions rather than "
        "artificially smoothing the curve."
    )


if __name__ == "__main__":
    main()
