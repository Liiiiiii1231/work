"""Stage 10B-v3-adaptive：合并 epsilon-constraint 与 weighted-sum 候选，提取最终非支配边界。

用途
----
不重新跑优化，只处理已有结果：

1. 读取 Stage 10B-v2 的 lexicographic epsilon-constraint 候选；
2. 读取 Stage 10A 的 FPA-UPA + FP weighted-sum 候选；
3. 默认只使用已经收敛的候选；
4. 合并两个候选池；
5. 用带数值容差的 Pareto dominance 检查删除 dominated points；
6. 输出“已知可行候选集上的 nondominated achievable boundary”。

注意
----
这里得到的是“当前候选池上的非支配可达边界”，不是全局 Pareto frontier 的证明。
epsilon-constraint 用来补 weighted-sum 漏掉的中间非支撑点；
weighted-sum 用来补充 supported/extreme points，尤其是通信端点。

运行
----
python experiments/plot_stage10b_final_boundary.py

如果想把未收敛点也放进诊断候选池：
python experiments/plot_stage10b_final_boundary.py --include-unconverged

输出
----
results/stage10b/
├─ stage10b_final_candidates.csv
├─ stage10b_final_boundary.csv
├─ stage10b_final_boundary.png
└─ stage10b_final_boundary.pdf
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_EPSILON_CSV = (
    ROOT
    / "results"
    / "stage10b"
    / "epsilon_lexicographic_upa_adaptive.csv"
)

DEFAULT_WEIGHTED_CSV = (
    ROOT
    / "results"
    / "stage10a"
    / "stage10a_tradeoff.csv"
)

RESULT_DIR = ROOT / "results" / "stage10b"

CANDIDATE_CSV = RESULT_DIR / "stage10b_final_candidates_adaptive.csv"
BOUNDARY_CSV = RESULT_DIR / "stage10b_final_boundary_adaptive.csv"
FIG_PNG = RESULT_DIR / "stage10b_final_boundary_adaptive.png"
FIG_PDF = RESULT_DIR / "stage10b_final_boundary_adaptive.pdf"


@dataclass
class Candidate:
    source: str
    source_parameter: float
    sum_rate_comm: float
    sum_rate_sensing: float
    converged: bool
    detail: str = ""
    is_pareto: bool = False


def _to_bool(value: str | bool | int | float) -> bool:
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def load_epsilon_candidates(
    path: Path,
) -> list[Candidate]:
    """读取 lexicographic epsilon-constraint 候选。"""

    if not path.exists():
        raise FileNotFoundError(
            f"Stage 10B epsilon CSV not found: {path}"
        )

    candidates: list[Candidate] = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        required = {
            "gamma_comm",
            "sum_rate_comm",
            "sum_rate_sensing",
            "stage1_converged",
            "stage2_converged",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(
                "epsilon CSV is missing columns: "
                + ", ".join(sorted(missing))
            )

        for row in reader:
            stage1_ok = _to_bool(row["stage1_converged"])
            stage2_ok = _to_bool(row["stage2_converged"])

            detail = (
                f"stage1={stage1_ok}, "
                f"stage2={stage2_ok}"
            )

            candidates.append(
                Candidate(
                    source="epsilon",
                    source_parameter=float(row["gamma_comm"]),
                    sum_rate_comm=float(row["sum_rate_comm"]),
                    sum_rate_sensing=float(
                        row["sum_rate_sensing"]
                    ),
                    converged=stage1_ok and stage2_ok,
                    detail=detail,
                )
            )

    return candidates


def load_weighted_candidates(
    path: Path,
    scheme: str,
) -> list[Candidate]:
    """读取 Stage 10A 指定 scheme 的 weighted-sum 候选。"""

    if not path.exists():
        raise FileNotFoundError(
            f"Stage 10A weighted-sum CSV not found: {path}"
        )

    candidates: list[Candidate] = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        required = {
            "omega_comm",
            "scheme",
            "sum_rate_comm",
            "sum_rate_sensing",
            "converged",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(
                "Stage 10A CSV is missing columns: "
                + ", ".join(sorted(missing))
            )

        for row in reader:
            if str(row["scheme"]).strip() != scheme:
                continue

            candidates.append(
                Candidate(
                    source="weighted_sum",
                    source_parameter=float(row["omega_comm"]),
                    sum_rate_comm=float(row["sum_rate_comm"]),
                    sum_rate_sensing=float(
                        row["sum_rate_sensing"]
                    ),
                    converged=_to_bool(row["converged"]),
                    detail=f"omega_C={float(row['omega_comm']):.6f}",
                )
            )

    if not candidates:
        raise RuntimeError(
            f"No Stage 10A candidates found for scheme={scheme!r}."
        )

    return candidates


def near_duplicate(
    a: Candidate,
    b: Candidate,
    tol: float,
) -> bool:
    """判断两个 operating points 是否仅有数值级差异。"""

    return (
        abs(a.sum_rate_comm - b.sum_rate_comm) <= tol
        and abs(
            a.sum_rate_sensing - b.sum_rate_sensing
        ) <= tol
    )


def collapse_duplicates(
    candidates: list[Candidate],
    tol: float,
) -> list[Candidate]:
    """合并几乎完全相同的点。

    优先保留：
    1. 已收敛点；
    2. 若收敛状态相同，保留先出现的候选。

    这里只处理真正近重复，不用于人为平滑曲线。
    """

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
            if near_duplicate(candidate, existing, tol):
                duplicate_index = index
                break

        if duplicate_index is None:
            kept.append(candidate)
            continue

        existing = kept[duplicate_index]

        if candidate.converged and not existing.converged:
            kept[duplicate_index] = candidate

    return kept


def dominates(
    a: Candidate,
    b: Candidate,
    tol: float,
) -> bool:
    """判断 a 是否在数值容差意义下支配 b。

    a 至少不比 b 差：
        R_C(a) >= R_C(b) - tol
        R_S(a) >= R_S(b) - tol

    并且至少一个指标有超过 tol 的严格改善。
    """

    not_worse_comm = (
        a.sum_rate_comm
        >= b.sum_rate_comm - tol
    )
    not_worse_sensing = (
        a.sum_rate_sensing
        >= b.sum_rate_sensing - tol
    )

    strictly_better = (
        a.sum_rate_comm
        > b.sum_rate_comm + tol
        or a.sum_rate_sensing
        > b.sum_rate_sensing + tol
    )

    return (
        not_worse_comm
        and not_worse_sensing
        and strictly_better
    )


def pareto_filter(
    candidates: list[Candidate],
    tol: float,
) -> list[Candidate]:
    """从候选池中提取非支配点。"""

    for candidate in candidates:
        candidate.is_pareto = False

    boundary: list[Candidate] = []

    for i, candidate in enumerate(candidates):
        is_dominated = False

        for j, other in enumerate(candidates):
            if i == j:
                continue

            if dominates(other, candidate, tol):
                is_dominated = True
                break

        if not is_dominated:
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
    """保存全部候选及 Pareto 标记。"""

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

        for candidate in sorted(
            candidates,
            key=lambda item: (
                item.sum_rate_comm,
                -item.sum_rate_sensing,
            ),
        ):
            writer.writerow(
                {
                    "source": candidate.source,
                    "source_parameter": (
                        candidate.source_parameter
                    ),
                    "sum_rate_comm": (
                        candidate.sum_rate_comm
                    ),
                    "sum_rate_sensing": (
                        candidate.sum_rate_sensing
                    ),
                    "converged": candidate.converged,
                    "is_pareto": candidate.is_pareto,
                    "detail": candidate.detail,
                }
            )


def save_boundary(
    boundary: list[Candidate],
) -> None:
    """保存最终非支配边界。"""

    fields = [
        "boundary_index",
        "source",
        "source_parameter",
        "sum_rate_comm",
        "sum_rate_sensing",
        "converged",
    ]

    with BOUNDARY_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )
        writer.writeheader()

        for index, candidate in enumerate(
            boundary,
            start=1,
        ):
            writer.writerow(
                {
                    "boundary_index": index,
                    "source": candidate.source,
                    "source_parameter": (
                        candidate.source_parameter
                    ),
                    "sum_rate_comm": (
                        candidate.sum_rate_comm
                    ),
                    "sum_rate_sensing": (
                        candidate.sum_rate_sensing
                    ),
                    "converged": candidate.converged,
                }
            )


def plot_boundary(
    epsilon_candidates: list[Candidate],
    weighted_candidates: list[Candidate],
    boundary: list[Candidate],
) -> None:
    """画候选点与最终 nondominated envelope。"""

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.0, 5.8))

    eps_rc = [
        item.sum_rate_comm
        for item in epsilon_candidates
    ]
    eps_rs = [
        item.sum_rate_sensing
        for item in epsilon_candidates
    ]

    weighted_rc = [
        item.sum_rate_comm
        for item in weighted_candidates
    ]
    weighted_rs = [
        item.sum_rate_sensing
        for item in weighted_candidates
    ]

    boundary_rc = [
        item.sum_rate_comm
        for item in boundary
    ]
    boundary_rs = [
        item.sum_rate_sensing
        for item in boundary
    ]

    ax.scatter(
        eps_rc,
        eps_rs,
        marker="o",
        s=45,
        alpha=0.55,
        label=r"Lexicographic $\epsilon$ candidates",
    )

    ax.scatter(
        weighted_rc,
        weighted_rs,
        marker="x",
        s=55,
        alpha=0.65,
        label="Stage 10A weighted-sum candidates",
    )

    ax.plot(
        boundary_rc,
        boundary_rs,
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
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epsilon-csv",
        type=Path,
        default=DEFAULT_EPSILON_CSV,
        help="Stage 10B-v2 lexicographic epsilon CSV.",
    )

    parser.add_argument(
        "--weighted-csv",
        type=Path,
        default=DEFAULT_WEIGHTED_CSV,
        help="Stage 10A weighted-sum CSV.",
    )

    parser.add_argument(
        "--scheme",
        type=str,
        default="b3",
        help="Stage 10A scheme code. FPA-UPA + FP is b3.",
    )

    parser.add_argument(
        "--pareto-tol",
        type=float,
        default=1.0e-4,
        help=(
            "Numerical dominance tolerance in bit/s/Hz. "
            "Default: 1e-4."
        ),
    )

    parser.add_argument(
        "--duplicate-tol",
        type=float,
        default=1.0e-6,
        help=(
            "Tolerance used only to remove numerically identical "
            "candidate points. Default: 1e-6."
        ),
    )

    parser.add_argument(
        "--include-unconverged",
        action="store_true",
        help=(
            "Include unconverged epsilon/weighted candidates in "
            "the Pareto candidate pool. Default excludes them."
        ),
    )

    args = parser.parse_args()

    if args.pareto_tol < 0.0:
        raise ValueError("--pareto-tol must be nonnegative.")

    if args.duplicate_tol < 0.0:
        raise ValueError(
            "--duplicate-tol must be nonnegative."
        )

    epsilon_all = load_epsilon_candidates(
        args.epsilon_csv
    )
    weighted_all = load_weighted_candidates(
        args.weighted_csv,
        scheme=args.scheme,
    )

    if args.include_unconverged:
        epsilon_pool = list(epsilon_all)
        weighted_pool = list(weighted_all)
    else:
        epsilon_pool = [
            item
            for item in epsilon_all
            if item.converged
        ]
        weighted_pool = [
            item
            for item in weighted_all
            if item.converged
        ]

    candidate_pool = collapse_duplicates(
        epsilon_pool + weighted_pool,
        tol=args.duplicate_tol,
    )

    boundary = pareto_filter(
        candidate_pool,
        tol=args.pareto_tol,
    )

    if len(boundary) < 2:
        raise RuntimeError(
            "Too few nondominated points were found."
        )

    save_candidates(candidate_pool)
    save_boundary(boundary)
    plot_boundary(
        epsilon_candidates=epsilon_all,
        weighted_candidates=weighted_all,
        boundary=boundary,
    )

    n_eps_unconverged = sum(
        not item.converged
        for item in epsilon_all
    )
    n_weighted_unconverged = sum(
        not item.converged
        for item in weighted_all
    )

    print("\n=== Stage 10B-v3 summary ===")
    print(
        f"epsilon candidates: "
        f"{len(epsilon_all)} "
        f"({n_eps_unconverged} unconverged)"
    )
    print(
        f"weighted candidates: "
        f"{len(weighted_all)} "
        f"({n_weighted_unconverged} unconverged)"
    )
    print(
        f"candidate pool used for Pareto filter: "
        f"{len(candidate_pool)}"
    )
    print(
        f"final nondominated points: "
        f"{len(boundary)}"
    )

    print("\nFinal boundary:")
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

    if n_eps_unconverged > 0:
        print(
            "\nNote: unconverged epsilon candidates are excluded "
            "from the final boundary by default."
        )
        print(
            "This avoids treating a max-iteration local iterate as "
            "a formal boundary point."
        )

    print(
        "\nInterpretation: this curve is the nondominated achievable "
        "boundary of the currently known feasible candidate set."
    )
    print(
        "Do not call it a globally optimal Pareto frontier unless "
        "global optimality is separately established."
    )


if __name__ == "__main__":
    main()
