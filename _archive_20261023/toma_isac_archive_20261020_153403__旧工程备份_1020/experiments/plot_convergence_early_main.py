"""绘制 Stage 8 收敛对比：主图突出前期，inset 显示完整收敛。"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = (
    ROOT
    / "results"
    / "stage8"
    / "convergence_comparison_scene_8001.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "stage8"
    / "figures"
)

PNG_PATH = (
    OUTPUT_DIR
    / "convergence_comparison_early_main_scene_8001.png"
)

PDF_PATH = (
    OUTPUT_DIR
    / "convergence_comparison_early_main_scene_8001.pdf"
)


# 主图只展示前 25 次迭代
MAIN_MAX_ITER = 25


SCHEME_ORDER = (
    "RCG-ToMA + Traditional",
    "Fixed-ToMA + FP",
    "FPA-UPA + FP",
    "MA + FP",
    "FP-aware RCG-ToMA + FP (Proposed)",
)


# 与其它 Stage 8 图保持一致
COLORS = {
    "RCG-ToMA + Traditional": "tab:blue",
    "Fixed-ToMA + FP": "tab:orange",
    "FPA-UPA + FP": "tab:green",
    "Fixed-ToMA + Traditional": "tab:red",
    "FPA-UPA + Traditional": "tab:purple",
    "MA + FP": "tab:brown",
    "FP-aware RCG-ToMA + FP (Proposed)": "tab:pink",
}


REFERENCE_ORDER = (
    "Fixed-ToMA + Traditional",
    "FPA-UPA + Traditional",
)


def load_results():
    """读取已有收敛 CSV，不重新运行任何优化算法。"""

    histories: dict[str, list[tuple[int, float]]] = {}
    references: dict[str, float] = {}

    with CSV_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            scheme = row["scheme"]
            curve_type = row["curve_type"]

            if curve_type == "iterative":
                histories.setdefault(
                    scheme,
                    [],
                ).append(
                    (
                        int(row["iteration"]),
                        float(row["wsr"]),
                    )
                )

            elif curve_type == "horizontal_reference":
                references[scheme] = float(
                    row["wsr"]
                )

    # 保证 iteration 顺序正确
    for scheme in histories:
        histories[scheme].sort(
            key=lambda item: item[0]
        )

    return histories, references


def main() -> None:
    histories, references = load_results()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(9.5, 5.8)
    )

    # ============================================================
    # 主图：只画前 MAIN_MAX_ITER 次
    # ============================================================

    for scheme in SCHEME_ORDER:
        if scheme not in histories:
            continue

        data = histories[scheme]

        iterations = np.asarray(
            [item[0] for item in data],
            dtype=int,
        )

        wsr = np.asarray(
            [item[1] for item in data],
            dtype=float,
        )

        mask = iterations <= MAIN_MAX_ITER

        ax.plot(
            iterations[mask],
            wsr[mask],
            color=COLORS[scheme],
            marker="o",
            linewidth=(
                2.5
                if "Proposed" in scheme
                else 1.8
            ),
            markersize=(
                5.5
                if "Proposed" in scheme
                else 4.5
            ),
            label=scheme,
            zorder=(
                6
                if "Proposed" in scheme
                else 3
            ),
        )

    # B4/B5 水平参考线
    for scheme in REFERENCE_ORDER:
        if scheme not in references:
            continue

        ax.hlines(
            references[scheme],
            xmin=0,
            xmax=MAIN_MAX_ITER,
            color=COLORS[scheme],
            linestyle="--",
            linewidth=1.6,
            label=f"{scheme} (final WSR)",
            zorder=2,
        )

    ax.set_xlim(
        0,
        MAIN_MAX_ITER,
    )

    ax.set_xlabel(
        "Iteration index"
    )

    ax.set_ylabel(
        "WSR (bit/s/Hz)"
    )

    ax.grid(
        alpha=0.30
    )

    # ============================================================
    # Inset：完整收敛过程
    # ============================================================

    axins = ax.inset_axes(
        [0.53, 0.16, 0.43, 0.38]
    )

    full_max_iter = 0

    for scheme in SCHEME_ORDER:
        if scheme not in histories:
            continue

        data = histories[scheme]

        iterations = np.asarray(
            [item[0] for item in data],
            dtype=int,
        )

        wsr = np.asarray(
            [item[1] for item in data],
            dtype=float,
        )

        full_max_iter = max(
            full_max_iter,
            int(iterations[-1]),
        )

        axins.plot(
            iterations,
            wsr,
            color=COLORS[scheme],
            linewidth=(
                2.0
                if "Proposed" in scheme
                else 1.2
            ),
        )

    # inset 中也保留 B4/B5
    for scheme in REFERENCE_ORDER:
        if scheme not in references:
            continue

        axins.hlines(
            references[scheme],
            xmin=0,
            xmax=full_max_iter,
            color=COLORS[scheme],
            linestyle="--",
            linewidth=1.0,
        )

    axins.set_xlim(
        0,
        full_max_iter,
    )

    axins.set_title(
        "Full convergence",
        fontsize=9,
    )

    axins.tick_params(
        axis="both",
        labelsize=8,
    )

    axins.grid(
        alpha=0.25
    )

    # ============================================================
    # 图例与布局
    # ============================================================

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        fontsize=8.2,
        frameon=True,
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.12,
        top=0.79,
    )

    fig.savefig(
        PNG_PATH,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.12,
    )

    fig.savefig(
        PDF_PATH,
        bbox_inches="tight",
        pad_inches=0.12,
    )

    plt.close(fig)

    print(f"Saved PNG: {PNG_PATH}")
    print(f"Saved PDF: {PDF_PATH}")


if __name__ == "__main__":
    main()