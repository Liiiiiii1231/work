"""绘制 Stage 10 最终非均匀权重通信-感知 trade-off。"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from experiments.run_stage10_tradeoff import (
    SCHEME_LABELS,
    SCHEME_ORDER,
)


CSV_PATH = (
    ROOT
    / "results"
    / "stage10_final_weights"
    / "stage10_final_weights.csv"
)

FIG_DIR = (
    ROOT
    / "results"
    / "stage10_final_weights"
    / "figures"
)

PNG_PATH = (
    FIG_DIR
    / "communication_sensing_tradeoff_final_weights.png"
)

PDF_PATH = (
    FIG_DIR
    / "communication_sensing_tradeoff_final_weights.pdf"
)


def load_rows() -> list[dict[str, str]]:
    """读取 Stage 10 最终结果。"""

    with CSV_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return list(
            csv.DictReader(file)
        )


def format_weight(
    value: float,
) -> str:
    """将 omega_C 格式化为论文图标注。"""

    if np.isclose(
        value,
        0.0,
        atol=1.0e-20,
    ):
        return r"$\omega_C=0$"

    exponent = int(
        np.floor(
            np.log10(value)
        )
    )

    coefficient = (
        value
        / (10.0 ** exponent)
    )

    if np.isclose(
        coefficient,
        1.0,
    ):
        return (
            rf"$\omega_C=10^{{{exponent}}}$"
        )

    return (
        rf"$\omega_C="
        rf"{coefficient:g}"
        rf"\times10^{{{exponent}}}$"
    )


def main() -> None:
    rows = load_rows()

    FIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(10.5, 7.0)
    )

    for scheme in SCHEME_ORDER:

        scheme_rows = sorted(
            [
                row
                for row in rows
                if row["scheme"] == scheme
            ],
            key=lambda row: float(
                row["omega_comm"]
            ),
        )

        if not scheme_rows:
            continue

        rc = np.asarray(
            [
                float(
                    row["sum_rate_comm"]
                )
                for row in scheme_rows
            ],
            dtype=float,
        )

        rs = np.asarray(
            [
                float(
                    row["sum_rate_sensing"]
                )
                for row in scheme_rows
            ],
            dtype=float,
        )

        omega = np.asarray(
            [
                float(
                    row["omega_comm"]
                )
                for row in scheme_rows
            ],
            dtype=float,
        )

        # Proposed 稍微加粗，其余方案正常显示。
        if scheme == "proposed":
            linewidth = 2.4
            markersize = 6.0
            zorder = 4
        else:
            linewidth = 1.9
            markersize = 5.0
            zorder = 2

        ax.plot(
            rc,
            rs,
            marker="o",
            markersize=markersize,
            linewidth=linewidth,
            label=SCHEME_LABELS[scheme],
            zorder=zorder,
        )

        # ----------------------------------------------------
        # 仅给 Proposed 标三个代表点。
        # 不再标所有权重，避免右下角文字堆叠。
        # ----------------------------------------------------
        if scheme == "proposed":

            label_settings = {
                0.0: (8, 8),
                1.0e-8: (8, 8),
                1.0e-4: (8, 8),
            }

            for target, offset in label_settings.items():

                idx = int(
                    np.argmin(
                        np.abs(
                            omega - target
                        )
                    )
                )

                if not np.isclose(
                    omega[idx],
                    target,
                    rtol=1.0e-6,
                    atol=1.0e-15,
                ):
                    continue

                ax.annotate(
                    format_weight(
                        omega[idx]
                    ),
                    (
                        rc[idx],
                        rs[idx],
                    ),
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=9,
                )

    ax.set_xlabel(
        "Sum Communication Rate (bit/s/Hz)"
    )

    ax.set_ylabel(
        "Sum Sensing Rate (bit/s/Hz)"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend(
        loc="best",
        frameon=True,
    )

    fig.tight_layout()

    fig.savefig(
        PNG_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        PDF_PATH,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"PNG saved to: {PNG_PATH}"
    )

    print(
        f"PDF saved to: {PDF_PATH}"
    )


if __name__ == "__main__":
    main()