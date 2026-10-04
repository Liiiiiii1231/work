"""绘制 Proposed 非均匀权重定位结果。"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = (
    ROOT
    / "results"
    / "stage10_probe_proposed"
    / "stage10_probe_proposed.csv"
)

FIG_DIR = (
    ROOT
    / "results"
    / "stage10_probe_proposed"
    / "figures"
)

PNG_PATH = (
    FIG_DIR
    / "communication_sensing_probe_proposed.png"
)

PDF_PATH = (
    FIG_DIR
    / "communication_sensing_probe_proposed.pdf"
)


def main() -> None:

    FIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with CSV_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        rows = list(csv.DictReader(file))

    rows = sorted(
        rows,
        key=lambda row: float(row["omega_comm"]),
    )

    omega = np.asarray(
        [float(row["omega_comm"]) for row in rows]
    )

    rc = np.asarray(
        [float(row["sum_rate_comm"]) for row in rows]
    )

    rs = np.asarray(
        [float(row["sum_rate_sensing"]) for row in rows]
    )

    fig, ax = plt.subplots(
        figsize=(10.0, 7.0)
    )

    ax.plot(
        rc,
        rs,
        marker="o",
        linewidth=2.0,
        label="FP-aware RCG-ToMA + FP (Proposed)",
    )

    # --------------------------------------------------------
    # 只标几个关键权重点，避免文字重叠
    # --------------------------------------------------------
    targets = (
        0.0,
        1.0e-9,
        1.0e-8,
        3.0e-8,
        1.0e-7,
        1.0e-6,
        1.0,
    )

    for target in targets:

        idx = int(
            np.argmin(
                np.abs(omega - target)
            )
        )

        if not np.isclose(
            omega[idx],
            target,
            rtol=1.0e-6,
            atol=1.0e-15,
        ):
            continue

        if target == 0.0:
            label = r"$\omega_C=0$"

        elif target == 1.0:
            label = r"$\omega_C=1$"

        else:
            exponent = int(
                np.floor(
                    np.log10(target)
                )
            )

            coefficient = (
                target / (10.0 ** exponent)
            )

            if np.isclose(coefficient, 1.0):
                label = (
                    rf"$\omega_C=10^{{{exponent}}}$"
                )
            else:
                label = (
                    rf"$\omega_C="
                    rf"{coefficient:g}"
                    rf"\times10^{{{exponent}}}$"
                )

        ax.annotate(
            label,
            (rc[idx], rs[idx]),
            xytext=(7, 7),
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

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        PNG_PATH,
        dpi=220,
        bbox_inches="tight",
    )

    fig.savefig(
        PDF_PATH,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"PNG saved to: {PNG_PATH}")
    print(f"PDF saved to: {PDF_PATH}")


if __name__ == "__main__":
    main()