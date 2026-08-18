"""Stage 8：读取 CSV 并生成论文实验图。"""

from pathlib import Path
import csv
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage8 import SCHEME_LABELS, SCHEME_ORDER


RESULT_DIR = ROOT / "results" / "stage8"
FIGURE_DIR = RESULT_DIR / "figures"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def plot_monte_carlo_bar() -> None:
    """绘制六方案平均 WSR 柱状图。"""

    path = RESULT_DIR / "stage8_monte_carlo.csv"
    rows = _read_csv(path)

    means = []
    stds = []

    for key in SCHEME_ORDER:
        values = np.asarray(
            [float(row[f"{key}_wsr"]) for row in rows],
            dtype=float,
        )
        means.append(np.mean(values))
        stds.append(np.std(values, ddof=0))

    labels = [SCHEME_LABELS[key] for key in SCHEME_ORDER]

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8.0, 5.0))
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, labels)
    plt.ylabel("Average WSR")
    plt.xlabel("Scheme")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "monte_carlo_wsr.png", dpi=300)
    plt.savefig(FIGURE_DIR / "monte_carlo_wsr.pdf")
    plt.close()


def plot_sweep(parameter_name: str, x_label: str, log_x: bool = False) -> None:
    """绘制一个参数的 Average WSR sweep。"""

    path = RESULT_DIR / f"stage8_sweep_{parameter_name}.csv"
    rows = _read_csv(path)

    values = sorted({float(row["parameter_value"]) for row in rows})

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.0, 5.0))

    for key in SCHEME_ORDER:
        means = []

        for value in values:
            samples = [
                float(row[f"{key}_wsr"])
                for row in rows
                if np.isclose(float(row["parameter_value"]), value)
            ]
            means.append(np.mean(samples))

        plt.plot(
            values,
            means,
            marker="o",
            label=SCHEME_LABELS[key],
        )

    if log_x:
        plt.xscale("log")

    plt.xlabel(x_label)
    plt.ylabel("Average WSR")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"wsr_vs_{parameter_name}.png", dpi=300)
    plt.savefig(FIGURE_DIR / f"wsr_vs_{parameter_name}.pdf")
    plt.close()


def main() -> None:
    monte_carlo_csv = RESULT_DIR / "stage8_monte_carlo.csv"

    if monte_carlo_csv.exists():
        plot_monte_carlo_bar()
        print("Generated Monte Carlo bar chart.")

    sweep_specs = (
        ("rho_si", r"Residual SI factor $\rho_{SI}$", True),
        ("p_dl_max", r"$P_{DL}^{max}$", False),
        ("cable_length", "Cable length (m)", False),
    )

    for parameter_name, x_label, log_x in sweep_specs:
        path = RESULT_DIR / f"stage8_sweep_{parameter_name}.csv"

        if path.exists():
            plot_sweep(
                parameter_name,
                x_label=x_label,
                log_x=log_x,
            )
            print(f"Generated sweep plot: {parameter_name}")

    print(f"Figures saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
