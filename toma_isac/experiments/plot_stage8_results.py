"""Stage 8：读取 CSV 并生成论文实验图。"""

from pathlib import Path
import csv
import sys
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage8 import SCHEME_ORDER


RESULT_DIR = ROOT / "results" / "stage8"
FIGURE_DIR = RESULT_DIR / "figures"

# 柱状图横轴使用换行，避免名称过长。
BAR_SCHEME_LABELS = {
    "b1": "RCG-ToMA\n+ Traditional",
    "b2": "Fixed-ToMA\n+ FP",
    "b3": "FPA-UPA\n+ FP",
    "b4": "Fixed-ToMA\n+ Traditional",
    "b5": "FPA-UPA\n+ Traditional",
    "b6": "MA\n+ FP",
    "proposed": "RCG-ToMA\n+ FP",
}

# 参数扫描和 system-load 图使用完整图例名称。
LEGEND_SCHEME_LABELS = {
    "b1": "RCG-ToMA + Traditional",
    "b2": "Fixed-ToMA + FP",
    "b3": "FPA-UPA + FP",
    "b4": "Fixed-ToMA + Traditional",
    "b5": "FPA-UPA + Traditional",
    "b6": "MA + FP",
    "proposed": "RCG-ToMA + FP (Proposed)",
}

WSR_LABEL = "Average WSR (bit/s/Hz)"


def _read_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def plot_monte_carlo_bar() -> None:
    """绘制七方案平均 WSR 柱状图，并标出平均 WSR 数值。"""

    path = RESULT_DIR / "stage8_monte_carlo.csv"
    rows = _read_csv(path)

    means = []
    stds = []

    for key in SCHEME_ORDER:
        values = np.asarray(
            [float(row[f"{key}_wsr"]) for row in rows],
            dtype=float,
        )

        means.append(float(np.mean(values)))
        stds.append(float(np.std(values, ddof=0)))

    labels = [BAR_SCHEME_LABELS[key] for key in SCHEME_ORDER]

    means_array = np.asarray(means, dtype=float)
    stds_array = np.asarray(stds, dtype=float)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11.0, 5.5))
    x = np.arange(len(labels))

    bars = plt.bar(
        x,
        means_array,
        yerr=stds_array,
        capsize=4,
    )

    # 在误差棒上方标出各方案的平均 WSR，保留两位小数。
    error_top = means_array + stds_array
    max_top = float(np.max(error_top))
    text_offset = max(0.08, 0.015 * max_top)

    for bar, mean, top in zip(
        bars,
        means_array,
        error_top,
    ):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            top + text_offset,
            f"{mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 给数值标签预留顶部空间，避免被裁剪。
    plt.ylim(
        0.0,
        max_top + 4.0 * text_offset,
    )

    plt.xticks(x, labels, fontsize=9)
    plt.xlabel("Scheme")
    plt.ylabel(WSR_LABEL)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "monte_carlo_wsr.png",
        dpi=300,
    )
    plt.savefig(
        FIGURE_DIR / "monte_carlo_wsr.pdf"
    )
    plt.close()


def plot_sweep(
    parameter_name: str,
    x_label: str,
    x_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    x_ticks: np.ndarray | None = None,
) -> None:
    """绘制一个参数的 Average WSR sweep。"""

    path = RESULT_DIR / f"stage8_sweep_{parameter_name}.csv"
    rows = _read_csv(path)

    values = np.asarray(
        sorted(
            {
                float(row["parameter_value"])
                for row in rows
            }
        ),
        dtype=float,
    )

    if x_transform is None:
        plot_values = values
    else:
        plot_values = np.asarray(
            x_transform(values),
            dtype=float,
        )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10.0, 5.5))

    for key in SCHEME_ORDER:
        means = []

        for value in values:
            samples = [
                float(row[f"{key}_wsr"])
                for row in rows
                if np.isclose(
                    float(row["parameter_value"]),
                    value,
                )
            ]
            means.append(np.mean(samples))

        plt.plot(
            plot_values,
            means,
            marker="o",
            label=LEGEND_SCHEME_LABELS[key],
        )

    if x_ticks is not None:
        plt.xticks(x_ticks)

    plt.xlabel(x_label)
    plt.ylabel(WSR_LABEL)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / f"wsr_vs_{parameter_name}.png",
        dpi=300,
    )
    plt.savefig(
        FIGURE_DIR / f"wsr_vs_{parameter_name}.pdf"
    )
    plt.close()


def plot_system_load() -> None:
    """绘制 Average WSR vs 用户/目标数量。"""

    path = RESULT_DIR / "stage8_sweep_system_load.csv"
    rows = _read_csv(path)

    loads = sorted(
        {
            int(row["system_load"])
            for row in rows
        }
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10.0, 5.5))

    for key in SCHEME_ORDER:
        means = []

        for load in loads:
            samples = [
                float(row[f"{key}_wsr"])
                for row in rows
                if int(row["system_load"]) == load
            ]
            means.append(np.mean(samples))

        plt.plot(
            loads,
            means,
            marker="o",
            label=LEGEND_SCHEME_LABELS[key],
        )

    plt.xticks(loads)
    plt.xlabel(r"Number of users/targets ($K=J=L$)")
    plt.ylabel(WSR_LABEL)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "wsr_vs_system_load.png",
        dpi=300,
    )
    plt.savefig(
        FIGURE_DIR / "wsr_vs_system_load.pdf"
    )
    plt.close()


def _rho_to_db(values: np.ndarray) -> np.ndarray:
    """将残余 SI 功率比例 rho_SI 转换为 dB。"""

    if np.any(values <= 0.0):
        raise ValueError(
            "rho_si must be positive when converting to dB."
        )

    return 10.0 * np.log10(values)


def main() -> None:
    monte_carlo_csv = RESULT_DIR / "stage8_monte_carlo.csv"

    if monte_carlo_csv.exists():
        plot_monte_carlo_bar()
        print("Generated Monte Carlo bar chart.")

    sweep_specs = (
        (
            "rho_si",
            r"Residual SI level $10\log_{10}(\rho_{\mathrm{SI}})$ (dB)",
            _rho_to_db,
            np.asarray([-30.0, -20.0, -10.0, 0.0]),
        ),
        (
            "p_dl_max",
            r"Maximum DL transmit power $P_{\mathrm{DL}}^{\max}$ (W)",
            None,
            None,
        ),
    )

    for (
        parameter_name,
        x_label,
        x_transform,
        x_ticks,
    ) in sweep_specs:
        path = RESULT_DIR / f"stage8_sweep_{parameter_name}.csv"

        if path.exists():
            plot_sweep(
                parameter_name,
                x_label=x_label,
                x_transform=x_transform,
                x_ticks=x_ticks,
            )
            print(
                "Generated sweep plot: "
                f"{parameter_name}"
            )

    system_load_csv = RESULT_DIR / "stage8_sweep_system_load.csv"

    if system_load_csv.exists():
        plot_system_load()
        print("Generated sweep plot: system_load")

    print(f"Figures saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
