"""Stage 8A 最终版：窗口平台停止敏感性比较。

核心思想：
- 先得到“原始停止规则”的完整联合 WSR 轨迹；
- 对同一条轨迹离线扫描多个 tol_window；
- 因为窗口停止只改变“何时退出”，不会改变退出前的更新轨迹，
  所以不需要为每个阈值重新跑一遍联合算法。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm import run_joint_algorithm
from config import load_config
from stage8 import load_stage8_experiment_config, make_scene_config


CONFIG_PATH = ROOT / "configs" / "stage8_monte_carlo.yaml"
RESULT_DIR = ROOT / "results" / "stage8" / "diagnostics"
FIGURE_DIR = RESULT_DIR / "figures"

DEFAULT_THRESHOLDS = (
    3.0e-3,
    2.5e-3,
    2.0e-3,
    1.5e-3,
    1.4e-3,
    1.3e-3,
    1.2e-3,
    1.0e-3,
    5.0e-4,
    3.0e-4,
)


def _parse_thresholds(text: str | None) -> tuple[float, ...]:
    if text is None:
        return DEFAULT_THRESHOLDS

    values = tuple(
        float(item.strip())
        for item in text.split(",")
        if item.strip()
    )

    if not values or any(value <= 0.0 for value in values):
        raise ValueError("All window thresholds must be positive.")

    return values


def _load_history_csv(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows or "wsr" not in rows[0]:
        raise ValueError("History CSV must contain a 'wsr' column.")

    history = np.asarray([float(row["wsr"]) for row in rows], dtype=float)

    if history.ndim != 1 or history.size < 2 or not np.all(np.isfinite(history)):
        raise ValueError("Invalid WSR history.")

    return history


def _save_history_csv(history: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["outer_iteration", "wsr", "relative_change"])

        for iteration, wsr in enumerate(history):
            if iteration == 0:
                change = np.nan
            else:
                previous = history[iteration - 1]
                change = abs(wsr - previous) / max(1.0, abs(previous))

            writer.writerow([iteration, float(wsr), float(change)])


def _run_reference(
    scene_seed: int,
    algorithm_seed: int,
) -> np.ndarray:
    cfg = load_config(CONFIG_PATH)
    experiment_cfg = load_stage8_experiment_config(CONFIG_PATH)

    # 强制关闭窗口停止，得到原始 outer-tolerance 基准轨迹。
    reference_cfg = replace(
        cfg,
        plateau_window=0,
        tol_window=0.0,
    )

    scene_cfg = make_scene_config(
        reference_cfg,
        scene_seed=scene_seed,
        position_jitter=experiment_cfg.position_jitter,
    )

    result = run_joint_algorithm(
        scene_cfg,
        rng=np.random.default_rng(algorithm_seed),
        verbose=False,
    )

    if not result.converged:
        raise RuntimeError("Reference run did not converge.")

    return np.asarray(result.joint_wsr_history, dtype=float)


def evaluate_window_thresholds(
    history: np.ndarray,
    window: int,
    thresholds: tuple[float, ...],
) -> list[dict[str, object]]:
    """在完整基准轨迹上离线评估窗口停止规则。"""

    if window < 2:
        raise ValueError("window must be at least 2.")
    if history.size <= window:
        raise ValueError("history is shorter than the requested window.")

    reference_iteration = int(history.size - 1)
    reference_wsr = float(history[-1])

    rows: list[dict[str, object]] = [
        {
            "tol_window": 0.0,
            "window": window,
            "stop_iteration": reference_iteration,
            "stop_reason": "outer_tolerance_reference",
            "window_gain_at_stop": float("nan"),
            "final_wsr": reference_wsr,
            "wsr_loss_pct": 0.0,
            "saved_iterations": 0,
            "saved_iterations_pct": 0.0,
        }
    ]

    for threshold in thresholds:
        stop_iteration: int | None = None
        stop_gain = float("nan")

        for iteration in range(window, history.size):
            reference = float(history[iteration - window])
            gain = (
                float(history[iteration]) - reference
            ) / max(1.0, abs(reference))
            gain = max(0.0, float(gain))

            if gain <= threshold:
                stop_iteration = iteration
                stop_gain = gain
                break

        if stop_iteration is None:
            stop_iteration = reference_iteration
            stop_reason = "outer_tolerance_reference"
        else:
            stop_reason = "window_plateau"

        candidate_wsr = float(history[stop_iteration])
        loss_pct = max(
            0.0,
            (reference_wsr - candidate_wsr)
            / max(abs(reference_wsr), 1.0e-15)
            * 100.0,
        )

        saved_iterations = reference_iteration - stop_iteration
        saved_pct = (
            saved_iterations / max(reference_iteration, 1) * 100.0
        )

        rows.append(
            {
                "tol_window": float(threshold),
                "window": window,
                "stop_iteration": int(stop_iteration),
                "stop_reason": stop_reason,
                "window_gain_at_stop": stop_gain,
                "final_wsr": candidate_wsr,
                "wsr_loss_pct": float(loss_pct),
                "saved_iterations": int(saved_iterations),
                "saved_iterations_pct": float(saved_pct),
            }
        )

    return rows


def _save_rows(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_comparison(
    history: np.ndarray,
    rows: list[dict[str, object]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    iterations = np.arange(history.size)
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    ax.plot(iterations, history, linewidth=1.8, label="Reference WSR")

    for row in rows[1:]:
        if row["stop_reason"] != "window_plateau":
            continue

        stop_iteration = int(row["stop_iteration"])
        ax.scatter(
            [stop_iteration],
            [history[stop_iteration]],
            s=28,
            label=(
                f"tol={float(row['tol_window']):.1e}, "
                f"iter={stop_iteration}"
            ),
        )

    ax.set_xlabel("Joint AO iteration")
    ax.set_ylabel("WSR")
    ax.set_title("Stage 8A window-plateau stopping sensitivity")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-seed", type=int, default=8001)
    parser.add_argument("--algorithm-seed", type=int, default=2029)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help="Comma-separated tol_window values.",
    )
    parser.add_argument(
        "--history-csv",
        type=Path,
        default=None,
        help="Reuse an existing convergence CSV instead of rerunning the reference.",
    )
    parser.add_argument(
        "--target-loss-pct",
        type=float,
        default=0.5,
    )
    args = parser.parse_args()

    thresholds = _parse_thresholds(args.thresholds)

    if args.history_csv is None:
        print("Running original outer-tolerance reference once ...")
        history = _run_reference(
            scene_seed=args.scene_seed,
            algorithm_seed=args.algorithm_seed,
        )
    else:
        history = _load_history_csv(args.history_csv)
        print(f"Reusing history: {args.history_csv}")

    rows = evaluate_window_thresholds(
        history,
        window=args.window,
        thresholds=thresholds,
    )

    stem = (
        f"window_plateau_scene_{args.scene_seed}_"
        f"seed_{args.algorithm_seed}_W{args.window}"
    )
    history_path = RESULT_DIR / f"{stem}_reference_history.csv"
    summary_path = RESULT_DIR / f"{stem}_comparison.csv"
    figure_path = FIGURE_DIR / f"{stem}.png"

    _save_history_csv(history, history_path)
    _save_rows(rows, summary_path)
    _plot_comparison(history, rows, figure_path)

    print("\nStage 8A window-plateau comparison")
    print(
        f"Reference: iteration={history.size - 1}, "
        f"WSR={history[-1]:.10f}"
    )
    print(
        "tol_window | stop_iter | WSR | loss(%) | saved(%) | reason"
    )

    for row in rows[1:]:
        print(
            f"{float(row['tol_window']):.1e} | "
            f"{int(row['stop_iteration']):4d} | "
            f"{float(row['final_wsr']):.10f} | "
            f"{float(row['wsr_loss_pct']):7.3f} | "
            f"{float(row['saved_iterations_pct']):7.2f} | "
            f"{row['stop_reason']}"
        )

    feasible = [
        row
        for row in rows[1:]
        if float(row["wsr_loss_pct"]) <= args.target_loss_pct
        and int(row["saved_iterations"]) > 0
    ]

    if feasible:
        best = max(
            feasible,
            key=lambda row: int(row["saved_iterations"]),
        )
        print(
            f"\nBest tested threshold with loss <= "
            f"{args.target_loss_pct:.3f}%: "
            f"tol_window={float(best['tol_window']):.1e}, "
            f"iteration={int(best['stop_iteration'])}, "
            f"loss={float(best['wsr_loss_pct']):.3f}%."
        )
    else:
        print(
            f"\nNo tested threshold both stops early and keeps "
            f"loss <= {args.target_loss_pct:.3f}%."
        )

    print(f"\ncomparison CSV: {summary_path}")
    print(f"reference CSV:  {history_path}")
    print(f"figure:         {figure_path}")


if __name__ == "__main__":
    main()
