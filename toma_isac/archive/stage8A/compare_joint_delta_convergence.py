"""Stage 8A-4a：比较 finite_diff_delta 对正式联合收敛曲线的影响。

只改变外层中央差分步长，其余 scene、初始化 seed、FP/RCG 参数、
WSR 外层停止条件全部保持一致。

默认比较：
- old delta = 1e-5
- new delta = 1e-6
- scene_seed = 8001
- algorithm_seed = 2029

输出：
1. 两条正式 FP+RCG 联合 WSR 收敛曲线；
2. 每次 outer iteration 的 WSR / relative change CSV；
3. 两个 delta 的最终迭代次数、WSR、运行时间 summary CSV。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm import AlgorithmResult, run_joint_algorithm
from config import SystemConfig, load_config
from stage8 import load_stage8_experiment_config, make_scene_config
from utils import relative_change


CONFIG_PATH = ROOT / "configs" / "stage8_monte_carlo.yaml"
RESULT_DIR = ROOT / "results" / "stage8" / "diagnostics"
FIGURE_DIR = RESULT_DIR / "figures"


def _run_one_delta(
    scene_cfg: SystemConfig,
    delta: float,
    algorithm_seed: int,
    verbose: bool,
) -> tuple[AlgorithmResult, float]:
    """在同一 scene/seed 下运行一次正式联合算法。"""

    cfg = replace(
        scene_cfg,
        finite_diff_delta=float(delta),
    )
    cfg.validate()

    start = perf_counter()
    result = run_joint_algorithm(
        cfg,
        rng=np.random.default_rng(algorithm_seed),
        verbose=verbose,
    )
    runtime_s = perf_counter() - start

    return result, float(runtime_s)


def _history_rows(
    label: str,
    delta: float,
    result: AlgorithmResult,
    tol_outer: float,
) -> list[dict[str, object]]:
    """将 joint_wsr_history 转成逐轮记录。"""

    rows: list[dict[str, object]] = []
    history = result.joint_wsr_history

    for iteration, wsr in enumerate(history):
        if iteration == 0:
            change = float("nan")
            tol_satisfied = False
        else:
            change = relative_change(
                float(wsr),
                float(history[iteration - 1]),
            )
            tol_satisfied = bool(change <= tol_outer)

        rows.append(
            {
                "label": label,
                "finite_diff_delta": float(delta),
                "outer_iteration": iteration,
                "wsr": float(wsr),
                "relative_change": change,
                "tol_outer_satisfied": tol_satisfied,
            }
        )

    return rows


def _first_two_consecutive_tol_iteration(
    history: list[float],
    tol_outer: float,
) -> int | None:
    """独立复核正式的“两次连续 WSR 容差”停止位置。"""

    consecutive = 0

    for iteration in range(1, len(history)):
        change = relative_change(
            float(history[iteration]),
            float(history[iteration - 1]),
        )

        if change <= tol_outer:
            consecutive += 1
        else:
            consecutive = 0

        if consecutive >= 2:
            return iteration

    return None


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError("rows must not be empty.")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_histories(
    old_delta: float,
    old_result: AlgorithmResult,
    new_delta: float,
    new_result: AlgorithmResult,
    output_path: Path,
) -> None:
    """画正式联合 FP+RCG 收敛曲线对比。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    old_x = np.arange(len(old_result.joint_wsr_history))
    new_x = np.arange(len(new_result.joint_wsr_history))

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    ax.plot(
        old_x,
        old_result.joint_wsr_history,
        linewidth=1.8,
        label=f"delta={old_delta:.0e} ({old_result.outer_iterations} iters)",
    )
    ax.plot(
        new_x,
        new_result.joint_wsr_history,
        linewidth=1.8,
        label=f"delta={new_delta:.0e} ({new_result.outer_iterations} iters)",
    )

    ax.set_xlabel("Joint outer iteration")
    ax.set_ylabel("Weighted sum rate (WSR)")
    ax.set_title("Formal FP + RCG convergence: finite-difference delta comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-seed", type=int, default=8001)
    parser.add_argument("--algorithm-seed", type=int, default=2029)
    parser.add_argument("--old-delta", type=float, default=1.0e-5)
    parser.add_argument("--new-delta", type=float, default=1.0e-6)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every formal joint iteration. Usually unnecessary.",
    )
    args = parser.parse_args()

    if args.old_delta <= 0.0 or args.new_delta <= 0.0:
        raise ValueError("Both finite-difference deltas must be positive.")

    base_cfg = load_config(CONFIG_PATH)
    experiment_cfg = load_stage8_experiment_config(CONFIG_PATH)

    scene_cfg = make_scene_config(
        base_cfg,
        scene_seed=int(args.scene_seed),
        position_jitter=experiment_cfg.position_jitter,
    )

    print("Formal joint convergence delta comparison")
    print(f"scene_seed     = {args.scene_seed}")
    print(f"algorithm_seed = {args.algorithm_seed}")
    print(f"tol_outer      = {scene_cfg.tol_outer:.1e}")
    print(f"old delta      = {args.old_delta:.1e}")
    print(f"new delta      = {args.new_delta:.1e}")
    print("Only finite_diff_delta is changed.\n")

    print("[1/2] Running old delta...")
    old_result, old_runtime = _run_one_delta(
        scene_cfg,
        args.old_delta,
        args.algorithm_seed,
        args.verbose,
    )

    print("[2/2] Running new delta...")
    new_result, new_runtime = _run_one_delta(
        scene_cfg,
        args.new_delta,
        args.algorithm_seed,
        args.verbose,
    )

    # 两个实验必须从完全相同的初始化 WSR 出发，否则不是单变量对照。
    if not np.isclose(
        old_result.joint_wsr_history[0],
        new_result.joint_wsr_history[0],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "The two delta runs did not start from the same initial WSR. "
            "The comparison is not controlled."
        )

    old_stop_check = _first_two_consecutive_tol_iteration(
        old_result.joint_wsr_history,
        scene_cfg.tol_outer,
    )
    new_stop_check = _first_two_consecutive_tol_iteration(
        new_result.joint_wsr_history,
        scene_cfg.tol_outer,
    )

    if old_result.converged and old_stop_check != old_result.outer_iterations:
        raise RuntimeError("Old-delta stop iteration check failed.")
    if new_result.converged and new_stop_check != new_result.outer_iterations:
        raise RuntimeError("New-delta stop iteration check failed.")

    stem = (
        f"joint_delta_compare_scene_{args.scene_seed}"
        f"_seed_{args.algorithm_seed}"
    )

    history_path = RESULT_DIR / f"{stem}.csv"
    summary_path = RESULT_DIR / f"{stem}_summary.csv"
    figure_path = FIGURE_DIR / f"{stem}.png"

    history_rows = (
        _history_rows("old", args.old_delta, old_result, scene_cfg.tol_outer)
        + _history_rows("new", args.new_delta, new_result, scene_cfg.tol_outer)
    )
    _write_csv(history_rows, history_path)

    initial_wsr = float(old_result.joint_wsr_history[0])
    old_final_wsr = float(old_result.performance.weighted_sum_rate)
    new_final_wsr = float(new_result.performance.weighted_sum_rate)

    iteration_reduction = old_result.outer_iterations - new_result.outer_iterations
    iteration_reduction_percent = (
        100.0 * iteration_reduction / old_result.outer_iterations
        if old_result.outer_iterations > 0
        else float("nan")
    )

    wsr_difference = new_final_wsr - old_final_wsr
    wsr_difference_percent = (
        100.0 * wsr_difference / abs(old_final_wsr)
        if abs(old_final_wsr) > 0.0
        else float("nan")
    )

    summary_rows = [
        {
            "label": "old",
            "finite_diff_delta": float(args.old_delta),
            "initial_wsr": initial_wsr,
            "final_wsr": old_final_wsr,
            "outer_iterations": old_result.outer_iterations,
            "converged": old_result.converged,
            "runtime_s": old_runtime,
            "verified_stop_iteration": old_stop_check,
        },
        {
            "label": "new",
            "finite_diff_delta": float(args.new_delta),
            "initial_wsr": initial_wsr,
            "final_wsr": new_final_wsr,
            "outer_iterations": new_result.outer_iterations,
            "converged": new_result.converged,
            "runtime_s": new_runtime,
            "verified_stop_iteration": new_stop_check,
        },
    ]
    _write_csv(summary_rows, summary_path)
    _plot_histories(
        args.old_delta,
        old_result,
        args.new_delta,
        new_result,
        figure_path,
    )

    print("\nComparison finished.")
    print(
        f"OLD delta={args.old_delta:.1e}: "
        f"iters={old_result.outer_iterations}, "
        f"converged={old_result.converged}, "
        f"final WSR={old_final_wsr:.10f}, "
        f"runtime={old_runtime:.2f}s"
    )
    print(
        f"NEW delta={args.new_delta:.1e}: "
        f"iters={new_result.outer_iterations}, "
        f"converged={new_result.converged}, "
        f"final WSR={new_final_wsr:.10f}, "
        f"runtime={new_runtime:.2f}s"
    )
    print(
        f"iteration reduction = {iteration_reduction} "
        f"({iteration_reduction_percent:.2f}%)"
    )
    print(
        f"final WSR difference (new-old) = "
        f"{wsr_difference:+.6e} ({wsr_difference_percent:+.4f}%)"
    )
    print(f"\nhistory CSV: {history_path}")
    print(f"summary CSV: {summary_path}")
    print(f"figure:      {figure_path}")


if __name__ == "__main__":
    main()
