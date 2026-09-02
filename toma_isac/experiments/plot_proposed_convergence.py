"""Stage 8：绘制 Proposed 的外层联合迭代收敛曲线。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm import run_joint_algorithm
from config import SystemConfig, load_config
from multistart import MultiStartResult, run_multi_start
from stage8 import (
    load_stage8_experiment_config,
    make_scene_config,
)


CONFIG_PATH = ROOT / "configs" / "stage8_monte_carlo.yaml"
RESULT_DIR = ROOT / "results" / "stage8"
FIGURE_DIR = RESULT_DIR / "figures"


def run_proposed_convergence(
    cfg: SystemConfig,
    scene_seed: int,
    multi_start_seeds: tuple[int, ...],
    position_jitter: np.ndarray,
    require_convergence: bool = True,
    verbose: bool = True,
) -> MultiStartResult:
    """在一个 Stage 8 scene 上运行 Proposed 的 Multi-start。"""

    scene_cfg = make_scene_config(
        cfg,
        scene_seed=scene_seed,
        position_jitter=position_jitter,
    )

    return run_multi_start(
        name="Proposed: RCG-ToMA + FP",
        seeds=multi_start_seeds,
        run_single=lambda seed: run_joint_algorithm(
                scene_cfg,
                rng=np.random.default_rng(seed),
                verbose=False,
                position_update_mode="fp_aware",
                fp_aware_candidate_iters=2,
            ),
        iteration_getter=lambda result: result.outer_iterations,
        cfg=scene_cfg,
        require_convergence=require_convergence,
        verbose=verbose,
    )


def save_convergence_csv(
    history: np.ndarray,
    path: Path,
) -> None:
    """保存 iteration、WSR 和相邻迭代相对变化。"""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "outer_iteration",
                "wsr",
                "relative_change",
            ]
        )

        for iteration, wsr in enumerate(history):
            if iteration == 0:
                change = np.nan
            else:
                previous = history[iteration - 1]
                change = abs(wsr - previous) / max(
                    1.0,
                    abs(previous),
                )

            writer.writerow(
                [
                    iteration,
                    float(wsr),
                    float(change),
                ]
            )


def plot_convergence(
    history: np.ndarray,
    output_stem: Path,
) -> None:
    """绘制完整 WSR 曲线，并放大后期收敛区域。"""

    if history.ndim != 1 or history.size < 2:
        raise ValueError(
            "history must be a 1-D array with at least two values."
        )

    if not np.all(np.isfinite(history)):
        raise ValueError(
            "history contains non-finite values."
        )

    iterations = np.arange(history.size)
    final_iteration = int(iterations[-1])
    final_wsr = float(history[-1])

    output_stem.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(9.0, 5.5)
    )

    ax.plot(
        iterations,
        history,
        linewidth=1.8,
        label="RCG-ToMA + FP (Proposed)",
    )

    # 最终停止点即该次运行的收敛迭代。
    ax.axvline(
        final_iteration,
        linestyle="--",
        linewidth=1.2,
        label=f"Stopped at iteration {final_iteration}",
    )

    ax.scatter(
        [final_iteration],
        [final_wsr],
        s=40,
        zorder=3,
    )

    ax.set_xlabel("Joint AO iteration")
    ax.set_ylabel("WSR (bit/s/Hz)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")

    # -------------------------------------------------
    # Inset：自动放大最后约 35% 的 outer iterations。
    # 对 305 次收敛的情况，大约显示 198~305。
    # -------------------------------------------------
    zoom_start = max(
        1,
        int(np.floor(0.65 * final_iteration)),
    )

    zoom_history = history[zoom_start:]
    zoom_min = float(np.min(zoom_history))
    zoom_max = float(np.max(zoom_history))

    y_span = zoom_max - zoom_min

    if y_span <= 0.0:
        y_padding = max(
            1.0e-3,
            0.01 * abs(zoom_min),
        )
    else:
        y_padding = 0.15 * y_span

    axins = ax.inset_axes(
        [0.50, 0.18, 0.45, 0.36]
    )

    axins.plot(
        iterations,
        history,
        linewidth=1.5,
    )

    axins.axvline(
        final_iteration,
        linestyle="--",
        linewidth=1.0,
    )

    axins.scatter(
        [final_iteration],
        [final_wsr],
        s=28,
        zorder=3,
    )

    axins.set_xlim(
        zoom_start,
        final_iteration + max(2, int(0.02 * final_iteration)),
    )

    axins.set_ylim(
        zoom_min - y_padding,
        zoom_max + y_padding,
    )

    axins.set_title(
        "Late-stage behavior",
        fontsize=9,
    )

    axins.tick_params(
        axis="both",
        labelsize=8,
    )

    axins.grid(alpha=0.25)

    # 在主图中框出被放大的区域，并连到 inset。
    mark_inset(
        ax,
        axins,
        loc1=2,
        loc2=4,
        fc="none",
        ec="0.5",
        linewidth=0.8,
    )

    fig.tight_layout()

    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=300,
    )

    fig.savefig(
        output_stem.with_suffix(".pdf"),
    )

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scene-seed",
        type=int,
        default=None,
        help=(
            "Stage 8 scene seed. "
            "Default: experiment.scene_seed_start."
        ),
    )

    args = parser.parse_args()

    print("[plot version] inset-v2")

    cfg = load_config(CONFIG_PATH)
    experiment_cfg = load_stage8_experiment_config(
        CONFIG_PATH
    )

    scene_seed = (
        experiment_cfg.scene_seed_start
        if args.scene_seed is None
        else int(args.scene_seed)
    )

    result = run_proposed_convergence(
        cfg,
        scene_seed=scene_seed,
        multi_start_seeds=experiment_cfg.multi_start_seeds,
        position_jitter=experiment_cfg.position_jitter,
        require_convergence=True,
        verbose=True,
    )

    best = result.best_start
    algorithm_result = best.result

    history = np.asarray(
        algorithm_result.joint_wsr_history,
        dtype=float,
    )

    expected_length = (
        algorithm_result.outer_iterations + 1
    )

    if history.size != expected_length:
        raise RuntimeError(
            "joint_wsr_history length does not match "
            "outer_iterations + 1."
        )

    stem = (
        f"proposed_convergence_scene_{scene_seed}_with_inset"
    )

    csv_path = RESULT_DIR / f"{stem}.csv"
    figure_stem = FIGURE_DIR / stem

    save_convergence_csv(
        history,
        csv_path,
    )

    plot_convergence(
        history,
        figure_stem,
    )

    print("\nBest Proposed start")
    print(f"  scene seed = {scene_seed}")
    print(f"  best multi-start seed = {best.seed}")
    print(
        "  outer iterations = "
        f"{algorithm_result.outer_iterations}"
    )
    print(
        f"  converged = {algorithm_result.converged}"
    )
    print(
        f"  initial WSR = {history[0]:.10f}"
    )
    print(
        f"  final WSR = {history[-1]:.10f}"
    )
    print(f"CSV saved to: {csv_path}")
    print(
        "Figures saved to: "
        f"{figure_stem.with_suffix('.png')} and "
        f"{figure_stem.with_suffix('.pdf')}"
    )


if __name__ == "__main__":
    main()
