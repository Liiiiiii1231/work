"""Stage 8：绘制 Proposed 的外层联合迭代收敛曲线。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

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
    """绘制 Proposed WSR vs Outer iteration。"""

    if history.ndim != 1 or history.size < 2:
        raise ValueError(
            "history must be a 1-D array with at least two values."
        )

    if not np.all(np.isfinite(history)):
        raise ValueError("history contains non-finite values.")

    iterations = np.arange(history.size)

    output_stem.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(8.0, 5.0))

    plt.plot(
        iterations,
        history,
        linewidth=1.8,
        label="RCG-ToMA + FP (Proposed)",
    )

    plt.scatter(
        [iterations[-1]],
        [history[-1]],
        s=35,
        zorder=3,
    )

    plt.xlabel("Outer iteration")
    plt.ylabel("WSR")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_stem.with_suffix(".png"),
        dpi=300,
    )

    plt.savefig(
        output_stem.with_suffix(".pdf"),
    )

    plt.close()


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
        f"proposed_convergence_scene_{scene_seed}"
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
    print(f"  outer iterations = {algorithm_result.outer_iterations}")
    print(f"  converged = {algorithm_result.converged}")
    print(f"  initial WSR = {history[0]:.10f}")
    print(f"  final WSR = {history[-1]:.10f}")
    print(f"CSV saved to: {csv_path}")
    print(
        "Figures saved to: "
        f"{figure_stem.with_suffix('.png')} and "
        f"{figure_stem.with_suffix('.pdf')}"
    )


if __name__ == "__main__":
    main()
