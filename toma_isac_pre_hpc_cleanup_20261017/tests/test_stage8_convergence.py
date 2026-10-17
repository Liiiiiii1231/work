"""测试 Proposed convergence 数据接口。"""

from dataclasses import replace
from pathlib import Path

import numpy as np

from config import load_config
from experiments.plot_proposed_convergence import (
    plot_convergence,
    run_proposed_convergence,
    save_convergence_csv,
)
from stage8 import load_stage8_experiment_config


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "stage8_monte_carlo.yaml"
)


def test_proposed_convergence_history_and_output(
    tmp_path,
) -> None:
    """验证 Proposed 已保存完整 outer WSR history。"""

    cfg = load_config(CONFIG_PATH)
    experiment_cfg = load_stage8_experiment_config(
        CONFIG_PATH
    )

    short_cfg = replace(
        cfg,
        max_inner_iter=2,
        max_outer_iter=2,
    )

    result = run_proposed_convergence(
        short_cfg,
        scene_seed=9401,
        multi_start_seeds=(51,),
        position_jitter=experiment_cfg.position_jitter,
        require_convergence=False,
        verbose=False,
    )

    best = result.best_start
    history = np.asarray(
        best.result.joint_wsr_history,
        dtype=float,
    )

    assert history.ndim == 1
    assert history.size == best.result.outer_iterations + 1
    assert np.all(np.isfinite(history))

    tolerance = (
        1.0e-8
        * np.maximum(
            1.0,
            np.abs(history[:-1]),
        )
    )

    assert np.all(
        np.diff(history) >= -tolerance
    )

    csv_path = tmp_path / "convergence.csv"
    figure_stem = tmp_path / "convergence"

    save_convergence_csv(
        history,
        csv_path,
    )

    plot_convergence(
        history,
        figure_stem,
    )

    assert csv_path.exists()
    assert figure_stem.with_suffix(".png").exists()
    assert figure_stem.with_suffix(".pdf").exists()
