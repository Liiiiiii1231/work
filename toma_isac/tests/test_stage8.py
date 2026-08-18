"""测试 Stage 8 scene、六方案预算与结果保存。"""

from dataclasses import replace
from pathlib import Path

import numpy as np

from config import load_config
from stage8 import (
    SCHEME_ORDER,
    load_stage8_experiment_config,
    make_scene_config,
    run_stage8_scene,
    save_monte_carlo_results,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "stage8_monte_carlo.yaml"
)


def test_scene_seed_is_reproducible_and_separate() -> None:
    """验证 scene seed 可复现，且只改变场景位置。"""

    cfg = load_config(CONFIG_PATH)
    experiment_cfg = load_stage8_experiment_config(CONFIG_PATH)

    scene_a = make_scene_config(
        cfg,
        scene_seed=9001,
        position_jitter=experiment_cfg.position_jitter,
    )
    scene_b = make_scene_config(
        cfg,
        scene_seed=9001,
        position_jitter=experiment_cfg.position_jitter,
    )
    scene_c = make_scene_config(
        cfg,
        scene_seed=9002,
        position_jitter=experiment_cfg.position_jitter,
    )

    assert np.allclose(scene_a.q_dl, scene_b.q_dl)
    assert np.allclose(scene_a.q_ul, scene_b.q_ul)
    assert np.allclose(scene_a.q_s, scene_b.q_s)

    assert not np.allclose(scene_a.q_dl, scene_c.q_dl)
    assert scene_a.random_seed == cfg.random_seed
    assert scene_a.p_dl_max == cfg.p_dl_max


def test_stage8_scene_uses_expected_start_budget() -> None:
    """验证 ToMA 多起点、固定 UPA 单起点及六方案接口。"""

    cfg = load_config(CONFIG_PATH)
    experiment_cfg = load_stage8_experiment_config(CONFIG_PATH)

    short_cfg = replace(
        cfg,
        max_inner_iter=2,
        max_outer_iter=2,
    )

    result = run_stage8_scene(
        short_cfg,
        scene_seed=9101,
        multi_start_seeds=(31, 32),
        fixed_upa_seed=31,
        position_jitter=experiment_cfg.position_jitter,
        require_convergence=False,
        verbose=False,
    )

    assert tuple(result.schemes) == SCHEME_ORDER

    for key in ("b1", "b2", "b4", "proposed"):
        assert result.schemes[key].n_starts == 2
        assert np.isfinite(result.schemes[key].wsr)

    for key in ("b3", "b5"):
        assert result.schemes[key].n_starts == 1
        assert np.isfinite(result.schemes[key].wsr)


def test_stage8_results_can_be_saved(tmp_path) -> None:
    """验证 Stage 8 CSV/NPZ 输出结构。"""

    cfg = load_config(CONFIG_PATH)
    experiment_cfg = load_stage8_experiment_config(CONFIG_PATH)

    short_cfg = replace(
        cfg,
        max_inner_iter=1,
        max_outer_iter=1,
    )

    scene = run_stage8_scene(
        short_cfg,
        scene_seed=9201,
        multi_start_seeds=(41,),
        fixed_upa_seed=41,
        position_jitter=experiment_cfg.position_jitter,
        require_convergence=False,
    )

    csv_path, npz_path = save_monte_carlo_results(
        [scene],
        tmp_path,
        stem="test_stage8",
    )

    assert csv_path.exists()
    assert npz_path.exists()

    data = np.load(npz_path)
    assert data["wsr"].shape == (1, 6)
    assert data["q_dl"].shape == (1, cfg.k_dl, 3)
    assert data["q_ul"].shape == (1, cfg.j_ul, 3)
    assert data["q_s"].shape == (1, cfg.l_target, 3)
