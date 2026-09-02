"""测试 Stage 8 scene、七方案预算与结果保存。"""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from config import load_config
from stage8 import (
    SCHEME_ORDER,
    load_stage8_experiment_config,
    make_scene_config,
    run_stage8_scene,
    save_monte_carlo_results,
)


CONFIG_PATH = (
    ROOT
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
    """验证 ToMA 多起点、固定 UPA 单起点、B6 多起点及七方案接口。"""

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
        ma_n_starts=experiment_cfg.ma_n_starts,
    )

    assert tuple(result.schemes) == SCHEME_ORDER

    for key in ("b1", "b2", "b4", "proposed"):
        assert result.schemes[key].n_starts == 2
        assert np.isfinite(result.schemes[key].wsr)

    for key in ("b3", "b5"):
        assert result.schemes[key].n_starts == 1
        assert np.isfinite(result.schemes[key].wsr)

    assert (
        result.schemes["b6"].n_starts
        == experiment_cfg.ma_n_starts
    )
    assert np.isfinite(result.schemes["b6"].wsr)


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
        ma_n_starts=1,
    )

    csv_path, npz_path = save_monte_carlo_results(
        [scene],
        tmp_path,
        stem="test_stage8",
    )

    assert csv_path.exists()
    assert npz_path.exists()

    data = np.load(npz_path)

    assert data["wsr"].shape == (
        1,
        len(SCHEME_ORDER),
    )
    assert data["q_dl"].shape == (
        1,
        cfg.k_dl,
        3,
    )
    assert data["q_ul"].shape == (
        1,
        cfg.j_ul,
        3,
    )
    assert data["q_s"].shape == (
        1,
        cfg.l_target,
        3,
    )
    assert data["alpha_dl"].shape == (
        1,
        cfg.k_dl,
    )
    assert data["alpha_ul"].shape == (
        1,
        cfg.j_ul,
    )
    assert data["xi_s"].shape == (
        1,
        cfg.l_target,
    )


def test_scene_level_fading_is_reproducible_and_fixed() -> None:
    """验证同 scene 的 fading 可复现，不同 scene 会改变 fading。"""

    cfg = load_config(CONFIG_PATH)
    experiment_cfg = load_stage8_experiment_config(CONFIG_PATH)

    scene_a = make_scene_config(
        cfg,
        scene_seed=9401,
        position_jitter=experiment_cfg.position_jitter,
        randomize_fading=True,
        fading_model=experiment_cfg.fading_model,
    )
    scene_b = make_scene_config(
        cfg,
        scene_seed=9401,
        position_jitter=experiment_cfg.position_jitter,
        randomize_fading=True,
        fading_model=experiment_cfg.fading_model,
    )
    scene_c = make_scene_config(
        cfg,
        scene_seed=9402,
        position_jitter=experiment_cfg.position_jitter,
        randomize_fading=True,
        fading_model=experiment_cfg.fading_model,
    )

    assert np.allclose(
        scene_a.alpha_dl,
        scene_b.alpha_dl,
    )
    assert np.allclose(
        scene_a.alpha_ul,
        scene_b.alpha_ul,
    )
    assert np.allclose(
        scene_a.xi_s,
        scene_b.xi_s,
    )

    assert not np.allclose(
        scene_a.alpha_dl,
        scene_c.alpha_dl,
    )
    assert not np.allclose(
        scene_a.alpha_ul,
        scene_c.alpha_ul,
    )
    assert not np.allclose(
        scene_a.xi_s,
        scene_c.xi_s,
    )

    # base_cfg 不被 scene 构造原地修改。
    assert np.allclose(
        cfg.alpha_dl,
        load_config(CONFIG_PATH).alpha_dl,
    )
    assert np.allclose(
        cfg.alpha_ul,
        load_config(CONFIG_PATH).alpha_ul,
    )
    assert np.allclose(
        cfg.xi_s,
        load_config(CONFIG_PATH).xi_s,
    )


def test_fading_has_unit_power_normalization_statistically() -> None:
    """快速验证 scalar Rayleigh multiplier 的平均功率归一化。"""

    from stage8 import _unit_power_complex_rayleigh

    rng = np.random.default_rng(12345)

    samples = _unit_power_complex_rayleigh(
        rng,
        200000,
    )

    mean_power = float(
        np.mean(
            np.abs(samples) ** 2
        )
    )

    assert abs(
        mean_power - 1.0
    ) < 0.02


def test_parameter_changes_do_not_change_same_scene_fading() -> None:
    """验证 sweep 中同一个 scene seed 会复用相同 fading realization。"""

    cfg = load_config(CONFIG_PATH)
    experiment_cfg = load_stage8_experiment_config(CONFIG_PATH)

    cfg_other_rho = replace(
        cfg,
        rho_si=0.1,
    )

    scene_a = make_scene_config(
        cfg,
        scene_seed=9501,
        position_jitter=experiment_cfg.position_jitter,
        randomize_fading=True,
        fading_model=experiment_cfg.fading_model,
    )

    scene_b = make_scene_config(
        cfg_other_rho,
        scene_seed=9501,
        position_jitter=experiment_cfg.position_jitter,
        randomize_fading=True,
        fading_model=experiment_cfg.fading_model,
    )

    assert np.allclose(
        scene_a.q_dl,
        scene_b.q_dl,
    )
    assert np.allclose(
        scene_a.q_ul,
        scene_b.q_ul,
    )
    assert np.allclose(
        scene_a.q_s,
        scene_b.q_s,
    )
    assert np.allclose(
        scene_a.alpha_dl,
        scene_b.alpha_dl,
    )
    assert np.allclose(
        scene_a.alpha_ul,
        scene_b.alpha_ul,
    )
    assert np.allclose(
        scene_a.xi_s,
        scene_b.xi_s,
    )