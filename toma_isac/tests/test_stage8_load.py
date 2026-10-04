"""测试 Stage 8 system-load sweep 配置与接口。"""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from config import load_config
from experiments.run_stage8_load_sweep import (
    save_system_load_results,
)
from stage8 import (
    SCHEME_ORDER,
    SceneSummary,
    SchemeSummary,
    load_stage8_experiment_config,
    run_stage8_scene,
)

ROOT = Path(__file__).resolve().parents[1]


def _config_path(load: int) -> Path:
    """返回对应 system load 的配置文件。"""

    return (
        ROOT
        / "configs"
        / f"stage8_load{load}.yaml"
    )


def test_system_load_yaml_shapes_are_consistent() -> None:
    """验证 load=3~7 的 shape、权重和感知流数。"""

    for load in (3, 4, 5, 6, 7):
        path = _config_path(load)

        cfg = load_config(path)

        experiment_cfg = (
            load_stage8_experiment_config(
                path
            )
        )

        assert cfg.k_dl == load
        assert cfg.j_ul == load
        assert cfg.l_target == load
        assert cfg.r_sensing == min(
            load,
            cfg.n_tx,
        )

        assert cfg.p_ul_max.shape == (
            load,
        )
        assert cfg.sigma_dl2.shape == (
            load,
        )

        assert cfg.q_dl.shape == (
            load,
            3,
        )
        assert cfg.q_ul.shape == (
            load,
            3,
        )
        assert cfg.q_s.shape == (
            load,
            3,
        )

        assert cfg.xi_s.shape == (
            load,
        )
        assert cfg.fading_dl.shape == (
            load,
        )
        assert cfg.fading_ul.shape == (
            load,
        )
        assert cfg.fading_s.shape == (
            load,
        )

        total_weight = (
            np.sum(cfg.weight_dl)
            + np.sum(cfg.weight_ul)
            + np.sum(cfg.weight_s)
        )

        assert np.isclose(
            total_weight,
            1.0,
        )

        assert experiment_cfg.n_mc == 10


def test_load3_runs_through_seven_scheme_interface() -> None:
    """验证 load=3 没有隐藏的二维写死。"""

    path = _config_path(3)

    cfg = load_config(path)

    experiment_cfg = (
        load_stage8_experiment_config(
            path
        )
    )

    short_cfg = replace(
        cfg,
        max_inner_iter=1,
        max_outer_iter=1,
    )

    result = run_stage8_scene(
        short_cfg,
        scene_seed=9301,
        multi_start_seeds=(51,),
        fixed_upa_seed=51,
        position_jitter=experiment_cfg.position_jitter,
        require_convergence=False,
        verbose=False,
        ma_n_starts=1,
    )

    assert tuple(
        result.schemes
    ) == SCHEME_ORDER

    for key in SCHEME_ORDER:
        assert np.isfinite(
            result.schemes[key].wsr
        )

        assert (
            result.schemes[key].n_starts
            == 1
        )


def test_system_load_results_can_be_saved(
    tmp_path,
) -> None:
    """验证合并 CSV/NPZ 保存。"""

    def _scene(
        load: int,
    ) -> SceneSummary:
        schemes = {
            key: SchemeSummary(
                wsr=float(load),
                best_seed=2026,
                iterations=1,
                runtime_s=0.1,
                n_starts=1,
            )
            for key in SCHEME_ORDER
        }

        return SceneSummary(
            scene_seed=8000 + load,
            q_dl=np.zeros(
                (load, 3)
            ),
            q_ul=np.zeros(
                (load, 3)
            ),
            q_s=np.zeros(
                (load, 3)
            ),
            alpha_dl=np.ones(
                load,
                dtype=np.complex128,
            ),
            alpha_ul=np.ones(
                load,
                dtype=np.complex128,
            ),
            xi_s=np.ones(
                load,
                dtype=np.complex128,
            ),
            schemes=schemes,
        )

    csv_path, npz_path = (
        save_system_load_results(
            {
                load: [_scene(load)]
                for load in (3, 4, 5, 6, 7)
            },
            tmp_path,
        )
    )

    assert csv_path.exists()
    assert npz_path.exists()

    data = np.load(
        npz_path
    )

    assert data[
        "system_load"
    ].tolist() == [
        3,
        4,
        5,
        6,
        7,
    ]

    assert data["wsr"].shape == (
        5,
        len(SCHEME_ORDER),
    )