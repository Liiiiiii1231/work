"""Stage 9F：正式 FP-aware 位置更新接口测试。"""

from pathlib import Path

import numpy as np
import pytest

from algorithm import run_joint_algorithm
from config import load_config
from stage8 import make_scene_config
from utils import check_resource_constraints


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "stage9_outer_debug.yaml"
POSITION_JITTER = np.array([1.5, 1.5, 0.25], dtype=float)


def _scene_cfg():
    cfg = load_config(CONFIG_PATH)
    return make_scene_config(
        cfg,
        scene_seed=8001,
        position_jitter=POSITION_JITTER,
        randomize_fading=False,
    )


def test_original_mode_keeps_default_behavior() -> None:
    cfg = _scene_cfg()

    default_result = run_joint_algorithm(
        cfg,
        rng=np.random.default_rng(2029),
        max_outer_iterations=2,
    )
    explicit_result = run_joint_algorithm(
        cfg,
        rng=np.random.default_rng(2029),
        max_outer_iterations=2,
        position_update_mode="original",
    )

    assert np.allclose(
        default_result.joint_wsr_history,
        explicit_result.joint_wsr_history,
        rtol=0.0,
        atol=0.0,
    )


def test_fp_aware_mode_is_feasible_and_monotone() -> None:
    cfg = _scene_cfg()

    result = run_joint_algorithm(
        cfg,
        rng=np.random.default_rng(2029),
        max_outer_iterations=2,
        position_update_mode="fp_aware",
        fp_aware_candidate_iters=2,
    )

    history = np.asarray(result.joint_wsr_history, dtype=float)
    assert np.all(np.diff(history) >= -1.0e-8)
    check_resource_constraints(result.resources, cfg)


def test_invalid_position_update_mode_is_rejected() -> None:
    cfg = _scene_cfg()

    with pytest.raises(ValueError):
        run_joint_algorithm(
            cfg,
            rng=np.random.default_rng(2029),
            max_outer_iterations=1,
            position_update_mode="unknown",
        )
