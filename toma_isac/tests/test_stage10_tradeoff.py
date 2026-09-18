"""Stage 10 权重分组与配置检查。"""

from pathlib import Path

import numpy as np

from config import load_config
from experiments.run_stage10_tradeoff import (
    load_tradeoff_config,
    make_group_weights,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "stage10_tradeoff.yaml"


def test_stage10_uses_k5_nominal_setting() -> None:
    cfg = load_config(CONFIG_PATH)
    assert (cfg.k_dl, cfg.j_ul, cfg.l_target, cfg.r_sensing) == (5, 5, 5, 4)


def test_group_weights_sum_to_one() -> None:
    cfg = load_config(CONFIG_PATH)

    for omega_comm in (0.1, 0.5, 0.9):
        weight_dl, weight_ul, weight_s = make_group_weights(cfg, omega_comm)

        assert np.isclose(np.sum(weight_dl), omega_comm / 2.0)
        assert np.isclose(np.sum(weight_ul), omega_comm / 2.0)
        assert np.isclose(np.sum(weight_s), 1.0 - omega_comm)
        assert np.isclose(
            np.sum(weight_dl) + np.sum(weight_ul) + np.sum(weight_s),
            1.0,
        )


def test_tradeoff_weight_grid() -> None:
    """验证 Stage 10A 正式通信权重网格。"""

    tradeoff = load_tradeoff_config(CONFIG_PATH)

    assert tradeoff.scene_seed == 8001

    expected = tuple(
        round(0.05 * i, 2)
        for i in range(1, 20)
    )

    assert tradeoff.omega_comm == expected