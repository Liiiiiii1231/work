"""测试 ToMA 基础几何映射与几何约束。"""

import numpy as np
import pytest

from config import load_config
from geometry import (
    build_geometry,
    min_endpoint_distance,
    min_tx_rx_element_distance,
)


def test_build_geometry_matches_cable_model() -> None:
    """验证 p_{m,n}=(n/N_c)c_m 以及 Tx/Rx 分组。"""

    cfg = load_config()

    # 构造一组简单、确定且满足约束的端点。
    endpoints = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=float,
    )

    geometry = build_geometry(
        endpoints,
        cfg,
    )

    # 当前 default.yaml 中 N_c=2，
    # 所以每根缆绳上的阵元位于 1/2 和 1 倍端点位置。
    expected_tx = np.array(
        [
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )

    expected_rx = np.array(
        [
            [-0.5, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -0.5, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=float,
    )

    assert np.allclose(
        geometry.tx_positions,
        expected_tx,
    )

    assert np.allclose(
        geometry.rx_positions,
        expected_rx,
    )

    # 四个端点中最近距离应为 sqrt(2)。
    assert np.isclose(
        min_endpoint_distance(endpoints),
        np.sqrt(2.0),
    )

    # Tx/Rx 阵元之间不能出现 SI 距离奇异。
    assert (
        min_tx_rx_element_distance(geometry)
        > cfg.min_si_distance
    )


def test_build_geometry_rejects_uav_collision() -> None:
    """验证端点距离过小时会触发防碰撞检查。"""

    cfg = load_config()

    # 前两个 UAV 位于同一个位置，
    # 因此必然违反防碰撞约束。
    endpoints = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=float,
    )

    with pytest.raises(
        ValueError,
        match="UAV collision constraint violated",
    ):
        build_geometry(
            endpoints,
            cfg,
        )