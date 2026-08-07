"""ToMA 几何模型。

本文件负责：

1. 根据拖曳子无人机端点 c_m 生成每根缆绳上的阵元位置；
2. 形成发射阵列位置 p_i^T；
3. 形成接收阵列位置 p_s^R；
4. 检查缆绳定长约束；
5. 检查子无人机防碰撞约束；
6. 防止直接 SI 信道中的 1/d 出现数值奇异。

以后 outer_solver.py 优化 c_m 后，
仍然直接调用 build_geometry() 重建阵列几何。
"""

import numpy as np

from config import SystemConfig
from state import GeometryState


def build_geometry(
    endpoints: np.ndarray,
    cfg: SystemConfig,
) -> GeometryState:
    """由所有 ToMA 端点 c_m 构造发射/接收阵列位置。

    参数
    ----------
    endpoints:
        所有拖曳子无人机端点。
        shape = (M, 3)

        endpoints[m] 对应数学中的 c_m。
        注意 Python 下标从 0 开始。

    cfg:
        系统参数。

    返回
    ----------
    GeometryState:
        包含：
        endpoints
        tx_positions
        rx_positions
    """

    endpoints = np.asarray(
        endpoints,
        dtype=float,
    )

    if endpoints.shape != (
        cfg.m_uav,
        3,
    ):
        raise ValueError(
            f"endpoints.shape={endpoints.shape}, "
            f"expected {(cfg.m_uav, 3)}."
        )

    # ----------------------------------------------------------
    # 数学模型：
    #
    # p_{m,n} = (n / N_c) c_m
    #
    # n = 1,...,N_c
    #
    # fractions =
    # [1/N_c, 2/N_c, ..., N_c/N_c]
    # ----------------------------------------------------------

    fractions = (
        np.arange(
            1,
            cfg.n_cable + 1,
            dtype=float,
        )
        / cfg.n_cable
    )

    # all_positions.shape = (M, N_c, 3)
    #
    # all_positions[m,n]
    # 表示第 m 根缆绳上的第 n 个阵元位置。
    all_positions = (
        endpoints[:, None, :]
        * fractions[
            None,
            :,
            None,
        ]
    )

    # ----------------------------------------------------------
    # 前 M/2 根缆绳属于 Tx。
    #
    # reshape 后：
    # tx_positions.shape = (N_T, 3)
    #
    # 顺序正好对应论文中的全局发射阵元编号 i。
    # ----------------------------------------------------------

    tx_positions = (
        all_positions[
            : cfg.n_tx_uav
        ]
        .reshape(
            cfg.n_tx,
            3,
        )
    )

    # ----------------------------------------------------------
    # 后 M/2 根缆绳属于 Rx。
    #
    # rx_positions.shape = (N_R, 3)
    # ----------------------------------------------------------

    rx_positions = (
        all_positions[
            cfg.n_tx_uav :
        ]
        .reshape(
            cfg.n_rx,
            3,
        )
    )

    geometry = GeometryState(
        endpoints=endpoints.copy(),
        tx_positions=tx_positions,
        rx_positions=rx_positions,
    )

    # 每次构建几何后马上检查是否合法。
    check_geometry_constraints(
        geometry,
        cfg,
    )

    return geometry


def check_geometry_constraints(
    geometry: GeometryState,
    cfg: SystemConfig,
    atol: float = 1e-10,
) -> None:
    """检查当前 ToMA 几何是否满足约束。"""

    endpoints = geometry.endpoints

    # ==========================================================
    # C1：缆绳完全展开
    #
    # ||c_m||_2 = L_c
    # ==========================================================

    endpoint_norms = np.linalg.norm(
        endpoints,
        axis=1,
    )

    if not np.allclose(
        endpoint_norms,
        cfg.cable_length,
        atol=atol,
    ):
        raise ValueError(
            "Cable-length constraint violated."
        )

    # ==========================================================
    # C2：子无人机防碰撞
    #
    # ||c_m - c_m'||_2 >= D_UAV
    # ==========================================================

    difference = (
        endpoints[:, None, :]
        - endpoints[
            None,
            :,
            :,
        ]
    )

    distances = np.linalg.norm(
        difference,
        axis=2,
    )

    # 排除 m=m' 时自己与自己的距离 0。
    mask = ~np.eye(
        cfg.m_uav,
        dtype=bool,
    )

    if np.any(
        distances[mask]
        < cfg.min_uav_distance - atol
    ):
        raise ValueError(
            "UAV collision constraint violated."
        )

    # ==========================================================
    # 数值保护：
    #
    # H_SI^0 中存在 1 / d_si。
    #
    # 所以任何 Tx/Rx 阵元都不能重合。
    #
    # 注意：
    # 这不是残余 SI 功率约束，
    # 只是避免除以 0。
    # ==========================================================

    si_difference = (
        geometry.rx_positions[
            :,
            None,
            :,
        ]
        - geometry.tx_positions[
            None,
            :,
            :,
        ]
    )

    si_distances = np.linalg.norm(
        si_difference,
        axis=2,
    )

    if np.min(
        si_distances
    ) < cfg.min_si_distance:
        raise ValueError(
            "Tx-Rx element distance is too small."
        )


def min_endpoint_distance(
    endpoints: np.ndarray,
) -> float:
    """返回任意两个不同子无人机端点之间的最小距离。"""

    difference = (
        endpoints[:, None, :]
        - endpoints[
            None,
            :,
            :,
        ]
    )

    distances = np.linalg.norm(
        difference,
        axis=2,
    )

    # 对角线设置成无穷大，
    # 避免把自己到自己的距离 0 当成最小距离。
    np.fill_diagonal(
        distances,
        np.inf,
    )

    return float(
        np.min(distances)
    )


def min_tx_rx_element_distance(
    geometry: GeometryState,
) -> float:
    """返回任意 Tx/Rx 阵元对之间的最小距离。"""

    difference = (
        geometry.rx_positions[
            :,
            None,
            :,
        ]
        - geometry.tx_positions[
            None,
            :,
            :,
        ]
    )

    distances = np.linalg.norm(
        difference,
        axis=2,
    )

    return float(
        np.min(distances)
    )