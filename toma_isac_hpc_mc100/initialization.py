"""ToMA-ISAC 优化变量初始化。

本文件负责：

1. 随机生成满足几何约束的初始 ToMA 端点；
2. 使用 MRT 初始化下行通信波束 W；
3. 使用目标阵列响应初始化感知波束 V；
4. 形成 Q=[W,V]；
5. 缩放 Q，使 ||Q||_F^2 <= P_D^max；
6. 初始化上行发射幅度 q_j；
7. 为第一大里程碑临时初始化 b_j 和 u_l。

注意：
b_j 和 u_l 的这里的值只是“初始值”。

以后 inner_solver.py 会根据推导的
max-SINR / MMSE 公式重新更新它们。
"""

import numpy as np

from channels import (
    transmit_array_response,
)
from config import SystemConfig
from geometry import build_geometry
from state import (
    ChannelState,
    GeometryState,
    ResourceState,
)
from utils import (
    hermitianize,
    normalize_vector,
)


def generate_feasible_endpoints(
    cfg: SystemConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """随机生成一组满足 ToMA 几何约束的初始端点。

    方法：
    1. 随机生成 M 个三维方向；
    2. 每个方向归一化；
    3. 乘 L_c，使所有端点位于半径 L_c 球面；
    4. 调用 build_geometry() 检查防碰撞等条件；
    5. 不合法则重新随机。
    """

    for _ in range(
        cfg.max_geometry_trials
    ):

        # 随机方向。
        directions = rng.normal(
            size=(
                cfg.m_uav,
                3,
            )
        )

        norms = np.linalg.norm(
            directions,
            axis=1,
            keepdims=True,
        )

        # 极低概率出现接近 0 的随机方向，
        # 直接重新生成。
        if np.any(
            norms <= 1e-14
        ):
            continue

        # ------------------------------------------------------
        # c_m =
        # L_c * direction / ||direction||
        #
        # 因此自动满足：
        #
        # ||c_m|| = L_c
        # ------------------------------------------------------

        endpoints = (
            cfg.cable_length
            * directions
            / norms
        )

        try:
            # build_geometry 会检查：
            # - 缆绳长度
            # - 防碰撞
            # - Tx/Rx 阵元最小距离
            build_geometry(
                endpoints,
                cfg,
            )

        except ValueError:
            # 当前随机几何不合法，
            # 换一组重新生成。
            continue

        return endpoints

    raise RuntimeError(
        "Failed to generate "
        "a feasible initial APV."
    )


def initialize_resources(
    geometry: GeometryState,
    channels: ChannelState,
    cfg: SystemConfig,
    rng: np.random.Generator,
) -> ResourceState:
    """初始化 Q、q_j、b_j 和 u_l。"""

    # ==========================================================
    # 1. 下行通信波束矩阵 W_bar
    #
    # MRT：
    #
    # w_bar_k =
    # h_k / ||h_k||
    #
    # shape(W_bar) = (N_T, K)
    # ==========================================================

    w_bar = np.empty(
        (
            cfg.n_tx,
            cfg.k_dl,
        ),
        dtype=np.complex128,
    )

    for k in range(
        cfg.k_dl
    ):
        w_bar[:, k] = (
            normalize_vector(
                channels.h[k]
            )
        )

    # ==========================================================
    # 2. 感知波束矩阵 V_bar
    #
    # shape(V_bar)
    # =
    # (N_T, R_s)
    # ==========================================================

    v_bar = np.empty(
        (
            cfg.n_tx,
            cfg.r_sensing,
        ),
        dtype=np.complex128,
    )

    # 有多少个感知波束能够直接对应目标。
    n_target_beams = min(
        cfg.r_sensing,
        cfg.l_target,
    )

    # ----------------------------------------------------------
    # 前 min(R_s,L) 个感知波束：
    #
    # v_bar_r
    # =
    # a_T(q_r^S)
    # / ||a_T(q_r^S)||
    # ----------------------------------------------------------

    for r in range(
        n_target_beams
    ):
        a_t = (
            transmit_array_response(
                geometry.tx_positions,
                cfg.q_s[r],
                cfg,
            )
        )

        v_bar[:, r] = (
            normalize_vector(
                a_t
            )
        )

    # ----------------------------------------------------------
    # 如果 R_s > L，
    # 剩余的感知波束用随机复单位向量初始化。
    # ----------------------------------------------------------

    for r in range(
        n_target_beams,
        cfg.r_sensing,
    ):
        random_vector = (
            rng.normal(
                size=cfg.n_tx
            )
            + 1j
            * rng.normal(
                size=cfg.n_tx
            )
        )

        v_bar[:, r] = (
            normalize_vector(
                random_vector
            )
        )

    # ==========================================================
    # 3. 合并发射波束：
    #
    # Q_bar = [W_bar, V_bar]
    #
    # shape(Q_bar)
    # =
    # (N_T, K+R_s)
    # ==========================================================

    q_bar = np.concatenate(
        [
            w_bar,
            v_bar,
        ],
        axis=1,
    )

    q_bar_power = (
        np.linalg.norm(
            q_bar,
            "fro",
        )
        ** 2
    )

    if (
        q_bar_power
        <= 1e-14
    ):
        raise ValueError(
            "Initial Q_bar "
            "has near-zero power."
        )

    # ==========================================================
    # 4. 总发射功率缩放
    #
    # chi_0 =
    # min{
    #     1,
    #     P_D^max / ||Q_bar||_F^2
    # }
    #
    # Q^(0)
    # =
    # sqrt(chi_0) Q_bar
    # ==========================================================

    chi_0 = min(
        1.0,
        cfg.p_dl_max
        / q_bar_power,
    )

    q_matrix = (
        np.sqrt(
            chi_0
        )
        * q_bar
    )

    # ==========================================================
    # 5. 上行发射幅度
    #
    # q_j^(0)
    # =
    # kappa_U
    # sqrt(P_U,j^max)
    #
    # 注意：
    # q_ul 存的是“幅度” q_j，
    # 不是功率 p_j。
    #
    # 实际 UL 功率：
    # p_j = q_ul[j]^2
    # ==========================================================

    q_ul = (
        cfg.kappa_ul
        * np.sqrt(
            cfg.p_ul_max
        )
    )

    # ==========================================================
    # 6. UL 接收波束 b_j 初值
    #
    # 第一阶段为了能够计算 UL SINR，
    # 暂时采用：
    #
    # b_j =
    # f_j / ||f_j||
    #
    # 后面 inner_solver.py 会覆盖。
    # ==========================================================

    b_ul = np.empty(
        (
            cfg.j_ul,
            cfg.n_rx,
        ),
        dtype=np.complex128,
    )

    for j in range(
        cfg.j_ul
    ):
        b_ul[j] = (
            normalize_vector(
                channels.f[j]
            )
        )

    # ==========================================================
    # 7. 感知接收波束 u_l 初值
    #
    # 第一阶段采用期望目标回波协方差
    # C_l =
    # G_l Q Q^H G_l^H
    #
    # 的最大特征向量。
    #
    # 注意：
    # 真正 inner_solver.py 中，
    # 会加入干扰协方差 D_l，
    # 使用广义特征向量求 max-SINR u_l。
    # ==========================================================

    u_s = np.empty(
        (
            cfg.l_target,
            cfg.n_rx,
        ),
        dtype=np.complex128,
    )

    # Q Q^H
    q_q_h = (
        q_matrix
        @ q_matrix.conj().T
    )

    for ell in range(
        cfg.l_target
    ):
        g_l = channels.g[ell]

        c_desired = (
            g_l
            @ q_q_h
            @ g_l.conj().T
        )

        # 理论上是 Hermitian，
        # 这里消除浮点误差。
        c_desired = (
            hermitianize(
                c_desired
            )
        )

        _, eigenvectors = (
            np.linalg.eigh(
                c_desired
            )
        )

        # eigh 返回特征值升序，
        # 最后一列对应最大特征值。
        candidate = (
            eigenvectors[:, -1]
        )

        # 极端情况下提供数值保护。
        if (
            np.linalg.norm(
                candidate
            )
            <= 1e-14
        ):
            candidate = (
                rng.normal(
                    size=cfg.n_rx
                )
                + 1j
                * rng.normal(
                    size=cfg.n_rx
                )
            )

        u_s[ell] = (
            normalize_vector(
                candidate
            )
        )

    return ResourceState(
        q_matrix=q_matrix,
        q_ul=q_ul,
        b_ul=b_ul,
        u_s=u_s,
    )