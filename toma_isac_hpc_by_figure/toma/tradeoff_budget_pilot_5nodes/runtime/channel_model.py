from __future__ import annotations


# ===== channels: shared implementation =====
"""ToMA 全双工近场 ISAC 信道模型。

本文件根据当前 GeometryState 构造：

h       -> 下行信道 h_k
f       -> 上行信道 f_j
g       -> 目标双程回波矩阵 G_l
h_si0   -> 未消除直接 SI 信道 H_SI^0
h_rsi   -> 等效残余 SI 信道 H_RSI

通信和感知链路采用 USW：公共幅度路径增益由公共锚点（原点）
到用户/目标的参考距离决定，逐阵元精确距离只用于球面波相位。
"""

import numpy as np

from system_model import SystemConfig
from system_model import ChannelState, GeometryState


def transmit_array_response(
    tx_positions: np.ndarray,
    q: np.ndarray,
    cfg: SystemConfig,
) -> np.ndarray:
    """计算发射阵列的 USW 球面波响应 a_T。"""

    distances = np.linalg.norm(
        tx_positions - q[None, :],
        axis=1,
    )

    return np.exp(1j * cfg.k0 * distances)


def receive_array_response(
    rx_positions: np.ndarray,
    q: np.ndarray,
    cfg: SystemConfig,
) -> np.ndarray:
    """计算接收阵列的 USW 球面波响应 a_R。"""

    distances = np.linalg.norm(
        rx_positions - q[None, :],
        axis=1,
    )

    return np.exp(1j * cfg.k0 * distances)


def free_space_amplitude_gain(
    q: np.ndarray,
    cfg: SystemConfig,
) -> float:
    """公共锚点到位置 q 的自由空间单程幅度增益 lambda/(4*pi*r)。"""

    distance = float(np.linalg.norm(q))

    if distance <= 0.0:
        raise ValueError(
            "User/target distance to the common anchor must be positive."
        )

    return float(
        cfg.wavelength
        / (4.0 * np.pi * distance)
    )


def communication_path_gains(
    cfg: SystemConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """返回当前场景 DL/UL 的公共复幅度路径增益。"""

    alpha_dl = np.asarray(
        [free_space_amplitude_gain(q, cfg) for q in cfg.q_dl],
        dtype=np.complex128,
    ) * cfg.fading_dl

    alpha_ul = np.asarray(
        [free_space_amplitude_gain(q, cfg) for q in cfg.q_ul],
        dtype=np.complex128,
    ) * cfg.fading_ul

    return alpha_dl, alpha_ul


def sensing_path_gains(
    cfg: SystemConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回感知单程功率增益 beta_T/beta_R 与双程幅度系数 alpha_S。

    当前模型中收发阵列共用位于原点的公共锚点，因此
    r_T,l^S = r_R,l^S = ||q_l^S||_2。
    """

    one_way_amp = np.asarray(
        [free_space_amplitude_gain(q, cfg) for q in cfg.q_s],
        dtype=float,
    )

    beta_t_s = one_way_amp ** 2
    beta_r_s = one_way_amp ** 2

    alpha_s = (
        cfg.xi_s
        * cfg.fading_s
        * np.sqrt(beta_t_s * beta_r_s)
    )

    return beta_t_s, beta_r_s, alpha_s


def build_channels(
    geometry: GeometryState,
    cfg: SystemConfig,
) -> ChannelState:
    """根据当前 ToMA 几何构造所有通信、感知和 SI 信道。"""

    h = np.empty(
        (cfg.k_dl, cfg.n_tx),
        dtype=np.complex128,
    )
    f = np.empty(
        (cfg.j_ul, cfg.n_rx),
        dtype=np.complex128,
    )
    g = np.empty(
        (cfg.l_target, cfg.n_rx, cfg.n_tx),
        dtype=np.complex128,
    )

    # 公共路径增益由用户/目标到公共锚点的参考距离决定。
    alpha_dl, alpha_ul = communication_path_gains(cfg)
    _, _, alpha_s = sensing_path_gains(cfg)

    # 1. 下行信道 h_k = alpha_k^D a_T
    for k in range(cfg.k_dl):
        a_t = transmit_array_response(
            geometry.tx_positions,
            cfg.q_dl[k],
            cfg,
        )
        h[k] = alpha_dl[k] * a_t

    # 2. 上行信道 f_j = alpha_j^U a_R
    for j in range(cfg.j_ul):
        a_r = receive_array_response(
            geometry.rx_positions,
            cfg.q_ul[j],
            cfg,
        )
        f[j] = alpha_ul[j] * a_r

    # 3. 目标双程回波 G_l = alpha_l^S a_R^* a_T^H
    for ell in range(cfg.l_target):
        a_t = transmit_array_response(
            geometry.tx_positions,
            cfg.q_s[ell],
            cfg,
        )
        a_r = receive_array_response(
            geometry.rx_positions,
            cfg.q_s[ell],
            cfg,
        )

        g[ell] = (
            alpha_s[ell]
            * np.outer(
                a_r.conj(),
                a_t.conj(),
            )
        )

    # 4. 直接 SI 阵元距离矩阵
    difference = (
        geometry.rx_positions[:, None, :]
        - geometry.tx_positions[None, :, :]
    )
    distance = np.linalg.norm(difference, axis=2)

    if np.min(distance) < cfg.min_si_distance:
        raise ValueError(
            "Direct SI distance is too small."
        )

    # 5. 未消除直接 SI
    h_si0 = (
        cfg.wavelength
        / (4.0 * np.pi * distance)
        * np.exp(-1j * cfg.k0 * distance)
    )

    # 6. 等效残余 SI：rho_SI 是功率比例。
    h_rsi = np.sqrt(cfg.rho_si) * h_si0

    channels = ChannelState(
        h=h,
        f=f,
        g=g,
        h_si0=h_si0,
        h_rsi=h_rsi,
    )

    check_channel_shapes(channels, cfg)
    return channels


def check_channel_shapes(
    channels: ChannelState,
    cfg: SystemConfig,
) -> None:
    """检查所有信道的尺寸以及是否出现 NaN/Inf。"""

    expected_shapes = {
        "h": (cfg.k_dl, cfg.n_tx),
        "f": (cfg.j_ul, cfg.n_rx),
        "g": (cfg.l_target, cfg.n_rx, cfg.n_tx),
        "h_si0": (cfg.n_rx, cfg.n_tx),
        "h_rsi": (cfg.n_rx, cfg.n_tx),
    }

    for name, expected_shape in expected_shapes.items():
        array = getattr(channels, name)

        if array.shape != expected_shape:
            raise ValueError(
                f"{name}.shape={array.shape}, expected {expected_shape}."
            )

        if not np.all(np.isfinite(array)):
            raise ValueError(
                f"{name} contains NaN or Inf."
            )


# ===== metrics: shared implementation =====
"""计算原始 ToMA-ISAC 问题的真实性能指标。

本文件只计算：

1. DL SINR
2. UL SINR
3. Sensing SINR
4. 三类 Rate
5. Weighted Sum Rate

非常重要：

以后 inner_solver.py 中会出现
LDT/QT 转换后的辅助目标。

那些“辅助目标”不要写进这里。

这里始终只计算原始联合优化问题中的
真实 SINR、Rate 和 WSR。

这样以后：
- 内层收敛检查
- 外层 Armijo 检查
- 最终仿真

都使用同一套真实目标函数。
"""

import numpy as np

from system_model import SystemConfig
from system_model import ChannelState, PerformanceResult, ResourceState
from system_model import check_finite, hermitianize, split_transmit_matrix


def build_uplink_interference_covariance(
    channels: ChannelState,
    q_matrix: np.ndarray,
) -> np.ndarray:
    """构造 UL 的回波与残余 SI 干扰协方差 C_U。

    数学模型：

    C_U
    =
    sum_l G_l R G_l^H
    + H_RSI R H_RSI^H,

    其中 R = Q Q^H。各分量在功率/协方差域累加，
    不包含 echo--echo 或 echo--RSI 交叉项。

    返回：
        shape = (N_R, N_R)
    """

    transmit_covariance = (
        q_matrix
        @ q_matrix.conj().T
    )

    covariance = (
        channels.h_rsi
        @ transmit_covariance
        @ channels.h_rsi.conj().T
    )

    for g_l in channels.g:
        covariance += (
            g_l
            @ transmit_covariance
            @ g_l.conj().T
        )

    return hermitianize(covariance)


def build_sensing_interference_covariance(
    channels: ChannelState,
    q_matrix: np.ndarray,
    ell: int,
) -> np.ndarray:
    """构造目标 ell 的其他回波与残余 SI 协方差 C_S,l。

    数学模型：

    C_S,l
    =
    sum_{i != l} G_i R G_i^H
    + H_RSI R H_RSI^H,

    其中 R = Q Q^H。各分量在功率/协方差域累加，
    不包含 echo--echo 或 echo--RSI 交叉项。

    返回：
        shape = (N_R, N_R)
    """

    transmit_covariance = (
        q_matrix
        @ q_matrix.conj().T
    )

    covariance = (
        channels.h_rsi
        @ transmit_covariance
        @ channels.h_rsi.conj().T
    )

    for index, g_i in enumerate(channels.g):
        if index == ell:
            continue

        covariance += (
            g_i
            @ transmit_covariance
            @ g_i.conj().T
        )

    return hermitianize(covariance)


def build_transmit_interference_gram(
    channels: ChannelState,
    combiner: np.ndarray,
    excluded_target: int | None = None,
) -> np.ndarray:
    """构造 Q 子问题中的无交叉项发射侧干扰 Gram 矩阵。

    返回

    sum_i G_i^H c c^H G_i
    + H_RSI^H c c^H H_RSI，

    其中 ``excluded_target`` 用于感知目标的干扰集合。
    """

    gram = np.outer(
        channels.h_rsi.conj().T @ combiner,
        (channels.h_rsi.conj().T @ combiner).conj(),
    )

    for index, g_i in enumerate(channels.g):
        if index == excluded_target:
            continue

        projected = g_i.conj().T @ combiner
        gram += np.outer(
            projected,
            projected.conj(),
        )

    return hermitianize(gram)


def compute_downlink_sinr(
    channels: ChannelState,
    resources: ResourceState,
    cfg: SystemConfig,
) -> np.ndarray:
    """计算全部下行用户的 SINR gamma_k^D。"""

    # Q = [W,V]
    w_matrix, v_matrix = (
        split_transmit_matrix(
            resources.q_matrix,
            cfg,
        )
    )

    gamma_dl = np.empty(
        cfg.k_dl,
        dtype=float,
    )

    for k in range(
        cfg.k_dl
    ):
        h_k = channels.h[k]

        # ======================================================
        # 期望 DL 信号功率：
        #
        # |h_k^H w_k|^2
        #
        # np.vdot(a,b)
        # =
        # a^H b
        # ======================================================

        desired_power = (
            np.abs(
                np.vdot(
                    h_k,
                    w_matrix[:, k],
                )
            )
            ** 2
        )

        # ======================================================
        # 其他下行用户干扰：
        #
        # sum_{i != k}
        # |h_k^H w_i|^2
        # ======================================================

        dl_interference = 0.0

        for i in range(
            cfg.k_dl
        ):
            if i == k:
                continue

            dl_interference += (
                np.abs(
                    np.vdot(
                        h_k,
                        w_matrix[:, i],
                    )
                )
                ** 2
            )

        # ======================================================
        # 感知波束对 DL 的干扰：
        #
        # ||h_k^H V||_2^2
        # ======================================================

        sensing_interference = (
            np.linalg.norm(
                h_k.conj()
                @ v_matrix
            )
            ** 2
        )

        denominator = (
            dl_interference
            + sensing_interference
            + cfg.sigma_dl2[k]
        )

        gamma_dl[k] = float(
            np.real(
                desired_power
                / denominator
            )
        )

    return gamma_dl


def compute_uplink_sinr(
    channels: ChannelState,
    resources: ResourceState,
    cfg: SystemConfig,
) -> np.ndarray:
    """计算全部上行用户 SINR gamma_j^U。"""

    gamma_ul = np.empty(
        cfg.j_ul,
        dtype=float,
    )

    # C_U：各目标回波与残余 SI 的协方差分别累加。
    c_u = build_uplink_interference_covariance(
        channels,
        resources.q_matrix,
    )

    for j in range(
        cfg.j_ul
    ):
        b_j = resources.b_ul[j]

        # ======================================================
        # 期望 UL 信号功率：
        #
        # q_j^2
        # |b_j^H f_j|^2
        #
        # q_ul[j]^2 才是真正 UL 功率。
        # ======================================================

        desired_power = (
            resources.q_ul[j]
            ** 2
            * np.abs(
                np.vdot(
                    b_j,
                    channels.f[j],
                )
            )
            ** 2
        )

        # ======================================================
        # 其他 UL 用户干扰：
        #
        # sum_{i != j}
        # q_i^2
        # |b_j^H f_i|^2
        # ======================================================

        ul_interference = 0.0

        for i in range(
            cfg.j_ul
        ):
            if i == j:
                continue

            ul_interference += (
                resources.q_ul[i]
                ** 2
                * np.abs(
                    np.vdot(
                        b_j,
                        channels.f[i],
                    )
                )
                ** 2
            )

        # ======================================================
        # 感知回波 + RSI 的无交叉项协方差功率：
        #
        # b_j^H C_U b_j
        # ======================================================

        fd_interference = (
            np.real(
                np.vdot(
                    b_j,
                    c_u @ b_j,
                )
            )
        )

        # 因为 ||b_j||=1，
        # 合并后的噪声功率就是 sigma_BS^2。
        denominator = (
            ul_interference
            + fd_interference
            + cfg.sigma_bs2
        )

        gamma_ul[j] = float(
            np.real(
                desired_power
                / denominator
            )
        )

    return gamma_ul


def compute_sensing_sinr(
    channels: ChannelState,
    resources: ResourceState,
    cfg: SystemConfig,
) -> np.ndarray:
    """计算全部感知目标 SINR gamma_l^S。"""

    gamma_s = np.empty(
        cfg.l_target,
        dtype=float,
    )

    for ell in range(
        cfg.l_target
    ):
        u_l = resources.u_s[ell]

        # C_S,l：其他目标回波与残余 SI 的协方差分别累加。
        c_s = build_sensing_interference_covariance(
            channels,
            resources.q_matrix,
            ell,
        )

        # ======================================================
        # 目标 ell 的期望回波功率：
        #
        # ||u_l^H G_l Q||_2^2
        # ======================================================

        desired_power = (
            np.linalg.norm(
                u_l.conj()
                @ channels.g[ell]
                @ resources.q_matrix
            )
            ** 2
        )

        # ======================================================
        # UL 用户对 sensing 的干扰：
        #
        # sum_j
        # q_j^2
        # |u_l^H f_j|^2
        # ======================================================

        ul_interference = 0.0

        for j in range(
            cfg.j_ul
        ):
            ul_interference += (
                resources.q_ul[j]
                ** 2
                * np.abs(
                    np.vdot(
                        u_l,
                        channels.f[j],
                    )
                )
                ** 2
            )

        # ======================================================
        # 其他目标回波 + RSI 的无交叉项协方差功率：
        #
        # u_l^H C_S,l u_l
        # ======================================================

        other_echo_and_si = (
            np.real(
                np.vdot(
                    u_l,
                    c_s @ u_l,
                )
            )
        )

        denominator = (
            ul_interference
            + other_echo_and_si
            + cfg.sigma_bs2
        )

        gamma_s[ell] = float(
            np.real(
                desired_power
                / denominator
            )
        )

    return gamma_s


def compute_performance(
    channels: ChannelState,
    resources: ResourceState,
    cfg: SystemConfig,
) -> PerformanceResult:
    """统一计算当前系统真实 SINR、Rate 和 WSR。"""

    gamma_dl = (
        compute_downlink_sinr(
            channels,
            resources,
            cfg,
        )
    )

    gamma_ul = (
        compute_uplink_sinr(
            channels,
            resources,
            cfg,
        )
    )

    gamma_s = (
        compute_sensing_sinr(
            channels,
            resources,
            cfg,
        )
    )

    # ==========================================================
    # SINR 理论上均应 >= 0。
    #
    # 极小负数可能来自浮点误差。
    # ==========================================================

    for (
        name,
        gamma,
    ) in (
        ("gamma_dl", gamma_dl),
        ("gamma_ul", gamma_ul),
        ("gamma_s", gamma_s),
    ):
        check_finite(
            gamma,
            name,
        )

        if np.any(
            gamma < -1e-12
        ):
            raise ValueError(
                f"{name} contains "
                "invalid negative values."
            )

    # 仅消除数值级别的极小负数。
    gamma_dl = np.maximum(
        gamma_dl,
        0.0,
    )

    gamma_ul = np.maximum(
        gamma_ul,
        0.0,
    )

    gamma_s = np.maximum(
        gamma_s,
        0.0,
    )

    # ==========================================================
    # R = log2(1 + gamma)
    # ==========================================================

    rate_dl = np.log2(
        1.0 + gamma_dl
    )

    rate_ul = np.log2(
        1.0 + gamma_ul
    )

    rate_s = np.log2(
        1.0 + gamma_s
    )

    # ==========================================================
    # Weighted Sum Rate
    #
    # F =
    # sum_k omega_k^D R_k^D
    # +
    # sum_j omega_j^U R_j^U
    # +
    # sum_l omega_l^S R_l^S
    # ==========================================================

    weighted_sum_rate = float(
        cfg.weight_dl
        @ rate_dl

        + cfg.weight_ul
        @ rate_ul

        + cfg.weight_s
        @ rate_s
    )

    check_finite(
        weighted_sum_rate,
        "weighted_sum_rate",
    )

    return PerformanceResult(
        gamma_dl=gamma_dl,
        gamma_ul=gamma_ul,
        gamma_s=gamma_s,

        rate_dl=rate_dl,
        rate_ul=rate_ul,
        rate_s=rate_s,

        weighted_sum_rate=(
            weighted_sum_rate
        ),
    )


# ===== initialization: shared implementation =====
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

pass
from system_model import SystemConfig
from system_model import build_geometry
from system_model import ChannelState, GeometryState, ResourceState
from system_model import hermitianize, normalize_vector


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