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

from config import SystemConfig
from state import ChannelState, GeometryState


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
