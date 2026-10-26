from __future__ import annotations


# ===== config: shared implementation =====


from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SystemConfig:
    # 系统规模
    m_uav: int
    n_cable: int
    k_dl: int
    j_ul: int
    l_target: int
    r_sensing: int

    # 几何参数
    wavelength: float
    cable_length: float
    min_uav_distance: float
    min_si_distance: float

    # 功率和噪声
    p_dl_max: float
    p_ul_max: np.ndarray
    sigma_dl2: np.ndarray
    sigma_bs2: float
    rho_si: float

    # WSR 权重
    weight_dl: np.ndarray
    weight_ul: np.ndarray
    weight_s: np.ndarray

    # 用户和目标位置
    q_dl: np.ndarray
    q_ul: np.ndarray
    q_s: np.ndarray

    # 信道参数
    xi_s: np.ndarray

    # 场景级可选小尺度复衰落乘子。
    # 正式纯 LoS 仿真默认全 1，不需要在 YAML 中配置。
    fading_dl: np.ndarray
    fading_ul: np.ndarray
    fading_s: np.ndarray

    # 初始化参数
    kappa_ul: float
    random_seed: int
    max_geometry_trials: int

    # 后续内层和外层算法参数
    max_inner_iter: int
    max_outer_iter: int

    tol_inner: float
    tol_outer: float

    max_bisection_iter: int
    tol_bisection: float

    max_backtracking_iter: int
    armijo_mu: float
    backtracking_factor: float

    finite_diff_delta: float
    initial_outer_step: float

    # 联合 AO 的性能平台停止条件。0 表示关闭。
    plateau_window: int = 0
    tol_window: float = 0.0

    @property
    def n_tx_uav(self) -> int:
        return self.m_uav // 2

    @property
    def n_rx_uav(self) -> int:
        return self.m_uav // 2

    @property
    def n_tx(self) -> int:
        return self.n_tx_uav * self.n_cable

    @property
    def n_rx(self) -> int:
        return self.n_rx_uav * self.n_cable

    @property
    def d_stream(self) -> int:
        return self.k_dl + self.r_sensing

    @property
    def k0(self) -> float:
        return 2.0 * np.pi / self.wavelength

    def validate(self) -> None:
        if self.m_uav <= 0 or self.m_uav % 2 != 0:
            raise ValueError(
                "M must be a positive even integer."
            )

        if self.n_cable <= 0:
            raise ValueError("N_c must be positive.")

        if min(
            self.k_dl,
            self.j_ul,
            self.l_target,
            self.r_sensing,
        ) <= 0:
            raise ValueError(
                "K, J, L and R_s must be positive."
            )

        if self.wavelength <= 0:
            raise ValueError("wavelength must be positive.")

        if self.cable_length <= 0:
            raise ValueError(
                "cable_length must be positive."
            )

        if self.p_dl_max <= 0:
            raise ValueError("p_dl_max must be positive.")

        if np.any(self.p_ul_max < 0):
            raise ValueError(
                "p_ul_max must be nonnegative."
            )

        if np.any(self.sigma_dl2 <= 0):
            raise ValueError(
                "sigma_dl2 must be positive."
            )

        if self.sigma_bs2 <= 0:
            raise ValueError(
                "sigma_bs2 must be positive."
            )

        if not 0.0 <= self.rho_si <= 1.0:
            raise ValueError(
                "rho_si must lie in [0, 1]."
            )

        if not 0.0 < self.kappa_ul < 1.0:
            raise ValueError(
                "kappa_ul must lie in (0, 1)."
            )

        _require_shape(
            self.p_ul_max,
            (self.j_ul,),
            "p_ul_max",
        )

        _require_shape(
            self.sigma_dl2,
            (self.k_dl,),
            "sigma_dl2",
        )

        _require_shape(
            self.q_dl,
            (self.k_dl, 3),
            "q_dl",
        )

        _require_shape(
            self.q_ul,
            (self.j_ul, 3),
            "q_ul",
        )

        _require_shape(
            self.q_s,
            (self.l_target, 3),
            "q_s",
        )

        _require_shape(
            self.xi_s,
            (self.l_target,),
            "xi_s",
        )

        _require_shape(
            self.fading_dl,
            (self.k_dl,),
            "fading_dl",
        )

        _require_shape(
            self.fading_ul,
            (self.j_ul,),
            "fading_ul",
        )

        _require_shape(
            self.fading_s,
            (self.l_target,),
            "fading_s",
        )

        all_weights = np.concatenate(
            [
                self.weight_dl,
                self.weight_ul,
                self.weight_s,
            ]
        )

        if np.any(all_weights < 0):
            raise ValueError(
                "WSR weights must be nonnegative."
            )

        if not np.isclose(
            np.sum(all_weights),
            1.0,
            atol=1e-12,
        ):
            raise ValueError(
                "All WSR weights must sum to 1."
            )
        if self.initial_outer_step <= 0:
            raise ValueError(
                "initial_outer_step must be positive."
            )

        if self.plateau_window < 0:
            raise ValueError(
                "plateau_window must be nonnegative."
            )

        if self.tol_window < 0.0:
            raise ValueError(
                "tol_window must be nonnegative."
            )

        plateau_enabled = (
            self.plateau_window > 0
            or self.tol_window > 0.0
        )

        if plateau_enabled:
            if self.plateau_window < 2:
                raise ValueError(
                    "plateau_window must be at least 2 "
                    "when plateau stopping is enabled."
                )
            if self.tol_window <= 0.0:
                raise ValueError(
                    "tol_window must be positive "
                    "when plateau stopping is enabled."
                )


def _require_shape(
    array: np.ndarray,
    shape: tuple[int, ...],
    name: str,
) -> None:
    if array.shape != shape:
        raise ValueError(
            f"{name}.shape={array.shape}, "
            f"expected {shape}."
        )


def _load_sensing_scattering(raw: dict) -> np.ndarray:
    """读取目标幅度散射系数。

    推荐配置 channel.rcs_m2: [A_1, ..., A_L]，其中 A_l 是目标 RCS (m^2)。
    按标准点目标双基地雷达方程，xi_l = sqrt(4*pi*A_l) / wavelength。
    不单独引入目标散射相位；旧配置 channel.xi_s 保持兼容。
    """
    channel = raw["channel"]
    has_rcs = "rcs_m2" in channel
    has_xi = "xi_s" in channel

    if has_rcs and has_xi:
        raise ValueError(
            "channel.rcs_m2 and channel.xi_s are mutually exclusive."
        )
    if not has_rcs:
        if not has_xi:
            raise ValueError("channel must contain rcs_m2 or xi_s.")
        return np.asarray(channel["xi_s"], dtype=np.complex128)

    rcs = np.asarray(channel["rcs_m2"], dtype=float)
    expected_shape = (int(raw["system"]["L"]),)
    if rcs.shape != expected_shape:
        raise ValueError(
            f"channel.rcs_m2.shape={rcs.shape}, "
            f"expected {expected_shape}."
        )
    if not np.all(np.isfinite(rcs)) or np.any(rcs < 0.0):
        raise ValueError("channel.rcs_m2 must be finite and nonnegative.")

    wavelength = float(raw["geometry"]["wavelength"])
    if not np.isfinite(wavelength) or wavelength <= 0.0:
        raise ValueError("geometry.wavelength must be finite and positive.")

    return np.asarray(
        np.sqrt(4.0 * np.pi * rcs) / wavelength,
        dtype=np.complex128,
    )


def load_config(
    path: str | Path | None = None,
) -> SystemConfig:
    if isinstance(path, dict):
        raw = path
    else:
        import json
        config_path = Path(path) if path is not None else Path(__file__).with_name('simulation_config.json')
        raw = json.loads(config_path.read_text(encoding='utf-8'))
        if 'models' in raw:
            raw = raw['models']['stage8_monte_carlo']

    cfg = SystemConfig(
        m_uav=int(raw["system"]["M"]),
        n_cable=int(raw["system"]["N_c"]),
        k_dl=int(raw["system"]["K"]),
        j_ul=int(raw["system"]["J"]),
        l_target=int(raw["system"]["L"]),
        r_sensing=int(raw["system"]["R_s"]),

        wavelength=float(
            raw["geometry"]["wavelength"]
        ),
        cable_length=float(
            raw["geometry"]["cable_length"]
        ),
        min_uav_distance=float(
            raw["geometry"]["min_uav_distance"]
        ),
        min_si_distance=float(
            raw["geometry"]["min_si_distance"]
        ),

        p_dl_max=float(
            raw["power"]["p_dl_max"]
        ),
        p_ul_max=np.asarray(
            raw["power"]["p_ul_max"],
            dtype=float,
        ),

        sigma_dl2=np.asarray(
            raw["noise"]["sigma_dl2"],
            dtype=float,
        ),
        sigma_bs2=float(
            raw["noise"]["sigma_bs2"]
        ),

        rho_si=float(
            raw["si"]["rho_si"]
        ),

        weight_dl=np.asarray(
            raw["weights"]["dl"],
            dtype=float,
        ),
        weight_ul=np.asarray(
            raw["weights"]["ul"],
            dtype=float,
        ),
        weight_s=np.asarray(
            raw["weights"]["sensing"],
            dtype=float,
        ),

        q_dl=np.asarray(
            raw["scene"]["q_dl"],
            dtype=float,
        ),
        q_ul=np.asarray(
            raw["scene"]["q_ul"],
            dtype=float,
        ),
        q_s=np.asarray(
            raw["scene"]["q_s"],
            dtype=float,
        ),

        xi_s=_load_sensing_scattering(raw),

        # 正式模型中的路径损耗由 channels.py 根据 q 与 lambda 实时计算。
        # 这里仅初始化可选 scene-level 小尺度衰落乘子。
        fading_dl=np.ones(
            int(raw["system"]["K"]),
            dtype=np.complex128,
        ),
        fading_ul=np.ones(
            int(raw["system"]["J"]),
            dtype=np.complex128,
        ),
        fading_s=np.ones(
            int(raw["system"]["L"]),
            dtype=np.complex128,
        ),

        kappa_ul=float(
            raw["initialization"]["kappa_ul"]
        ),
        random_seed=int(
            raw["initialization"]["random_seed"]
        ),
        max_geometry_trials=int(
            raw["initialization"][
                "max_geometry_trials"
            ]
        ),

        max_inner_iter=int(
            raw["algorithm"]["max_inner_iter"]
        ),
        max_outer_iter=int(
            raw["algorithm"]["max_outer_iter"]
        ),

        tol_inner=float(
            raw["algorithm"]["tol_inner"]
        ),
        tol_outer=float(
            raw["algorithm"]["tol_outer"]
        ),

        max_bisection_iter=int(
            raw["algorithm"][
                "max_bisection_iter"
            ]
        ),
        tol_bisection=float(
            raw["algorithm"]["tol_bisection"]
        ),

        max_backtracking_iter=int(
            raw["algorithm"][
                "max_backtracking_iter"
            ]
        ),
        armijo_mu=float(
            raw["algorithm"]["armijo_mu"]
        ),
        backtracking_factor=float(
            raw["algorithm"][
                "backtracking_factor"
            ]
        ),

        finite_diff_delta=float(
            raw["algorithm"][
                "finite_diff_delta"
            ]
        ),
        initial_outer_step=float(
            raw["algorithm"][
                "initial_outer_step"
            ]
        ),

        plateau_window=int(
            raw["algorithm"].get(
                "plateau_window",
                0,
            )
        ),
        tol_window=float(
            raw["algorithm"].get(
                "tol_window",
                0.0,
            )
        ),
    )

    cfg.validate()

    return cfg

# ===== state: shared implementation =====
from dataclasses import dataclass

import numpy as np


@dataclass
class GeometryState:
    # c_m
    endpoints: np.ndarray

    # p_i^T
    tx_positions: np.ndarray

    # p_s^R
    rx_positions: np.ndarray


@dataclass
class ChannelState:
    # h_k
    h: np.ndarray

    # f_j
    f: np.ndarray

    # G_l
    g: np.ndarray

    # H_SI^0
    h_si0: np.ndarray

    # H_RSI
    h_rsi: np.ndarray


@dataclass
class ResourceState:
    # Q
    q_matrix: np.ndarray

    # q_j
    q_ul: np.ndarray

    # b_j
    b_ul: np.ndarray

    # u_l
    u_s: np.ndarray


@dataclass
class PerformanceResult:
    gamma_dl: np.ndarray
    gamma_ul: np.ndarray
    gamma_s: np.ndarray

    rate_dl: np.ndarray
    rate_ul: np.ndarray
    rate_s: np.ndarray

    weighted_sum_rate: float

@dataclass
class FPState:
    """内层 Fractional Programming 的辅助变量。

    这里保存的是 LDT 和 QT 为了求解原问题而引入的辅助变量。

   
    """

    # ============================================================
    # LDT 辅助变量
    #
    # eta_k^D = gamma_k^D
    # eta_j^U = gamma_j^U
    # eta_l^S = gamma_l^S
    # ============================================================

    eta_dl: np.ndarray
    # shape = (K,)
    # 对应 {eta_k^D}

    eta_ul: np.ndarray
    # shape = (J,)
    # 对应 {eta_j^U}

    eta_s: np.ndarray
    # shape = (L,)
    # 对应 {eta_l^S}

    # ============================================================
    # QT 辅助变量
    # ============================================================

    phi_dl: np.ndarray
    # shape = (K,)
    # complex
    #
    # 对应标量：
    # phi_k^D

    phi_ul: np.ndarray
    # shape = (J,)
    # complex
    #
    # 对应标量：
    # phi_j^U

    phi_s: np.ndarray
    # shape = (L, D)
    # complex
    #
    # 每一行 phi_s[l]
    # 对应向量：
    # boldsymbol phi_l^S ∈ C^D

# ===== utils: shared implementation =====
import numpy as np

pass
pass


def normalize_vector(
    x: np.ndarray,
    eps: float = 1e-14,
) -> np.ndarray:
    """向量单位范数归一化。"""

    norm = np.linalg.norm(x)

    if norm <= eps:
        raise ValueError(
            "Cannot normalize a near-zero vector."
        )

    return x / norm


def hermitianize(
    a: np.ndarray,
) -> np.ndarray:
    """消除数值误差产生的非 Hermitian 部分。"""

    return 0.5 * (a + a.conj().T)


def relative_change(
    new: float,
    old: float,
) -> float:
    """以后内外层算法的相对停止判据。"""

    return abs(new - old) / max(
        1.0,
        abs(old),
    )


def check_finite(
    x: np.ndarray | float,
    name: str,
) -> None:
    """检查 NaN 和 Inf。"""

    if not np.all(np.isfinite(x)):
        raise ValueError(
            f"{name} contains NaN or Inf."
        )


def check_resource_constraints(
    resources: ResourceState,
    cfg: SystemConfig,
    atol: float = 1e-8,
) -> None:
    """检查当前资源变量是否合法。"""

    q_matrix = resources.q_matrix
    q_ul = resources.q_ul
    b_ul = resources.b_ul
    u_s = resources.u_s

    if q_matrix.shape != (
        cfg.n_tx,
        cfg.d_stream,
    ):
        raise ValueError(
            "Q dimension is incorrect."
        )

    if q_ul.shape != (cfg.j_ul,):
        raise ValueError(
            "q_ul dimension is incorrect."
        )

    if b_ul.shape != (
        cfg.j_ul,
        cfg.n_rx,
    ):
        raise ValueError(
            "b_ul dimension is incorrect."
        )

    if u_s.shape != (
        cfg.l_target,
        cfg.n_rx,
    ):
        raise ValueError(
            "u_s dimension is incorrect."
        )

    tx_power = (
        np.linalg.norm(q_matrix, "fro") ** 2
    )

    if tx_power > cfg.p_dl_max + atol:
        raise ValueError(
            "Transmit power constraint violated."
        )

    if np.any(q_ul < -atol):
        raise ValueError(
            "Negative UL amplitude."
        )

    if np.any(
        q_ul
        > np.sqrt(cfg.p_ul_max) + atol
    ):
        raise ValueError(
            "UL power constraint violated."
        )

    b_norms = np.linalg.norm(
        b_ul,
        axis=1,
    )

    u_norms = np.linalg.norm(
        u_s,
        axis=1,
    )

    if not np.allclose(
        b_norms,
        1.0,
        atol=atol,
    ):
        raise ValueError(
            "b_j is not unit norm."
        )

    if not np.allclose(
        u_norms,
        1.0,
        atol=atol,
    ):
        raise ValueError(
            "u_l is not unit norm."
        )
def split_transmit_matrix(
    q_matrix: np.ndarray,
    cfg: SystemConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """从合并发射矩阵 Q=[W,V] 中拆出 W 和 V。

    参数
    ----------
    q_matrix:
        合并发射波束矩阵 Q。
        shape = (N_T, K + R_s)

        前 K 列：
            W = [w_1, ..., w_K]

        后 R_s 列：
            V = [v_1, ..., v_Rs]

    cfg:
        系统参数。

    返回
    ----------
    w_matrix:
        下行通信波束矩阵 W。
        shape = (N_T, K)

    v_matrix:
        感知波束矩阵 V。
        shape = (N_T, R_s)
    """

    expected_shape = (
        cfg.n_tx,
        cfg.d_stream,
    )

    if q_matrix.shape != expected_shape:
        raise ValueError(
            f"Q.shape={q_matrix.shape}, "
            f"expected {expected_shape}."
        )

    # Q 的前 K 列是下行通信波束矩阵 W。
    w_matrix = q_matrix[
        :,
        : cfg.k_dl,
    ]

    # Q 的后 R_s 列是感知波束矩阵 V。
    v_matrix = q_matrix[
        :,
        cfg.k_dl :,
    ]

    return w_matrix, v_matrix
    
    

# ===== geometry: shared implementation =====
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

pass
pass


def build_geometry(
    endpoints: np.ndarray,
    cfg: SystemConfig,
    check_constraints: bool = True,
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
    if check_constraints:
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

from dataclasses import replace


def _unit_power_complex_rayleigh(
    rng: np.random.Generator,
    size: int,
) -> np.ndarray:
    """生成 E[|z|^2]=1 的复标量 Rayleigh 衰落系数。"""

    return (
        rng.standard_normal(size)
        + 1j * rng.standard_normal(size)
    ) / np.sqrt(2.0)


def make_scene_config(
    base_cfg: SystemConfig,
    scene_seed: int,
    position_jitter: np.ndarray,
    randomize_fading: bool = False,
    fading_model: str = "scalar_rayleigh",
) -> SystemConfig:
    """由 scene seed 生成位置与 scene-level 随机衰落。

    位置扰动继续使用原来的 scene_seed 随机流，从而保持 Stage 8A
    已使用场景的位置可复现性。随机衰落使用独立的确定性随机流。

    路径损耗不在本文件中计算，而由 channels.py 根据当前 q 与波长
    统一构造。当 randomize_fading=True 时这里只生成 scene-level 的
    单位功率复衰落乘子；正式纯 LoS 仿真保持关闭。
    同一 cfg 会在整个 scene 的 FP/RCG 过程中反复复用，因此 fading
    realization 不会随 outer/inner iteration 重新采样。
    """

    jitter = np.asarray(position_jitter, dtype=float)

    if jitter.shape != (3,) or np.any(jitter < 0.0):
        raise ValueError(
            "position_jitter must be a nonnegative array of shape (3,)."
        )

    position_rng = np.random.default_rng(int(scene_seed))

    def _perturb(points: np.ndarray) -> np.ndarray:
        delta = position_rng.uniform(
            low=-jitter,
            high=jitter,
            size=points.shape,
        )
        return np.asarray(points + delta, dtype=float)

    q_dl = _perturb(base_cfg.q_dl)
    q_ul = _perturb(base_cfg.q_ul)
    q_s = _perturb(base_cfg.q_s)

    # 当前场景都位于 z > 0；过大的扰动直接报错，不静默截断。
    if (
        np.any(q_dl[:, 2] <= 0.0)
        or np.any(q_ul[:, 2] <= 0.0)
        or np.any(q_s[:, 2] <= 0.0)
    ):
        raise ValueError(
            "Scene perturbation produced a non-positive z coordinate."
        )

    fading_dl = np.ones(
        base_cfg.k_dl,
        dtype=np.complex128,
    )
    fading_ul = np.ones(
        base_cfg.j_ul,
        dtype=np.complex128,
    )
    fading_s = np.ones(
        base_cfg.l_target,
        dtype=np.complex128,
    )

    if randomize_fading:
        if fading_model != "scalar_rayleigh":
            raise ValueError(
                "fading_model must be 'scalar_rayleigh'."
            )

        # 与位置随机流分离；不会改变原 scene 的位置。
        fading_seed = np.random.SeedSequence(
            [int(scene_seed), 20260826]
        )
        fading_rng = np.random.default_rng(fading_seed)

        fading_dl = _unit_power_complex_rayleigh(
            fading_rng,
            base_cfg.k_dl,
        )
        fading_ul = _unit_power_complex_rayleigh(
            fading_rng,
            base_cfg.j_ul,
        )
        fading_s = _unit_power_complex_rayleigh(
            fading_rng,
            base_cfg.l_target,
        )

    cfg = replace(
        base_cfg,
        q_dl=q_dl,
        q_ul=q_ul,
        q_s=q_s,
        fading_dl=fading_dl,
        fading_ul=fading_ul,
        fading_s=fading_s,
    )
    cfg.validate()
    return cfg
