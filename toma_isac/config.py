from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


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
    alpha_dl: np.ndarray
    alpha_ul: np.ndarray
    xi_s: np.ndarray
    beta_t_s: np.ndarray
    beta_r_s: np.ndarray

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
            self.alpha_dl,
            (self.k_dl,),
            "alpha_dl",
        )

        _require_shape(
            self.alpha_ul,
            (self.j_ul,),
            "alpha_ul",
        )

        _require_shape(
            self.xi_s,
            (self.l_target,),
            "xi_s",
        )

        _require_shape(
            self.beta_t_s,
            (self.l_target,),
            "beta_t_s",
        )

        _require_shape(
            self.beta_r_s,
            (self.l_target,),
            "beta_r_s",
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


def load_config(
    path: str | Path | None = None,
) -> SystemConfig:
    if path is None:
        path = (
            Path(__file__).resolve().parent
            / "configs"
            / "default.yaml"
        )
    else:
        path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        raw = yaml.safe_load(file)

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

        alpha_dl=np.asarray(
            raw["channel"]["alpha_dl"],
            dtype=np.complex128,
        ),
        alpha_ul=np.asarray(
            raw["channel"]["alpha_ul"],
            dtype=np.complex128,
        ),
        xi_s=np.asarray(
            raw["channel"]["xi_s"],
            dtype=np.complex128,
        ),
        beta_t_s=np.asarray(
            raw["channel"]["beta_t_s"],
            dtype=float,
        ),
        beta_r_s=np.asarray(
            raw["channel"]["beta_r_s"],
            dtype=float,
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
    )

    cfg.validate()

    return cfg