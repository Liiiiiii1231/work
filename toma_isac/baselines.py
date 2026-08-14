"""Stage 6：Baseline 对比方案。

本文件实现：
B1: RCG-ToMA + Traditional Resource Design
B2: Fixed-ToMA + FP Resource Optimization
B3: FPA-UPA + FP Resource Optimization

Proposed 继续直接调用 algorithm.py 中的 run_joint_algorithm()，
因此这里不重复实现完整联合算法。

实现原则：
1. 不重复实现 SINR / Rate / WSR；
2. 不重复实现 FP；
3. 不重复实现 RCG；
4. Baseline 只重新组合现有模块。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from channels import build_channels
from config import SystemConfig
from geometry import build_geometry
from initialization import (
    generate_feasible_endpoints,
    initialize_resources,
)
from inner_solver import (
    solve_inner_problem,
    update_sensing_combiners,
    update_uplink_combiners,
)
from metrics import compute_performance
from outer_solver import (
    create_outer_rcg_state,
    update_positions_once,
)
from state import (
    ChannelState,
    GeometryState,
    PerformanceResult,
    ResourceState,
)
from utils import (
    check_resource_constraints,
    relative_change,
)


@dataclass
class BaselineResult:
    """保存一个 Stage 6 baseline 的单起点结果。"""

    name: str

    geometry: GeometryState
    channels: ChannelState
    resources: ResourceState
    performance: PerformanceResult

    wsr_history: list[float]

    converged: bool
    iterations: int

    # 保存 ToMA 方法最开始的端点，
    # 用于检查 Fixed-ToMA 和不同方案的公平初始化。
    initial_endpoints: np.ndarray | None = None


def _inner_history_converged(
    history: list[float],
    cfg: SystemConfig,
) -> bool:
    """根据 solve_inner_problem 使用的同一停止准则判断是否收敛。"""

    if len(history) < 2:
        return False

    return (
        relative_change(
            history[-1],
            history[-2],
        )
        <= cfg.tol_inner
    )


# ============================================================
# 1. Traditional Resource Design
# ============================================================

def build_traditional_resources(
    geometry: GeometryState,
    channels: ChannelState,
    cfg: SystemConfig,
) -> ResourceState:
    """构造 Baseline 1 使用的传统资源方案。

    发射侧：
    - DL: MRT；
    - sensing: target steering；
    - UL: q_j = kappa_U sqrt(P_U,j^max)。

    接收侧：
    - sensing: max-SINR combiner；
    - UL: MMSE / max-SINR combiner。

    不执行：
    - LDT；
    - QT；
    - FP Q 更新；
    - FP UL 幅度更新。
    """

    # initialize_resources() 已经实现 MRT、target steering
    # 和固定 UL 幅度。
    #
    # 使用固定局部随机种子，使 R_s > L 时的额外 sensing
    # 初始化波束在不同外层轮次中保持一致，避免 baseline
    # 因随机刷新产生额外波动。
    local_rng = np.random.default_rng(
        cfg.random_seed
    )

    resources = initialize_resources(
        geometry,
        channels,
        cfg,
        local_rng,
    )

    # 不故意弱化 baseline 的接收端。
    resources.u_s = update_sensing_combiners(
        channels,
        resources,
        cfg,
    )

    resources.b_ul = update_uplink_combiners(
        channels,
        resources,
        cfg,
    )

    check_resource_constraints(
        resources,
        cfg,
    )

    return resources


# ============================================================
# 2. Baseline 1
#    RCG-ToMA + Traditional Resource Design
# ============================================================

def run_rcg_toma_traditional(
    cfg: SystemConfig,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
    max_outer_iterations: int | None = None,
) -> BaselineResult:
    """运行 RCG-ToMA + Traditional Resource Design。

    与 Proposed 相同：
    - 使用 ToMA；
    - 使用相同 RCG 位置更新；
    - 采用位置—资源交替。

    与 Proposed 不同：
    - 资源侧不调用 solve_inner_problem()；
    - 每次 ToMA 改变后重新生成传统资源。
    """

    if rng is None:
        rng = np.random.default_rng(
            cfg.random_seed
        )

    number_of_outer_iterations = (
        cfg.max_outer_iter
        if max_outer_iterations is None
        else max_outer_iterations
    )

    if number_of_outer_iterations <= 0:
        raise ValueError(
            "max_outer_iterations must be positive."
        )

    # --------------------------------------------------------
    # 初始 ToMA
    # --------------------------------------------------------

    endpoints = generate_feasible_endpoints(
        cfg,
        rng,
    )
    # 保存 RCG 优化之前的初始 ToMA。
    initial_endpoints = endpoints.copy()

    geometry = build_geometry(
        endpoints,
        cfg,
    )

    channels = build_channels(
        geometry,
        cfg,
    )

    resources = build_traditional_resources(
        geometry,
        channels,
        cfg,
    )

    performance = compute_performance(
        channels,
        resources,
        cfg,
    )

    previous_wsr = float(
        performance.weighted_sum_rate
    )

    wsr_history = [
        previous_wsr
    ]

    # 与 Proposed 复用完全相同的 RCG 状态和位置更新函数。
    rcg_state = create_outer_rcg_state(
        cfg
    )

    consecutive_small_changes = 0

    converged = False
    iterations = 0

    # --------------------------------------------------------
    # Traditional resources <-> RCG-ToMA
    # --------------------------------------------------------

    for outer_iter in range(
        1,
        number_of_outer_iterations + 1,
    ):
        # ToMA 改变以后，h / f / G / H_RSI 都会改变，
        # 因此 traditional resources 必须重新生成。
        resources = build_traditional_resources(
            geometry,
            channels,
            cfg,
        )

        (
            geometry,
            channels,
            _,
            rcg_state,
            accepted_count,
        ) = update_positions_once(
            geometry,
            resources,
            cfg,
            rcg_state=rcg_state,
            verbose=False,
        )

        performance = compute_performance(
            channels,
            resources,
            cfg,
        )

        current_wsr = float(
            performance.weighted_sum_rate
        )

        wsr_history.append(
            current_wsr
        )

        change = relative_change(
            current_wsr,
            previous_wsr,
        )

        if verbose:
            print(
                f"B1 iteration {outer_iter}: "
                f"accepted={accepted_count}, "
                f"WSR={current_wsr:.10f}, "
                f"change={change:.3e}"
            )

        # Traditional resource refresh 不是 WSR 最优更新，
        # 因此 B1 不要求整个 history 严格单调。
        if change <= cfg.tol_outer:
            consecutive_small_changes += 1
        else:
            consecutive_small_changes = 0

        iterations = outer_iter

        if consecutive_small_changes >= 2:
            converged = True
            break

        previous_wsr = current_wsr

    check_resource_constraints(
        resources,
        cfg,
    )

    # 最终重新执行一次完整 ToMA 几何合法性检查。
    geometry = build_geometry(
        geometry.endpoints,
        cfg,
    )

    channels = build_channels(
        geometry,
        cfg,
    )

    performance = compute_performance(
        channels,
        resources,
        cfg,
    )

    return BaselineResult(
    name=(
        "RCG-ToMA + "
        "Traditional Resource Design"
    ),
    geometry=geometry,
    channels=channels,
    resources=resources,
    performance=performance,
    wsr_history=wsr_history,
    converged=converged,
    iterations=iterations,
    initial_endpoints=initial_endpoints,
)


# ============================================================
# 3. Baseline 2
#    Fixed-ToMA + FP
# ============================================================

def run_fixed_toma_fp(
    cfg: SystemConfig,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> BaselineResult:
    """固定初始 ToMA，只执行完整 FP resource optimization。"""

    if rng is None:
        rng = np.random.default_rng(
            cfg.random_seed
        )

    endpoints = generate_feasible_endpoints(
        cfg,
        rng,
    )
    # Fixed-ToMA 的初始端点之后应始终保持不变。
    initial_endpoints = endpoints.copy()
    geometry = build_geometry(
        endpoints,
        cfg,
    )

    channels = build_channels(
        geometry,
        cfg,
    )

    resources = initialize_resources(
        geometry,
        channels,
        cfg,
        rng,
    )

    resources, inner_history = solve_inner_problem(
        channels,
        resources,
        cfg,
        verbose=verbose,
    )

    check_resource_constraints(
        resources,
        cfg,
    )

    performance = compute_performance(
        channels,
        resources,
        cfg,
    )

    return BaselineResult(
    name=(
        "Fixed-ToMA + "
        "FP Resource Optimization"
    ),
    geometry=geometry,
    channels=channels,
    resources=resources,
    performance=performance,
    wsr_history=list(
        inner_history
    ),
    converged=_inner_history_converged(
        inner_history,
        cfg,
    ),
    iterations=(
        len(inner_history)
        - 1
    ),
    initial_endpoints=initial_endpoints,
)

# ============================================================
# 4. FPA-UPA baseline geometry
# ============================================================

def _factor_grid(
    n_elements: int,
) -> tuple[int, int]:
    """生成乘积为 n_elements 且尽量接近方形的二维网格。"""

    if n_elements <= 0:
        raise ValueError(
            "n_elements must be positive."
        )

    n_rows = int(
        np.floor(
            np.sqrt(n_elements)
        )
    )

    while n_rows > 1:
        if (
            n_elements
            % n_rows
            == 0
        ):
            break

        n_rows -= 1

    n_cols = (
        n_elements
        // n_rows
    )

    return (
        n_rows,
        n_cols,
    )


def _centered_upa_positions(
    n_elements: int,
    spacing: float,
    center: np.ndarray,
) -> np.ndarray:
    """在 y-z 平面生成中心化固定 UPA 坐标。"""

    n_rows, n_cols = (
        _factor_grid(
            n_elements
        )
    )

    y_offsets = (
        np.arange(
            n_rows,
            dtype=float,
        )
        - 0.5
        * (
            n_rows
            - 1
        )
    ) * spacing

    z_offsets = (
        np.arange(
            n_cols,
            dtype=float,
        )
        - 0.5
        * (
            n_cols
            - 1
        )
    ) * spacing

    positions = []

    for y_offset in y_offsets:
        for z_offset in z_offsets:
            positions.append(
                center
                + np.array(
                    [
                        0.0,
                        y_offset,
                        z_offset,
                    ],
                    dtype=float,
                )
            )

    return np.asarray(
        positions,
        dtype=float,
    )


def build_fpa_upa_geometry(
    cfg: SystemConfig,
    spacing: float | None = None,
    tx_rx_separation: float | None = None,
) -> GeometryState:
    """构造 Baseline 3 的固定 Tx/Rx UPA。

    默认：
    spacing = lambda / 2

    Tx 和 Rx UPA 的中心沿 x 轴分离：
    tx_rx_separation = cable_length

    这样做的目的：
    1. 避免 Tx/Rx 阵元重合导致直接 SI 的 1/d 奇异；
    2. 让 Tx/Rx 阵列中心的空间尺度与当前 ToMA 系统处于同一量级。

    注意：
    这是 Stage 6 第一版 dense UPA baseline。
    后续若需要控制孔径因素，可再加入 aperture-matched FPA。
    """

    if spacing is None:
        spacing = (
            0.5
            * cfg.wavelength
        )

    if tx_rx_separation is None:
        tx_rx_separation = (
            cfg.cable_length
        )

    if spacing <= 0.0:
        raise ValueError(
            "UPA spacing must be positive."
        )

    if tx_rx_separation <= 0.0:
        raise ValueError(
            "Tx/Rx UPA separation "
            "must be positive."
        )

    tx_center = np.array(
        [
            -0.5
            * tx_rx_separation,
            0.0,
            0.0,
        ],
        dtype=float,
    )

    rx_center = np.array(
        [
            0.5
            * tx_rx_separation,
            0.0,
            0.0,
        ],
        dtype=float,
    )

    tx_positions = (
        _centered_upa_positions(
            cfg.n_tx,
            spacing,
            tx_center,
        )
    )

    rx_positions = (
        _centered_upa_positions(
            cfg.n_rx,
            spacing,
            rx_center,
        )
    )

    distances = np.linalg.norm(
        rx_positions[
            :,
            None,
            :,
        ]
        - tx_positions[
            None,
            :,
            :,
        ],
        axis=2,
    )

    if (
        np.min(distances)
        < cfg.min_si_distance
    ):
        raise ValueError(
            "FPA-UPA Tx/Rx separation "
            "is too small."
        )

    # FPA-UPA 不存在 ToMA endpoint。
    # build_channels() 只使用 tx_positions 和 rx_positions，
    # 因此这里使用空 endpoints 作显式标记。
    return GeometryState(
        endpoints=np.empty(
            (
                0,
                3,
            ),
            dtype=float,
        ),
        tx_positions=tx_positions,
        rx_positions=rx_positions,
    )


# ============================================================
# 5. Baseline 3
#    FPA-UPA + FP
# ============================================================

def run_fpa_upa_fp(
    cfg: SystemConfig,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
    spacing: float | None = None,
    tx_rx_separation: float | None = None,
) -> BaselineResult:
    """固定 UPA 几何，并使用与 Proposed 相同的 FP 资源优化。"""

    if rng is None:
        rng = np.random.default_rng(
            cfg.random_seed
        )

    geometry = build_fpa_upa_geometry(
        cfg,
        spacing=spacing,
        tx_rx_separation=(
            tx_rx_separation
        ),
    )

    channels = build_channels(
        geometry,
        cfg,
    )

    resources = initialize_resources(
        geometry,
        channels,
        cfg,
        rng,
    )

    resources, inner_history = solve_inner_problem(
        channels,
        resources,
        cfg,
        verbose=verbose,
    )

    check_resource_constraints(
        resources,
        cfg,
    )

    performance = compute_performance(
        channels,
        resources,
        cfg,
    )

    return BaselineResult(
        name=(
            "FPA-UPA + "
            "FP Resource Optimization"
        ),
        geometry=geometry,
        channels=channels,
        resources=resources,
        performance=performance,
        wsr_history=list(
            inner_history
        ),
        converged=_inner_history_converged(
            inner_history,
            cfg,
        ),
        iterations=(
            len(inner_history)
            - 1
        ),
    )
