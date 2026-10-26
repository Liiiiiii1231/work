from __future__ import annotations


# ===== algorithm: shared implementation =====
"""ToMA-ISAC 完整单起点联合优化算法。

每个外层迭代执行：
内层资源优化 -> Tx/Rx ToMA 更新 -> 真实 WSR 检查。

多起点 Monte Carlo 和仿真实验留到后续阶段实现。
"""



from dataclasses import dataclass
from copy import deepcopy

import numpy as np

from channel_model import build_channels
from system_model import SystemConfig
from system_model import build_geometry
from channel_model import generate_feasible_endpoints, initialize_resources
from inner_solver import solve_inner_problem
from channel_model import compute_performance
from outer_solver import OuterRCGState, create_outer_rcg_state, update_positions_once, update_positions_once_fp_aware
from system_model import ChannelState, GeometryState, PerformanceResult, ResourceState
from system_model import check_resource_constraints, relative_change


@dataclass
class AlgorithmResult:
    """保存一次完整单起点联合优化的结果。"""

    geometry: GeometryState
    channels: ChannelState
    resources: ResourceState
    performance: PerformanceResult

    joint_wsr_history: list[float]
    inner_wsr_histories: list[list[float]]
    accepted_position_counts: list[int]
    window_gain_history: list[float]

    converged: bool
    outer_iterations: int
    stop_reason: str


# ============================================================
# 1. 单起点联合优化
# ============================================================

def run_joint_algorithm(
    cfg: SystemConfig,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
    max_outer_iterations: int | None = None,
    position_update_mode: str = "original",
    fp_aware_candidate_iters: int = 2,
    initial_endpoints: np.ndarray | None = None,
    initial_resources: ResourceState | None = None,
) -> AlgorithmResult:
    """执行一次完整的单起点 ToMA-ISAC 联合优化。

    position_update_mode="original" 保持原固定资源 Armijo；
    position_update_mode="fp_aware" 使用候选位置 warm-start FP Armijo。
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

    if position_update_mode not in {
        "original",
        "fp_aware",
    }:
        raise ValueError(
            "position_update_mode must be "
            "'original' or 'fp_aware'."
        )

    if (
        position_update_mode == "fp_aware"
        and fp_aware_candidate_iters <= 0
    ):
        raise ValueError(
            "fp_aware_candidate_iters must be positive."
        )

    # ========================================================
    # 1. 初始化几何、信道和资源
    # ========================================================

    if initial_endpoints is None:
        endpoints = generate_feasible_endpoints(
            cfg,
            rng,
        )
    else:
        endpoints = np.asarray(
            initial_endpoints,
            dtype=float,
        ).copy()

    geometry = build_geometry(
        endpoints,
        cfg,
    )

    channels = build_channels(
        geometry,
        cfg,
    )

    if initial_resources is None:
        resources = initialize_resources(
            geometry,
            channels,
            cfg,
            rng,
        )
    else:
        resources = deepcopy(initial_resources)

    check_resource_constraints(
        resources,
        cfg,
    )

    performance = compute_performance(
        channels,
        resources,
        cfg,
    )

    initial_wsr = float(
        performance.weighted_sum_rate
    )

    joint_wsr_history = [
        initial_wsr
    ]

    inner_wsr_histories: list[
        list[float]
    ] = []

    accepted_position_counts: list[
        int
    ] = []

    window_gain_history: list[float] = []

    # 保存各端点上一轮的 RCG 信息。
    rcg_state: OuterRCGState = (
        create_outer_rcg_state(
            cfg
        )
    )

    previous_joint_wsr = initial_wsr
    consecutive_small_changes = 0
    converged = False
    outer_iterations = 0
    stop_reason = "max_outer_iterations"

    plateau_enabled = (
        cfg.plateau_window >= 2
        and cfg.tol_window > 0.0
    )

    if verbose:
        print(
            f"Joint iteration 0: "
            f"WSR = {initial_wsr:.10f}"
        )

    # ========================================================
    # 2. 完整外层联合迭代
    # ========================================================

    for outer_iter in range(
        1,
        number_of_outer_iterations + 1,
    ):
        if verbose:
            print(
                "\n"
                + "-" * 64
            )
            print(
                f"Joint iteration "
                f"{outer_iter}"
            )
            print(
                "-" * 64
            )

        # ----------------------------------------------------
        # Step 1：固定当前 ToMA，优化所有资源
        # ----------------------------------------------------

        (
            resources,
            inner_history,
        ) = solve_inner_problem(
            channels,
            resources,
            cfg,
            verbose=False,
        )

        inner_wsr_histories.append(
            inner_history
        )

        wsr_after_inner = (
            inner_history[-1]
        )

        if verbose:
            print(
                "[Inner]"
            )
            print(
                f"  iterations = "
                f"{len(inner_history) - 1}"
            )
            print(
                f"  WSR = "
                f"{wsr_after_inner:.10f}"
            )

        # ----------------------------------------------------
        # Step 2：按所选模式更新 Tx 和 Rx ToMA
        # ----------------------------------------------------

        if position_update_mode == "original":
            (
                geometry,
                channels,
                wsr_after_positions,
                rcg_state,
                accepted_count,
            ) = update_positions_once(
                geometry,
                resources,
                cfg,
                rcg_state=rcg_state,
                verbose=False,
            )
        else:
            (
                geometry,
                channels,
                resources,
                wsr_after_positions,
                rcg_state,
                accepted_count,
            ) = update_positions_once_fp_aware(
                geometry,
                resources,
                cfg,
                rcg_state=rcg_state,
                candidate_fp_iterations=(
                    fp_aware_candidate_iters
                ),
                verbose=False,
            )

        accepted_position_counts.append(
            accepted_count
        )

        # ----------------------------------------------------
        # Step 3：重新计算完整真实系统性能
        # ----------------------------------------------------

        performance = compute_performance(
            channels,
            resources,
            cfg,
        )

        current_joint_wsr = float(
            performance.weighted_sum_rate
        )

        # outer_solver 返回的 WSR 与重新计算值应一致。
        if not np.isclose(
            current_joint_wsr,
            wsr_after_positions,
            rtol=1e-9,
            atol=1e-10,
        ):
            raise RuntimeError(
                "Outer WSR does not match "
                "the recomputed true WSR."
            )

        joint_wsr_history.append(
            current_joint_wsr
        )

        outer_iterations = outer_iter

        # ----------------------------------------------------
        # Step 4：检查完整联合迭代是否保持非下降
        # ----------------------------------------------------

        decrease_tolerance = (
            1e-8
            * max(
                1.0,
                abs(previous_joint_wsr),
            )
        )

        if (
            current_joint_wsr
            < previous_joint_wsr
            - decrease_tolerance
        ):
            raise RuntimeError(
                "Joint WSR decreased "
                "more than numerical tolerance."
            )

        change = relative_change(
            current_joint_wsr,
            previous_joint_wsr,
        )

        if verbose:
            print(
                "[Positions]"
            )
            print(
                f"  accepted endpoints = "
                f"{accepted_count}"
            )
            print(
                f"  WSR = "
                f"{current_joint_wsr:.10f}"
            )

            print(
                "[Joint]"
            )
            print(
                f"  relative change = "
                f"{change:.3e}"
            )

        # ----------------------------------------------------
        # Step 5：原始单步 WSR 收敛条件
        # ----------------------------------------------------

        if change <= cfg.tol_outer:
            consecutive_small_changes += 1
        else:
            consecutive_small_changes = 0

        # ----------------------------------------------------
        # Step 6：窗口平台条件
        # ----------------------------------------------------

        window_gain = float("nan")

        if (
            plateau_enabled
            and outer_iter >= cfg.plateau_window
        ):
            reference_wsr = joint_wsr_history[
                outer_iter - cfg.plateau_window
            ]

            window_gain = (
                current_joint_wsr - reference_wsr
            ) / max(
                1.0,
                abs(reference_wsr),
            )

            # 联合 WSR 理论上非下降；只消除机器精度级负值。
            window_gain = max(0.0, float(window_gain))

        window_gain_history.append(window_gain)

        if verbose and np.isfinite(window_gain):
            print(
                f"  window({cfg.plateau_window}) gain = "
                f"{window_gain:.3e}"
            )

        # 优先保留原始停止规则；窗口规则只作为附加早停。
        if consecutive_small_changes >= 2:
            converged = True
            stop_reason = "outer_tolerance"

            if verbose:
                print(
                    "Joint algorithm converged by "
                    "outer tolerance at iteration "
                    f"{outer_iter}."
                )

            break

        if (
            np.isfinite(window_gain)
            and window_gain <= cfg.tol_window
        ):
            converged = True
            stop_reason = "window_plateau"

            if verbose:
                print(
                    "Joint algorithm reached a WSR plateau "
                    f"at iteration {outer_iter}: "
                    f"window_gain={window_gain:.3e}."
                )

            break

        previous_joint_wsr = current_joint_wsr

    # ========================================================
    # 3. 最终检查
    # ========================================================

    check_resource_constraints(
        resources,
        cfg,
    )

    # 重新通过 build_geometry 的完整几何检查。
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

    return AlgorithmResult(
        geometry=geometry,
        channels=channels,
        resources=resources,
        performance=performance,
        joint_wsr_history=joint_wsr_history,
        inner_wsr_histories=inner_wsr_histories,
        accepted_position_counts=(
            accepted_position_counts
        ),
        window_gain_history=window_gain_history,
        converged=converged,
        outer_iterations=outer_iterations,
        stop_reason=stop_reason,
    )


# ===== baselines: shared implementation =====
"""Stage 6：Baseline 对比方案。

本文件实现：
B1: RCG-ToMA + Traditional Resource Design
B2: Fixed-ToMA + FP Resource Optimization
B3: FPA-UPA + FP Resource Optimization
B4: Fixed-ToMA + Traditional Resource Design
B5: FPA-UPA + Traditional Resource Design

Proposed 继续直接调用 algorithm.py 中的 run_joint_algorithm()，
因此这里不重复实现完整联合算法。

实现原则：
1. 不重复实现 SINR / Rate / WSR；
2. 不重复实现 FP；
3. 不重复实现 RCG；
4. Baseline 只重新组合现有模块。
"""



from dataclasses import dataclass
from copy import deepcopy

import numpy as np

from channel_model import build_channels
from system_model import SystemConfig
from system_model import build_geometry
from channel_model import generate_feasible_endpoints, initialize_resources
from inner_solver import solve_inner_problem, update_sensing_combiners, update_uplink_combiners
from channel_model import compute_performance
from outer_solver import create_outer_rcg_state, update_positions_once
from system_model import ChannelState, GeometryState, PerformanceResult, ResourceState
from system_model import check_resource_constraints, relative_change


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


def _evaluate_traditional_wsr(
    endpoints: np.ndarray,
    cfg: SystemConfig,
) -> float:
    """按候选 ToMA 位置重新生成传统资源并计算 B1 的真实 WSR。"""

    geometry = build_geometry(
        endpoints,
        cfg,
        check_constraints=False,
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

    return float(
        performance.weighted_sum_rate
    )


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
            objective_evaluator=lambda candidate_endpoints: (
                _evaluate_traditional_wsr(
                    candidate_endpoints,
                    cfg,
                )
            ),
        )

        # ToMA 改变后信道已经变化，
        # traditional resources 必须按新信道重新生成。
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
    initial_endpoints: np.ndarray | None = None,
    initial_resources: ResourceState | None = None,
) -> BaselineResult:
    """固定初始 ToMA，只执行完整 FP resource optimization。"""

    if rng is None:
        rng = np.random.default_rng(
            cfg.random_seed
        )

    if initial_endpoints is None:
        endpoints = generate_feasible_endpoints(
            cfg,
            rng,
        )
    else:
        endpoints = np.asarray(
            initial_endpoints,
            dtype=float,
        ).copy()

    # Fixed-ToMA 的初始端点之后应始终保持不变。
    initial_endpoints_saved = endpoints.copy()
    geometry = build_geometry(
        endpoints,
        cfg,
    )

    channels = build_channels(
        geometry,
        cfg,
    )

    if initial_resources is None:
        resources = initialize_resources(
            geometry,
            channels,
            cfg,
            rng,
        )
    else:
        resources = deepcopy(initial_resources)

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
    initial_endpoints=initial_endpoints_saved,
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
    initial_resources: ResourceState | None = None,
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

    if initial_resources is None:
        resources = initialize_resources(
            geometry,
            channels,
            cfg,
            rng,
        )
    else:
        resources = deepcopy(initial_resources)

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


# ============================================================
# 6. Baseline 4
#    Fixed-ToMA + Traditional Resource Design
# ============================================================

def run_fixed_toma_traditional(
    cfg: SystemConfig,
    rng: np.random.Generator | None = None,
) -> BaselineResult:
    """固定随机 ToMA，并使用传统波束与资源设计。"""

    if rng is None:
        rng = np.random.default_rng(
            cfg.random_seed
        )

    endpoints = generate_feasible_endpoints(
        cfg,
        rng,
    )
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

    check_resource_constraints(
        resources,
        cfg,
    )

    wsr = float(
        performance.weighted_sum_rate
    )

    return BaselineResult(
        name=(
            "Fixed-ToMA + "
            "Traditional Resource Design"
        ),
        geometry=geometry,
        channels=channels,
        resources=resources,
        performance=performance,
        wsr_history=[wsr],
        converged=True,
        iterations=0,
        initial_endpoints=initial_endpoints,
    )


# ============================================================
# 7. Baseline 5
#    FPA-UPA + Traditional Resource Design
# ============================================================

def run_fpa_upa_traditional(
    cfg: SystemConfig,
    spacing: float | None = None,
    tx_rx_separation: float | None = None,
) -> BaselineResult:
    """固定 UPA，并使用传统波束与资源设计。"""

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

    check_resource_constraints(
        resources,
        cfg,
    )

    wsr = float(
        performance.weighted_sum_rate
    )

    return BaselineResult(
        name=(
            "FPA-UPA + "
            "Traditional Resource Design"
        ),
        geometry=geometry,
        channels=channels,
        resources=resources,
        performance=performance,
        wsr_history=[wsr],
        converged=True,
        iterations=0,
        initial_endpoints=None,
    )


# ===== ma_baseline: shared implementation =====
"""Stage 8B-2：MA + FP 对比方案。

B6: MA + FP Resource Optimization

定义：
- Tx / Rx 都从与 B3 相同的固定 UPA 初始化；
- Tx / Rx 阵元分别位于两个固定 x 平面的二维可移动区域；
- 每个阵元只在 y-z 平面独立移动；
- 同阵列阵元满足最小间距约束；
- Tx/Rx 直接 SI 继续使用真实阵元距离；
- 资源侧完全复用现有 FP；
- 位置侧使用投影梯度 + Armijo 回溯，不使用 ToMA 的球面 RCG。
"""



from dataclasses import dataclass
from copy import deepcopy

import numpy as np

pass
from channel_model import build_channels
from system_model import SystemConfig
from channel_model import initialize_resources
from inner_solver import solve_inner_problem
from channel_model import compute_performance
from system_model import ChannelState, GeometryState, PerformanceResult, ResourceState
from system_model import check_resource_constraints, relative_change


@dataclass
class MABaselineResult:
    """保存 MA + FP 单次运行结果。"""

    name: str
    geometry: GeometryState
    channels: ChannelState
    resources: ResourceState
    performance: PerformanceResult
    wsr_history: list[float]
    converged: bool
    iterations: int
    stop_reason: str
    initial_tx_positions: np.ndarray
    initial_rx_positions: np.ndarray


def _build_ma_geometry(
    tx_positions: np.ndarray,
    rx_positions: np.ndarray,
    cfg: SystemConfig,
) -> GeometryState:
    """直接由 MA 阵元坐标构造 GeometryState。"""

    tx_positions = np.asarray(tx_positions, dtype=float)
    rx_positions = np.asarray(rx_positions, dtype=float)

    if tx_positions.shape != (cfg.n_tx, 3):
        raise ValueError(
            f"tx_positions.shape={tx_positions.shape}, "
            f"expected {(cfg.n_tx, 3)}."
        )

    if rx_positions.shape != (cfg.n_rx, 3):
        raise ValueError(
            f"rx_positions.shape={rx_positions.shape}, "
            f"expected {(cfg.n_rx, 3)}."
        )

    si_distances = np.linalg.norm(
        rx_positions[:, None, :]
        - tx_positions[None, :, :],
        axis=2,
    )

    if np.min(si_distances) < cfg.min_si_distance:
        raise ValueError("MA Tx/Rx element distance is too small.")

    return GeometryState(
        endpoints=np.empty((0, 3), dtype=float),
        tx_positions=tx_positions.copy(),
        rx_positions=rx_positions.copy(),
    )


def _minimum_pairwise_distance(
    positions: np.ndarray,
) -> float:
    """返回同一阵列内任意两个不同阵元的最小距离。"""

    if len(positions) <= 1:
        return np.inf

    difference = (
        positions[:, None, :]
        - positions[None, :, :]
    )
    distances = np.linalg.norm(difference, axis=2)
    np.fill_diagonal(distances, np.inf)
    return float(np.min(distances))


def _position_in_region(
    position: np.ndarray,
    center: np.ndarray,
    region_size: float,
    atol: float = 1e-12,
) -> bool:
    """检查一个阵元是否位于指定 y-z 正方形移动区域内。"""

    half = 0.5 * region_size

    return bool(
        np.isclose(position[0], center[0], atol=atol)
        and center[1] - half - atol <= position[1] <= center[1] + half + atol
        and center[2] - half - atol <= position[2] <= center[2] + half + atol
    )


def _ma_geometry_is_feasible(
    tx_positions: np.ndarray,
    rx_positions: np.ndarray,
    tx_center: np.ndarray,
    rx_center: np.ndarray,
    region_size: float,
    min_spacing: float,
    cfg: SystemConfig,
) -> bool:
    """检查 MA 移动区域、阵元间距和直接 SI 数值安全。"""

    if not all(
        _position_in_region(position, tx_center, region_size)
        for position in tx_positions
    ):
        return False

    if not all(
        _position_in_region(position, rx_center, region_size)
        for position in rx_positions
    ):
        return False

    if _minimum_pairwise_distance(tx_positions) < min_spacing - 1e-12:
        return False

    if _minimum_pairwise_distance(rx_positions) < min_spacing - 1e-12:
        return False

    si_distances = np.linalg.norm(
        rx_positions[:, None, :]
        - tx_positions[None, :, :],
        axis=2,
    )

    return bool(np.min(si_distances) >= cfg.min_si_distance)



def _randomize_initial_ma_geometry(
    base_geometry: GeometryState,
    tx_center: np.ndarray,
    rx_center: np.ndarray,
    region_size: float,
    min_spacing: float,
    initial_jitter_fraction: float,
    cfg: SystemConfig,
    rng: np.random.Generator,
) -> GeometryState:
    """在固定 UPA 周围生成一组随机可行 MA 初始位置。

    - Tx/Rx 仍位于各自固定 x 平面；
    - 只对 y-z 坐标加随机扰动；
    - 扰动范围由 initial_jitter_fraction * (region_size/2) 控制；
    - 最终必须满足移动区域、同阵列最小间距和 Tx/Rx SI 数值安全。
    """

    if not (0.0 < initial_jitter_fraction <= 1.0):
        raise ValueError(
            "initial_jitter_fraction must lie in (0, 1]."
        )

    jitter_half = (
        0.5 * float(region_size) * float(initial_jitter_fraction)
    )
    half = 0.5 * float(region_size)

    for _ in range(int(cfg.max_geometry_trials)):
        tx_candidate = base_geometry.tx_positions.copy()
        rx_candidate = base_geometry.rx_positions.copy()

        tx_candidate[:, 1:] += rng.uniform(
            low=-jitter_half,
            high=jitter_half,
            size=(cfg.n_tx, 2),
        )
        rx_candidate[:, 1:] += rng.uniform(
            low=-jitter_half,
            high=jitter_half,
            size=(cfg.n_rx, 2),
        )

        tx_candidate[:, 1] = np.clip(
            tx_candidate[:, 1],
            tx_center[1] - half,
            tx_center[1] + half,
        )
        tx_candidate[:, 2] = np.clip(
            tx_candidate[:, 2],
            tx_center[2] - half,
            tx_center[2] + half,
        )
        rx_candidate[:, 1] = np.clip(
            rx_candidate[:, 1],
            rx_center[1] - half,
            rx_center[1] + half,
        )
        rx_candidate[:, 2] = np.clip(
            rx_candidate[:, 2],
            rx_center[2] - half,
            rx_center[2] + half,
        )

        # 固定两个 MA 平面的 x 坐标。
        tx_candidate[:, 0] = tx_center[0]
        rx_candidate[:, 0] = rx_center[0]

        if _ma_geometry_is_feasible(
            tx_candidate,
            rx_candidate,
            tx_center,
            rx_center,
            region_size,
            min_spacing,
            cfg,
        ):
            return _build_ma_geometry(
                tx_candidate,
                rx_candidate,
                cfg,
            )

    raise RuntimeError(
        "Failed to generate a feasible randomized MA initial geometry."
    )

def _evaluate_ma_wsr(
    tx_positions: np.ndarray,
    rx_positions: np.ndarray,
    resources: ResourceState,
    cfg: SystemConfig,
) -> tuple[float, GeometryState, ChannelState]:
    """固定 FP resources，计算候选 MA 几何的真实 WSR。"""

    geometry = _build_ma_geometry(
        tx_positions,
        rx_positions,
        cfg,
    )
    channels = build_channels(geometry, cfg)
    performance = compute_performance(
        channels,
        resources,
        cfg,
    )

    return (
        float(performance.weighted_sum_rate),
        geometry,
        channels,
    )


def _element_gradient_yz(
    tx_positions: np.ndarray,
    rx_positions: np.ndarray,
    resources: ResourceState,
    cfg: SystemConfig,
    group: str,
    index: int,
    center: np.ndarray,
    region_size: float,
) -> np.ndarray:
    """用中央差分/边界单边差分计算单阵元的 y-z 梯度。"""

    delta = float(cfg.finite_diff_delta)
    half = 0.5 * region_size

    if delta <= 0.0:
        raise ValueError("finite_diff_delta must be positive.")

    positions = tx_positions if group == "tx" else rx_positions

    base_wsr, _, _ = _evaluate_ma_wsr(
        tx_positions,
        rx_positions,
        resources,
        cfg,
    )

    gradient = np.zeros(2, dtype=float)

    for local_axis, global_axis in enumerate((1, 2)):
        lower = center[global_axis] - half
        upper = center[global_axis] + half
        value = positions[index, global_axis]

        plus_ok = value + delta <= upper
        minus_ok = value - delta >= lower

        if plus_ok:
            tx_plus = tx_positions.copy()
            rx_plus = rx_positions.copy()
            target = tx_plus if group == "tx" else rx_plus
            target[index, global_axis] += delta
            f_plus, _, _ = _evaluate_ma_wsr(
                tx_plus,
                rx_plus,
                resources,
                cfg,
            )
        else:
            f_plus = base_wsr

        if minus_ok:
            tx_minus = tx_positions.copy()
            rx_minus = rx_positions.copy()
            target = tx_minus if group == "tx" else rx_minus
            target[index, global_axis] -= delta
            f_minus, _, _ = _evaluate_ma_wsr(
                tx_minus,
                rx_minus,
                resources,
                cfg,
            )
        else:
            f_minus = base_wsr

        if plus_ok and minus_ok:
            gradient[local_axis] = (
                f_plus - f_minus
            ) / (2.0 * delta)
        elif plus_ok:
            gradient[local_axis] = (
                f_plus - base_wsr
            ) / delta
        elif minus_ok:
            gradient[local_axis] = (
                base_wsr - f_minus
            ) / delta

    return gradient




def _project_to_ma_feasible_tangent(
    gradient: np.ndarray,
    positions: np.ndarray,
    index: int,
    center: np.ndarray,
    region_size: float,
    min_spacing: float,
) -> np.ndarray:
    """计算二维梯度在盒边界及活动阵元间距约束下的可行锥投影。

    对活动阵元间距约束 ||p_i-p_j|| >= d_min，线性化后
    一阶可行方向 v 满足 (p_i-p_j)·v >= 0（固定其他阵元）。
    这里仅生成额外候选方向；原有梯度搜索仍然保留。
    所有最终候选依旧经过完整非线性几何约束检查。
    """
    g = np.asarray(gradient, dtype=float).reshape(2)
    p = np.asarray(positions[index, 1:], dtype=float)
    normals = []
    half = region_size / 2.0
    boundary_tol = 1.0e-10
    active_tol = max(1.0e-10, 1.0e-5 * min_spacing)

    for ax in range(2):
        if p[ax] <= center[ax + 1] - half + boundary_tol:
            n = np.zeros(2, dtype=float)
            n[ax] = 1.0
            normals.append(n)
        if p[ax] >= center[ax + 1] + half - boundary_tol:
            n = np.zeros(2, dtype=float)
            n[ax] = -1.0
            normals.append(n)

    for j in range(len(positions)):
        if j == index:
            continue
        sep = p - np.asarray(positions[j, 1:], dtype=float)
        d = float(np.linalg.norm(sep))
        if 1.0e-15 < d <= min_spacing + active_tol:
            normals.append(sep / d)

    if not normals:
        return g.copy()

    def direction_allowed(d: np.ndarray) -> bool:
        tol = 1.0e-12 * max(1.0, float(np.linalg.norm(d)))
        return all(float(n @ d) >= -tol for n in normals)

    # 在 R^2 中，投影到半空间交得到的凸锥，只可能落在
    # 原梯度本身、某个活动边界对应的射线上，或原点。
    candidates = [np.zeros(2, dtype=float)]
    if direction_allowed(g):
        candidates.append(g.copy())
    for n in normals:
        tangent = np.array([-n[1], n[0]], dtype=float)
        v = float(np.dot(g, tangent)) * tangent
        if direction_allowed(v):
            candidates.append(v)

    return min(candidates, key=lambda v: float(np.dot(v - g, v - g)))


def _update_ma_positions_once(
    geometry: GeometryState,
    resources: ResourceState,
    cfg: SystemConfig,
    tx_center: np.ndarray,
    rx_center: np.ndarray,
    region_size: float,
    min_spacing: float,
    initial_step: float,
) -> tuple[GeometryState, ChannelState, float, int]:
    """固定 resources，对所有 Tx/Rx MA 阵元做一次位置 sweep。"""

    tx_positions = geometry.tx_positions.copy()
    rx_positions = geometry.rx_positions.copy()

    current_wsr, current_geometry, current_channels = _evaluate_ma_wsr(
        tx_positions,
        rx_positions,
        resources,
        cfg,
    )

    accepted_count = 0

    for group, center, n_elements in (
        ("tx", tx_center, cfg.n_tx),
        ("rx", rx_center, cfg.n_rx),
    ):
        for index in range(n_elements):
            gradient = _element_gradient_yz(
                tx_positions,
                rx_positions,
                resources,
                cfg,
                group=group,
                index=index,
                center=center,
                region_size=region_size,
            )

            grad_norm = float(np.linalg.norm(gradient))

            if grad_norm <= 1.0e-12:
                continue

            # 原始方向仍保留，用于避免改动现有 MA 基线的有效搜索路径。
            # 额外搜索投影到活动几何约束的可行方向。
            original_direction = gradient / grad_norm
            current_positions = tx_positions if group == "tx" else rx_positions
            feasible_grad = _project_to_ma_feasible_tangent(
                gradient, current_positions, index, center,
                region_size, min_spacing,
            )
            feasible_norm = float(np.linalg.norm(feasible_grad))
            candidate_directions = [original_direction]
            if feasible_norm > 1.0e-12:
                feasible_direction = feasible_grad / feasible_norm
                if not np.allclose(feasible_direction, original_direction, atol=1e-10):
                    candidate_directions.append(feasible_direction)

            # 两条方向从同一状态出发。各自进行原有的 Armijo 回溯，
            # 最后接受本次单阵元搜索中真实 WSR 最高的可行候选。
            best_move = None
            best_wsr = current_wsr
            for direction in candidate_directions:
                step = float(initial_step)
                for _ in range(cfg.max_backtracking_iter):
                    tx_candidate = tx_positions.copy()
                    rx_candidate = rx_positions.copy()
                    candidate_positions = (
                        tx_candidate if group == "tx" else rx_candidate
                    )

                    previous_position = candidate_positions[index].copy()
                    candidate_position = previous_position.copy()
                    candidate_position[1:] += step * direction

                    half = 0.5 * region_size
                    candidate_position[1] = np.clip(
                        candidate_position[1],
                        center[1] - half, center[1] + half,
                    )
                    candidate_position[2] = np.clip(
                        candidate_position[2],
                        center[2] - half, center[2] + half,
                    )
                    candidate_position[0] = center[0]
                    candidate_positions[index] = candidate_position

                    displacement = candidate_position[1:] - previous_position[1:]
                    if np.linalg.norm(displacement) <= 1.0e-15:
                        step *= cfg.backtracking_factor
                        continue

                    if not _ma_geometry_is_feasible(
                        tx_candidate, rx_candidate, tx_center, rx_center,
                        region_size, min_spacing, cfg,
                    ):
                        step *= cfg.backtracking_factor
                        continue

                    candidate_wsr, candidate_geometry, candidate_channels = (
                        _evaluate_ma_wsr(
                            tx_candidate, rx_candidate, resources, cfg,
                        )
                    )
                    armijo_rhs = (
                        current_wsr
                        + cfg.armijo_mu * float(np.dot(gradient, displacement))
                    )
                    if candidate_wsr >= max(current_wsr, armijo_rhs):
                        if candidate_wsr > best_wsr:
                            best_wsr = candidate_wsr
                            best_move = (
                                tx_candidate, rx_candidate,
                                candidate_geometry, candidate_channels,
                            )
                        break
                    step *= cfg.backtracking_factor

            if best_move is not None:
                tx_positions, rx_positions, current_geometry, current_channels = best_move
                current_wsr = best_wsr
                accepted_count += 1

    return (
        current_geometry,
        current_channels,
        current_wsr,
        accepted_count,
    )


def run_ma_fp(
    cfg: SystemConfig,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
    spacing: float | None = None,
    tx_rx_separation: float | None = None,
    region_size: float | None = None,
    min_spacing: float | None = None,
    initial_step: float | None = None,
    max_outer_iterations: int | None = None,
    randomize_initial_geometry: bool = False,
    initial_jitter_fraction: float = 0.5,
    initial_tx_positions: np.ndarray | None = None,
    initial_rx_positions: np.ndarray | None = None,
    initial_resources: ResourceState | None = None,
) -> MABaselineResult:
    """运行 B6：MA + FP Resource Optimization。

    公平性：
    - 与 B3 使用同一 UPA 作为初始阵列；
    - 与 Proposed 使用同一个 FP 内层；
    - MA 只改变阵元位置可行域和位置优化方式。
    """

    if rng is None:
        rng = np.random.default_rng(cfg.random_seed)

    if region_size is None:
        region_size = cfg.cable_length

    if min_spacing is None:
        min_spacing = 0.5 * cfg.wavelength

    if initial_step is None:
        initial_step = 0.1 * region_size

    if region_size <= 0.0:
        raise ValueError("MA region_size must be positive.")

    if min_spacing <= 0.0:
        raise ValueError("MA min_spacing must be positive.")

    if initial_step <= 0.0:
        raise ValueError("MA initial_step must be positive.")

    number_of_outer_iterations = (
        cfg.max_outer_iter
        if max_outer_iterations is None
        else int(max_outer_iterations)
    )

    base_geometry = build_fpa_upa_geometry(
        cfg,
        spacing=spacing,
        tx_rx_separation=tx_rx_separation,
    )

    tx_center = np.mean(base_geometry.tx_positions, axis=0)
    rx_center = np.mean(base_geometry.rx_positions, axis=0)

    if not _ma_geometry_is_feasible(
        base_geometry.tx_positions,
        base_geometry.rx_positions,
        tx_center,
        rx_center,
        region_size,
        min_spacing,
        cfg,
    ):
        raise ValueError(
            "Initial UPA is not feasible for the selected MA region/min_spacing."
        )

    if (initial_tx_positions is None) != (initial_rx_positions is None):
        raise ValueError(
            "initial_tx_positions and initial_rx_positions "
            "must be provided together."
        )

    if initial_tx_positions is not None:
        geometry = _build_ma_geometry(
            initial_tx_positions,
            initial_rx_positions,
            cfg,
        )
        if not _ma_geometry_is_feasible(
            geometry.tx_positions,
            geometry.rx_positions,
            tx_center,
            rx_center,
            region_size,
            min_spacing,
            cfg,
        ):
            raise ValueError(
                "Provided MA warm-start geometry is infeasible."
            )
    elif randomize_initial_geometry:
        geometry = _randomize_initial_ma_geometry(
            base_geometry,
            tx_center=tx_center,
            rx_center=rx_center,
            region_size=region_size,
            min_spacing=min_spacing,
            initial_jitter_fraction=initial_jitter_fraction,
            cfg=cfg,
            rng=rng,
        )
    else:
        geometry = base_geometry

    initial_tx_positions_saved = geometry.tx_positions.copy()
    initial_rx_positions_saved = geometry.rx_positions.copy()

    channels = build_channels(geometry, cfg)
    if initial_resources is None:
        resources = initialize_resources(
            geometry,
            channels,
            cfg,
            rng,
        )
    else:
        resources = deepcopy(initial_resources)

    check_resource_constraints(resources, cfg)

    performance = compute_performance(
        channels,
        resources,
        cfg,
    )
    previous_wsr = float(performance.weighted_sum_rate)
    wsr_history = [previous_wsr]

    consecutive_small_changes = 0
    converged = False
    iterations = 0
    stop_reason = "max_outer_iterations"

    plateau_enabled = (
        cfg.plateau_window >= 2
        and cfg.tol_window > 0.0
    )

    for outer_iter in range(1, number_of_outer_iterations + 1):
        resources, _ = solve_inner_problem(
            channels,
            resources,
            cfg,
            verbose=False,
        )

        geometry, channels, _, accepted_count = _update_ma_positions_once(
            geometry,
            resources,
            cfg,
            tx_center=tx_center,
            rx_center=rx_center,
            region_size=region_size,
            min_spacing=min_spacing,
            initial_step=initial_step,
        )

        performance = compute_performance(
            channels,
            resources,
            cfg,
        )
        current_wsr = float(performance.weighted_sum_rate)
        wsr_history.append(current_wsr)

        change = relative_change(
            current_wsr,
            previous_wsr,
        )

        if verbose:
            print(
                f"MA+FP iteration {outer_iter}: "
                f"accepted={accepted_count}/{cfg.n_tx + cfg.n_rx}, "
                f"WSR={current_wsr:.10f}, "
                f"change={change:.3e}"
            )

        if change <= cfg.tol_outer:
            consecutive_small_changes += 1
        else:
            consecutive_small_changes = 0

        iterations = outer_iter

        if consecutive_small_changes >= 2:
            converged = True
            stop_reason = "outer_tolerance"
            break

        if (
            plateau_enabled
            and len(wsr_history) > cfg.plateau_window
        ):
            old_wsr = wsr_history[-1 - cfg.plateau_window]
            window_gain = (
                current_wsr - old_wsr
            ) / max(1.0, abs(old_wsr))

            if 0.0 <= window_gain <= cfg.tol_window:
                converged = True
                stop_reason = "window_plateau"
                break

        previous_wsr = current_wsr

    check_resource_constraints(resources, cfg)

    return MABaselineResult(
        name="MA + FP Resource Optimization",
        geometry=geometry,
        channels=channels,
        resources=resources,
        performance=performance,
        wsr_history=wsr_history,
        converged=converged,
        iterations=iterations,
        stop_reason=stop_reason,
        initial_tx_positions=initial_tx_positions_saved,
        initial_rx_positions=initial_rx_positions_saved,
    )
