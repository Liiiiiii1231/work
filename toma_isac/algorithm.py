"""ToMA-ISAC 完整单起点联合优化算法。

每个外层迭代执行：
内层资源优化 -> Tx/Rx ToMA 更新 -> 真实 WSR 检查。

多起点 Monte Carlo 和仿真实验留到后续阶段实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy

import numpy as np

from channels import build_channels
from config import SystemConfig
from geometry import build_geometry
from initialization import (
    generate_feasible_endpoints,
    initialize_resources,
)
from inner_solver import solve_inner_problem
from metrics import compute_performance
from outer_solver import (
    OuterRCGState,
    create_outer_rcg_state,
    update_positions_once,
    update_positions_once_fp_aware,
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
