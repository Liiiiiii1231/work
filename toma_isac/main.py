"""ToMA-ISAC 第三阶段主程序。

先完成固定 ToMA 的内层资源优化，
再固定资源执行 ToMA 外层位置优化。
"""

import numpy as np

from channels import build_channels
from config import load_config
from geometry import (
    build_geometry,
    min_endpoint_distance,
)
from initialization import (
    generate_feasible_endpoints,
    initialize_resources,
)
from inner_solver import solve_inner_problem
from metrics import compute_performance
from outer_solver import solve_outer_problem
from utils import check_resource_constraints


def main() -> None:
    """运行第三阶段外层 ToMA 位置优化测试。"""

    cfg = load_config()

    rng = np.random.default_rng(
        cfg.random_seed
    )

    # 初始几何和信道
    endpoints = generate_feasible_endpoints(
        cfg,
        rng,
    )

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

    print("=" * 64)
    print(
        "ToMA-ISAC Stage 3: "
        "Outer Position Optimization"
    )
    print("=" * 64)

    # ==========================================================
    # Stage 2：固定初始 ToMA，优化资源
    # ==========================================================

    initial_performance = compute_performance(
        channels,
        resources,
        cfg,
    )

    print(
        "\nInitial WSR =",
        initial_performance.weighted_sum_rate,
    )

    print("\n[Stage 2: Inner Solver]")

    (
        optimized_resources,
        inner_history,
    ) = solve_inner_problem(
        channels,
        resources,
        cfg,
        verbose=True,
    )

    check_resource_constraints(
        optimized_resources,
        cfg,
    )

    print(
        "\nWSR after inner solver =",
        inner_history[-1],
    )

    # ==========================================================
    # Stage 3：固定资源，优化 ToMA 位置
    # ==========================================================

    print(
        "\n[Stage 3: Outer Solver]"
    )

    (
        optimized_geometry,
        optimized_channels,
        outer_history,
    ) = solve_outer_problem(
        geometry,
        optimized_resources,
        cfg,
        verbose=True,
    )

    final_performance = compute_performance(
        optimized_channels,
        optimized_resources,
        cfg,
    )

    print(
        "\n[Final Geometry]"
    )

    print(
        optimized_geometry.endpoints
    )

    print(
        "endpoint norms =",
        np.linalg.norm(
            optimized_geometry.endpoints,
            axis=1,
        ),
    )

    print(
        "min endpoint distance =",
        min_endpoint_distance(
            optimized_geometry.endpoints
        ),
    )

    print(
        "\n[Outer WSR]"
    )

    print(
        "Before outer =",
        outer_history[0],
    )

    print(
        "After outer  =",
        outer_history[-1],
    )

    print(
        "Outer sweeps =",
        len(outer_history) - 1,
    )

    print(
        "\n[Final SINR]"
    )

    print(
        "DL      =",
        final_performance.gamma_dl,
    )

    print(
        "UL      =",
        final_performance.gamma_ul,
    )

    print(
        "Sensing =",
        final_performance.gamma_s,
    )

    print(
        "\nFinal WSR =",
        final_performance.weighted_sum_rate,
    )

    print(
        "\nStage 3 passed."
    )


if __name__ == "__main__":
    main()