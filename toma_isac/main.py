"""ToMA-ISAC 第二阶段主程序。

固定 ToMA 几何，完成内层 FP 资源优化，
并比较优化前后的 SINR、Rate 和 WSR。
"""

import numpy as np

from channels import build_channels
from config import load_config
from geometry import (
    build_geometry,
    min_endpoint_distance,
    min_tx_rx_element_distance,
)
from initialization import (
    generate_feasible_endpoints,
    initialize_resources,
)
from inner_solver import solve_inner_problem
from metrics import compute_performance
from utils import check_resource_constraints


def main() -> None:
    """运行固定 ToMA 下的内层资源优化。"""

    cfg = load_config()

    rng = np.random.default_rng(
        cfg.random_seed
    )

    # 生成固定的初始 ToMA 几何
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

    # 初始化 Q、q_j、b_j、u_l
    initial_resources = initialize_resources(
        geometry,
        channels,
        cfg,
        rng,
    )

    check_resource_constraints(
        initial_resources,
        cfg,
    )

    initial_performance = compute_performance(
        channels,
        initial_resources,
        cfg,
    )

    print("=" * 64)
    print(
        "ToMA-ISAC Stage 2: "
        "Inner Resource Optimization"
    )
    print("=" * 64)

    print(
        f"M={cfg.m_uav}, "
        f"N_c={cfg.n_cable}, "
        f"N_T={cfg.n_tx}, "
        f"N_R={cfg.n_rx}, "
        f"D={cfg.d_stream}"
    )

    print("\n[Geometry]")

    print(
        "min endpoint distance =",
        min_endpoint_distance(
            geometry.endpoints
        ),
    )

    print(
        "min Tx-Rx element distance =",
        min_tx_rx_element_distance(
            geometry
        ),
    )

    print("\n[Initial Performance]")

    print(
        "DL SINR      =",
        initial_performance.gamma_dl,
    )

    print(
        "UL SINR      =",
        initial_performance.gamma_ul,
    )

    print(
        "Sensing SINR =",
        initial_performance.gamma_s,
    )

    print(
        "Initial WSR  =",
        initial_performance.weighted_sum_rate,
    )

    # -------------------- 内层优化 --------------------

    print("\n[Inner Iteration]")

    (
        optimized_resources,
        wsr_history,
    ) = solve_inner_problem(
        channels,
        initial_resources,
        cfg,
        verbose=True,
    )

    check_resource_constraints(
        optimized_resources,
        cfg,
    )

    final_performance = compute_performance(
        channels,
        optimized_resources,
        cfg,
    )

    # -------------------- 最终资源 --------------------

    print("\n[Optimized Resources]")

    print(
        "||Q||_F^2 =",
        np.linalg.norm(
            optimized_resources.q_matrix,
            "fro",
        ) ** 2,
    )

    print(
        "UL amplitudes q_j =",
        optimized_resources.q_ul,
    )

    print(
        "UL powers q_j^2 =",
        optimized_resources.q_ul ** 2,
    )

    print(
        "||b_j|| =",
        np.linalg.norm(
            optimized_resources.b_ul,
            axis=1,
        ),
    )

    print(
        "||u_l|| =",
        np.linalg.norm(
            optimized_resources.u_s,
            axis=1,
        ),
    )

    # -------------------- 最终性能 --------------------

    print("\n[Final SINR]")

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
        "\n[Final Rate: bit/s/Hz]"
    )

    print(
        "DL      =",
        final_performance.rate_dl,
    )

    print(
        "UL      =",
        final_performance.rate_ul,
    )

    print(
        "Sensing =",
        final_performance.rate_s,
    )

    print("\n[Weighted Sum Rate]")

    print(
        "Initial WSR =",
        wsr_history[0],
    )

    print(
        "Final WSR   =",
        wsr_history[-1],
    )

    print(
        "Inner iterations =",
        len(wsr_history) - 1,
    )

    print("\nStage 2 passed.")


if __name__ == "__main__":
    main()