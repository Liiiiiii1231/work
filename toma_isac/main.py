"""ToMA-ISAC 完整单起点联合优化主程序。"""

import numpy as np

from algorithm import run_joint_algorithm
from config import load_config
from geometry import (
    min_endpoint_distance,
    min_tx_rx_element_distance,
)


def main() -> None:
    """运行完整的内层资源 + 外层 ToMA 联合优化。"""

    cfg = load_config()

    rng = np.random.default_rng(
        cfg.random_seed
    )

    print("=" * 64)
    print(
        "ToMA-ISAC Joint Optimization"
    )
    print("=" * 64)

    result = run_joint_algorithm(
        cfg,
        rng=rng,
        verbose=True,
    )

    geometry = result.geometry
    resources = result.resources
    performance = result.performance

    print(
        "\n[Final Geometry]"
    )

    print(
        geometry.endpoints
    )

    print(
        "endpoint norms =",
        np.linalg.norm(
            geometry.endpoints,
            axis=1,
        ),
    )

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

    print(
        "\n[Final Resources]"
    )

    print(
        "||Q||_F^2 =",
        np.linalg.norm(
            resources.q_matrix,
            "fro",
        ) ** 2,
    )

    print(
        "UL amplitudes q_j =",
        resources.q_ul,
    )

    print(
        "UL powers q_j^2 =",
        resources.q_ul ** 2,
    )

    print(
        "||b_j|| =",
        np.linalg.norm(
            resources.b_ul,
            axis=1,
        ),
    )

    print(
        "||u_l|| =",
        np.linalg.norm(
            resources.u_s,
            axis=1,
        ),
    )

    print(
        "\n[Final SINR]"
    )

    print(
        "DL      =",
        performance.gamma_dl,
    )

    print(
        "UL      =",
        performance.gamma_ul,
    )

    print(
        "Sensing =",
        performance.gamma_s,
    )

    print(
        "\n[Final Rate: bit/s/Hz]"
    )

    print(
        "DL      =",
        performance.rate_dl,
    )

    print(
        "UL      =",
        performance.rate_ul,
    )

    print(
        "Sensing =",
        performance.rate_s,
    )

    print(
        "\n[Joint Optimization]"
    )

    print(
        "Initial WSR =",
        result.joint_wsr_history[0],
    )

    print(
        "Final WSR   =",
        performance.weighted_sum_rate,
    )

    print(
        "Outer iterations =",
        result.outer_iterations,
    )

    print(
        "Converged =",
        result.converged,
    )

    print(
        "WSR history =",
        np.asarray(
            result.joint_wsr_history
        ),
    )

    print(
        "\nStage 4 passed."
    )


if __name__ == "__main__":
    main()