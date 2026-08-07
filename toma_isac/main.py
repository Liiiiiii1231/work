"""ToMA-ISAC 第一大里程碑主程序。

当前版本暂时不运行 inner/outer 优化。

目的只是验证：

SystemConfig
    ↓
GeometryState
    ↓
ChannelState
    ↓
ResourceState
    ↓
PerformanceResult

整个系统模型是否能够正确数值运行。
"""

import numpy as np

from channels import (
    build_channels,
)
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
from metrics import (
    compute_performance,
)
from utils import (
    check_resource_constraints,
)


def main() -> None:
    """运行固定 ToMA 条件下的第一大里程碑。"""

    # ==========================================================
    # 1. 读取系统参数
    # ==========================================================

    cfg = load_config()

    # 固定随机数种子，
    # 保证每次调试得到同样的随机初始化结果。
    rng = np.random.default_rng(
        cfg.random_seed
    )

    # ==========================================================
    # 2. 生成初始 ToMA 端点
    # ==========================================================

    endpoints = (
        generate_feasible_endpoints(
            cfg,
            rng,
        )
    )

    # 根据 c_m 生成所有 Tx/Rx 阵元位置。
    geometry = build_geometry(
        endpoints,
        cfg,
    )

    # ==========================================================
    # 3. 根据当前 ToMA 几何生成所有信道
    # ==========================================================

    channels = build_channels(
        geometry,
        cfg,
    )

    # ==========================================================
    # 4. 初始化 Q、q_j、b_j、u_l
    # ==========================================================

    resources = (
        initialize_resources(
            geometry,
            channels,
            cfg,
            rng,
        )
    )

    # ==========================================================
    # 5. 检查资源变量是否满足约束
    # ==========================================================

    check_resource_constraints(
        resources,
        cfg,
    )

    # ==========================================================
    # 6. 计算真实 SINR、Rate 和 WSR
    # ==========================================================

    performance = (
        compute_performance(
            channels,
            resources,
            cfg,
        )
    )

    # ==========================================================
    # 7. 输出结果
    # ==========================================================

    print("=" * 64)
    print(
        "ToMA-ISAC First Milestone"
    )
    print("=" * 64)

    print(
        f"M={cfg.m_uav}, "
        f"N_c={cfg.n_cable}, "
        f"N_T={cfg.n_tx}, "
        f"N_R={cfg.n_rx}, "
        f"D={cfg.d_stream}"
    )

    # ----------------------------------------------------------
    # 数组尺寸检查
    # ----------------------------------------------------------

    print("\n[Shapes]")

    print(
        "endpoints :",
        geometry.endpoints.shape,
    )

    print(
        "tx_pos    :",
        geometry.tx_positions.shape,
    )

    print(
        "rx_pos    :",
        geometry.rx_positions.shape,
    )

    print(
        "h         :",
        channels.h.shape,
    )

    print(
        "f         :",
        channels.f.shape,
    )

    print(
        "G         :",
        channels.g.shape,
    )

    print(
        "H_SI0     :",
        channels.h_si0.shape,
    )

    print(
        "H_RSI     :",
        channels.h_rsi.shape,
    )

    print(
        "Q         :",
        resources.q_matrix.shape,
    )

    print(
        "q_ul      :",
        resources.q_ul.shape,
    )

    print(
        "b_ul      :",
        resources.b_ul.shape,
    )

    print(
        "u_s       :",
        resources.u_s.shape,
    )

    # ----------------------------------------------------------
    # 几何检查
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    # 资源约束
    # ----------------------------------------------------------

    print("\n[Resources]")

    print(
        "||Q||_F^2 =",
        np.linalg.norm(
            resources.q_matrix,
            "fro",
        )
        ** 2,
    )

    # 注意：
    # q_ul 是幅度。
    # q_ul**2 才是真实上行功率。
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

    # ----------------------------------------------------------
    # 三类 SINR
    # ----------------------------------------------------------

    print("\n[SINR]")

    print(
        "DL      :",
        performance.gamma_dl,
    )

    print(
        "UL      :",
        performance.gamma_ul,
    )

    print(
        "Sensing :",
        performance.gamma_s,
    )

    # ----------------------------------------------------------
    # 三类速率
    # ----------------------------------------------------------

    print(
        "\n[Rate: bit/s/Hz]"
    )

    print(
        "DL      :",
        performance.rate_dl,
    )

    print(
        "UL      :",
        performance.rate_ul,
    )

    print(
        "Sensing :",
        performance.rate_s,
    )

    # ----------------------------------------------------------
    # WSR
    # ----------------------------------------------------------

    print(
        "\n[Weighted Sum Rate]"
    )

    print(
        "WSR =",
        performance.weighted_sum_rate,
    )

    print(
        "\nFirst milestone passed."
    )


if __name__ == "__main__":
    main()