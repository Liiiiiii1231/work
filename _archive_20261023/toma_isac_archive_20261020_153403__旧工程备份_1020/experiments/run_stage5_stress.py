"""Stage 5：多用户 / 多目标联合优化压力测试。"""

from pathlib import Path
from time import perf_counter

import numpy as np

from algorithm import run_joint_algorithm
from config import load_config
from utils import check_resource_constraints


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    ROOT
    / "configs"
    / "stage5_multiuser.yaml"
)

# 同一多用户场景使用多个随机初始 ToMA。
TEST_SEEDS = (
    2026,
    2027,
    2028,
)


def main() -> None:
    """运行多个随机种子的多用户压力测试。"""

    cfg = load_config(
        CONFIG_PATH
    )

    print("=" * 72)
    print(
        "Stage 5: Multi-user / "
        "Multi-target Stress Test"
    )
    print("=" * 72)

    print(
        f"K={cfg.k_dl}, "
        f"J={cfg.j_ul}, "
        f"L={cfg.l_target}, "
        f"R_S={cfg.r_sensing}"
    )

    print(
        f"N_T={cfg.n_tx}, "
        f"N_R={cfg.n_rx}, "
        f"D={cfg.d_stream}"
    )

    final_wsrs = []

    for seed in TEST_SEEDS:
        print(
            "\n"
            + "-" * 72
        )

        print(
            f"Seed = {seed}"
        )

        rng = np.random.default_rng(
            seed
        )

        start_time = perf_counter()

        result = run_joint_algorithm(
            cfg,
            rng=rng,
            verbose=False,
        )

        elapsed = (
            perf_counter()
            - start_time
        )

        check_resource_constraints(
            result.resources,
            cfg,
        )

        history = np.asarray(
            result.joint_wsr_history,
            dtype=float,
        )
        
        if not np.all(
            np.isfinite(history)
        ):
            raise RuntimeError(
                "Non-finite WSR detected."
            )

        if np.any(
            np.diff(history) < -1e-7
        ):
            raise RuntimeError(
                "Joint WSR decreased."
            )

        final_wsr = float(
            result.performance
            .weighted_sum_rate
        )

        final_wsrs.append(
            final_wsr
        )

        print(
            "Initial WSR =",
            history[0],
        )

        print(
            "Final WSR   =",
            final_wsr,
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
            "Runtime [s] =",
            elapsed,
        )

        print(
            "DL rates =",
            result.performance.rate_dl,
        )

        print(
            "UL rates =",
            result.performance.rate_ul,
        )

        print(
            "Sensing rates =",
            result.performance.rate_s,
        )

    final_wsrs = np.asarray(
        final_wsrs,
        dtype=float,
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "[Stress Test Summary]"
    )

    print(
        "Final WSRs =",
        final_wsrs,
    )

    print(
        "Mean WSR =",
        np.mean(final_wsrs),
    )

    print(
        "Min WSR =",
        np.min(final_wsrs),
    )

    print(
        "Max WSR =",
        np.max(final_wsrs),
    )

    print(
        "\nStage 5 stress test passed."
    )


if __name__ == "__main__":
    main()