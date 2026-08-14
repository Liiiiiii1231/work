"""Stage 6：四方案单起点 Baseline 对比。"""

from pathlib import Path
from time import perf_counter

import numpy as np

from algorithm import run_joint_algorithm
from baselines import (
    run_fixed_toma_fp,
    run_fpa_upa_fp,
    run_rcg_toma_traditional,
)
from config import load_config
from utils import check_resource_constraints


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    ROOT
    / "configs"
    / "stage6_baselines.yaml"
)


def _run_timed(name, function):
    """执行一个方案并返回结果与运行时间。"""

    print("\n" + "-" * 72)
    print(name)

    start = perf_counter()
    result = function()
    elapsed = perf_counter() - start

    return result, elapsed


def main() -> None:
    """运行 B1、B2、B3 和 Proposed 的单起点公平对比。"""

    cfg = load_config(CONFIG_PATH)
    seed = cfg.random_seed

    print("=" * 72)
    print("Stage 6: Baseline Comparison")
    print("=" * 72)

    print(
        f"K={cfg.k_dl}, J={cfg.j_ul}, "
        f"L={cfg.l_target}, R_S={cfg.r_sensing}"
    )
    print(f"Single-start seed = {seed}")

    # 每个 ToMA 方案都重新创建同 seed RNG。
    # 因为三者的第一步都是 generate_feasible_endpoints()，
    # 所以初始 ToMA 完全一致。
    b1, time_b1 = _run_timed(
        "B1: RCG-ToMA + Traditional Resource Design",
        lambda: run_rcg_toma_traditional(
            cfg,
            rng=np.random.default_rng(seed),
            verbose=False,
        ),
    )

    b2, time_b2 = _run_timed(
        "B2: Fixed-ToMA + FP Resource Optimization",
        lambda: run_fixed_toma_fp(
            cfg,
            rng=np.random.default_rng(seed),
            verbose=False,
        ),
    )

    b3, time_b3 = _run_timed(
        "B3: FPA-UPA + FP Resource Optimization",
        lambda: run_fpa_upa_fp(
            cfg,
            rng=np.random.default_rng(seed),
            verbose=False,
        ),
    )

    proposed, time_proposed = _run_timed(
        "Proposed: RCG-ToMA + FP Resource Optimization",
        lambda: run_joint_algorithm(
            cfg,
            rng=np.random.default_rng(seed),
            verbose=False,
        ),
    )

    # --------------------------------------------------------
    # 基础合法性与收敛检查
    # --------------------------------------------------------

    for result in (b1, b2, b3):
        check_resource_constraints(
            result.resources,
            cfg,
        )

        if not np.isfinite(
            result.performance.weighted_sum_rate
        ):
            raise RuntimeError(
                f"{result.name} returned non-finite WSR."
            )

    check_resource_constraints(
        proposed.resources,
        cfg,
    )

    if not np.isfinite(
        proposed.performance.weighted_sum_rate
    ):
        raise RuntimeError(
            "Proposed returned non-finite WSR."
        )

    # B2/B3 和 Proposed 都使用收敛型优化器，正式实验要求充分收敛。
    if not b2.converged:
        raise RuntimeError(
            "B2 FP solver did not converge."
        )

    if not b3.converged:
        raise RuntimeError(
            "B3 FP solver did not converge."
        )

    if not proposed.converged:
        raise RuntimeError(
            "Proposed joint algorithm did not converge."
        )

    # B1 也给出收敛标记；若未收敛，说明需要增加外层上限，
    # 而不是直接拿截断值进行公平比较。
    if not b1.converged:
        raise RuntimeError(
            "B1 did not converge within max_outer_iter."
        )

    wsr_b1 = float(
        b1.performance.weighted_sum_rate
    )
    wsr_b2 = float(
        b2.performance.weighted_sum_rate
    )
    wsr_b3 = float(
        b3.performance.weighted_sum_rate
    )
    wsr_proposed = float(
        proposed.performance.weighted_sum_rate
    )

    print("\n" + "=" * 72)
    print("[Stage 6 Summary]")

    print(f"B1 WSR       = {wsr_b1:.10f}")
    print(f"B2 WSR       = {wsr_b2:.10f}")
    print(f"B3 WSR       = {wsr_b3:.10f}")
    print(f"Proposed WSR = {wsr_proposed:.10f}")

    print("\n[Convergence]")
    print(
        f"B1: converged={b1.converged}, "
        f"iterations={b1.iterations}"
    )
    print(
        f"B2: converged={b2.converged}, "
        f"iterations={b2.iterations}"
    )
    print(
        f"B3: converged={b3.converged}, "
        f"iterations={b3.iterations}"
    )
    print(
        f"Proposed: converged={proposed.converged}, "
        f"iterations={proposed.outer_iterations}"
    )

    print("\n[Runtime / s]")
    print(f"B1       = {time_b1:.3f}")
    print(f"B2       = {time_b2:.3f}")
    print(f"B3       = {time_b3:.3f}")
    print(f"Proposed = {time_proposed:.3f}")

    print("\n[Main Ablation Differences]")
    print(
        "Proposed - B1 "
        "(FP resource-design advantage) = "
        f"{wsr_proposed - wsr_b1:.10f}"
    )
    print(
        "Proposed - B2 "
        "(ToMA position-optimization gain) = "
        f"{wsr_proposed - wsr_b2:.10f}"
    )
    print(
        "Proposed - B3 "
        "(optimized ToMA vs dense FPA-UPA) = "
        f"{wsr_proposed - wsr_b3:.10f}"
    )

    print("\nStage 6 baseline comparison passed.")


if __name__ == "__main__":
    main()
