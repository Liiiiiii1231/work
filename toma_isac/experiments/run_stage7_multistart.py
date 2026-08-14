"""Stage 7：四方案 Multi-start 多起点公平对比。"""

from pathlib import Path

import numpy as np

from config import load_config
from multistart import run_stage7_multistart


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    ROOT
    / "configs"
    / "stage7_multistart.yaml"
)

# Stage 7 使用同一组起点公平比较全部方案。
# 这里的 seed 表示算法初始化，不是 Monte Carlo 场景随机性。
MULTI_START_SEEDS = (
    2026,
    2027,
    2028,
    2029,
    2030,
)


def _print_seed_table(results) -> None:
    """打印每个共同起点下四方案的最终 WSR。"""

    print("\n[Per-start Final WSR]")
    print(
        f"{'Seed':>6}  "
        f"{'B1':>12}  "
        f"{'B2':>12}  "
        f"{'B3':>12}  "
        f"{'Proposed':>12}"
    )

    for index, seed in enumerate(MULTI_START_SEEDS):
        print(
            f"{seed:>6d}  "
            f"{results['b1'].starts[index].final_wsr:>12.6f}  "
            f"{results['b2'].starts[index].final_wsr:>12.6f}  "
            f"{results['b3'].starts[index].final_wsr:>12.6f}  "
            f"{results['proposed'].starts[index].final_wsr:>12.6f}"
        )


def _print_best_summary(results) -> None:
    """打印各方案 best-of-S 结果。"""

    print("\n[Best-of-S Summary]")

    for key in ("b1", "b2", "b3", "proposed"):
        scheme = results[key]
        best = scheme.best_start

        print(
            f"{key.upper():>8}: "
            f"best WSR={best.final_wsr:.10f}, "
            f"seed={best.seed}, "
            f"iterations={best.iterations}, "
            f"best-run time={best.runtime_s:.3f}s, "
            f"total time={scheme.total_runtime_s:.3f}s"
        )

    proposed = results["proposed"].best_wsr

    print("\n[Best-of-S Differences]")
    print(
        "Proposed(best) - B1(best) = "
        f"{proposed - results['b1'].best_wsr:.10f}"
    )
    print(
        "Proposed(best) - B2(best) = "
        f"{proposed - results['b2'].best_wsr:.10f}"
    )
    print(
        "Proposed(best) - B3(best) = "
        f"{proposed - results['b3'].best_wsr:.10f}"
    )


def main() -> None:
    """执行 Stage 7 Multi-start。"""

    cfg = load_config(CONFIG_PATH)

    print("=" * 76)
    print("Stage 7: Multi-start Comparison")
    print("=" * 76)

    print(
        f"K={cfg.k_dl}, J={cfg.j_ul}, "
        f"L={cfg.l_target}, R_S={cfg.r_sensing}"
    )
    print(f"Seeds = {MULTI_START_SEEDS}")
    print(f"Number of starts = {len(MULTI_START_SEEDS)}")

    results = run_stage7_multistart(
        cfg,
        seeds=MULTI_START_SEEDS,
        verbose=True,
        require_convergence=True,
    )

    _print_seed_table(results)
    _print_best_summary(results)

    # B3 的几何固定。如果所有 seed 得到同一结果，说明当前配置下
    # 它没有实际的初始化敏感性；保留重复运行只是为了统一验证接口。
    if np.allclose(
        results["b3"].wsrs,
        results["b3"].wsrs[0],
        rtol=1e-10,
        atol=1e-10,
    ):
        print(
            "\nB3 produced identical WSRs across seeds; "
            "the current fixed-UPA setup is effectively "
            "seed-insensitive."
        )

    print("\nStage 7 multi-start comparison passed.")


if __name__ == "__main__":
    main()
