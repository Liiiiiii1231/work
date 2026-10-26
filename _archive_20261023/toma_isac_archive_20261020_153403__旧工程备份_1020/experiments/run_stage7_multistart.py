"""Stage 7：五个 baseline 与 Proposed 的 Multi-start 公平对比。"""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from multistart import run_stage7_multistart


CONFIG_PATH = (
    ROOT
    / "configs"
    / "stage7_multistart.yaml"
)

# seed 表示算法初始化，不是 Monte Carlo scene seed。
MULTI_START_SEEDS = (
    2026,
    2027,
    2028,
    2029,
    2030,
)

SCHEME_ORDER = (
    "b1",
    "b2",
    "b3",
    "b4",
    "b5",
    "proposed",
)


def _print_seed_table(results) -> None:
    """打印每个共同起点下六方案的最终 WSR。"""

    print("\n[Per-start Final WSR]")
    print(
        f"{'Seed':>6}  "
        f"{'B1':>10}  "
        f"{'B2':>10}  "
        f"{'B3':>10}  "
        f"{'B4':>10}  "
        f"{'B5':>10}  "
        f"{'Proposed':>10}"
    )

    for index, seed in enumerate(MULTI_START_SEEDS):
        print(
            f"{seed:>6d}  "
            f"{results['b1'].starts[index].final_wsr:>10.6f}  "
            f"{results['b2'].starts[index].final_wsr:>10.6f}  "
            f"{results['b3'].starts[index].final_wsr:>10.6f}  "
            f"{results['b4'].starts[index].final_wsr:>10.6f}  "
            f"{results['b5'].starts[index].final_wsr:>10.6f}  "
            f"{results['proposed'].starts[index].final_wsr:>10.6f}"
        )


def _print_best_summary(results) -> None:
    """打印各方案 best-of-S 结果及主要差值。"""

    print("\n[Best-of-S Summary]")

    for key in SCHEME_ORDER:
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
        "Proposed - B1 = "
        f"{proposed - results['b1'].best_wsr:.10f}"
    )
    print(
        "Proposed - B2 = "
        f"{proposed - results['b2'].best_wsr:.10f}"
    )
    print(
        "Proposed - B3 = "
        f"{proposed - results['b3'].best_wsr:.10f}"
    )
    print(
        "B1 - B4       = "
        f"{results['b1'].best_wsr - results['b4'].best_wsr:.10f}"
    )
    print(
        "B2 - B4       = "
        f"{results['b2'].best_wsr - results['b4'].best_wsr:.10f}"
    )
    print(
        "B3 - B5       = "
        f"{results['b3'].best_wsr - results['b5'].best_wsr:.10f}"
    )


def _report_fixed_upa_seed_sensitivity(results) -> None:
    """检查固定 UPA 方案是否对 Stage 7 seed 敏感。"""

    for key in ("b3", "b5"):
        values = results[key].wsrs

        if np.allclose(
            values,
            values[0],
            rtol=1e-10,
            atol=1e-10,
        ):
            print(
                f"\n{key.upper()} produced identical WSRs "
                "across all seeds; the current fixed-UPA "
                "setup is effectively seed-insensitive."
            )
        else:
            print(
                f"\n{key.upper()} is seed-sensitive under "
                "the current configuration; keep Multi-start "
                "for this scheme in later experiments."
            )


def main() -> None:
    """执行 Stage 7 Multi-start。"""

    cfg = load_config(CONFIG_PATH)

    print("=" * 90)
    print("Stage 7: Multi-start Comparison")
    print("=" * 90)

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
    _report_fixed_upa_seed_sensitivity(results)

    print("\nStage 7 multi-start comparison passed.")


if __name__ == "__main__":
    main()
