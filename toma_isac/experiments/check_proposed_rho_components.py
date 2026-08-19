"""诊断 Proposed 在 rho_si 扫描下的性能分量变化。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm import AlgorithmResult, run_joint_algorithm
from config import load_config
from stage8 import (
    load_stage8_experiment_config,
    make_scene_config,
)


CONFIG_PATH = (
    ROOT
    / "configs"
    / "stage8_monte_carlo.yaml"
)

OUTPUT_PATH = (
    ROOT
    / "results"
    / "stage8"
    / "check_proposed_rho_components.csv"
)


def run_best_proposed(
    cfg,
    seeds: tuple[int, ...],
) -> tuple[int, AlgorithmResult]:
    """按 Stage 8 的 best-of-S 规则运行 Proposed。"""

    best_seed = None
    best_result = None
    best_wsr = -np.inf

    for seed in seeds:
        result = run_joint_algorithm(
            cfg,
            rng=np.random.default_rng(seed),
            verbose=False,
        )

        if not result.converged:
            raise RuntimeError(
                "Proposed did not converge: "
                f"seed={seed}, "
                f"outer_iterations="
                f"{result.outer_iterations}."
            )

        wsr = float(
            result.performance.weighted_sum_rate
        )

        if not np.isfinite(wsr):
            raise RuntimeError(
                f"Non-finite WSR at seed={seed}."
            )

        if wsr > best_wsr:
            best_wsr = wsr
            best_seed = seed
            best_result = result

    if best_seed is None or best_result is None:
        raise RuntimeError(
            "No valid Proposed result."
        )

    return best_seed, best_result


def summarize_result(
    rho_si: float,
    best_seed: int,
    result: AlgorithmResult,
    cfg,
) -> dict:
    """提取 DL / UL / sensing 分量。"""

    performance = result.performance

    weighted_dl = float(
        cfg.weight_dl
        @ performance.rate_dl
    )

    weighted_ul = float(
        cfg.weight_ul
        @ performance.rate_ul
    )

    weighted_s = float(
        cfg.weight_s
        @ performance.rate_s
    )

    component_sum = (
        weighted_dl
        + weighted_ul
        + weighted_s
    )

    if not np.isclose(
        component_sum,
        performance.weighted_sum_rate,
        rtol=1e-10,
        atol=1e-10,
    ):
        raise RuntimeError(
            "Rate components do not sum to WSR."
        )

    return {
        "rho_si": float(rho_si),

        "best_seed": int(best_seed),

        "outer_iterations": int(
            result.outer_iterations
        ),

        "h_si0_norm": float(
            np.linalg.norm(
                result.channels.h_si0
            )
        ),

        "h_rsi_norm": float(
            np.linalg.norm(
                result.channels.h_rsi
            )
        ),

        "mean_gamma_dl": float(
            np.mean(
                performance.gamma_dl
            )
        ),

        "mean_gamma_ul": float(
            np.mean(
                performance.gamma_ul
            )
        ),

        "mean_gamma_s": float(
            np.mean(
                performance.gamma_s
            )
        ),

        "mean_rate_dl": float(
            np.mean(
                performance.rate_dl
            )
        ),

        "mean_rate_ul": float(
            np.mean(
                performance.rate_ul
            )
        ),

        "mean_rate_s": float(
            np.mean(
                performance.rate_s
            )
        ),

        "weighted_dl": weighted_dl,
        "weighted_ul": weighted_ul,
        "weighted_s": weighted_s,

        "wsr": float(
            performance.weighted_sum_rate
        ),
    }


def print_summary(
    row: dict,
) -> None:
    """打印一个 rho_si 点的结果。"""

    print("-" * 88)

    print(
        f"rho_si={row['rho_si']:.1e}, "
        f"best_seed={row['best_seed']}, "
        f"outer_iter="
        f"{row['outer_iterations']}"
    )

    print(
        "SI norms: "
        f"||H_SI0||_F="
        f"{row['h_si0_norm']:.6e}, "
        f"||H_RSI||_F="
        f"{row['h_rsi_norm']:.6e}"
    )

    print(
        "Mean SINR: "
        f"DL={row['mean_gamma_dl']:.6e}, "
        f"UL={row['mean_gamma_ul']:.6e}, "
        f"S={row['mean_gamma_s']:.6e}"
    )

    print(
        "Mean rate: "
        f"DL={row['mean_rate_dl']:.6f}, "
        f"UL={row['mean_rate_ul']:.6f}, "
        f"S={row['mean_rate_s']:.6f}"
    )

    print(
        "Weighted components: "
        f"DL={row['weighted_dl']:.6f}, "
        f"UL={row['weighted_ul']:.6f}, "
        f"S={row['weighted_s']:.6f}"
    )

    print(
        f"WSR={row['wsr']:.6f}"
    )


def save_csv(
    rows: list[dict],
) -> None:
    """保存诊断数据。"""

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scene-seed",
        type=int,
        default=8001,
    )

    parser.add_argument(
        "--n-starts",
        type=int,
        default=None,
        help=(
            "Use the first N Multi-start "
            "seeds. Omit for full Stage 8."
        ),
    )

    args = parser.parse_args()

    cfg = load_config(
        CONFIG_PATH
    )

    experiment_cfg = (
        load_stage8_experiment_config(
            CONFIG_PATH
        )
    )

    seeds = (
        experiment_cfg
        .multi_start_seeds
    )

    if args.n_starts is not None:

        if (
            args.n_starts <= 0
            or args.n_starts
            > len(seeds)
        ):
            raise ValueError(
                "--n-starts out of range."
            )

        seeds = seeds[
            :args.n_starts
        ]

    # 所有 rho_si 使用同一个物理场景。
    base_scene_cfg = (
        make_scene_config(
            cfg,
            scene_seed=args.scene_seed,
            position_jitter=(
                experiment_cfg
                .position_jitter
            ),
        )
    )

    print("=" * 88)

    print(
        "Proposed rho_si "
        "component sensitivity check"
    )

    print("=" * 88)

    print(
        f"Scene seed = "
        f"{args.scene_seed}"
    )

    print(
        f"Multi-start seeds = "
        f"{seeds}"
    )

    print(
        "rho_si values = "
        f"{experiment_cfg.rho_si_values}"
    )

    rows = []

    for rho_si in (
        experiment_cfg
        .rho_si_values
    ):

        test_cfg = replace(
            base_scene_cfg,
            rho_si=float(rho_si),
        )

        test_cfg.validate()

        (
            best_seed,
            best_result,
        ) = run_best_proposed(
            test_cfg,
            seeds=seeds,
        )

        row = summarize_result(
            rho_si=float(rho_si),
            best_seed=best_seed,
            result=best_result,
            cfg=test_cfg,
        )

        rows.append(
            row
        )

        print_summary(
            row
        )

    save_csv(
        rows
    )

    print("=" * 88)

    print(
        f"CSV saved to: "
        f"{OUTPUT_PATH}"
    )

    print(
        "Check completed."
    )


if __name__ == "__main__":
    main()