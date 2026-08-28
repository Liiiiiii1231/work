"""只重跑 B6 multi-start MA + FP 的 rho_si 专项验证。"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from ma_baseline import run_ma_fp
from multistart import run_multi_start
from stage8 import (
    load_stage8_experiment_config,
    make_ma_start_seeds,
    make_scene_config,
)


CONFIG_PATH = ROOT / "configs" / "stage8_monte_carlo.yaml"
OUTPUT_DIR = ROOT / "results" / "stage8" / "diagnostics"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-mc", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config(CONFIG_PATH)
    exp = load_stage8_experiment_config(CONFIG_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int]] = []
    rho_values = tuple(float(v) for v in exp.rho_si_values)

    print("\nB6 multi-start MA + FP rho_si check")
    print(f"n_mc={args.n_mc}")
    print(f"MA n_starts={exp.ma_n_starts}")
    print(
        "rho_si=",
        ", ".join(f"{v:.1e}" for v in rho_values),
    )

    for offset in range(args.n_mc):
        scene_seed = exp.scene_seed_start + offset
        ma_seeds = make_ma_start_seeds(
            scene_seed=scene_seed,
            fixed_upa_seed=exp.fixed_upa_seed,
            n_starts=exp.ma_n_starts,
        )

        print(
            f"\nscene {scene_seed} | "
            f"MA start seeds={ma_seeds}"
        )

        for rho in rho_values:
            sweep_cfg = replace(cfg, rho_si=rho)
            sweep_cfg.validate()

            scene_cfg = make_scene_config(
                sweep_cfg,
                scene_seed=scene_seed,
                position_jitter=exp.position_jitter,
                randomize_fading=exp.fading_enabled,
                fading_model=exp.fading_model,
            )

            result = run_multi_start(
                name="B6: Multi-start MA + FP",
                seeds=ma_seeds,
                run_single=lambda seed: run_ma_fp(
                    scene_cfg,
                    rng=np.random.default_rng(seed),
                    verbose=False,
                    region_size=exp.ma_region_size,
                    min_spacing=exp.ma_min_spacing,
                    initial_step=exp.ma_initial_step,
                    randomize_initial_geometry=(
                        int(seed) != int(exp.fixed_upa_seed)
                    ),
                    initial_jitter_fraction=(
                        exp.ma_initial_jitter_fraction
                    ),
                ),
                iteration_getter=lambda item: item.iterations,
                cfg=scene_cfg,
                require_convergence=True,
                verbose=False,
            )

            start_wsrs = result.wsrs
            best = float(result.best_wsr)

            row: dict[str, float | int] = {
                "scene_seed": int(scene_seed),
                "rho_si": float(rho),
                "best_wsr": best,
                "best_seed": int(result.best_seed),
            }

            for index, value in enumerate(start_wsrs, start=1):
                row[f"start_{index}_wsr"] = float(value)

            rows.append(row)

            start_text = ", ".join(
                f"{value:.6f}" for value in start_wsrs
            )
            print(
                f"  rho={rho:.1e} | "
                f"starts=[{start_text}] | "
                f"best={best:.6f}"
            )

    # Scene x rho matrix.
    matrix: dict[int, dict[float, float]] = {}
    for row in rows:
        matrix.setdefault(int(row["scene_seed"]), {})[
            float(row["rho_si"])
        ] = float(row["best_wsr"])

    abnormal_scenes: list[int] = []

    print("\n" + "=" * 88)
    print("Monotonicity check")
    print("=" * 88)

    for scene_seed in sorted(matrix):
        values = np.asarray(
            [matrix[scene_seed][rho] for rho in rho_values],
            dtype=float,
        )
        steps = np.diff(values)
        abnormal = bool(np.any(steps > 1.0e-8))

        if abnormal:
            abnormal_scenes.append(scene_seed)

        print(
            f"scene {scene_seed}: "
            f"{'ABNORMAL' if abnormal else 'OK'} | "
            f"values={values} | steps={steps}"
        )

    means = np.asarray(
        [
            np.mean(
                [matrix[seed][rho] for seed in sorted(matrix)]
            )
            for rho in rho_values
        ],
        dtype=float,
    )

    print("\nAverage B6 WSR:")
    for rho, value in zip(rho_values, means):
        print(f"rho={rho:.1e}: {value:.6f}")

    print(
        f"\nAbnormal scenes: {abnormal_scenes}\n"
        f"Number abnormal = "
        f"{len(abnormal_scenes)}/{len(matrix)}"
    )

    csv_path = OUTPUT_DIR / "b6_multistart_rho_check.csv"
    fieldnames = [
        "scene_seed",
        "rho_si",
        "best_wsr",
        "best_seed",
        *[
            f"start_{index}_wsr"
            for index in range(1, exp.ma_n_starts + 1)
        ],
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
