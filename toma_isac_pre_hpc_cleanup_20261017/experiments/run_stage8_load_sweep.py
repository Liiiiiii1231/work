"""Stage 8：System-load sweep。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from stage8 import (
    SCHEME_LABELS,
    SCHEME_ORDER,
    SceneSummary,
    load_stage8_experiment_config,
    run_monte_carlo,
    save_monte_carlo_results,
    summarize_monte_carlo,
)


CONFIG_PATHS = {
    3: ROOT / "configs" / "stage8_load3.yaml",
    4: ROOT / "configs" / "stage8_load4.yaml",
    5: ROOT / "configs" / "stage8_load5.yaml",
    6: ROOT / "configs" / "stage8_load6.yaml",
    7: ROOT / "configs" / "stage8_load7.yaml",
}

OUTPUT_DIR = ROOT / "results" / "stage8"
DETAIL_DIR = OUTPUT_DIR / "system_load"


def _validate_load_config(load: int, cfg) -> None:
    """确认 YAML 与指定 system load 一致。"""

    dimensions = (
        cfg.k_dl,
        cfg.j_ul,
        cfg.l_target,
    )

    if dimensions != (load, load, load):
        raise ValueError(
            f"load={load} requires K=J=L={load}, "
            f"but got {dimensions}."
        )

    expected_r_s = min(load, cfg.n_tx)

    if cfg.r_sensing != expected_r_s:
        raise ValueError(
            f"load={load} requires "
            f"R_s=min(L,N_T)={expected_r_s}, "
            f"but got R_s={cfg.r_sensing}."
        )


def _combined_fieldnames() -> list[str]:
    return [
        "system_load",
        "K",
        "J",
        "L",
        "R_s",
        "scene_seed",
        *[f"{key}_wsr" for key in SCHEME_ORDER],
    ]


def save_system_load_results(
    results_by_load: dict[int, list[SceneSummary]],
    output_dir: str | Path,
    stem: str = "stage8_sweep_system_load",
) -> tuple[Path, Path]:
    """保存 system-load sweep 的 CSV 和 NPZ。"""

    if len(results_by_load) == 0:
        raise ValueError("results_by_load must not be empty.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{stem}.csv"
    npz_path = output_dir / f"{stem}.npz"

    rows = []

    for load in sorted(results_by_load):
        scenes = results_by_load[load]

        if len(scenes) == 0:
            raise ValueError(f"load={load} has no scenes.")

        for scene in scenes:
            row: dict[str, float | int] = {
                "system_load": int(load),
                "K": int(load),
                "J": int(load),
                "L": int(load),
                "R_s": int(min(load, 4)),
                "scene_seed": int(scene.scene_seed),
            }

            for key in SCHEME_ORDER:
                row[f"{key}_wsr"] = float(
                    scene.schemes[key].wsr
                )

            rows.append(row)

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=_combined_fieldnames(),
        )
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(
        npz_path,
        system_load=np.asarray(
            [row["system_load"] for row in rows],
            dtype=int,
        ),
        scene_seed=np.asarray(
            [row["scene_seed"] for row in rows],
            dtype=int,
        ),
        scheme_order=np.asarray(SCHEME_ORDER),
        wsr=np.asarray(
            [
                [
                    row[f"{key}_wsr"]
                    for key in SCHEME_ORDER
                ]
                for row in rows
            ],
            dtype=float,
        ),
    )

    return csv_path, npz_path


def run_system_load_sweep(
    loads: tuple[int, ...] = (3, 4, 5, 6, 7),
    n_mc: int | None = None,
    verbose: bool = True,
) -> dict[int, list[SceneSummary]]:
    """运行 K=J=L=3/4/5/6/7 的七方案 Monte Carlo。"""

    if len(loads) == 0:
        raise ValueError("loads must not be empty.")

    if len(set(loads)) != len(loads):
        raise ValueError("loads must be unique.")

    results_by_load: dict[int, list[SceneSummary]] = {}

    for load in loads:
        if load not in CONFIG_PATHS:
            raise ValueError(
                f"Unsupported system load: {load}"
            )

        config_path = CONFIG_PATHS[load]

        cfg = load_config(config_path)
        experiment_cfg = load_stage8_experiment_config(
            config_path
        )

        _validate_load_config(load, cfg)

        if verbose:
            print("\n" + "=" * 80)
            print(
                f"System load = {load} "
                f"(K=J=L={load}, R_s={cfg.r_sensing})"
            )
            print("=" * 80)

        scenes = run_monte_carlo(
            cfg,
            experiment_cfg,
            n_mc=n_mc,
            require_convergence=True,
            verbose=verbose,
        )

        results_by_load[load] = scenes

        detail_dir = DETAIL_DIR / f"load_{load}"

        save_monte_carlo_results(
            scenes,
            detail_dir,
            stem=f"stage8_load{load}",
        )

        summary = summarize_monte_carlo(scenes)

        text = "  ".join(
            f"{SCHEME_LABELS[key]}="
            f"{summary[key][0]:.6f}"
            for key in SCHEME_ORDER
        )

        print(f"[load={load}] {text}")

        # 每跑完一个 load 就保存一次。
        save_system_load_results(
            results_by_load,
            OUTPUT_DIR,
        )

    return results_by_load


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--loads",
        nargs="+",
        type=int,
        choices=tuple(CONFIG_PATHS),
        default=[3, 4, 5, 6, 7],
        help="User/target loads to run. Default: 3 4 5 6 7.",
    )

    parser.add_argument(
        "--n-mc",
        type=int,
        default=None,
        help="Override experiment.n_mc for every load.",
    )

    args = parser.parse_args()

    loads = tuple(args.loads)

    results_by_load = run_system_load_sweep(
        loads=loads,
        n_mc=args.n_mc,
        verbose=True,
    )

    csv_path, npz_path = save_system_load_results(
        results_by_load,
        OUTPUT_DIR,
    )

    print(f"\nCSV saved to: {csv_path}")
    print(f"NPZ saved to: {npz_path}")
    print("Stage 8 system-load sweep passed.")


if __name__ == "__main__":
    main()