"""检查 Proposed 最终 ToMA 孔径与用户/目标距离是否匹配。

默认只检查正式场景 scene 8001。
如需检查 Stage 8 的 10 个 Monte Carlo 场景，把
SCENE_SEEDS 改为 tuple(range(8001, 8011))。

运行：
    python experiments/check_proposed_near_field.py

输出：
    results/near_field/proposed_near_field_check.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm import run_joint_algorithm  # noqa: E402
from config import load_config  # noqa: E402
from multistart import run_multi_start  # noqa: E402
from stage8 import (  # noqa: E402
    load_stage8_experiment_config,
    make_scene_config,
)


CONFIG_PATH = ROOT / "configs" / "stage8_monte_carlo.yaml"
OUTPUT_DIR = ROOT / "results" / "near_field"
CSV_PATH = OUTPUT_DIR / "proposed_near_field_check.csv"

# 先只跑一个正式场景，速度更快。
SCENE_SEEDS = (8001,)

# 如果要检查 Stage 8 全部 10 个场景，改成：
# SCENE_SEEDS = tuple(range(8001, 8011))


def array_aperture(positions: np.ndarray) -> float:
    """阵列孔径：最远两个阵元之间的距离。"""

    positions = np.asarray(positions, dtype=float)

    difference = (
        positions[:, None, :]
        - positions[None, :, :]
    )
    distances = np.linalg.norm(
        difference,
        axis=2,
    )

    return float(np.max(distances))


def near_field_bounds(
    aperture: float,
    wavelength: float,
) -> tuple[float, float]:
    """返回辐射近场下界和 Rayleigh 距离。"""

    r_f = 0.62 * np.sqrt(
        aperture**3 / wavelength
    )
    r_ray = 2.0 * aperture**2 / wavelength

    return float(r_f), float(r_ray)


def classify_range(
    distance: float,
    r_f: float,
    r_ray: float,
) -> str:
    """按当前论文定义判断链路所在区域。"""

    if distance <= r_f:
        return "below_radiative_near_field"
    if distance >= r_ray:
        return "far_field"
    return "radiative_near_field"


def append_link_rows(
    rows: list[dict],
    *,
    scene_seed: int,
    best_seed: int,
    wsr: float,
    object_type: str,
    points: np.ndarray,
    array_side: str,
    array_center: np.ndarray,
    aperture: float,
    r_f: float,
    r_ray: float,
) -> None:
    """保存某类用户/目标相对某一侧阵列的距离。"""

    for index, point in enumerate(
        np.asarray(points, dtype=float),
        start=1,
    ):
        distance = float(
            np.linalg.norm(point - array_center)
        )
        anchor_distance = float(
            np.linalg.norm(point)
        )

        rows.append(
            {
                "scene_seed": scene_seed,
                "best_seed": best_seed,
                "wsr": wsr,
                "object_type": object_type,
                "object_index": index,
                "array_side": array_side,
                "distance_from_array_center_m": distance,
                "distance_from_anchor_m": anchor_distance,
                "aperture_m": aperture,
                "fresnel_lower_bound_m": r_f,
                "rayleigh_distance_m": r_ray,
                "region": classify_range(
                    distance,
                    r_f,
                    r_ray,
                ),
            }
        )


def print_group(
    name: str,
    points: np.ndarray,
    center: np.ndarray,
    r_f: float,
    r_ray: float,
) -> None:
    """打印一组链路的距离和区域判断。"""

    print(f"\n{name}")

    for index, point in enumerate(
        np.asarray(points, dtype=float),
        start=1,
    ):
        distance = float(
            np.linalg.norm(point - center)
        )
        region = classify_range(
            distance,
            r_f,
            r_ray,
        )

        print(
            f"  {index}: "
            f"distance={distance:.4f} m, "
            f"region={region}"
        )


def run_one_scene(
    base_cfg,
    experiment_cfg,
    scene_seed: int,
) -> list[dict]:
    """只运行 Proposed，多起点后取 WSR 最大结果。"""

    scene_cfg = make_scene_config(
        base_cfg,
        scene_seed=scene_seed,
        position_jitter=experiment_cfg.position_jitter,
        randomize_fading=experiment_cfg.fading_enabled,
        fading_model=experiment_cfg.fading_model,
    )

    multi = run_multi_start(
        name="Proposed near-field check",
        seeds=experiment_cfg.multi_start_seeds,
        run_single=lambda seed: run_joint_algorithm(
            scene_cfg,
            rng=np.random.default_rng(seed),
            verbose=False,
            position_update_mode="fp_aware",
            fp_aware_candidate_iters=2,
        ),
        iteration_getter=lambda result: result.outer_iterations,
        cfg=scene_cfg,
        require_convergence=True,
        verbose=True,
    )

    best = multi.best_result
    geometry = best.geometry

    d_tx = array_aperture(
        geometry.tx_positions
    )
    d_rx = array_aperture(
        geometry.rx_positions
    )

    r_f_tx, r_ray_tx = near_field_bounds(
        d_tx,
        scene_cfg.wavelength,
    )
    r_f_rx, r_ray_rx = near_field_bounds(
        d_rx,
        scene_cfg.wavelength,
    )

    tx_center = np.mean(
        geometry.tx_positions,
        axis=0,
    )
    rx_center = np.mean(
        geometry.rx_positions,
        axis=0,
    )

    print("\n" + "=" * 72)
    print(f"Scene {scene_seed}")
    print(
        f"best seed = {multi.best_seed}, "
        f"WSR = {multi.best_wsr:.10f}"
    )

    print("\nFinal apertures")
    print(
        f"  D_T = {d_tx:.6f} m"
    )
    print(
        f"  D_R = {d_rx:.6f} m"
    )

    print("\nRadiative near-field ranges")
    print(
        f"  Tx: {r_f_tx:.6f} < r "
        f"< {r_ray_tx:.6f} m"
    )
    print(
        f"  Rx: {r_f_rx:.6f} < r "
        f"< {r_ray_rx:.6f} m"
    )

    print("\nArray centers")
    print(
        "  Tx center = "
        f"{np.array2string(tx_center, precision=4)}"
    )
    print(
        "  Rx center = "
        f"{np.array2string(rx_center, precision=4)}"
    )

    print_group(
        "DL users relative to Tx array",
        scene_cfg.q_dl,
        tx_center,
        r_f_tx,
        r_ray_tx,
    )

    print_group(
        "UL users relative to Rx array",
        scene_cfg.q_ul,
        rx_center,
        r_f_rx,
        r_ray_rx,
    )

    print_group(
        "Targets relative to Tx array",
        scene_cfg.q_s,
        tx_center,
        r_f_tx,
        r_ray_tx,
    )

    print_group(
        "Targets relative to Rx array",
        scene_cfg.q_s,
        rx_center,
        r_f_rx,
        r_ray_rx,
    )

    rows: list[dict] = []

    common = {
        "scene_seed": scene_seed,
        "best_seed": multi.best_seed,
        "wsr": multi.best_wsr,
    }

    append_link_rows(
        rows,
        **common,
        object_type="DL",
        points=scene_cfg.q_dl,
        array_side="Tx",
        array_center=tx_center,
        aperture=d_tx,
        r_f=r_f_tx,
        r_ray=r_ray_tx,
    )

    append_link_rows(
        rows,
        **common,
        object_type="UL",
        points=scene_cfg.q_ul,
        array_side="Rx",
        array_center=rx_center,
        aperture=d_rx,
        r_f=r_f_rx,
        r_ray=r_ray_rx,
    )

    append_link_rows(
        rows,
        **common,
        object_type="Target",
        points=scene_cfg.q_s,
        array_side="Tx",
        array_center=tx_center,
        aperture=d_tx,
        r_f=r_f_tx,
        r_ray=r_ray_tx,
    )

    append_link_rows(
        rows,
        **common,
        object_type="Target",
        points=scene_cfg.q_s,
        array_side="Rx",
        array_center=rx_center,
        aperture=d_rx,
        r_f=r_f_rx,
        r_ray=r_ray_rx,
    )

    bad_rows = [
        row
        for row in rows
        if row["region"]
        != "radiative_near_field"
    ]

    print("\nSummary")
    if not bad_rows:
        print(
            "  PASS: all checked links are in "
            "the radiative near-field range."
        )
    else:
        print(
            f"  WARNING: {len(bad_rows)} checked "
            "links are outside the radiative "
            "near-field range."
        )
        for row in bad_rows:
            print(
                "   - "
                f"{row['object_type']}"
                f"{row['object_index']} / "
                f"{row['array_side']}: "
                f"{row['distance_from_array_center_m']:.4f} m, "
                f"{row['region']}"
            )

    return rows


def main() -> None:
    base_cfg = load_config(CONFIG_PATH)
    experiment_cfg = (
        load_stage8_experiment_config(
            CONFIG_PATH
        )
    )

    print("Configuration")
    print(
        f"  lambda = {base_cfg.wavelength} m"
    )
    print(
        f"  cable length = "
        f"{base_cfg.cable_length} m"
    )
    print(
        f"  multi-start seeds = "
        f"{experiment_cfg.multi_start_seeds}"
    )

    all_rows: list[dict] = []

    for scene_seed in SCENE_SEEDS:
        all_rows.extend(
            run_one_scene(
                base_cfg,
                experiment_cfg,
                int(scene_seed),
            )
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "scene_seed",
        "best_seed",
        "wsr",
        "object_type",
        "object_index",
        "array_side",
        "distance_from_array_center_m",
        "distance_from_anchor_m",
        "aperture_m",
        "fresnel_lower_bound_m",
        "rayleigh_distance_m",
        "region",
    ]

    with CSV_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nCSV saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()
