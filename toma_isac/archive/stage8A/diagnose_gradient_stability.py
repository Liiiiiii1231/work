"""Stage 8A-3：数值黎曼梯度稳定性测试。

目的：
1. 在同一个固定资源、固定位置点上，仅改变 finite_diff_delta；
2. 比较不同 delta 得到的黎曼梯度大小和方向是否稳定；
3. 用球面回缩后的真实 WSR 局部变化，检查该梯度方向是否真的是上升方向；
4. 为后续选择 finite_diff_delta 和设置梯度停止准则提供依据。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from channels import build_channels
from config import SystemConfig, load_config
from geometry import build_geometry
from initialization import generate_feasible_endpoints, initialize_resources
from inner_solver import solve_inner_problem
from metrics import compute_performance
from outer_solver import (
    EndpointRCGDiagnostic,
    central_difference_gradient,
    create_outer_rcg_state,
    evaluate_fixed_resource_wsr,
    project_to_tangent,
    sphere_retraction,
    update_positions_once,
)
from stage8 import load_stage8_experiment_config, make_scene_config
from state import ResourceState
from utils import check_resource_constraints


CONFIG_PATH = ROOT / "configs" / "stage8_monte_carlo.yaml"
RESULT_DIR = ROOT / "results" / "stage8" / "diagnostics"
FIGURE_DIR = RESULT_DIR / "figures"
MONTE_CARLO_CSV = ROOT / "results" / "stage8" / "stage8_monte_carlo.csv"

DEFAULT_DELTAS = (1.0e-5, 3.0e-6, 1.0e-6, 3.0e-7, 1.0e-7)


def _find_previous_best_seed(scene_seed: int, fallback_seed: int) -> int:
    """优先复用 Stage 8 已保存的 Proposed best seed。"""

    if not MONTE_CARLO_CSV.exists():
        return fallback_seed

    with MONTE_CARLO_CSV.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if int(row["scene_seed"]) == scene_seed:
                return int(row["proposed_best_seed"])

    return fallback_seed


def _copy_resources(resources: ResourceState) -> dict[str, np.ndarray]:
    return {
        "q_matrix": resources.q_matrix.copy(),
        "q_ul": resources.q_ul.copy(),
        "b_ul": resources.b_ul.copy(),
        "u_s": resources.u_s.copy(),
    }


def _assert_resources_frozen(
    resources: ResourceState,
    snapshot: dict[str, np.ndarray],
) -> None:
    for name, expected in snapshot.items():
        if not np.array_equal(getattr(resources, name), expected):
            raise RuntimeError(f"Resource {name} changed during gradient test.")


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError("rows must not be empty.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-15 or nb <= 1e-15:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _retraction_directional_slope(
    endpoints: np.ndarray,
    endpoint_index: int,
    unit_direction: np.ndarray,
    resources: ResourceState,
    cfg: SystemConfig,
    probe_step: float,
) -> tuple[float, float, float]:
    """用球面回缩 ±h 计算真实局部方向导数。"""

    point = endpoints[endpoint_index]

    plus_endpoints = endpoints.copy()
    minus_endpoints = endpoints.copy()

    plus_endpoints[endpoint_index] = sphere_retraction(
        point,
        unit_direction,
        probe_step,
        cfg,
    )
    minus_endpoints[endpoint_index] = sphere_retraction(
        point,
        -unit_direction,
        probe_step,
        cfg,
    )

    f_plus = evaluate_fixed_resource_wsr(plus_endpoints, resources, cfg)
    f_minus = evaluate_fixed_resource_wsr(minus_endpoints, resources, cfg)

    slope = (f_plus - f_minus) / (2.0 * probe_step)
    return float(slope), float(f_plus), float(f_minus)


def _prepare_probe_state(
    cfg: SystemConfig,
    scene_seed: int,
    algorithm_seed: int,
    position_jitter: np.ndarray,
    max_probe_sweeps: int,
    verbose: bool,
) -> tuple[SystemConfig, np.ndarray, ResourceState, int, float, int, bool]:
    """运行一次 FP，再按当前 delta 推进纯 RCG，优先定位首次全端点线搜索失败点。"""

    scene_cfg = make_scene_config(
        cfg,
        scene_seed=scene_seed,
        position_jitter=position_jitter,
    )

    rng = np.random.default_rng(algorithm_seed)
    endpoints = generate_feasible_endpoints(scene_cfg, rng)
    geometry = build_geometry(endpoints, scene_cfg)
    channels = build_channels(geometry, scene_cfg)
    resources = initialize_resources(geometry, channels, scene_cfg, rng)

    check_resource_constraints(resources, scene_cfg)

    resources, inner_history = solve_inner_problem(
        channels,
        resources,
        scene_cfg,
        verbose=False,
    )
    inner_iterations = len(inner_history) - 1
    resource_snapshot = _copy_resources(resources)

    rcg_state = create_outer_rcg_state(scene_cfg)
    stalled = False
    reached_sweep = 0

    for sweep in range(1, max_probe_sweeps + 1):
        _assert_resources_frozen(resources, resource_snapshot)
        endpoint_records: list[EndpointRCGDiagnostic] = []

        geometry, channels, _, rcg_state, accepted_count = update_positions_once(
            geometry,
            resources,
            scene_cfg,
            rcg_state=rcg_state,
            verbose=False,
            diagnostic_records=endpoint_records,
        )
        reached_sweep = sweep

        max_grad = max(record.grad_norm for record in endpoint_records)
        max_backtrack = max(record.backtracking_steps for record in endpoint_records)

        if verbose:
            print(
                f"probe RCG {sweep:3d} | "
                f"accepted={accepted_count}/{scene_cfg.m_uav} | "
                f"grad_max={max_grad:.3e} | "
                f"backtrack_max={max_backtrack:2d}"
            )

        if accepted_count == 0:
            stalled = True
            break

    _assert_resources_frozen(resources, resource_snapshot)

    probe_wsr = float(
        compute_performance(channels, resources, scene_cfg).weighted_sum_rate
    )

    return (
        scene_cfg,
        geometry.endpoints.copy(),
        resources,
        inner_iterations,
        probe_wsr,
        reached_sweep,
        stalled,
    )


def run_gradient_stability_test(
    cfg: SystemConfig,
    scene_seed: int,
    algorithm_seed: int,
    position_jitter: np.ndarray,
    deltas: tuple[float, ...],
    max_probe_sweeps: int = 30,
    probe_step: float = 1.0e-7,
    verbose: bool = True,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """在同一个 probe 点上比较不同 finite_diff_delta 的黎曼梯度。"""

    if not deltas or any(delta <= 0.0 for delta in deltas):
        raise ValueError("All deltas must be positive.")
    if max_probe_sweeps <= 0:
        raise ValueError("max_probe_sweeps must be positive.")
    if probe_step <= 0.0:
        raise ValueError("probe_step must be positive.")

    (
        scene_cfg,
        endpoints,
        resources,
        inner_iterations,
        probe_wsr,
        reached_sweep,
        stalled,
    ) = _prepare_probe_state(
        cfg,
        scene_seed,
        algorithm_seed,
        position_jitter,
        max_probe_sweeps,
        verbose,
    )

    gradients: dict[tuple[int, float], np.ndarray] = {}

    # 先计算所有 delta 的梯度，确保比较时位置和资源完全相同。
    for endpoint_index in range(scene_cfg.m_uav):
        for delta in deltas:
            delta_cfg = replace(scene_cfg, finite_diff_delta=float(delta))
            euclidean_gradient = central_difference_gradient(
                endpoints,
                endpoint_index,
                resources,
                delta_cfg,
            )
            riemannian_gradient = project_to_tangent(
                endpoints[endpoint_index],
                euclidean_gradient,
                delta_cfg,
            )
            gradients[(endpoint_index, float(delta))] = riemannian_gradient

    # 最小 delta 作为方向比较参考；同时保留相邻 delta 的相似度，避免只依赖单一参考。
    reference_delta = min(deltas)
    sorted_deltas_desc = tuple(sorted(deltas, reverse=True))

    detailed_rows: list[dict[str, object]] = []

    for endpoint_index in range(scene_cfg.m_uav):
        reference_grad = gradients[(endpoint_index, float(reference_delta))]
        previous_grad: np.ndarray | None = None

        for delta in sorted_deltas_desc:
            grad = gradients[(endpoint_index, float(delta))]
            grad_norm = float(np.linalg.norm(grad))

            if grad_norm > 1e-15:
                unit_direction = grad / grad_norm
                observed_slope, f_plus, f_minus = _retraction_directional_slope(
                    endpoints,
                    endpoint_index,
                    unit_direction,
                    resources,
                    scene_cfg,
                    probe_step,
                )
                predicted_slope = grad_norm
                slope_relative_error = abs(observed_slope - predicted_slope) / max(
                    1.0,
                    abs(observed_slope),
                    abs(predicted_slope),
                )
                ascent_consistent = observed_slope > 0.0
            else:
                observed_slope = 0.0
                predicted_slope = 0.0
                slope_relative_error = 0.0
                f_plus = probe_wsr
                f_minus = probe_wsr
                ascent_consistent = True

            detailed_rows.append(
                {
                    "endpoint_index": endpoint_index,
                    "group": "Tx" if endpoint_index < scene_cfg.n_tx_uav else "Rx",
                    "delta": float(delta),
                    "grad_x": float(grad[0]),
                    "grad_y": float(grad[1]),
                    "grad_z": float(grad[2]),
                    "grad_norm": grad_norm,
                    "cos_to_smallest_delta": _cosine_similarity(grad, reference_grad),
                    "cos_to_previous_delta": (
                        float("nan")
                        if previous_grad is None
                        else _cosine_similarity(grad, previous_grad)
                    ),
                    "predicted_slope": predicted_slope,
                    "observed_retraction_slope": observed_slope,
                    "slope_relative_error": float(slope_relative_error),
                    "probe_f_plus": f_plus,
                    "probe_f_minus": f_minus,
                    "ascent_consistent": bool(ascent_consistent),
                }
            )

            previous_grad = grad

    summary_rows: list[dict[str, object]] = []
    for delta in sorted_deltas_desc:
        rows = [row for row in detailed_rows if float(row["delta"]) == float(delta)]
        norms = np.asarray([float(row["grad_norm"]) for row in rows])
        cosines = np.asarray([float(row["cos_to_smallest_delta"]) for row in rows])
        errors = np.asarray([float(row["slope_relative_error"]) for row in rows])

        summary_rows.append(
            {
                "delta": float(delta),
                "max_grad_norm": float(np.max(norms)),
                "mean_grad_norm": float(np.mean(norms)),
                "min_cos_to_smallest_delta": float(np.nanmin(cosines)),
                "mean_cos_to_smallest_delta": float(np.nanmean(cosines)),
                "max_slope_relative_error": float(np.max(errors)),
                "mean_slope_relative_error": float(np.mean(errors)),
                "ascent_consistent_count": sum(bool(row["ascent_consistent"]) for row in rows),
                "endpoint_count": scene_cfg.m_uav,
            }
        )

    metadata = {
        "scene_seed": scene_seed,
        "algorithm_seed": algorithm_seed,
        "one_time_fp_iterations": inner_iterations,
        "probe_wsr": probe_wsr,
        "probe_sweep": reached_sweep,
        "probe_is_stalled": stalled,
        "probe_step": probe_step,
        "reference_delta": reference_delta,
    }

    return detailed_rows, summary_rows, metadata


def _save_figures(
    detailed_rows: list[dict[str, object]],
    stem: str,
) -> tuple[Path, Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    norm_path = FIGURE_DIR / f"{stem}_norm.png"
    cosine_path = FIGURE_DIR / f"{stem}_cosine.png"

    endpoint_indices = sorted({int(row["endpoint_index"]) for row in detailed_rows})

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for endpoint_index in endpoint_indices:
        rows = sorted(
            [row for row in detailed_rows if int(row["endpoint_index"]) == endpoint_index],
            key=lambda row: float(row["delta"]),
        )
        x = np.asarray([float(row["delta"]) for row in rows])
        y = np.asarray([float(row["grad_norm"]) for row in rows])
        ax.plot(x, y, marker="o", label=f"endpoint {endpoint_index}")

    ax.set_xscale("log")
    ax.set_xlabel("finite_diff_delta")
    ax.set_ylabel("Riemannian gradient norm")
    ax.set_title("Gradient norm stability across finite-difference deltas")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(norm_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for endpoint_index in endpoint_indices:
        rows = sorted(
            [row for row in detailed_rows if int(row["endpoint_index"]) == endpoint_index],
            key=lambda row: float(row["delta"]),
        )
        x = np.asarray([float(row["delta"]) for row in rows])
        y = np.asarray([float(row["cos_to_smallest_delta"]) for row in rows])
        ax.plot(x, y, marker="o", label=f"endpoint {endpoint_index}")

    ax.set_xscale("log")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("finite_diff_delta")
    ax.set_ylabel("Cosine similarity to smallest-delta gradient")
    ax.set_title("Gradient direction stability across finite-difference deltas")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(cosine_path, dpi=180)
    plt.close(fig)

    return norm_path, cosine_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-seed", type=int, default=None)
    parser.add_argument("--algorithm-seed", type=int, default=None)
    parser.add_argument(
        "--max-probe-sweeps",
        type=int,
        default=30,
        help="Advance pure RCG until the first all-endpoint line-search stall, up to this budget.",
    )
    parser.add_argument(
        "--probe-step",
        type=float,
        default=1.0e-7,
        help="Small retraction step used to verify the true local directional derivative.",
    )
    args = parser.parse_args()

    cfg = load_config(CONFIG_PATH)
    experiment_cfg = load_stage8_experiment_config(CONFIG_PATH)

    scene_seed = (
        experiment_cfg.scene_seed_start
        if args.scene_seed is None
        else int(args.scene_seed)
    )
    fallback_seed = int(experiment_cfg.multi_start_seeds[0])
    algorithm_seed = (
        _find_previous_best_seed(scene_seed, fallback_seed)
        if args.algorithm_seed is None
        else int(args.algorithm_seed)
    )

    detailed_rows, summary_rows, metadata = run_gradient_stability_test(
        cfg,
        scene_seed=scene_seed,
        algorithm_seed=algorithm_seed,
        position_jitter=experiment_cfg.position_jitter,
        deltas=DEFAULT_DELTAS,
        max_probe_sweeps=args.max_probe_sweeps,
        probe_step=args.probe_step,
        verbose=True,
    )

    stem = f"gradient_stability_scene_{scene_seed}_seed_{algorithm_seed}"
    detail_path = RESULT_DIR / f"{stem}.csv"
    summary_path = RESULT_DIR / f"{stem}_summary.csv"

    _write_csv(detailed_rows, detail_path)
    _write_csv(summary_rows, summary_path)
    norm_figure, cosine_figure = _save_figures(detailed_rows, stem)

    print("\nStage 8A-3 gradient stability test finished.")
    print(f"probe sweep: {metadata['probe_sweep']}")
    print(f"probe stalled: {metadata['probe_is_stalled']}")
    print(f"probe WSR: {metadata['probe_wsr']:.10f}")
    print(f"reference delta: {metadata['reference_delta']:.1e}")
    print(f"directional probe step: {metadata['probe_step']:.1e}")
    print("\nDelta summary:")

    for row in summary_rows:
        print(
            f"delta={float(row['delta']):.1e} | "
            f"grad_max={float(row['max_grad_norm']):.3e} | "
            f"cos_min={float(row['min_cos_to_smallest_delta']):+.6f} | "
            f"cos_mean={float(row['mean_cos_to_smallest_delta']):+.6f} | "
            f"dir_err_mean={float(row['mean_slope_relative_error']):.3e} | "
            f"ascent={int(row['ascent_consistent_count'])}/{int(row['endpoint_count'])}"
        )

    print(f"\ndetail CSV:  {detail_path}")
    print(f"summary CSV: {summary_path}")
    print(f"norm figure: {norm_figure}")
    print(f"cos figure:  {cosine_figure}")


if __name__ == "__main__":
    main()
