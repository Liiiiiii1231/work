"""Stage 10：最终候选非均匀权重 weighted-sum 扫描（不使用 epsilon-constraint）。

目的：
1. 使用根据本模型敏感区确定的 16 个最终候选权重；
2. 每个权重点先执行原来的 formal multi-start；
3. 再用相邻权重点的最好解做 warm start，降低非凸局部最优敏感性；
4. 只按当前 omega_C 下的真实 weighted-sum objective 选择更好的候选；
5. 输出稳定化后的 R_C-R_S 数据，并检查 dominated / 非单调点。

注意：
- 本脚本仍然是 weighted-sum 方法；
- 不使用 epsilon-constraint；
- 不改变 FP、FP-aware RCG、MA baseline 的算法定义；
- warm start 只是额外初值，不改变各方案的可行域和目标函数。

运行：
    python experiments/run_stage10_final_weights.py --dry-run
    python experiments/run_stage10_final_weights.py --smoke
    python experiments/run_stage10_final_weights.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm import run_joint_algorithm
from baselines import run_fixed_toma_fp, run_fpa_upa_fp
from config import SystemConfig, load_config
from ma_baseline import run_ma_fp
from stage8 import load_stage8_experiment_config, make_scene_config
from experiments.run_stage10_tradeoff import (
    SCHEME_LABELS,
    SCHEME_ORDER,
    load_tradeoff_config,
    run_one_scheme,
)

CONFIG_PATH = ROOT / "configs" / "stage10_tradeoff.yaml"
RESULT_DIR = ROOT / "results" / "stage10_final_weights"
CSV_PATH = RESULT_DIR / "stage10_final_weights.csv"
DETAIL_CSV_PATH = RESULT_DIR / "stage10_final_weights_candidates.csv"
NPZ_PATH = RESULT_DIR / "stage10_final_weights.npz"

FINAL_OMEGA = (
    0.0, 0.05, 0.10, 0.20, 0.225, 0.25, 0.30, 0.40,
    0.50, 0.60, 0.65, 0.675, 0.70, 0.725, 0.75, 1.0,
)


@dataclass
class Candidate:
    omega_comm: float
    scheme: str
    source: str
    warm_from_omega: float | None
    result: object
    runtime_s: float

    @property
    def wsr(self) -> float:
        return float(self.result.performance.weighted_sum_rate)

    @property
    def converged(self) -> bool:
        return bool(self.result.converged)


@dataclass(frozen=True)
class StabilizedRecord:
    omega_comm: float
    omega_sensing: float
    scheme: str
    scheme_label: str
    sum_rate_dl: float
    sum_rate_ul: float
    sum_rate_comm: float
    sum_rate_sensing: float
    weighted_sum_rate: float
    selected_source: str
    warm_from_omega: float
    iterations: int
    runtime_s: float
    n_candidates: int
    converged: bool


def _distance_range(points: np.ndarray) -> tuple[float, float]:
    distance = np.linalg.norm(np.asarray(points, dtype=float), axis=1)
    return float(distance.min()), float(distance.max())


def _iteration_count(scheme: str, result: object) -> int:
    if scheme == "proposed":
        return int(result.outer_iterations)
    return int(result.iterations)


def _validate_formal_setting(base_cfg, experiment_cfg, tradeoff_cfg) -> None:
    if (base_cfg.k_dl, base_cfg.j_ul, base_cfg.l_target, base_cfg.r_sensing) != (
        5,
        5,
        5,
        4,
    ):
        raise RuntimeError("Stage 10 requires K=J=L=5 and R_s=4.")

    if experiment_cfg.fading_enabled:
        raise RuntimeError("Stage 10 requires fading.enabled=false.")

    expected_region = 30.0 * base_cfg.wavelength
    if not np.isclose(
        experiment_cfg.ma_region_size,
        expected_region,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            f"MA region_size={experiment_cfg.ma_region_size}, "
            f"expected 30lambda={expected_region}."
        )

    expected_spacing = 0.5 * base_cfg.wavelength
    if not np.isclose(
        experiment_cfg.ma_min_spacing,
        expected_spacing,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            f"MA min_spacing={experiment_cfg.ma_min_spacing}, "
            f"expected lambda/2={expected_spacing}."
        )


def _make_group_weights_with_endpoints(
    cfg: SystemConfig,
    omega_comm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """与原 Stage 10 权重定义一致，但允许 omega_C=0 和 1。"""

    omega_comm = float(omega_comm)
    if not 0.0 <= omega_comm <= 1.0:
        raise ValueError("omega_comm must lie in [0, 1].")

    omega_sensing = 1.0 - omega_comm
    weight_dl = np.full(cfg.k_dl, omega_comm / (2.0 * cfg.k_dl), dtype=float)
    weight_ul = np.full(cfg.j_ul, omega_comm / (2.0 * cfg.j_ul), dtype=float)
    weight_s = np.full(cfg.l_target, omega_sensing / cfg.l_target, dtype=float)

    total = float(weight_dl.sum() + weight_ul.sum() + weight_s.sum())
    if not np.isclose(total, 1.0, atol=1.0e-12):
        raise RuntimeError(f"Trade-off weights sum to {total}, expected 1.")

    return weight_dl, weight_ul, weight_s


def _cfg_at_weight(base_cfg: SystemConfig, omega: float) -> SystemConfig:
    weight_dl, weight_ul, weight_s = _make_group_weights_with_endpoints(base_cfg, omega)
    cfg = replace(
        base_cfg,
        weight_dl=weight_dl,
        weight_ul=weight_ul,
        weight_s=weight_s,
    )
    cfg.validate()
    return cfg


def _run_warm_candidate(
    scheme: str,
    cfg: SystemConfig,
    experiment_cfg,
    source_result: object,
    warm_from_omega: float,
    phase: str,
    rng_seed: int,
) -> Candidate:
    """用相邻权重点的完整最好解作为当前权重点的额外初值。"""

    rng = np.random.default_rng(int(rng_seed))
    start = perf_counter()

    if scheme == "b2":
        result = run_fixed_toma_fp(
            cfg,
            rng=rng,
            verbose=False,
            initial_endpoints=source_result.geometry.endpoints,
            initial_resources=source_result.resources,
        )

    elif scheme == "b3":
        # B3 的 UPA 几何固定，只 warm-start FP 资源。
        result = run_fpa_upa_fp(
            cfg,
            rng=rng,
            verbose=False,
            initial_resources=source_result.resources,
        )

    elif scheme == "b6":
        result = run_ma_fp(
            cfg,
            rng=rng,
            verbose=False,
            region_size=experiment_cfg.ma_region_size,
            min_spacing=experiment_cfg.ma_min_spacing,
            initial_step=experiment_cfg.ma_initial_step,
            randomize_initial_geometry=False,
            initial_tx_positions=source_result.geometry.tx_positions,
            initial_rx_positions=source_result.geometry.rx_positions,
            initial_resources=source_result.resources,
        )

    elif scheme == "proposed":
        result = run_joint_algorithm(
            cfg,
            rng=rng,
            verbose=False,
            position_update_mode="fp_aware",
            fp_aware_candidate_iters=2,
            initial_endpoints=source_result.geometry.endpoints,
            initial_resources=source_result.resources,
        )

    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    runtime_s = perf_counter() - start

    return Candidate(
        omega_comm=float(cfg.weight_dl.sum() + cfg.weight_ul.sum()),
        scheme=scheme,
        source=phase,
        warm_from_omega=float(warm_from_omega),
        result=result,
        runtime_s=float(runtime_s),
    )


def _base_candidate(
    omega: float,
    scheme: str,
    multi_result,
) -> Candidate:
    return Candidate(
        omega_comm=float(omega),
        scheme=scheme,
        source="formal_multistart",
        warm_from_omega=None,
        result=multi_result.best_result,
        runtime_s=float(multi_result.total_runtime_s),
    )


def _select_better(current: Candidate, challenger: Candidate) -> Candidate:
    """只按当前权重下的真实 WSR 选择候选；不做曲线形状筛选。"""

    if not challenger.converged:
        return current

    if challenger.wsr > current.wsr + 1.0e-10:
        return challenger

    return current


def _rates(candidate: Candidate) -> tuple[float, float, float, float]:
    performance = candidate.result.performance
    sum_rate_dl = float(np.sum(performance.rate_dl))
    sum_rate_ul = float(np.sum(performance.rate_ul))
    sum_rate_comm = sum_rate_dl + sum_rate_ul
    sum_rate_sensing = float(np.sum(performance.rate_s))
    return sum_rate_dl, sum_rate_ul, sum_rate_comm, sum_rate_sensing


def _make_record(
    omega: float,
    scheme: str,
    selected: Candidate,
    candidates: list[Candidate],
) -> StabilizedRecord:
    sum_rate_dl, sum_rate_ul, sum_rate_comm, sum_rate_sensing = _rates(selected)

    total_runtime = float(sum(c.runtime_s for c in candidates))

    return StabilizedRecord(
        omega_comm=float(omega),
        omega_sensing=float(1.0 - omega),
        scheme=scheme,
        scheme_label=SCHEME_LABELS[scheme],
        sum_rate_dl=sum_rate_dl,
        sum_rate_ul=sum_rate_ul,
        sum_rate_comm=sum_rate_comm,
        sum_rate_sensing=sum_rate_sensing,
        weighted_sum_rate=selected.wsr,
        selected_source=selected.source,
        warm_from_omega=(
            float("nan")
            if selected.warm_from_omega is None
            else float(selected.warm_from_omega)
        ),
        iterations=_iteration_count(scheme, selected.result),
        runtime_s=total_runtime,
        n_candidates=len(candidates),
        converged=selected.converged,
    )


def _save_records(
    records: list[StabilizedRecord],
    all_candidates: dict[tuple[str, float], list[Candidate]],
) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = list(StabilizedRecord.__dataclass_fields__)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))

    detail_fields = [
        "omega_comm",
        "scheme",
        "scheme_label",
        "candidate_source",
        "warm_from_omega",
        "sum_rate_comm",
        "sum_rate_sensing",
        "weighted_sum_rate",
        "iterations",
        "runtime_s",
        "converged",
    ]
    with DETAIL_CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=detail_fields)
        writer.writeheader()

        for (scheme, omega), candidates in sorted(
            all_candidates.items(), key=lambda item: (item[0][1], item[0][0])
        ):
            for candidate in candidates:
                _, _, rc, rs = _rates(candidate)
                writer.writerow(
                    {
                        "omega_comm": omega,
                        "scheme": scheme,
                        "scheme_label": SCHEME_LABELS[scheme],
                        "candidate_source": candidate.source,
                        "warm_from_omega": (
                            ""
                            if candidate.warm_from_omega is None
                            else candidate.warm_from_omega
                        ),
                        "sum_rate_comm": rc,
                        "sum_rate_sensing": rs,
                        "weighted_sum_rate": candidate.wsr,
                        "iterations": _iteration_count(scheme, candidate.result),
                        "runtime_s": candidate.runtime_s,
                        "converged": candidate.converged,
                    }
                )

    np.savez(
        NPZ_PATH,
        omega_comm=np.asarray([r.omega_comm for r in records]),
        omega_sensing=np.asarray([r.omega_sensing for r in records]),
        scheme=np.asarray([r.scheme for r in records]),
        sum_rate_comm=np.asarray([r.sum_rate_comm for r in records]),
        sum_rate_sensing=np.asarray([r.sum_rate_sensing for r in records]),
        weighted_sum_rate=np.asarray([r.weighted_sum_rate for r in records]),
    )

    print(f"\nCSV saved to: {CSV_PATH}")
    print(f"Candidate details saved to: {DETAIL_CSV_PATH}")
    print(f"NPZ saved to: {NPZ_PATH}")


def _print_diagnostics(records: list[StabilizedRecord]) -> None:
    print("\nStabilized trade-off diagnostic")

    for scheme in SCHEME_ORDER:
        rows = sorted(
            [r for r in records if r.scheme == scheme],
            key=lambda row: row.omega_comm,
        )

        comm = np.asarray([r.sum_rate_comm for r in rows], dtype=float)
        sensing = np.asarray([r.sum_rate_sensing for r in rows], dtype=float)

        comm_nondec = bool(np.all(np.diff(comm) >= -1.0e-8))
        sensing_noninc = bool(np.all(np.diff(sensing) <= 1.0e-8))

        dominated: list[float] = []
        for i, row in enumerate(rows):
            rc_i = row.sum_rate_comm
            rs_i = row.sum_rate_sensing
            for j, other in enumerate(rows):
                if i == j:
                    continue
                if (
                    other.sum_rate_comm >= rc_i - 1.0e-10
                    and other.sum_rate_sensing >= rs_i - 1.0e-10
                    and (
                        other.sum_rate_comm > rc_i + 1.0e-10
                        or other.sum_rate_sensing > rs_i + 1.0e-10
                    )
                ):
                    dominated.append(row.omega_comm)
                    break

        print(f"\n  {SCHEME_LABELS[scheme]}")
        print(f"    communication nondecreasing = {comm_nondec}")
        print(f"    sensing nonincreasing       = {sensing_noninc}")
        print(f"    dominated weights           = {dominated}")

        print("    adjacent jumps:")
        for left, right in zip(rows[:-1], rows[1:]):
            delta_c = right.sum_rate_comm - left.sum_rate_comm
            delta_s = right.sum_rate_sensing - left.sum_rate_sensing
            jump = float(np.hypot(delta_c, delta_s))
            print(
                f"      {left.omega_comm:.2f}->{right.omega_comm:.2f}: "
                f"dRC={delta_c:+.6f}, dRS={delta_s:+.6f}, jump={jump:.6f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate settings; no optimization.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use omega_C={0, 0.5, 1}, one formal start, and short iterations.",
    )
    args = parser.parse_args()

    base_cfg = load_config(CONFIG_PATH)
    experiment_cfg = load_stage8_experiment_config(CONFIG_PATH)
    tradeoff_cfg = load_tradeoff_config(CONFIG_PATH)
    _validate_formal_setting(base_cfg, experiment_cfg, tradeoff_cfg)

    physical_scene_cfg = make_scene_config(
        base_cfg,
        scene_seed=tradeoff_cfg.scene_seed,
        position_jitter=experiment_cfg.position_jitter,
        randomize_fading=False,
        fading_model=experiment_cfg.fading_model,
    )

    if args.smoke:
        omega_values = (0.0, 0.50, 1.0)
        working_cfg = replace(
            physical_scene_cfg,
            max_inner_iter=min(8, physical_scene_cfg.max_inner_iter),
            max_outer_iter=min(5, physical_scene_cfg.max_outer_iter),
        )
        formal_multistart = False
        require_convergence = False
    else:
        omega_values = FINAL_OMEGA
        working_cfg = physical_scene_cfg
        formal_multistart = True
        require_convergence = True

    dl_min, dl_max = _distance_range(working_cfg.q_dl)
    ul_min, ul_max = _distance_range(working_cfg.q_ul)
    s_min, s_max = _distance_range(working_cfg.q_s)

    print("Stage 10 final-candidate nonuniform weighted-sum sweep")
    print(f"  scene_seed={tradeoff_cfg.scene_seed}")
    print(f"  omega_C={omega_values}")
    print("  weights=16-point model-adaptive candidate set")
    print("  method=weighted-sum only")
    print("  stabilization=formal multistart + forward/backward neighbor warm start")
    print("  epsilon-constraint=NOT USED")
    print(f"  MA region={experiment_cfg.ma_region_size:.3f} m = 30lambda")
    print(f"  DL distance={dl_min:.3f}~{dl_max:.3f} m")
    print(f"  UL distance={ul_min:.3f}~{ul_max:.3f} m")
    print(f"  S  distance={s_min:.3f}~{s_max:.3f} m")

    if args.dry_run:
        print("\nExecution plan per scheme:")
        print("  1) run the original multi-start at each omega_C")
        print("  2) forward pass: warm-start omega_i from omega_{i-1}")
        print("  3) backward pass: warm-start omega_i from omega_{i+1}")
        print("  4) select the candidate with the largest current weighted WSR")
        print("\n[PASS] Dry run passed. No optimization was executed.")
        return

    selected: dict[tuple[str, float], Candidate] = {}
    all_candidates: dict[tuple[str, float], list[Candidate]] = {}

    # --------------------------------------------------------
    # Pass 1: 原 formal multi-start + forward continuation
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("PASS 1: formal multi-start + forward warm start")
    print("=" * 80)

    for scheme_index, scheme in enumerate(SCHEME_ORDER):
        print(f"\n[{SCHEME_LABELS[scheme]}]")
        previous: Candidate | None = None

        for weight_index, omega in enumerate(omega_values):
            cfg = _cfg_at_weight(working_cfg, omega)

            multi_result = run_one_scheme(
                scheme=scheme,
                cfg=cfg,
                experiment_cfg=experiment_cfg,
                scene_seed=tradeoff_cfg.scene_seed,
                require_convergence=require_convergence,
                formal_multistart=formal_multistart,
            )
            base = _base_candidate(omega, scheme, multi_result)
            candidates = [base]
            best = base

            if previous is not None:
                warm = _run_warm_candidate(
                    scheme=scheme,
                    cfg=cfg,
                    experiment_cfg=experiment_cfg,
                    source_result=previous.result,
                    warm_from_omega=previous.omega_comm,
                    phase="forward_warm",
                    rng_seed=(
                        910000
                        + 10000 * scheme_index
                        + 100 * weight_index
                        + int(round(omega * 100))
                    ),
                )
                candidates.append(warm)
                best = _select_better(best, warm)

            selected[(scheme, omega)] = best
            all_candidates[(scheme, omega)] = candidates
            previous = best

            print(
                f"  omega={omega:.2f}: "
                f"base={base.wsr:.8f}, "
                f"selected={best.wsr:.8f} ({best.source})"
            )

    # --------------------------------------------------------
    # Pass 2: backward continuation
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("PASS 2: backward warm start")
    print("=" * 80)

    for scheme_index, scheme in enumerate(SCHEME_ORDER):
        print(f"\n[{SCHEME_LABELS[scheme]}]")

        for reverse_index in range(len(omega_values) - 2, -1, -1):
            omega = omega_values[reverse_index]
            right_omega = omega_values[reverse_index + 1]
            cfg = _cfg_at_weight(working_cfg, omega)

            current = selected[(scheme, omega)]
            right = selected[(scheme, right_omega)]

            warm = _run_warm_candidate(
                scheme=scheme,
                cfg=cfg,
                experiment_cfg=experiment_cfg,
                source_result=right.result,
                warm_from_omega=right_omega,
                phase="backward_warm",
                rng_seed=(
                    920000
                    + 10000 * scheme_index
                    + 100 * reverse_index
                    + int(round(omega * 100))
                ),
            )

            all_candidates[(scheme, omega)].append(warm)
            better = _select_better(current, warm)
            selected[(scheme, omega)] = better

            print(
                f"  omega={omega:.2f}: "
                f"current={current.wsr:.8f}, "
                f"backward={warm.wsr:.8f}, "
                f"selected={better.wsr:.8f} ({better.source})"
            )

    records: list[StabilizedRecord] = []
    for omega in omega_values:
        for scheme in SCHEME_ORDER:
            key = (scheme, omega)
            records.append(
                _make_record(
                    omega=omega,
                    scheme=scheme,
                    selected=selected[key],
                    candidates=all_candidates[key],
                )
            )

    _save_records(records, all_candidates)
    _print_diagnostics(records)

    print("\n[PASS] Final-candidate weighted-sum sweep finished.")


if __name__ == "__main__":
    main()
