r"""Diagnose Proposed at omega_C=0.40/0.42 using 20 independent cold starts.

Place in toma_isac/experiments/. No formal result files are changed.
Run: python -u .\experiments\run_stage10_multistart20_probe.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from algorithm import run_joint_algorithm
from channels import build_channels
from config import load_config
from geometry import build_geometry, check_geometry_constraints
from initialization import generate_feasible_endpoints, initialize_resources
from metrics import compute_performance
from stage8 import load_stage8_experiment_config, make_scene_config
from utils import check_resource_constraints
from experiments.run_stage10_cold20 import make_group_weights

CONFIG = ROOT / "configs" / "stage10_tradeoff_cold20.yaml"
FORMAL_CSV = ROOT / "results/stage10/cold_wide20_dl114_tol1e4/stage10_tradeoff.csv"


def write_csv(path, rows):
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def weight_cfg(scene, omega):
    wd, wu, ws = make_group_weights(scene, omega)
    cfg = replace(scene, weight_dl=wd, weight_ul=wu, weight_s=ws)
    cfg.validate()
    return cfg


def initial_state(cfg, seed):
    rng = np.random.default_rng(seed)
    endpoints = generate_feasible_endpoints(cfg, rng)
    geometry = build_geometry(endpoints, cfg)
    channels = build_channels(geometry, cfg)
    resources = initialize_resources(geometry, channels, cfg, rng)
    check_resource_constraints(resources, cfg)
    return geometry, resources


def formal_wsrs():
    if not FORMAL_CSV.exists():
        return {}
    with FORMAL_CSV.open(encoding="utf-8-sig", newline="") as f:
        return {
            round(float(r["omega_comm"]), 12): float(r["weighted_sum_rate"])
            for r in csv.DictReader(f) if r["scheme"] == "proposed"
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", nargs="+", type=float, default=[0.40, 0.42])
    parser.add_argument("--dry-run", action="store_true",
                        help="Check all initializations without solving or writing results.")
    parser.add_argument("--smoke", action="store_true",
                        help="One seed and one outer iteration; NOT a performance experiment.")
    args = parser.parse_args()
    if args.dry_run and args.smoke:
        parser.error("Choose --dry-run or --smoke, not both.")
    if len(set(args.weights)) != len(args.weights) or any(
        not np.isfinite(w) or not 0 < w < 1 for w in args.weights
    ):
        parser.error("Weights must be unique finite values strictly between 0 and 1.")

    cfg = load_config(CONFIG)
    exp = load_stage8_experiment_config(CONFIG)
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if (cfg.k_dl, cfg.j_ul, cfg.l_target, cfg.r_sensing) != (5, 5, 5, 4):
        raise ValueError("Expected K=J=L=5, R_s=4.")
    if tuple(exp.multi_start_seeds) != tuple(range(2026, 2031)):
        raise ValueError("Original five seeds changed; review before running this probe.")
    if exp.fading_enabled:
        raise ValueError("Expected formal LoS experiment with fading disabled.")
    scene_seed = int(raw["tradeoff"]["scene_seed"])
    scene = make_scene_config(
        cfg, scene_seed=scene_seed, position_jitter=exp.position_jitter,
        randomize_fading=False, fading_model=exp.fading_model,
    )
    seeds = tuple(range(2026, 2046))
    print(f"scene={scene_seed}, weights={args.weights}, seeds=2026..2045", flush=True)
    print(f"tol_inner={cfg.tol_inner:g}, tol_outer={cfg.tol_outer:g}", flush=True)

    # Verify that the weight sweep uses identical initial states for each seed.
    for seed in seeds:
        reference = None
        for omega in args.weights:
            g, r = initial_state(weight_cfg(scene, omega), seed)
            fields = [g.endpoints, r.q_matrix, r.q_ul, r.b_ul, r.u_s]
            if reference is None:
                reference = [a.copy() for a in fields]
            elif not all(np.array_equal(a, b) for a, b in zip(reference, fields)):
                raise RuntimeError(f"Initial state differs across weights: seed={seed}")
    print("PASS: initial geometries/resources feasible and identical across weights per seed.", flush=True)
    if args.dry_run:
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suffix = "smoke" if args.smoke else "formal_settings"
    out = ROOT / "results/stage10/multistart20_probe" / f"{stamp}_{suffix}"
    out.mkdir(parents=True, exist_ok=False)
    hashes = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ["algorithm.py", "inner_solver.py", "outer_solver.py",
                     "initialization.py", "channels.py", "metrics.py",
                     "experiments/run_stage10_cold20.py"]
    }
    metadata = dict(config=raw, config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
                    code_sha256=hashes, weights=args.weights,
                    seeds=[2026] if args.smoke else list(seeds),
                    smoke=args.smoke, independent_cold_start=True,
                    position_update_mode="fp_aware", candidate_fp_iters=2)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Output: {out}", flush=True)
    if args.smoke:
        print("SMOKE ONLY: max_outer_iterations=1, no best-result selection.", flush=True)

    rows, summaries, winners = [], [], {}
    official = formal_wsrs()
    for omega in args.weights:
        current_cfg = weight_cfg(scene, omega)
        results = []
        for number, seed in enumerate([2026] if args.smoke else seeds, 1):
            print(f"omega={omega:g}: start {number}/{1 if args.smoke else 20}, seed={seed}", flush=True)
            row = dict(omega_comm=omega, seed=seed, initial_wsr="", final_wsr="",
                       rate_comm="", rate_sensing="", iterations="", converged=False,
                       stop_reason="", runtime_s=0.0, state_file="", error="")
            start = perf_counter()
            try:
                # Use the exact formal solver entry: fresh RNG, no warm-start state.
                result = run_joint_algorithm(
                    current_cfg, rng=np.random.default_rng(seed), verbose=False,
                    position_update_mode="fp_aware", fp_aware_candidate_iters=2,
                    max_outer_iterations=1 if args.smoke else None,
                )
                check_geometry_constraints(result.geometry, current_cfg)
                check_resource_constraints(result.resources, current_cfg)
                performance = compute_performance(result.channels, result.resources, current_cfg)
                rc = float(np.sum(performance.rate_dl) + np.sum(performance.rate_ul))
                rs = float(np.sum(performance.rate_s))
                wsr = float(performance.weighted_sum_rate)
                expected = omega * rc / 10 + (1 - omega) * rs / 5
                if not np.isfinite(wsr) or not np.isclose(wsr, expected, rtol=1e-10, atol=1e-10):
                    raise RuntimeError("Invalid final WSR or inconsistent group weights.")
                state_name = f"omega_{omega:.12g}_seed_{seed}.npz"
                np.savez_compressed(
                    out / state_name, endpoints=result.geometry.endpoints,
                    tx_positions=result.geometry.tx_positions, rx_positions=result.geometry.rx_positions,
                    q_matrix=result.resources.q_matrix, q_ul=result.resources.q_ul,
                    b_ul=result.resources.b_ul, u_s=result.resources.u_s,
                    joint_wsr_history=np.asarray(result.joint_wsr_history),
                    accepted_position_counts=np.asarray(result.accepted_position_counts),
                    inner_iteration_counts=np.asarray([len(h)-1 for h in result.inner_wsr_histories]),
                    omega_comm=omega, seed=seed, converged=result.converged,
                    stop_reason=result.stop_reason,
                )
                row.update(initial_wsr=result.joint_wsr_history[0], final_wsr=wsr,
                           rate_comm=rc, rate_sensing=rs, iterations=result.outer_iterations,
                           converged=bool(result.converged), stop_reason=result.stop_reason,
                           state_file=state_name)
                if result.converged and not args.smoke:
                    results.append((seed, wsr, result, row.copy()))
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["stop_reason"] = "exception"
            row["runtime_s"] = perf_counter() - start
            rows.append(row)
            write_csv(out / "all_starts.csv", rows)
            print(f"  WSR={row['final_wsr']}, converged={row['converged']}, "
                  f"reason={row['stop_reason']}, time={row['runtime_s']:.1f}s "
                  f"{row['error']}", flush=True)

        if args.smoke:
            continue
        best = max(results, key=lambda item: item[1]) if results else None
        first_five = [item for item in results if item[0] <= 2030]
        best5 = max(first_five, key=lambda item: item[1]) if first_five else None
        old_wsr = official.get(round(omega, 12))
        summary = dict(
            omega_comm=omega, n_attempted=20, n_converged=len(results),
            best20_seed=best[0] if best else "", best20_wsr=best[1] if best else "",
            best5_seed=best5[0] if best5 else "", best5_wsr=best5[1] if best5 else "",
            gain_vs_rerun5=best[1]-best5[1] if best and best5 else "",
            original_csv_wsr=old_wsr if old_wsr is not None else "",
            gain_vs_original=best[1]-old_wsr if best and old_wsr is not None else "",
            rerun5_minus_original=best5[1]-old_wsr if best5 and old_wsr is not None else "",
            best20_rate_comm=best[3]["rate_comm"] if best else "",
            best20_rate_sensing=best[3]["rate_sensing"] if best else "",
        )
        summaries.append(summary)
        write_csv(out / "summary.csv", summaries)
        print(f"SUMMARY: {summary}", flush=True)
        if best:
            winners[omega] = best[2]

    # Evaluate every selected final state at each other weight, without reoptimizing.
    # Same scene/constraints: these are genuine feasible candidate values.
    cross = []
    for source, result in winners.items():
        for target, target_result in winners.items():
            target_cfg = weight_cfg(scene, target)
            channels = build_channels(result.geometry, target_cfg)
            performance = compute_performance(channels, result.resources, target_cfg)
            gap = float(performance.weighted_sum_rate - target_result.performance.weighted_sum_rate)
            cross.append(dict(source_weight=source, target_weight=target,
                              cross_wsr=float(performance.weighted_sum_rate),
                              target_selected_wsr=float(target_result.performance.weighted_sum_rate),
                              improvement_over_target=gap))
            if source != target and gap > 1e-8:
                print(f"CROSS CHECK: state from {source:g} improves target {target:g} "
                      f"by {gap:.8g} WSR (no reoptimization).", flush=True)
    write_csv(out / "cross_evaluation.csv", cross)
    print(f"DONE: {out}", flush=True)
    if not args.smoke and len(winners) != len(args.weights):
        raise SystemExit("Some weights have no converged valid candidate; inspect all_starts.csv.")
    if any(r["error"] for r in rows):
        raise SystemExit("Some starts failed; available results saved. Inspect error column before interpreting.")


if __name__ == "__main__":
    main()
