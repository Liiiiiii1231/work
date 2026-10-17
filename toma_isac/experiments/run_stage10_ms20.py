r"""Stage 10 only: 20 cold ToMA starts, tol_inner=1e-6, tol_outer=1e-5.

Run from the project root:
    python -u .\experiments\run_stage10_ms20.py
    python -u .\experiments\run_stage10_ms20.py --resume
No old results are reused, except checkpoints belonging to this exact new run.
MA keeps its original 3 starts; FPA-UPA keeps 1 start.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from algorithm import run_joint_algorithm
from baselines import run_fixed_toma_fp, run_fpa_upa_fp
from channels import build_channels
from config import load_config
from geometry import check_geometry_constraints
from ma_baseline import run_ma_fp
from metrics import compute_performance
from stage8 import load_stage8_experiment_config, make_ma_start_seeds, make_scene_config
from utils import check_resource_constraints
from experiments.run_stage10_cold20 import make_group_weights

CONFIG = ROOT / "configs/stage10_tradeoff_ms20_in1e6_out1e5.yaml"
RUN_DIR = ROOT / "results/stage10/cold20_ms20_in1e6_out1e5"
SCHEMES = ("b2", "b3", "b6", "proposed")
LABELS = {"b2": "Fixed-ToMA + FP", "b3": "FPA-UPA + FP", "b6": "MA + FP",
          "proposed": "FP-aware RCG-ToMA + FP (Proposed)"}
START_FIELDS = ["omega_comm", "scheme", "seed", "sum_rate_dl", "sum_rate_ul",
                "sum_rate_comm", "sum_rate_sensing", "weighted_sum_rate", "iterations",
                "converged", "stop_reason", "runtime_s", "state_file", "state_sha256"]
PLOT_FIELDS = ["omega_comm", "omega_sensing", "scheme", "scheme_label", "sum_rate_dl",
               "sum_rate_ul", "sum_rate_comm", "sum_rate_sensing", "weighted_sum_rate",
               "best_seed", "iterations", "runtime_s", "n_starts", "converged"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows, fields):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def key(row):
    return round(float(row["omega_comm"]), 12), row["scheme"], int(row["seed"])


def is_true(value):
    return str(value).lower() in {"true", "1"}


def save_state(path, result, omega, scheme, seed):
    history = getattr(result, "joint_wsr_history", getattr(result, "wsr_history", []))
    tmp = path.with_suffix(".npz.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(
            f, omega_comm=omega, scheme=scheme, seed=seed,
            endpoints=result.geometry.endpoints,
            tx_positions=result.geometry.tx_positions, rx_positions=result.geometry.rx_positions,
            q_matrix=result.resources.q_matrix, q_ul=result.resources.q_ul,
            b_ul=result.resources.b_ul, u_s=result.resources.u_s,
            wsr_history=np.asarray(history),
            accepted_position_counts=np.asarray(getattr(result, "accepted_position_counts", [])),
            converged=result.converged,
        )
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="Resume only this new experiment.")
    parser.add_argument("--dry-run", action="store_true", help="Validate/print settings; no solving or writing.")
    parser.add_argument("--smoke", action="store_true", help="3 weights, 1 start, short run; separate folder.")
    parser.add_argument("--no-plot", action="store_true", help="Do not automatically draw the completed figure.")
    args = parser.parse_args()
    base = load_config(CONFIG)
    exp = load_stage8_experiment_config(CONFIG)
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    weights = tuple(float(w) for w in raw["tradeoff"]["omega_comm"])
    expected = (0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.35,
                0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9)
    if weights != expected:
        raise ValueError("Expected the unchanged formal 20-weight grid.")
    if tuple(exp.multi_start_seeds) != tuple(range(2026, 2046)):
        raise ValueError("Expected 20 ToMA seeds: 2026..2045.")
    if exp.ma_n_starts != 3:
        raise ValueError("This experiment retains the original 3 MA starts.")
    if (base.k_dl, base.j_ul, base.l_target, base.r_sensing) != (5, 5, 5, 4):
        raise ValueError("Expected K=J=L=5, R_s=4.")
    if exp.fading_enabled or base.tol_inner != 1e-6 or base.tol_outer != 1e-5:
        raise ValueError("Expected fading OFF, tol_inner=1e-6, tol_outer=1e-5.")
    scene_seed = int(raw["tradeoff"]["scene_seed"])
    scene = make_scene_config(base, scene_seed, exp.position_jitter,
                              randomize_fading=False, fading_model=exp.fading_model)
    ma_seeds = make_ma_start_seeds(scene_seed, exp.fixed_upa_seed, exp.ma_n_starts)
    plans = {"b2": tuple(exp.multi_start_seeds), "b3": (exp.fixed_upa_seed,),
             "b6": ma_seeds, "proposed": tuple(exp.multi_start_seeds)}
    if args.smoke:
        weights = (0.1, 0.5, 0.9)
        plans = {s: (seeds[0],) for s, seeds in plans.items()}
        scene = replace(scene, max_inner_iter=5, max_outer_iter=3)
    out = RUN_DIR.with_name(RUN_DIR.name + "_smoke") if args.smoke else RUN_DIR
    planned = len(weights) * sum(map(len, plans.values()))
    print(f"Stage 10 only: scene={scene_seed}, weights={weights}", flush=True)
    print(f"Starts: Fixed-ToMA={len(plans['b2'])}, FPA={len(plans['b3'])}, "
          f"MA={len(plans['b6'])}, Proposed={len(plans['proposed'])}", flush=True)
    print(f"tol_inner={scene.tol_inner:g}, tol_outer={scene.tol_outer:g}; "
          f"total single-start runs={planned}", flush=True)
    print(f"Independent cold starts; output={out}", flush=True)
    if args.dry_run:
        for omega in weights:
            wd, wu, ws = make_group_weights(scene, omega)
            replace(scene, weight_dl=wd, weight_ul=wu, weight_s=ws).validate()
        print("PASS. No files written and no optimization executed.", flush=True)
        return

    code_names = ["algorithm.py", "inner_solver.py", "outer_solver.py", "initialization.py",
                  "geometry.py", "channels.py", "config.py", "metrics.py", "utils.py",
                  "baselines.py", "ma_baseline.py", "stage8.py", "state.py",
                  "experiments/run_stage10_cold20.py"]
    fingerprint = dict(config_sha256=digest(CONFIG), code_sha256={n: digest(ROOT/n) for n in code_names},
                       runner_sha256=digest(Path(__file__)), smoke=args.smoke,
                       weights=list(weights), seeds={s: list(v) for s, v in plans.items()},
                       scene_seed=scene_seed, position_update_mode="fp_aware", candidate_fp_iters=2)
    manifest = out / "run_metadata.json"
    if out.exists():
        if not args.resume:
            raise FileExistsError(f"Output already exists: {out}. Use --resume.")
        if not manifest.exists():
            raise RuntimeError("Existing folder lacks metadata; choose a clean output folder.")
        prior = json.loads(manifest.read_text(encoding="utf-8"))
        if prior.get("fingerprint") != fingerprint:
            raise RuntimeError("Configuration/code differs from this checkpoint; cannot resume.")
    else:
        out.mkdir(parents=True)
        manifest.write_text(json.dumps(dict(fingerprint=fingerprint, config=raw), indent=2), encoding="utf-8")
    (out / "states").mkdir(exist_ok=True)
    rows = read_csv(out / "all_starts.csv")
    completed = {key(r) for r in rows}
    allowed = {(round(w, 12), s, int(seed)) for w in weights for s in SCHEMES for seed in plans[s]}
    if len(completed) != len(rows) or not completed.issubset(allowed):
        raise ValueError("Duplicate/unexpected checkpoint rows.")
    for r in rows:
        state_path = out / r["state_file"]
        if not state_path.is_file() or digest(state_path) != r["state_sha256"]:
            raise ValueError(f"Missing/changed checkpoint state: {state_path}")
        vals = [float(r[n]) for n in ["sum_rate_comm", "sum_rate_sensing", "weighted_sum_rate"]]
        w = float(r["omega_comm"])
        if not all(np.isfinite(vals)) or not np.isclose(vals[2], w*vals[0]/10+(1-w)*vals[1]/5, atol=1e-9, rtol=1e-9):
            raise ValueError("Invalid checkpoint objective values.")
    print(f"Checkpoint: {len(completed)}/{planned} starts finished.", flush=True)

    def summaries():
        result = []
        for w in weights:
            for s in SCHEMES:
                group = [r for r in rows if key(r)[:2] == (round(w, 12), s)]
                valid = [r for r in group if is_true(r["converged"])]
                if len(group) != len(plans[s]) or not valid:
                    continue
                best = max(valid, key=lambda r: float(r["weighted_sum_rate"]))
                result.append(dict(
                    omega_comm=w, omega_sensing=1-w, scheme=s, scheme_label=LABELS[s],
                    **{n: float(best[n]) for n in ["sum_rate_dl", "sum_rate_ul", "sum_rate_comm",
                                                  "sum_rate_sensing", "weighted_sum_rate"]},
                    best_seed=int(best["seed"]), iterations=int(best["iterations"]),
                    runtime_s=sum(float(r["runtime_s"]) for r in group),
                    n_starts=len(group), converged=True))
        write_csv(out / "stage10_tradeoff.csv", result, PLOT_FIELDS)
        return result

    total_start = perf_counter()
    for omega in weights:
        wd, wu, ws = make_group_weights(scene, omega)
        cfg = replace(scene, weight_dl=wd, weight_ul=wu, weight_s=ws)
        cfg.validate()
        for scheme in SCHEMES:
            for seed in plans[scheme]:
                current_key = (round(omega, 12), scheme, int(seed))
                if current_key in completed:
                    print(f"[RESUME] omega={omega:g}, {scheme}, seed={seed}: done", flush=True)
                    continue
                print(f"[{len(completed)+1}/{planned}] omega={omega:g}, "
                      f"{LABELS[scheme]}, seed={seed}", flush=True)
                start = perf_counter()
                rng = np.random.default_rng(seed)
                try:
                    if scheme == "b2":
                        r = run_fixed_toma_fp(cfg, rng=rng, verbose=False)
                    elif scheme == "b3":
                        r = run_fpa_upa_fp(cfg, rng=rng, verbose=False)
                    elif scheme == "b6":
                        r = run_ma_fp(cfg, rng=rng, verbose=False, region_size=exp.ma_region_size,
                                      min_spacing=exp.ma_min_spacing, initial_step=exp.ma_initial_step,
                                      randomize_initial_geometry=(seed != exp.fixed_upa_seed),
                                      initial_jitter_fraction=exp.ma_initial_jitter_fraction)
                    else:
                        r = run_joint_algorithm(cfg, rng=rng, verbose=False,
                                                position_update_mode="fp_aware", fp_aware_candidate_iters=2)
                    elapsed = perf_counter()-start
                    check_resource_constraints(r.resources, cfg)
                    if scheme in {"b2", "proposed"}:
                        check_geometry_constraints(r.geometry, cfg)
                    perf = compute_performance(build_channels(r.geometry, cfg), r.resources, cfg)
                    dl, ul, sensing = map(float, (np.sum(perf.rate_dl), np.sum(perf.rate_ul), np.sum(perf.rate_s)))
                    wsr = float(perf.weighted_sum_rate)
                    if not all(np.isfinite([dl, ul, sensing, wsr])):
                        raise RuntimeError("Nonfinite final rate/WSR.")
                    if not np.isclose(wsr, omega*(dl+ul)/10+(1-omega)*sensing/5, atol=1e-9, rtol=1e-9):
                        raise RuntimeError("Incorrect weighted objective.")
                    state_name = f"states/omega_{omega:.12g}_{scheme}_seed_{seed}.npz"
                    save_state(out / state_name, r, omega, scheme, seed)
                    history = getattr(r, "joint_wsr_history", getattr(r, "wsr_history", []))
                    reason = getattr(r, "stop_reason", "inner_tolerance" if r.converged else "max_inner_iterations")
                    if scheme == "proposed":
                        inner = r.inner_wsr_histories
                        trace = [dict(iteration=0, wsr=history[0], inner_end_wsr="", accepted_endpoints="")]
                        trace += [dict(iteration=i, wsr=v, inner_end_wsr=inner[i-1][-1],
                                       accepted_endpoints=r.accepted_position_counts[i-1])
                                  for i, v in enumerate(history[1:], 1)]
                        write_csv(out / f"states/omega_{omega:.12g}_{scheme}_seed_{seed}_history.csv",
                                  trace, ["iteration", "wsr", "inner_end_wsr", "accepted_endpoints"])
                    row = dict(omega_comm=omega, scheme=scheme, seed=int(seed), sum_rate_dl=dl,
                               sum_rate_ul=ul, sum_rate_comm=dl+ul, sum_rate_sensing=sensing,
                               weighted_sum_rate=wsr,
                               iterations=r.outer_iterations if scheme == "proposed" else r.iterations,
                               converged=bool(r.converged), stop_reason=reason, runtime_s=elapsed,
                               state_file=state_name, state_sha256=digest(out/state_name))
                    rows.append(row)
                    completed.add(current_key)
                    write_csv(out / "all_starts.csv", rows, START_FIELDS)
                    summaries()
                    print(f"  WSR={wsr:.10f}, converged={r.converged}, "
                          f"reason={reason}, time={elapsed:.1f}s", flush=True)
                except Exception as exc:
                    with (out / "errors.log").open("a", encoding="utf-8") as f:
                        f.write(f"omega={omega}, scheme={scheme}, seed={seed}: {type(exc).__name__}: {exc}\n")
                    print("Saved completed starts. Inspect errors.log; use --resume after resolving the error.", flush=True)
                    raise
    records = summaries()
    nonconverged = [r for r in rows if not is_true(r["converged"])]
    write_csv(out / "nonconverged_starts.csv", nonconverged, START_FIELDS)
    print(f"Completed starts={len(rows)}, converged={len(rows)-len(nonconverged)}, "
          f"plot points={len(records)}; elapsed this invocation={perf_counter()-total_start:.1f}s", flush=True)
    if args.smoke:
        print("SMOKE ONLY. No figure or formal measurements produced.", flush=True)
        return
    if len(records) != 80:
        raise RuntimeError("Some scheme/weight has no converged result; inspect nonconverged_starts.csv. No plot produced.")

    cross_rows = []
    for target in records:
        for source in records:
            if source["scheme"] != target["scheme"]:
                continue
            w = target["omega_comm"]
            cross = w*source["sum_rate_comm"]/10+(1-w)*source["sum_rate_sensing"]/5
            cross_rows.append(dict(scheme=target["scheme"], source_weight=source["omega_comm"],
                                   target_weight=w, cross_wsr=cross, target_wsr=target["weighted_sum_rate"],
                                   improvement_over_target=cross-target["weighted_sum_rate"]))
    write_csv(out / "cross_evaluation.csv", cross_rows,
              ["scheme", "source_weight", "target_weight", "cross_wsr", "target_wsr", "improvement_over_target"])
    gains = [r for r in cross_rows if r["improvement_over_target"] > 1e-8]
    print(f"Cross evaluation: {len(gains)} comparisons reveal a better sampled feasible candidate.", flush=True)
    if not args.no_plot:
        from experiments.plot_stage10_cold20 import load_data, render
        figures = out / "figures"
        figures.mkdir(exist_ok=True)
        render(load_data(out / "stage10_tradeoff.csv"), figures)
    print(f"DONE: {out}", flush=True)


if __name__ == "__main__":
    main()
