r"""Continue selected saved Stage 10 candidates, then summarize and plot.

Place in toma_isac/experiments/; run from the project root:
    python -u .\experiments\run_stage10_continue_and_plot.py
    python -u .\experiments\run_stage10_continue_and_plot.py --resume

Positions/resources are inherited. Outer search history is reset per segment.
The source 880-start experiment, YAML files and solvers are never overwritten.
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
from baselines import build_fpa_upa_geometry, run_fixed_toma_fp, run_fpa_upa_fp
from channels import build_channels
from config import load_config
from geometry import build_geometry, check_geometry_constraints
from ma_baseline import run_ma_fp, _build_ma_geometry
from metrics import compute_performance
from stage8 import load_stage8_experiment_config, make_scene_config
from state import ResourceState
from utils import check_resource_constraints, relative_change
from experiments.run_stage10_cold20 import make_group_weights
from experiments.plot_stage10_cold20 import load_data, render

SOURCE_DIR = ROOT / "results/stage10/cold20_ms20_in1e6_out1e5"
OUTPUT_DIR = ROOT / "results/stage10/cold20_ms20_in1e6_out1e5_continued"
CONFIG = ROOT / "configs/stage10_tradeoff_ms20_in1e6_out1e5.yaml"
SCHEMES = ("b2", "b3", "b6", "proposed")
LABELS = {"b2": "Fixed-ToMA + FP", "b3": "FPA-UPA + FP", "b6": "MA + FP",
          "proposed": "FP-aware RCG-ToMA + FP (Proposed)"}
PLOT_FIELDS = ["omega_comm", "omega_sensing", "scheme", "scheme_label", "sum_rate_dl",
               "sum_rate_ul", "sum_rate_comm", "sum_rate_sensing", "weighted_sum_rate",
               "best_seed", "iterations", "runtime_s", "n_starts", "converged"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def dump_json(path, obj):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def truth(value):
    return str(value).lower() in {"true", "1"}


def group_key(row):
    return round(float(row["omega_comm"]), 12), row["scheme"]


def candidate_id(row):
    return f"omega_{float(row['omega_comm']):.12g}_{row['scheme']}_seed_{int(row['seed'])}"


def find_tasks(rows):
    """One highest unfinished candidate per missing / improvable group."""
    tasks = []
    for group in sorted({group_key(r) for r in rows}, key=lambda x: (x[0], SCHEMES.index(x[1]))):
        members = [r for r in rows if group_key(r) == group]
        valid = [r for r in members if truth(r["converged"])]
        unfinished = [r for r in members if not truth(r["converged"])]
        if not unfinished:
            continue
        top = max(unfinished, key=lambda r: float(r["weighted_sum_rate"]))
        best_valid = max((float(r["weighted_sum_rate"]) for r in valid), default=None)
        if best_valid is None or float(top["weighted_sum_rate"]) > best_valid + 1e-8:
            tasks.append(dict(candidate_id=candidate_id(top), omega_comm=group[0], scheme=group[1],
                              seed=int(top["seed"]), source_wsr=float(top["weighted_sum_rate"]),
                              source_iterations=int(top["iterations"]), source_state_file=top["state_file"],
                              source_state_sha256=top["state_sha256"],
                              reason="missing_point" if best_valid is None else "better_unconverged",
                              previous_best_converged=best_valid))
    return tasks


def weighted_cfg(scene, omega):
    wd, wu, ws = make_group_weights(scene, omega)
    cfg = replace(scene, weight_dl=wd, weight_ul=wu, weight_s=ws)
    cfg.validate()
    return cfg


def load_state(path, cfg, scheme, omega, seed):
    with np.load(path, allow_pickle=False) as z:
        if not np.isclose(float(z["omega_comm"]), omega) or str(z["scheme"]) != scheme or int(z["seed"]) != seed:
            raise ValueError(f"State identity mismatch: {path}")
        resources = ResourceState(**{n: z[n].copy() for n in ["q_matrix", "q_ul", "b_ul", "u_s"]})
        endpoints, tx, rx = z["endpoints"].copy(), z["tx_positions"].copy(), z["rx_positions"].copy()
    if scheme in {"b2", "proposed"}:
        geometry = build_geometry(endpoints, cfg)
    elif scheme == "b3":
        geometry = build_fpa_upa_geometry(cfg)
    else:
        geometry = _build_ma_geometry(tx, rx, cfg)
    if not (np.allclose(geometry.tx_positions, tx, atol=1e-12, rtol=0)
            and np.allclose(geometry.rx_positions, rx, atol=1e-12, rtol=0)):
        raise ValueError("Reconstructed geometry differs from stored array positions.")
    check_resource_constraints(resources, cfg)
    performance = compute_performance(build_channels(geometry, cfg), resources, cfg)
    if not np.isfinite(performance.weighted_sum_rate):
        raise ValueError("Nonfinite initial WSR.")
    return geometry, resources, performance


def save_state(path, geometry, resources, task, converged, history):
    temp = path.with_suffix(".npz.tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, omega_comm=task["omega_comm"], scheme=task["scheme"], seed=task["seed"],
                            endpoints=geometry.endpoints, tx_positions=geometry.tx_positions,
                            rx_positions=geometry.rx_positions, q_matrix=resources.q_matrix,
                            q_ul=resources.q_ul, b_ul=resources.b_ul, u_s=resources.u_s,
                            converged=converged, wsr_history=np.asarray(history))
    temp.replace(path)


def execute_segment(cfg, exp, task, geometry, resources, inner_chunk, outer_chunk):
    scheme, seed = task["scheme"], task["seed"]
    rng = np.random.default_rng(seed)
    if scheme in {"b2", "b3"}:
        local = replace(cfg, max_inner_iter=inner_chunk)
        if scheme == "b2":
            result = run_fixed_toma_fp(local, rng=rng, initial_endpoints=geometry.endpoints.copy(),
                                       initial_resources=resources, verbose=False)
        else:
            result = run_fpa_upa_fp(local, rng=rng, initial_resources=resources, verbose=False)
    elif scheme == "b6":
        result = run_ma_fp(cfg, rng=rng, verbose=False, max_outer_iterations=outer_chunk,
                           initial_tx_positions=geometry.tx_positions.copy(),
                           initial_rx_positions=geometry.rx_positions.copy(), initial_resources=resources,
                           region_size=exp.ma_region_size, min_spacing=exp.ma_min_spacing,
                           initial_step=exp.ma_initial_step, initial_jitter_fraction=exp.ma_initial_jitter_fraction)
    else:
        result = run_joint_algorithm(cfg, rng=rng, verbose=False, max_outer_iterations=outer_chunk,
                                     initial_endpoints=geometry.endpoints.copy(), initial_resources=resources,
                                     position_update_mode="fp_aware", fp_aware_candidate_iters=2)
    check_resource_constraints(result.resources, cfg)
    if scheme in {"b2", "proposed"}:
        check_geometry_constraints(result.geometry, cfg)
    history = getattr(result, "joint_wsr_history", getattr(result, "wsr_history", []))
    iterations = result.outer_iterations if scheme == "proposed" else result.iterations
    reason = getattr(result, "stop_reason", "inner_tolerance" if result.converged else "max_inner_iterations")
    perf = compute_performance(build_channels(result.geometry, cfg), result.resources, cfg)
    return result, perf, list(history), int(iterations), reason


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="List candidate plan; no NPZ loading, solving or writing.")
    parser.add_argument("--plot-only", action="store_true", help="Summarize/replot current saved continuation results.")
    parser.add_argument("--smoke", action="store_true", help="Short FPA interface check, separate output directory.")
    parser.add_argument("--inner-chunk", type=int, default=5000)
    parser.add_argument("--outer-chunk", type=int, default=200)
    parser.add_argument("--inner-rounds", type=int, default=4)
    parser.add_argument("--outer-rounds", type=int, default=15)
    args = parser.parse_args()
    if min(args.inner_chunk, args.outer_chunk, args.inner_rounds, args.outer_rounds) <= 0:
        parser.error("All segment budgets must be positive.")
    if args.smoke and args.plot_only:
        parser.error("Smoke runs cannot produce a figure.")
    source, out = args.source_dir.resolve(), args.output_dir.resolve()
    if args.smoke:
        out = out.with_name(out.name + "_smoke")
    if source == out or source in out.parents or out in source.parents:
        raise ValueError("Output must be separate from the source result tree.")
    source_csv = source / "all_starts.csv"
    metadata = json.loads((source / "run_metadata.json").read_text(encoding="utf-8"))
    old_fp = metadata["fingerprint"]
    if old_fp["smoke"]:
        raise ValueError("Cannot continue a smoke experiment.")
    if sha(CONFIG) != old_fp["config_sha256"]:
        raise ValueError("Current Stage 10 config differs from the source run. Restore it first.")
    for name, value in old_fp["code_sha256"].items():
        if sha(ROOT / name) != value:
            raise ValueError(f"Solver code changed since the source run: {name}")
    rows = read_csv(source_csv)
    expected_keys = {(round(float(w), 12), s, int(seed)) for w in old_fp["weights"]
                     for s, seeds in old_fp["seeds"].items() for seed in seeds}
    actual_keys = [(group_key(r)[0], r["scheme"], int(r["seed"])) for r in rows]
    if len(rows) != 880 or len(set(actual_keys)) != 880 or set(actual_keys) != expected_keys:
        raise ValueError("Expected all 880 unique original starts. Check source-dir.")
    for row in rows:
        w = float(row["omega_comm"])
        values = [float(row[n]) for n in ["sum_rate_comm", "sum_rate_sensing", "weighted_sum_rate"]]
        if not all(np.isfinite(values)) or not np.isclose(values[2], w*values[0]/10+(1-w)*values[1]/5, atol=1e-9, rtol=1e-9):
            raise ValueError("Invalid source CSV rates/objective.")
    base, exp = load_config(CONFIG), load_stage8_experiment_config(CONFIG)
    if base.tol_inner != 1e-6 or base.tol_outer != 1e-5 or exp.fading_enabled:
        raise ValueError("Expected unchanged tol_inner=1e-6, tol_outer=1e-5, fading OFF.")
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    scene = make_scene_config(base, int(raw["tradeoff"]["scene_seed"]), exp.position_jitter,
                              randomize_fading=False, fading_model=exp.fading_model)
    tasks = find_tasks(rows)
    if args.smoke:
        tasks = next(([t] for t in tasks if t["scheme"] == "b3"), tasks[:1])
    print(f"Original starts={len(rows)}; priority continuation candidates={len(tasks)}", flush=True)
    for task in tasks:
        print(f"  {task['candidate_id']}: {task['reason']}, WSR={task['source_wsr']:.9f}", flush=True)
    print("Tolerances unchanged. No new random starts and no cross-weight warm starts.", flush=True)
    if args.dry_run:
        return
    fingerprint = dict(source_dir=str(source), source_csv_sha256=sha(source_csv),
                       original_fingerprint=old_fp, script_sha256=sha(Path(__file__)),
                       tasks=tasks, smoke=args.smoke,
                       inner_chunk=5 if args.smoke else args.inner_chunk,
                       outer_chunk=1 if args.smoke else args.outer_chunk,
                       reset_outer_search_history_per_segment=True)
    manifest = out / "continuation_metadata.json"
    progress_path = out / "continuation_progress.json"
    if out.exists():
        if not args.resume and not args.plot_only:
            raise FileExistsError(f"Output exists: {out}. Use --resume or --plot-only.")
        if not manifest.exists() or json.loads(manifest.read_text(encoding="utf-8"))["fingerprint"] != fingerprint:
            raise ValueError("Output checkpoint differs from source/code/segment settings.")
        progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
    else:
        if args.plot_only:
            raise FileNotFoundError("Continuation output not found. Run continuation first.")
        out.mkdir(parents=True)
        (out / "states").mkdir()
        dump_json(manifest, dict(fingerprint=fingerprint, config=raw))
        progress = {}
        dump_json(progress_path, progress)
    # Round caps may be increased on --resume; source/config/segment sizes remain checked.
    dump_json(out / "latest_budget.json", dict(inner_rounds=args.inner_rounds, outer_rounds=args.outer_rounds))
    for task in tasks:
        state = progress.get(task["candidate_id"])
        if state:
            if sha(out / state["state_file"]) != state["state_sha256"]:
                raise ValueError("Continuation NPZ changed since checkpoint.")
        elif not args.plot_only:
            path = source / task["source_state_file"]
            if not path.exists() or sha(path) != task["source_state_sha256"]:
                raise FileNotFoundError(f"Missing/changed saved source NPZ: {path}")

    segment_log = read_csv(out / "segment_log.csv") if (out / "segment_log.csv").exists() else []
    if not args.plot_only:
        for number, task in enumerate(tasks, 1):
            cid = task["candidate_id"]
            state = progress.get(cid)
            if state and state["converged"]:
                print(f"[{number}/{len(tasks)}] {cid}: converged checkpoint, skip", flush=True)
                continue
            cfg = weighted_cfg(scene, task["omega_comm"])
            start_path = out / state["state_file"] if state else source / task["source_state_file"]
            geometry, resources, initial_perf = load_state(start_path, cfg, task["scheme"], task["omega_comm"], task["seed"])
            initial = float(initial_perf.weighted_sum_rate)
            expected_wsr = state["weighted_sum_rate"] if state else task["source_wsr"]
            if not np.isclose(initial, expected_wsr, rtol=1e-8, atol=1e-9):
                raise ValueError(f"Source state does not reproduce CSV WSR: {cid}, {initial} != {expected_wsr}")
            rounds_done = state["rounds_done"] if state else 0
            rounds_limit = 1 if args.smoke else (args.inner_rounds if task["scheme"] in {"b2", "b3"} else args.outer_rounds)
            extra_iter = state["extra_iterations"] if state else 0
            extra_time = state["extra_runtime_s"] if state else 0.0
            for round_index in range(rounds_done+1, rounds_limit+1):
                print(f"[{number}/{len(tasks)}] {cid}, segment {round_index}/{rounds_limit}, "
                      f"initial WSR={initial:.10f}", flush=True)
                start = perf_counter()
                try:
                    result, perf, history, iterations, reason = execute_segment(
                        cfg, exp, task, geometry, resources,
                        fingerprint["inner_chunk"], fingerprint["outer_chunk"])
                    elapsed = perf_counter()-start
                    final = float(perf.weighted_sum_rate)
                    if not np.isfinite(final) or final < initial - 1e-8*max(1., abs(initial)):
                        raise RuntimeError("Continued WSR decreased/nonfinite; previous state preserved.")
                    if not np.isclose(history[0], initial, rtol=1e-9, atol=1e-9):
                        raise RuntimeError("Solver did not start from the loaded full state.")
                    dl, ul, rs = map(float, (np.sum(perf.rate_dl), np.sum(perf.rate_ul), np.sum(perf.rate_s)))
                    extra_iter += iterations
                    extra_time += elapsed
                    # Version each segment so an interrupted write cannot destroy the preceding checkpoint.
                    file_name = f"states/{cid}_segment_{round_index:03d}.npz"
                    save_state(out/file_name, result.geometry, result.resources, task, bool(result.converged), history)
                    write_csv(out / f"states/{cid}_segment_{round_index:03d}_history.csv",
                              [dict(iteration=i, wsr=v, relative_change="" if i == 0 else relative_change(v, history[i-1]))
                               for i, v in enumerate(history)], ["iteration", "wsr", "relative_change"])
                    state = dict(candidate_id=cid, omega_comm=task["omega_comm"], scheme=task["scheme"], seed=task["seed"],
                                 sum_rate_dl=dl, sum_rate_ul=ul, sum_rate_comm=dl+ul, sum_rate_sensing=rs,
                                 weighted_sum_rate=final, converged=bool(result.converged), stop_reason=reason,
                                 rounds_done=round_index, extra_iterations=extra_iter, extra_runtime_s=extra_time,
                                 total_iterations=task["source_iterations"]+extra_iter,
                                 state_file=file_name, state_sha256=sha(out/file_name))
                    progress[cid] = state
                    dump_json(progress_path, progress)
                    segment_log.append(dict(candidate_id=cid, segment=round_index, initial_wsr=initial,
                                            final_wsr=final, gain=final-initial, iterations=iterations,
                                            converged=bool(result.converged), stop_reason=reason, runtime_s=elapsed))
                    write_csv(out / "segment_log.csv", segment_log,
                              ["candidate_id", "segment", "initial_wsr", "final_wsr", "gain", "iterations",
                               "converged", "stop_reason", "runtime_s"])
                    print(f"  WSR={final:.10f}, gain={final-initial:.3e}, "
                          f"converged={result.converged}, reason={reason}, time={elapsed:.1f}s", flush=True)
                    geometry, resources, initial = result.geometry, result.resources, final
                    if result.converged:
                        break
                except Exception as exc:
                    with (out / "errors.log").open("a", encoding="utf-8") as handle:
                        handle.write(f"{cid}, segment={round_index}: {type(exc).__name__}: {exc}\n")
                    print(f"Previous segments saved. Error details: {out/'errors.log'}", flush=True)
                    raise

    # Merge continued results by original seed identity; all other original results are retained.
    updated = []
    for original in rows:
        row = dict(original)
        cid = candidate_id(row)
        row["source_state_file"] = str(source / original["state_file"])
        row["extra_iterations"], row["extra_runtime_s"] = 0, 0.0
        if cid in progress:
            state = progress[cid]
            for name in ["sum_rate_dl", "sum_rate_ul", "sum_rate_comm", "sum_rate_sensing",
                         "weighted_sum_rate", "converged", "stop_reason"]:
                row[name] = state[name]
            row["iterations"] = state["total_iterations"]
            row["runtime_s"] = float(original["runtime_s"])+state["extra_runtime_s"]
            row["state_file"], row["state_sha256"] = str(out/state["state_file"]), state["state_sha256"]
            row["extra_iterations"], row["extra_runtime_s"] = state["extra_iterations"], state["extra_runtime_s"]
        else:
            row["state_file"] = str(source / original["state_file"])
        updated.append(row)
    write_csv(out / "all_starts_updated.csv", updated, list(updated[0]))
    report = [dict(task, **{k: state[k] for k in ["weighted_sum_rate", "converged", "stop_reason",
                                                 "rounds_done", "extra_iterations", "extra_runtime_s"]})
              for task in tasks if (state := progress.get(task["candidate_id"]))]
    if report:
        write_csv(out / "continuation_summary.csv", report, list(report[0]))
    if args.smoke:
        print(f"SMOKE complete: {out}. No formal figure produced.", flush=True)
        return
    records, missing, unresolved = [], [], []
    for w in old_fp["weights"]:
        for scheme in SCHEMES:
            group = [r for r in updated if group_key(r) == (round(float(w), 12), scheme)]
            valid = [r for r in group if truth(r["converged"])]
            if not valid:
                missing.append(dict(omega_comm=w, scheme=scheme, reason="no_converged_candidate"))
                continue
            best = max(valid, key=lambda r: float(r["weighted_sum_rate"]))
            for r in group:
                if not truth(r["converged"]) and float(r["weighted_sum_rate"]) > float(best["weighted_sum_rate"])+1e-8:
                    unresolved.append(dict(omega_comm=w, scheme=scheme, seed=int(r["seed"]),
                                           unfinished_wsr=float(r["weighted_sum_rate"]),
                                           selected_wsr=float(best["weighted_sum_rate"]),
                                           gap=float(r["weighted_sum_rate"])-float(best["weighted_sum_rate"])))
            records.append(dict(omega_comm=w, omega_sensing=1-float(w), scheme=scheme, scheme_label=LABELS[scheme],
                                **{n: float(best[n]) for n in ["sum_rate_dl", "sum_rate_ul", "sum_rate_comm",
                                                             "sum_rate_sensing", "weighted_sum_rate"]},
                                best_seed=int(best["seed"]), iterations=int(best["iterations"]),
                                runtime_s=sum(float(r["runtime_s"]) for r in group), n_starts=len(group), converged=True))
    write_csv(out / "stage10_tradeoff.csv", records, PLOT_FIELDS)
    write_csv(out / "missing_points.csv", missing, ["omega_comm", "scheme", "reason"])
    write_csv(out / "higher_unconverged_candidates.csv", unresolved,
              ["omega_comm", "scheme", "seed", "unfinished_wsr", "selected_wsr", "gap"])
    print(f"Selected points={len(records)}/80; missing={len(missing)}; "
          f"higher unfinished candidates={len(unresolved)}", flush=True)
    if missing:
        print(f"No complete figure drawn. Inspect {out/'missing_points.csv'}.", flush=True)
        print("To extend the SAME checkpoints: --resume --inner-rounds 8 --outer-rounds 30", flush=True)
        raise SystemExit(2)

    cross = []
    for target in records:
        for source_row in records:
            if target["scheme"] != source_row["scheme"]:
                continue
            w = float(target["omega_comm"])
            objective = w*source_row["sum_rate_comm"]/10+(1-w)*source_row["sum_rate_sensing"]/5
            cross.append(dict(scheme=target["scheme"], source_weight=source_row["omega_comm"],
                              target_weight=w, cross_wsr=objective, target_wsr=target["weighted_sum_rate"],
                              improvement_over_target=objective-target["weighted_sum_rate"]))
    write_csv(out / "cross_evaluation.csv", cross,
              ["scheme", "source_weight", "target_weight", "cross_wsr", "target_wsr", "improvement_over_target"])
    # Preserve every raw selected point. Diagnostic guides never replace solver outputs.
    data = load_data(out / "stage10_tradeoff.csv")
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    render(data, figures)
    if unresolved:
        for extension in ["png", "pdf", "svg"]:
            path = figures / f"communication_sensing_tradeoff.{extension}"
            if path.exists():
                path.replace(path.with_name(f"communication_sensing_tradeoff_diagnostic.{extension}"))
        print("Figure saved with _diagnostic suffix: higher unfinished candidates remain.", flush=True)
    print(f"Cross evaluations with better candidates: {sum(r['improvement_over_target'] > 1e-8 for r in cross)}", flush=True)
    print(f"DONE: {out}", flush=True)


if __name__ == "__main__":
    main()
