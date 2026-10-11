"""Stage 10 transition: add two mode anchors without overwriting previous results.

Requires prior 7-weight transition output produced by run_stage10_transition_probe.py
and its complete state files (NPZ). Existing CSV files alone are insufficient.

Run from project root:
 python -u experiments/run_stage10_anchor_extension.py --dry-run
 python -u experiments/run_stage10_anchor_extension.py --phase anchors
 python -u experiments/run_stage10_anchor_extension.py --phase refine
 python -u experiments/run_stage10_anchor_extension.py --phase report

New files are written ONLY inside results/stage10/transition_050_060_tol1e4/anchor_extension/
Never rewrites old 7-weight results or formal Stage 10 data.
"""
from __future__ import annotations

import argparse
import csv
import json
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from stage8 import load_stage8_experiment_config, make_scene_config
from experiments import run_stage10_pool_refinement as pool
from experiments import run_stage10_transition_probe as probe
from experiments.run_stage10_tradeoff import load_tradeoff_config

# These are independent per-scheme source states. Never cross-initialize schemes.
ANCHORS = {
    "proposed": {"weight": 0.005, "seed": 2028, "min_expected_sensing": 20.0},
    "b6": {"weight": 0.2, "seed": 1562647582, "min_expected_sensing": 4.0},
}
SCHEMES = ("proposed", "b6")
WEIGHTS = probe.WEIGHTS
DEFAULT_ORIGINAL = ROOT / "results" / "stage10" / "transition_050_060_tol1e4"


def csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def valid_attempt(row: dict) -> bool:
    return row.get("status") == "ok" and row.get("converged") is True


def same_weight(a, b) -> bool:
    return abs(float(a) - float(b)) < 1e-11


def existing_attempts(original: Path) -> list[dict]:
    rows = []
    for f in sorted((original / "attempts").glob("*.json")):
        row = read_json(f)
        if valid_attempt(row):
            state = original / row.get("state_file", "")
            if not state.is_file():
                raise FileNotFoundError(f"Original converged attempt missing NPZ: {state}")
            rows.append(dict(row, source="previous_transition", absolute_state=state))
    return rows


def verify_old_run(original: Path, scene, exp, schemes=SCHEMES):
    """Block mixing results from different physical configurations or solver revisions."""
    old_manifest = original / "manifest.json"
    if not old_manifest.is_file():
        raise FileNotFoundError(
            f"Missing {old_manifest}. You must keep the original seven-point output "
            "directory (including manifest, attempts and NPZ files)."
        )
    old = read_json(old_manifest)
    params = old.get("parameters", {})
    if tuple(params.get("weights", [])) != tuple(WEIGHTS):
        raise RuntimeError("Original weight grid differs from this seven-point test.")
    if tuple(params.get("schemes", [])) != SCHEMES:
        raise RuntimeError("Original schemes are not exactly Proposed and B6.")
    if params.get("physical_scene") != repr(scene):
        raise RuntimeError("Original and current physical scene differ; cannot reweight states.")
    if list(params.get("ma_params", [])) != [exp.ma_region_size, exp.ma_min_spacing,
                                             exp.ma_initial_step, exp.ma_initial_jitter_fraction]:
        raise RuntimeError("Original MA parameters differ; do not mix solutions.")
    for name, old_hash in params.get("code_hashes", {}).items():
        path = ROOT / name
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != old_hash:
            raise RuntimeError(f"Solver/configuration changed since old results: {name}. "
                               "Use the same files as the completed transition test, or start a new diagnostic.")
    # Validate the previous manifest's own signature, including starts/refine_top.
    cfg_path = ROOT / "configs" / "stage10_tradeoff.yaml"
    _, current_stamp, _ = probe._manifest(
        original, cfg_path, scene, exp, WEIGHTS,
        int(params["starts"]), int(params["refine_top"]),
    )
    if current_stamp != old.get("sha256"):
        raise RuntimeError("Previous run's source or parameters do not match its manifest.")
    rows = existing_attempts(original)
    for scheme in schemes:
        for w in WEIGHTS:
            if not any(r["scheme"] == scheme and same_weight(r["omega_comm"], w) for r in rows):
                raise RuntimeError(f"No converged previous result for {scheme} at omega={w:g}.")
    return rows, old.get("sha256")


def ensure_new_manifest(sub: Path, old_sha: str, scene, schemes):
    spec = {
        "original_manifest_sha256": old_sha,
        "physical_scene": repr(scene),
        "schemes": list(schemes),
        "targets": list(WEIGHTS),
        "anchors": {s: ANCHORS[s] for s in schemes},
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    digest = sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    file = sub / "manifest.json"
    if file.exists() and read_json(file).get("sha256") != digest:
        raise RuntimeError(f"Existing anchor extension has different settings/code: {sub}. "
                           "Choose a new --original folder or restore the same script.")
    return file, {"sha256": digest, "parameters": spec}


def anchor_path(sub: Path, scheme: str) -> Path:
    return sub / "attempts" / f"anchor_{scheme}.json"


def refined_path(sub: Path, scheme: str, weight: float) -> Path:
    return sub / "attempts" / f"refined_{scheme}_w{weight:.3f}.json"


def run_anchor(sub, scene, exp, schemes):
    for scheme in schemes:
        source = ANCHORS[scheme]
        omega, seed = source["weight"], source["seed"]
        info = anchor_path(sub, scheme)
        if info.exists():
            cached = read_json(info)
            if cached.get("status") == "ok":
                state = sub / cached["state_file"]
                if not state.is_file():
                    raise FileNotFoundError(f"Anchor status exists but state is missing: {state}")
                pool.verify_state(state, pool.target_cfg(scene, omega), cached["weighted_sum_rate"])
                print(f"[RESUME] {scheme} anchor state: {state}", flush=True)
                continue
        cfg = pool.target_cfg(scene, omega)
        print(f"[ANCHOR] {scheme}, source omega={omega:g}, seed={seed}", flush=True)
        begin = perf_counter()
        try:
            result = pool.run_solver(scheme, cfg, exp, seed)
            meta = pool.extract_row(scheme, omega, seed, result, "anchor_cold", "", omega)
            meta["runtime_s"] = round(perf_counter() - begin, 4)
            meta["status"] = "ok" if bool(result.converged) else "nonconverged"
            if result.converged:
                rel = Path("anchor_states") / f"anchor_{scheme}.npz"
                pool.save_state(sub / rel, result)
                meta["state_file"] = str(rel)
                pool.verify_state(sub / rel, cfg, meta["weighted_sum_rate"])
                if meta["sum_rate_sensing"] < source["min_expected_sensing"]:
                    print(f"[WARNING] {scheme} did not reproduce the expected sensing mode "
                          f"(RS={meta['sum_rate_sensing']:.4f}); check historical seed/config.", flush=True)
            pool.json_write(info, meta)
            print(f"    converged={meta['converged']}, WSR={meta['weighted_sum_rate']:.6f}, "
                  f"RC={meta['sum_rate_comm']:.4f}, RS={meta['sum_rate_sensing']:.4f}", flush=True)
            if not result.converged:
                raise RuntimeError(f"Anchor {scheme} failed convergence; no NPZ was used.")
        except Exception as exc:
            if not info.exists():
                pool.json_write(info, {"status": "error", "scheme": scheme,
                                            "omega_comm": omega, "seed": seed,
                                            "error": f"{type(exc).__name__}: {exc}"})
            raise


def anchor_rows(sub, schemes):
    rows = {}
    for scheme in schemes:
        file = anchor_path(sub, scheme)
        if not file.exists():
            raise FileNotFoundError(f"Run --phase anchors first; missing {file}")
        row = read_json(file)
        if not valid_attempt(row):
            raise RuntimeError(f"Anchor {scheme} did not converge: {row.get('status')}")
        state = sub / row.get("state_file", "")
        if not state.is_file():
            raise FileNotFoundError(str(state))
        rows[scheme] = dict(row, absolute_state=state)
    return rows


def write_anchor_evaluations(sub, scene, anchors, schemes):
    results = []
    for scheme in schemes:
        a = anchors[scheme]
        for w in WEIGHTS:
            cfg = pool.target_cfg(scene, w)
            # Evaluate the exact stored geometry + resources under the target weight.
            exact = pool.verify_state(a["absolute_state"], cfg)
            from_rates = (w/(2*cfg.k_dl) * a["sum_rate_dl"]
                          + w/(2*cfg.j_ul) * a["sum_rate_ul"]
                          + (1-w)/cfg.l_target * a["sum_rate_sensing"])
            if not np.isclose(exact, from_rates, atol=2e-7, rtol=1e-6):
                raise RuntimeError(f"Reweighted objective mismatch for {scheme}, target={w}")
            results.append(dict(scheme=scheme, anchor_omega=a["omega_comm"],
                                anchor_seed=a["seed"], target_omega=w,
                                direct_feasible_wsr=exact, sum_rate_comm=a["sum_rate_comm"],
                                sum_rate_sensing=a["sum_rate_sensing"],
                                target_converged=False))
    pool.csv_write(sub / "anchor_reweighted_scores.csv", results,
                   ["scheme", "anchor_omega", "anchor_seed", "target_omega",
                    "direct_feasible_wsr", "sum_rate_comm", "sum_rate_sensing",
                    "target_converged"])
    return results


def run_refinement(sub, scene, exp, anchors, schemes):
    for scheme in schemes:
        a = anchors[scheme]
        g, r, _ = pool.read_state(a["absolute_state"])
        for w in WEIGHTS:
            cfg = pool.target_cfg(scene, w)
            info = refined_path(sub, scheme, w)
            if info.exists():
                cached = read_json(info)
                if cached.get("status") == "ok":
                    state = sub / cached["state_file"]
                    if not state.is_file():
                        raise FileNotFoundError(f"Resume state missing: {state}")
                    pool.verify_state(state, cfg, cached["weighted_sum_rate"])
                    print(f"[RESUME] {scheme}, target={w:g}", flush=True)
                    continue
            print(f"[REFINE] {scheme}, anchor omega={a['omega_comm']:g} -> target={w:g}", flush=True)
            initial = pool.verify_state(a["absolute_state"], cfg)
            begin = perf_counter()
            try:
                result = pool.run_solver(scheme, cfg, exp, int(a["seed"]), warm=(g, r))
                meta = pool.extract_row(scheme, w, int(a["seed"]), result,
                                        "anchor_refined", "", a["omega_comm"])
                meta["initial_feasible_wsr"] = initial
                meta["runtime_s"] = round(perf_counter() - begin, 4)
                meta["status"] = "ok" if bool(result.converged) else "nonconverged"
                if result.converged:
                    rel = Path("refined_states") / f"{scheme}_target_{w:.3f}.npz"
                    pool.save_state(sub / rel, result)
                    meta["state_file"] = str(rel)
                    pool.verify_state(sub / rel, cfg, meta["weighted_sum_rate"])
                pool.json_write(info, meta)
                print(f"    initial={initial:.6f}, final={meta['weighted_sum_rate']:.6f}, "
                      f"converged={meta['converged']}", flush=True)
                if meta["weighted_sum_rate"] < initial - max(1e-6, abs(initial)*1e-5):
                    print("    [WARNING] target solver reduced actual WSR from initial feasible state.", flush=True)
            except Exception as exc:
                if not info.exists():
                    pool.json_write(info, {"status": "error", "scheme": scheme,
                                               "omega_comm": w, "seed": a["seed"],
                                               "error": f"{type(exc).__name__}: {exc}"})
                raise


def collect_report(sub, original, scene, schemes):
    prior = existing_attempts(original)
    anchors = anchor_rows(sub, schemes)
    anchor_evals = write_anchor_evaluations(sub, scene, anchors, schemes)
    final, detail = [], []
    for scheme in schemes:
        anchor = anchors[scheme]
        for w in WEIGHTS:
            target_cfg = pool.target_cfg(scene, w)
            old_target = [r for r in prior if r["scheme"] == scheme and
                          same_weight(r["omega_comm"], w)]
            old_best = max(old_target, key=lambda x: float(x["weighted_sum_rate"]))
            pool.verify_state(old_best["absolute_state"], target_cfg, old_best["weighted_sum_rate"])
            selected = dict(old_best)
            selected_type = "previous_target_converged"
            refined_file = refined_path(sub, scheme, w)
            ref = read_json(refined_file) if refined_file.exists() else None
            if ref is not None and valid_attempt(ref):
                ref_state = sub / ref["state_file"]
                pool.verify_state(ref_state, target_cfg, ref["weighted_sum_rate"])
                if float(ref["weighted_sum_rate"]) > float(selected["weighted_sum_rate"]):
                    selected = dict(ref, absolute_state=ref_state)
                    selected_type = "anchor_refined_target_converged"
            direct = next(q for q in anchor_evals if q["scheme"] == scheme and same_weight(q["target_omega"], w))
            # Original cold states from OTHER weights are also feasible here.
            # This identifies gaps missed by target-weight refinement, not only gaps
            # against the two new anchors.
            old_cold = [r for r in prior if r["scheme"] == scheme and r.get("method") == "cold"]
            old_reweighted = [(
                w/(2*target_cfg.k_dl) * float(r["sum_rate_dl"])
                + w/(2*target_cfg.j_ul) * float(r["sum_rate_ul"])
                + (1-w)/target_cfg.l_target * float(r["sum_rate_sensing"]), r)
                for r in old_cold]
            prior_best_feasible, prior_best_source = max(old_reweighted, key=lambda item: item[0])
            pool.verify_state(prior_best_source["absolute_state"], target_cfg, prior_best_feasible)
            known_feasible = max(prior_best_feasible, float(direct["direct_feasible_wsr"]),
                                 float(old_best["weighted_sum_rate"]))
            selected_wsr = float(selected["weighted_sum_rate"])
            row = dict(scheme=scheme, omega_comm=w,
                       sum_rate_dl=selected["sum_rate_dl"], sum_rate_ul=selected["sum_rate_ul"],
                       sum_rate_comm=selected["sum_rate_comm"],
                       sum_rate_sensing=selected["sum_rate_sensing"],
                       weighted_sum_rate=selected_wsr, best_seed=selected["seed"],
                       iterations=selected["iterations"], converged=True,
                       selection_kind=selected_type, origin_omega=selected["origin_omega"],
                       source_state=str(selected.get("absolute_state", "")))
            final.append(row)
            detail.append(dict(scheme=scheme, omega_comm=w,
                               old_target_best_wsr=float(old_best["weighted_sum_rate"]),
                               anchor_direct_feasible_wsr=direct["direct_feasible_wsr"],
                               previous_cold_feasible_best_wsr=prior_best_feasible,
                               previous_cold_feasible_origin=prior_best_source["omega_comm"],
                               anchor_refined_wsr=ref.get("weighted_sum_rate", "") if ref else "",
                               anchor_refined_converged=ref.get("converged", False) if ref else False,
                               selected_target_converged_wsr=selected_wsr,
                               selected_kind=selected_type,
                               improvement_vs_old=selected_wsr-float(old_best["weighted_sum_rate"]),
                               remaining_known_feasible_gap=max(0., known_feasible - selected_wsr),
                               anchor_quality_warning=anchor["sum_rate_sensing"] < ANCHORS[scheme]["min_expected_sensing"]))
    pool.csv_write(sub / "anchor_comparison.csv", detail,
                   ["scheme", "omega_comm", "old_target_best_wsr", "anchor_direct_feasible_wsr",
                    "previous_cold_feasible_best_wsr", "previous_cold_feasible_origin",
                    "anchor_refined_wsr", "anchor_refined_converged",
                    "selected_target_converged_wsr", "selected_kind", "improvement_vs_old",
                    "remaining_known_feasible_gap", "anchor_quality_warning"])
    pool.csv_write(sub / "selected_after_anchors.csv", final,
                   ["scheme", "omega_comm", "sum_rate_dl", "sum_rate_ul", "sum_rate_comm",
                    "sum_rate_sensing", "weighted_sum_rate", "best_seed", "iterations",
                    "converged", "selection_kind", "origin_omega", "source_state"])
    print("\n[REPORT] old / feasible anchor / selected target-converged WSR")
    for d in detail:
        print(f"  {d['scheme']:8s} omega={d['omega_comm']:.2f}: "
              f"{d['old_target_best_wsr']:.5f} / {d['anchor_direct_feasible_wsr']:.5f} / "
              f"{d['selected_target_converged_wsr']:.5f}; "
              f"gap={d['remaining_known_feasible_gap']:.3e}")
    print("[REPORT] written:", sub / "anchor_comparison.csv")
    return final, detail


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("anchors", "refine", "report", "all"), default="all")
    p.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL,
                   help="Directory with original seven-point NPZ states and manifest")
    p.add_argument("--schemes", nargs="+", choices=SCHEMES, default=list(SCHEMES))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    schemes = tuple(dict.fromkeys(args.schemes))
    orig = args.original.resolve()
    cfg_path = ROOT / "configs" / "stage10_tradeoff.yaml"
    base = load_config(cfg_path)
    exp = load_stage8_experiment_config(cfg_path)
    tr = load_tradeoff_config(cfg_path)
    if (base.k_dl, base.j_ul, base.l_target, base.r_sensing) != (5,5,5,4):
        raise RuntimeError("This test expects K=J=L=5, R_s=4")
    if tr.scene_seed != 8001 or exp.fading_enabled:
        raise RuntimeError("The original scene must be scene_seed=8001 with LOS fading OFF")
    if not (np.isclose(base.tol_inner, 1e-4) and np.isclose(base.tol_outer, 1e-4)):
        raise RuntimeError("Expected tol_inner = tol_outer = 1e-4")
    scene = make_scene_config(base, scene_seed=tr.scene_seed,
                              position_jitter=exp.position_jitter,
                              randomize_fading=False, fading_model=exp.fading_model)
    prior, old_sha = verify_old_run(orig, scene, exp, schemes)
    sub = orig / "anchor_extension"
    manifest_file, manifest_body = ensure_new_manifest(sub, old_sha, scene, schemes)
    print("Stage 10 ANCHOR EXTENSION; no original outputs overwritten")
    print("Prior converged attempts:", len(prior), "schemes:", schemes)
    print("Anchor sources:", {s: (ANCHORS[s]["weight"], ANCHORS[s]["seed"]) for s in schemes})
    print("Target weights:", WEIGHTS)
    print("Tolerance 1e-4, output:", sub)
    if args.dry_run:
        print("[OK] Input manifest, physical scene, existing state files and source hashes validated.")
        return
    pool.json_write(manifest_file, manifest_body)
    if args.phase in ("anchors", "all"):
        run_anchor(sub, scene, exp, schemes)
    if args.phase in ("refine", "all"):
        anchors = anchor_rows(sub, schemes)
        write_anchor_evaluations(sub, scene, anchors, schemes)
        run_refinement(sub, scene, exp, anchors, schemes)
    if args.phase in ("report", "anchors", "refine", "all"):
        collect_report(sub, orig, scene, schemes)
    print("[COMPLETE] Outputs at", sub)


if __name__ == "__main__":
    main()
