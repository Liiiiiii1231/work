"""Stage 10 Proposed-only cross-weight warm-start diagnostic.

This is NOT a replacement for the Stage 10 sweep. It does not modify any
existing Stage 10 CSV, NPZ, plot, config, or optimization code.

Steps:
    1. Rebuild the fixed scene used by Stage 10.
    2. Re-run the best source seed at omega_C=0.005 to recover full states.
    3. Re-evaluate that SAME feasible state at omega_C=0.3.
    4. Warm-start FP-aware RCG-ToMA + FP from both positions and resources.
    5. Save numerical outputs and histories for auditing.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG = ROOT / "configs" / "stage10_tradeoff.yaml"
ORIGINAL_CSV = ROOT / "results" / "stage10" / "stage10_tradeoff.csv"
OUT = ROOT / "results" / "stage10" / "cross_weight_warmstart_0005_to_03"
SOURCE_OMEGA = 0.005
TARGET_OMEGA = 0.3
DEFAULT_SEED = 2028  # Recorded best Proposed seed at omega=0.005 in supplied CSV.


def read_original_record(csv_path: Path, omega: float) -> dict | None:
    if not csv_path.is_file():
        return None
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if (row.get("scheme") == "proposed"
                    and abs(float(row["omega_comm"]) - omega) < 1e-12):
                return row
    return None


def rate_summary(performance) -> dict[str, float]:
    dl = float(np.sum(performance.rate_dl))
    ul = float(np.sum(performance.rate_ul))
    sensing = float(np.sum(performance.rate_s))
    return {
        "rate_dl": dl,
        "rate_ul": ul,
        "rate_comm": dl + ul,
        "rate_sensing": sensing,
        "wsr": float(performance.weighted_sum_rate),
    }


def save_history(path: Path, values: list[float], tol: float) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", "WSR", "relative_change", "below_outer_tolerance"])
        for index, value in enumerate(values):
            change = (abs(value - values[index - 1]) /
                      max(1.0, abs(values[index - 1]))) if index > 0 else float("nan")
            writer.writerow([index, "%.16g" % value,
                             "%.12g" % change if index > 0 else "",
                             bool(change <= tol) if index > 0 else ""])


def save_state(path: Path, result, omega: float) -> None:
    r = result.resources
    np.savez_compressed(
        path,
        omega_comm=np.array([omega], dtype=float),
        endpoints=np.array(result.geometry.endpoints, copy=True),
        q_matrix=np.array(r.q_matrix, copy=True),
        q_ul=np.array(r.q_ul, copy=True),
        b_ul=np.array(r.b_ul, copy=True),
        u_s=np.array(r.u_s, copy=True),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Proposed: omega_C=0.005 -> 0.3 warm start")
    parser.add_argument("--seed", type=int, default=None,
                        help="Source seed; default reads best seed from existing Stage 10 CSV")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate paths and print settings without running optimization")
    parser.add_argument("--strict-csv", action="store_true",
                        help="Stop if reproduced source differs >1% from saved CSV")
    parser.add_argument("--skip-cold", action="store_true",
                        help="Only run source and warm target, without a new cold-target control")
    args = parser.parse_args()

    if not CONFIG.exists():
        raise FileNotFoundError(f"Stage 10 config not found: {CONFIG}")
    from config import load_config
    base_cfg = load_config(CONFIG)
    with CONFIG.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    scene_seed = int(raw["tradeoff"]["scene_seed"])
    position_jitter = np.asarray(raw["experiment"]["position_jitter"], dtype=float)
    fading_enabled = bool(raw["experiment"].get("fading", {}).get("enabled", False))
    fading_model = str(raw["experiment"].get("fading", {}).get("model", "scalar_rayleigh"))

    source_record = read_original_record(ORIGINAL_CSV, SOURCE_OMEGA)
    target_record = read_original_record(ORIGINAL_CSV, TARGET_OMEGA)
    seed = (args.seed if args.seed is not None else
            int(source_record["best_seed"]) if source_record else DEFAULT_SEED)

    print("Stage 10 cross-weight warm start | Proposed only")
    print(f"project:          {ROOT}")
    print(f"scene seed:       {scene_seed}, proposed seed: {seed}")
    print(f"source -> target: {SOURCE_OMEGA:g} -> {TARGET_OMEGA:g}")
    print(f"RCS xi_s:         {np.asarray(base_cfg.xi_s)}")
    print(f"BS noise [W]:     {base_cfg.sigma_bs2:.12g}")
    print(f"DL noise [W]:     {np.asarray(base_cfg.sigma_dl2)}")
    print(f"tolerances:       inner={base_cfg.tol_inner:g}, outer={base_cfg.tol_outer:g}")
    print(f"iteration caps:   inner={base_cfg.max_inner_iter}, outer={base_cfg.max_outer_iter}")
    print(f"original CSV:     {ORIGINAL_CSV if ORIGINAL_CSV.exists() else 'not found'}")
    print(f"output directory: {OUT}")
    if source_record:
        print("saved source rates: C={:.9f}, S={:.9f}, WSR={:.9f}".format(
            float(source_record["sum_rate_comm"]),
            float(source_record["sum_rate_sensing"]),
            float(source_record["weighted_sum_rate"])))
    else:
        print("WARNING: existing CSV has no Proposed omega=0.005 record; ")
        print("         seed will be used, but prior-run numerical comparison is unavailable.")
    if args.dry_run:
        return

    # Imports deferred so --dry-run works without the FP dependencies.
    from algorithm import run_joint_algorithm
    from channels import build_channels
    from geometry import build_geometry
    from metrics import compute_performance
    from stage8 import make_scene_config
    from state import ResourceState
    from utils import check_resource_constraints

    def weighted_scene(omega: float):
        k, j, l = base_cfg.k_dl, base_cfg.j_ul, base_cfg.l_target
        cfg = replace(
            scene_cfg,
            weight_dl=np.full(k, omega / (2.0 * k)),
            weight_ul=np.full(j, omega / (2.0 * j)),
            weight_s=np.full(l, (1.0 - omega) / l),
        )
        cfg.validate()
        return cfg

    if fading_enabled:
        raise ValueError("This diagnostic expects the Stage 10 LoS setting fading.enabled=false")
    scene_cfg = make_scene_config(
        base_cfg,
        scene_seed=scene_seed,
        position_jitter=position_jitter,
        randomize_fading=fading_enabled,
        fading_model=fading_model,
    )
    cfg_source = weighted_scene(SOURCE_OMEGA)
    cfg_target = weighted_scene(TARGET_OMEGA)

    OUT.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    print("\n[1/3] Recomputing source at omega_C=0.005 to recover actual states ...", flush=True)
    source = run_joint_algorithm(
        cfg_source,
        rng=np.random.default_rng(seed),
        verbose=False,
        position_update_mode="fp_aware",
        fp_aware_candidate_iters=2,
    )
    source_stats = rate_summary(source.performance)
    save_history(OUT / "source_history.csv", source.joint_wsr_history, cfg_source.tol_outer)
    save_state(OUT / "source_state.npz", source, SOURCE_OMEGA)
    print(f" source converged={source.converged}, stop={source.stop_reason}, outer={source.outer_iterations}")
    print(" source C={rate_comm:.9f}, S={rate_sensing:.9f}, WSR={wsr:.9f}".format(**source_stats))
    if not source.converged:
        raise RuntimeError("Source did not satisfy the stopping rule. Histories/state saved; "
                           "do not use its state as a validated warm-start source.")

    csv_matches = False
    source_csv_errors = None
    if source_record is not None:
        old_c = float(source_record["sum_rate_comm"])
        old_s = float(source_record["sum_rate_sensing"])
        errors = [abs(source_stats["rate_comm"] - old_c) / max(1.0, abs(old_c)),
                  abs(source_stats["rate_sensing"] - old_s) / max(1.0, abs(old_s))]
        source_csv_errors = errors
        csv_matches = max(errors) <= 0.01
        if not csv_matches:
            warning = ("Recomputed source differs >1% from saved Stage 10 CSV. "
                       "Check changed YAML, tolerance, scene, and algorithm version.")
            print("\nWARNING:", warning, "relative errors:", errors)
            if args.strict_csv:
                raise RuntimeError(warning + " (--strict-csv requested)")
            print("Proceeding with internally consistent current-run comparison; "
                  "the old CSV is historical reference only.")

    # Verify feasibility on the same geometry under a different objective weight.
    check_resource_constraints(source.resources, cfg_target)
    reconstructed = build_geometry(source.geometry.endpoints.copy(), cfg_target)
    target_channels = build_channels(reconstructed, cfg_target)
    before = compute_performance(target_channels, source.resources, cfg_target)
    before_stats = rate_summary(before)
    if not np.isclose(before_stats["rate_comm"], source_stats["rate_comm"], rtol=1e-7, atol=1e-8):
        raise RuntimeError("Communication rate changed upon reweighting only: scene mismatch")
    if not np.isclose(before_stats["rate_sensing"], source_stats["rate_sensing"], rtol=1e-7, atol=1e-8):
        raise RuntimeError("Sensing rate changed upon reweighting only: scene mismatch")

    expected = (TARGET_OMEGA * (source_stats["rate_dl"] / (2 * cfg_target.k_dl)
                                + source_stats["rate_ul"] / (2 * cfg_target.j_ul))
                + (1 - TARGET_OMEGA) * source_stats["rate_sensing"] / cfg_target.l_target)
    if not np.isclose(expected, before_stats["wsr"], rtol=1e-9, atol=1e-9):
        raise RuntimeError("WSR under new weights does not match original Stage 10 formulation")
    print("\n[2/3] Identical state, only objective weights changed:")
    print(" at target weight (BEFORE optimization): WSR={:.9f}".format(before_stats["wsr"]))
    if target_record is not None:
        print(" old target search WSR (historical CSV): {:.9f}".format(
            float(target_record["weighted_sum_rate"])))

    # Optional internal control: repeat omega=0.3 from the SAME random seed,
    # but without transferring any endpoints or resources from omega=0.005.
    # This comparison is apples-to-apples even when prior CSV was produced with
    # a different code/config version.
    cold = None
    cold_stats = None
    if not args.skip_cold:
        print("\n[CONTROL] Running omega_C=0.3 cold start using same seed ...", flush=True)
        cold = run_joint_algorithm(
            cfg_target,
            rng=np.random.default_rng(seed),
            position_update_mode="fp_aware",
            fp_aware_candidate_iters=2,
            verbose=False,
        )
        cold_stats = rate_summary(cold.performance)
        save_history(OUT / "cold_target_history.csv", cold.joint_wsr_history, cfg_target.tol_outer)
        save_state(OUT / "cold_target_state.npz", cold, TARGET_OMEGA)
        print(" cold C={rate_comm:.9f}, S={rate_sensing:.9f}, WSR={wsr:.9f}, "
              "converged={c}, stop={r}".format(**cold_stats, c=cold.converged, r=cold.stop_reason))

    print("\n[3/3] Running target omega_C=0.3 with BOTH positions + resources warm-started ...", flush=True)
    # No random geometry or random resources here: these are transferred verbatim.
    warm = run_joint_algorithm(
        cfg_target,
        rng=np.random.default_rng(seed),
        position_update_mode="fp_aware",
        fp_aware_candidate_iters=2,
        initial_endpoints=source.geometry.endpoints.copy(),
        initial_resources=ResourceState(
            q_matrix=source.resources.q_matrix.copy(),
            q_ul=source.resources.q_ul.copy(),
            b_ul=source.resources.b_ul.copy(),
            u_s=source.resources.u_s.copy(),
        ),
        verbose=False,
    )
    final_stats = rate_summary(warm.performance)
    save_history(OUT / "warm_target_history.csv", warm.joint_wsr_history, cfg_target.tol_outer)
    save_state(OUT / "warm_target_state.npz", warm, TARGET_OMEGA)
    recorded_first = float(warm.joint_wsr_history[0])
    if not np.isclose(recorded_first, before_stats["wsr"], rtol=1e-9, atol=1e-8):
        raise RuntimeError("Warm-start initial WSR differs from precomputed candidate WSR")

    info = {
        "source_omega": SOURCE_OMEGA,
        "target_omega": TARGET_OMEGA,
        "scene_seed": scene_seed,
        "source_seed": seed,
        "source_converged": bool(source.converged),
        "source_stop_reason": source.stop_reason,
        "source_iterations": source.outer_iterations,
        "target_converged": bool(warm.converged),
        "target_stop_reason": warm.stop_reason,
        "target_iterations": warm.outer_iterations,
        "source_rates": source_stats,
        "source_state_reweighted_at_target": before_stats,
        "warm_optimized_target_rates": final_stats,
        "wsr_gain_after_warm_optimization": final_stats["wsr"] - before_stats["wsr"],
        "runtime_seconds": perf_counter() - start,
        "old_target_wsr_from_csv": (float(target_record["weighted_sum_rate"])
                                    if target_record else None),
        "prior_csv_source_matches_current_run_1pct": csv_matches,
        "source_csv_relative_errors": source_csv_errors,
        "old_target_cold_comparable": bool(csv_matches and target_record is not None),
        "new_cold_target_rates": cold_stats,
        "cold_target_converged": bool(cold.converged) if cold is not None else None,
        "new_warm_minus_cold_wsr": (final_stats["wsr"] - cold_stats["wsr"])
                                    if cold_stats is not None else None,
    }
    with (OUT / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2, ensure_ascii=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.plot(range(len(warm.joint_wsr_history)), warm.joint_wsr_history, "o-", label="Warm start (omega=0.3)")
        ax.axhline(before_stats["wsr"], linestyle="--", label="Transferred-state WSR")
        if cold is not None:
            ax.plot(range(len(cold.joint_wsr_history)), cold.joint_wsr_history, "s-", label="Cold start (omega=0.3)")
        if target_record is not None and csv_matches:
            ax.axhline(float(target_record["weighted_sum_rate"]), linestyle=":", label="Prior CSV, omega=0.3")
        ax.set(xlabel="Outer AO iteration", ylabel="True WSR", title="Cross-weight warm start: 0.005 -> 0.3")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / "warm_target_wsr.png", dpi=180)
        plt.close(fig)
    except ImportError:
        print("matplotlib unavailable; CSV and NPZ were saved, plot skipped")

    print("\n========== RESULT ==========")
    print("Warm initial at omega=0.3: WSR={:.9f}".format(before_stats["wsr"]))
    print("Warm final   at omega=0.3: WSR={:.9f}, C={:.9f}, S={:.9f}".format(
        final_stats["wsr"], final_stats["rate_comm"], final_stats["rate_sensing"]))
    print(f"converged={warm.converged}, stop={warm.stop_reason}, outer={warm.outer_iterations}")
    print("WSR improvement over transferred state: {:.9f}".format(final_stats["wsr"] - before_stats["wsr"]))
    if cold_stats is not None:
        print("Same-seed cold target WSR: {:.9f}, converged={}".format(cold_stats["wsr"], cold.converged))
        print("Warm minus cold target WSR: {:.9f}".format(final_stats["wsr"] - cold_stats["wsr"]))
    if final_stats["wsr"] < before_stats["wsr"] - 1e-7 * max(1, abs(before_stats["wsr"])):
        print("WARNING: Final WSR decreased; investigate numerical monotonicity!")
    print(f"Saved results in: {OUT}")
    if not warm.converged:
        print("WARNING: Target run did not meet stopping rule. Inspect history; do not use as formal converged result.")


if __name__ == "__main__":
    main()
