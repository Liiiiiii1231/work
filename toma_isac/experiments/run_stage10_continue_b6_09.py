"""Continue only B6 omega=0.9, seed=1562647582, then summarize and plot.

Put in experiments/. Uses the saved top10_abs1e4_scene8001 metadata, geometry
and resources. Both stopping rules stay absolute 1e-4; outer requires two
consecutive small changes. Source code and original results stay untouched.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

import numpy as np
import yaml

OMEGA, SEED = 0.9, 1562647582
STATE_NAME = "omega_0.9_b6_seed_1562647582.npz"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def is_target(row):
    return (row["scheme"] == "b6" and abs(float(row["omega_comm"])-OMEGA) < 1e-12
            and int(row.get("seed", row.get("best_seed", -1))) == SEED)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="results/stage10/top10_abs1e4_scene8001")
    ap.add_argument("--output", default="results/stage10/top10_abs1e4_scene8001_b6_09_continued")
    ap.add_argument("--extra-outer", type=int, default=500,
                    help="Additional outer budget for this invocation; early stop remains enabled.")
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--project-root", default=None)
    args = ap.parse_args()
    if args.extra_outer < 1 or args.checkpoint_every < 1:
        ap.error("Iteration budgets must be positive.")
    root = Path(args.project_root).resolve() if args.project_root else Path.cwd().resolve()
    if not (root/"ma_baseline.py").is_file():
        root = Path(__file__).resolve().parent.parent
    if not (root/"ma_baseline.py").is_file():
        raise RuntimeError("Run from project root, or pass --project-root.")
    sys.path.insert(0, str(root))
    source = (root/args.source).resolve()
    out = (root/args.output).resolve()
    if source == out or source in out.parents:
        raise RuntimeError("Output must be a separate sibling directory, outside original results.")
    if args.plot_only:
        plot(read_csv(out/"stage10_tradeoff.csv"), out,
             json.loads((out/"continuation_summary.json").read_text(encoding="utf-8"))["converged"])
        print(f"Replotted: {out}")
        return

    metadata_path = source/"run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    fp, raw = metadata["fingerprint"], metadata["config"]
    expected = fp["code_sha256"]
    code_files = ("ma_baseline.py", "baselines.py", "inner_solver.py", "channels.py",
                  "config.py", "geometry.py", "metrics.py", "utils.py", "state.py", "stage8.py", "initialization.py")
    for name in code_files:
        if sha(root/name) != expected[name]:
            raise RuntimeError(f"Code differs from original experiment: {name}; use matching source.")
    if fp.get("stopping") != "absolute_wsr_change" or fp.get("inner_consecutive") != 1 or fp.get("outer_consecutive") != 2:
        raise RuntimeError("Expected original absolute stopping rules: inner once, outer twice.")
    if raw["algorithm"]["tol_inner"] != 1e-4 or raw["algorithm"]["tol_outer"] != 1e-4:
        raise RuntimeError("Expected original inner/outer absolute tolerance 1e-4.")
    if raw["algorithm"].get("plateau_window", 0) or raw["experiment"].get("fading", {}).get("enabled", False):
        raise RuntimeError("Expected no plateau stopping and deterministic LoS scene.")
    original_starts = read_csv(source/"all_starts.csv")
    candidates = [r for r in original_starts if is_target(r)]
    if len(candidates) != 1:
        raise RuntimeError("Expected exactly one original target candidate.")
    original_row = candidates[0]
    source_state = source/original_row["state_file"]
    if sha(source_state) != original_row["state_sha256"]:
        raise RuntimeError("Original state checksum mismatch.")
    identity = {"metadata": sha(metadata_path), "state": sha(source_state),
                "all_starts": sha(source/"all_starts.csv"),
                "selected": sha(source/"stage10_tradeoff.csv"), "code": {n:sha(root/n) for n in code_files},
                "runner":sha(Path(__file__).resolve())}
    if out.exists() and not args.resume:
        raise RuntimeError(f"Output exists: {out}. Use --resume or --plot-only.")
    out.mkdir(parents=True, exist_ok=True)

    import inner_solver as inner
    import ma_baseline as ma
    from baselines import build_fpa_upa_geometry
    from channels import build_channels
    from config import load_config
    from metrics import compute_performance
    from stage8 import make_scene_config
    from state import ResourceState
    from utils import check_resource_constraints

    with TemporaryDirectory(prefix="b6_continue_config_") as tmp:
        p = Path(tmp)/"original.yaml"
        p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
        base = load_config(p)
    scene = make_scene_config(base, int(fp["scene_seed"]),
                              np.asarray(raw["experiment"]["position_jitter"], dtype=float), randomize_fading=False)
    cfg = replace(scene, weight_dl=np.full(scene.k_dl, OMEGA/(2*scene.k_dl)),
                  weight_ul=np.full(scene.j_ul, OMEGA/(2*scene.j_ul)),
                  weight_s=np.full(scene.l_target, (1-OMEGA)/scene.l_target))
    cfg.validate()
    ma_settings = raw["experiment"]["ma"]
    region, spacing, step = (float(ma_settings[k]) for k in ("region_size", "min_spacing", "initial_step"))
    upa = build_fpa_upa_geometry(cfg)
    tx_center, rx_center = np.mean(upa.tx_positions, axis=0), np.mean(upa.rx_positions, axis=0)
    progress_path = out/"progress.json"
    checkpoint_path = out/"states"/STATE_NAME
    (out/"states").mkdir(exist_ok=True)
    original_iterations = int(original_row["iterations"])
    if args.resume:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress["identity"] != identity:
            raise RuntimeError("Input or code changed since checkpoint; cannot resume.")
        if sha(checkpoint_path) != progress["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint checksum mismatch.")
        state_path = checkpoint_path
    else:
        progress = dict(identity=identity, extra_iterations=0, extra_runtime_s=0.0,
                        converged=False, consecutive_small_changes=0)
        state_path = source_state
    with np.load(state_path, allow_pickle=False) as a:
        if a["scheme"].item() != "b6" or int(a["seed"].item()) != SEED or abs(float(a["omega_comm"])-OMEGA)>1e-12:
            raise RuntimeError("State identity mismatch.")
        geometry = ma._build_ma_geometry(a["tx_positions"].copy(), a["rx_positions"].copy(), cfg)
        resources = ResourceState(q_matrix=a["q_matrix"].copy(), q_ul=a["q_ul"].copy(),
                                  b_ul=a["b_ul"].copy(), u_s=a["u_s"].copy())
        history = a["wsr_history"].astype(float).tolist()
        accepts = a["accepted_position_counts"].astype(int).tolist() if "accepted_position_counts" in a else []
    if not args.resume and not accepts:
        # Original B6 files did not record this count. -1 means unavailable,
        # rather than incorrectly reporting that every old iteration accepted zero moves.
        accepts = [-1] * (len(history)-1)
    if not ma._ma_geometry_is_feasible(geometry.tx_positions, geometry.rx_positions,
                                       tx_center, rx_center, region, spacing, cfg):
        raise RuntimeError("Saved MA geometry violates original region/spacing constraints.")
    check_resource_constraints(resources, cfg)
    channels = build_channels(geometry, cfg)
    performance = compute_performance(channels, resources, cfg)
    current = float(performance.weighted_sum_rate)
    if not np.isclose(current, history[-1], rtol=1e-9, atol=1e-9):
        raise RuntimeError("Saved WSR does not reproduce under original scene.")
    if not args.resume:
        if not np.isclose(current, float(original_row["weighted_sum_rate"]), rtol=1e-9, atol=1e-9):
            raise RuntimeError("Original CSV/state mismatch.")
        # Preserve any existing single small-change count across continuation.
        for i in range(len(history)-1, 0, -1):
            if abs(history[i]-history[i-1]) > cfg.tol_outer:
                break
            progress["consecutive_small_changes"] += 1
    extra_history = read_csv(out/"continuation_history.csv") if args.resume else []

    def checkpoint():
        buffer = io.BytesIO()
        np.savez_compressed(buffer, omega_comm=OMEGA, scheme="b6", seed=SEED,
                            endpoints=geometry.endpoints, tx_positions=geometry.tx_positions,
                            rx_positions=geometry.rx_positions, q_matrix=resources.q_matrix,
                            q_ul=resources.q_ul, b_ul=resources.b_ul, u_s=resources.u_s,
                            converged=progress["converged"], wsr_history=np.asarray(history),
                            accepted_position_counts=np.asarray(accepts, dtype=int))
        temporary = checkpoint_path.with_suffix(".tmp")
        temporary.write_bytes(buffer.getvalue())
        os.replace(temporary, checkpoint_path)
        progress["checkpoint_sha256"] = sha(checkpoint_path)
        temporary = progress_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(progress, indent=2), encoding="utf-8")
        os.replace(temporary, progress_path)
        write_csv(out/"continuation_history.csv", extra_history,
                  fields=["total_outer_iteration", "extra_outer_iteration", "wsr_before", "wsr_after_inner", "wsr",
                          "abs_change", "consecutive_small_changes", "inner_iterations", "accepted_positions", "runtime_s"])

    if not args.resume:
        checkpoint()
    print(f"B6 only: weight={OMEGA}, seed={SEED}, starting F={current:.9f}; absolute inner/outer tolerance=1e-4", flush=True)
    if progress["converged"]:
        print("Checkpoint already converged; summarizing without more optimization.", flush=True)
    else:
        with patch.object(inner, "relative_change", lambda new, old: abs(float(new)-float(old))):
            for local_iteration in range(1, args.extra_outer+1):
                tick = perf_counter()
                before = current
                resources, inner_history = inner.solve_inner_problem(channels, resources, cfg, verbose=False)
                after_inner = float(inner_history[-1])
                geometry, channels, _, count = ma._update_ma_positions_once(
                    geometry, resources, cfg, tx_center=tx_center, rx_center=rx_center,
                    region_size=region, min_spacing=spacing, initial_step=step)
                performance = compute_performance(channels, resources, cfg)
                current = float(performance.weighted_sum_rate)
                if not np.isfinite(current) or current < before-1e-8*max(1,abs(before)):
                    raise RuntimeError("True WSR decreased materially; do not adopt this continuation.")
                elapsed = perf_counter()-tick
                history.append(current)
                accepts.append(int(count))
                change = abs(current-before)
                progress["extra_iterations"] += 1
                progress["extra_runtime_s"] += elapsed
                progress["consecutive_small_changes"] = progress["consecutive_small_changes"]+1 if change<=cfg.tol_outer else 0
                progress["converged"] = progress["consecutive_small_changes"]>=2
                extra_history.append(dict(total_outer_iteration=original_iterations+progress["extra_iterations"],
                                          extra_outer_iteration=progress["extra_iterations"], wsr_before=before,
                                          wsr_after_inner=after_inner, wsr=current, abs_change=change,
                                          consecutive_small_changes=progress["consecutive_small_changes"],
                                          inner_iterations=len(inner_history)-1, accepted_positions=int(count), runtime_s=elapsed))
                if local_iteration % args.checkpoint_every == 0 or progress["converged"] or local_iteration==args.extra_outer:
                    checkpoint()
                    print(f"  total outer={original_iterations+progress['extra_iterations']}, F={current:.9f}, abs_change={change:.3e}, converged={progress['converged']}", flush=True)
                if progress["converged"]:
                    break

    check_resource_constraints(resources, cfg)
    new_row = dict(original_row)
    new_row.update(sum_rate_dl=float(np.sum(performance.rate_dl)), sum_rate_ul=float(np.sum(performance.rate_ul)),
                   sum_rate_comm=float(np.sum(performance.rate_dl)+np.sum(performance.rate_ul)),
                   sum_rate_sensing=float(np.sum(performance.rate_s)), weighted_sum_rate=current,
                   iterations=original_iterations+progress["extra_iterations"], converged=bool(progress["converged"]),
                   stop_reason="outer_tolerance" if progress["converged"] else "additional_budget_exhausted",
                   runtime_s=float(original_row["runtime_s"])+progress["extra_runtime_s"],
                   state_file="states/"+STATE_NAME, state_sha256=sha(checkpoint_path),
                   last_abs_change=abs(history[-1]-history[-2]),
                   source_state_file=os.path.relpath(source_state, out),
                   extra_iterations=progress["extra_iterations"], extra_runtime_s=progress["extra_runtime_s"])
    updated_starts = []
    for row in original_starts:
        if is_target(row):
            updated_starts.append(new_row)
        else:
            copied = dict(row)
            copied["state_file"] = os.path.relpath(source/row["state_file"], out)
            updated_starts.append(copied)
    write_csv(out/"all_starts.csv", updated_starts)
    nonconverged = [r for r in updated_starts if str(r["converged"]).lower() != "true"]
    all_fields = list(dict.fromkeys(k for row in updated_starts for k in row))
    write_csv(out/"nonconverged_starts.csv", nonconverged, fields=all_fields)
    selected = read_csv(source/"stage10_tradeoff.csv")
    old_selected = next(r for r in selected if r["scheme"]=="b6" and abs(float(r["omega_comm"])-OMEGA)<1e-12)
    replaced_point = progress["converged"] and current>float(old_selected["weighted_sum_rate"])
    if replaced_point:
        replacement = dict(old_selected)
        for k in ("sum_rate_dl", "sum_rate_ul", "sum_rate_comm", "sum_rate_sensing", "weighted_sum_rate", "iterations", "runtime_s", "converged"):
            replacement[k] = new_row[k]
        replacement["best_seed"] = SEED
        selected = [replacement if r is old_selected else r for r in selected]
    write_csv(out/"stage10_tradeoff.csv", selected)
    summary = dict(omega_comm=OMEGA, scheme="b6", seed=SEED,
                   source_wsr=float(original_row["weighted_sum_rate"]), final_wsr=current,
                   original_selected_wsr=float(old_selected["weighted_sum_rate"]),
                   converged=bool(progress["converged"]), replaced_selected_point=bool(replaced_point),
                   extra_outer_iterations=progress["extra_iterations"], extra_runtime_s=progress["extra_runtime_s"],
                   total_outer_iterations=original_iterations+progress["extra_iterations"],
                   sum_rate_comm=new_row["sum_rate_comm"], sum_rate_sensing=new_row["sum_rate_sensing"],
                   stop_reason=new_row["stop_reason"], tol_inner=cfg.tol_inner, tol_outer=cfg.tol_outer,
                   stopping="absolute", outer_consecutive=2)
    (out/"continuation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out/"README.txt").write_text(
        "仅续跑 B6，通信权重0.9，种子1562647582。原结果未修改。\n"
        "仅在候选满足绝对差值1e-4连续两次且WSR更高时替换选中工作点。\n"
        "其他起点状态通过相对路径引用原目录，请保留原目录。\n"
        "checkpoint保留累计历史及连续满足次数；恢复从最近一次完整保存状态开始。\n"
        "accepted_position_counts中的-1表示原实验未保存该轮位置接受次数。\n"
        "达到追加上限不代表收敛；此时图名带_diagnostic，运行同一命令加--resume继续。\n",
        encoding="utf-8")
    plot(selected, out, bool(progress["converged"]))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Output: {out}", flush=True)
    if not progress["converged"]:
        print("Not converged. Use --resume to add another budget without changing tolerances.", flush=True)


def plot(rows, out, complete):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.5, 6))
    styles = [("b6", "MA + FP", "#279b3b", "^"),
              ("proposed", "FP-aware RCG-ToMA + FP (Proposed)", "#df2228", "D")]
    for scheme, label, color, marker in styles:
        data = sorted((r for r in rows if r["scheme"]==scheme), key=lambda r:float(r["omega_comm"]))
        ax.plot([float(r["sum_rate_comm"]) for r in data], [float(r["sum_rate_sensing"]) for r in data],
                color=color, marker=marker, markersize=5.5, markeredgecolor="white", linewidth=2, label=label)
        if scheme=="proposed":
            point = next((r for r in data if abs(float(r["omega_comm"])-.1)<1e-12),None)
            if point:
                ax.annotate(r"$\omega_C=0.1$", (float(point["sum_rate_comm"]),float(point["sum_rate_sensing"])),
                            xytext=(12,14), textcoords="offset points", arrowprops=dict(arrowstyle="-",color="gray"))
    ax.set_xlabel("Sum Communication Rate (bit/s/Hz)")
    ax.set_ylabel("Sum Sensing Rate (bit/s/Hz)")
    ax.grid(alpha=.3)
    ax.legend(loc="lower left", bbox_to_anchor=(0,1.01,1,0.1), mode="expand", ncol=2, fontsize=9)
    fig.tight_layout()
    folder = out/"figures"
    folder.mkdir(exist_ok=True)
    stem = "communication_sensing_tradeoff"+("" if complete else "_diagnostic")
    for ext in ("png","pdf","svg"):
        fig.savefig(folder/f"{stem}.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
