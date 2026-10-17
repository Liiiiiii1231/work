"""Extend the verified scene-8001, 25 mW multi-start run to 35/40 dBm.

The script reuses the completed 20/25/30 dBm checkpoints and performs only
the 20 new optimizations at 35 and 40 dBm:

    (5 Proposed + 3 MA + 1 Fixed-ToMA + 1 Fixed-UPA) * 2 powers = 20

Initialization, geometry seeds, stopping rules and solver limits are inherited
from ``run_regular25_multistart_pilot.py``.  Existing checkpoints are verified
before any new job is submitted.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import patch

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]

import run_regular25_multistart_pilot as high25

pilot = high25.pilot
engine = high25.engine

SCENE_SEED = 8001
BASE_POWERS_DBM = [20, 25, 30]
NEW_POWERS_DBM = [35, 40]
ALL_POWERS_DBM = BASE_POWERS_DBM + NEW_POWERS_DBM
SCHEMES = [("proposed", 5), ("b6", 3), ("b2", 1), ("b3", 1)]
HIGH_SETTINGS = {**high25.LOW_SETTINGS, "initial_ul_power_W": 0.025}


def make_task(dbm: float, scheme: str) -> dict:
    """Build the exact task dictionary expected by the shared engine."""
    return {
        "figure": "p_dl_max",
        "parameter_value": 10 ** ((float(dbm) - 30.0) / 10.0),
        "scene_seed": SCENE_SEED,
        "scheme": scheme,
    }


def matched_pair(cfg, protocol, scheme: str, start_id: int):
    """Generate one of the frozen 5/3/1/1 geometry starts at 25 mW/user."""
    with patch.object(pilot, "SETTINGS", HIGH_SETTINGS):
        return pilot.pair(cfg, protocol, scheme, start_id)


def run_job(args):
    """Keep the 25 mW setting explicit inside each worker process."""
    with patch.object(pilot, "SETTINGS", HIGH_SETTINGS):
        return pilot.run_job(args)


def verify_converged_record(folder: Path, protocol: dict, task: dict, start_id: int) -> dict:
    """Verify checkpoint identity, initialization, constraints and final WSR."""
    record_file = folder / "record.json"
    if not record_file.exists():
        raise FileNotFoundError(f"Missing checkpoint record: {record_file}")

    record = engine.read_json(record_file)
    row = record["row"]
    cfg = engine.config_for_task(ROOT, protocol, task)
    expected_g, expected_r = matched_pair(cfg, protocol, task["scheme"], start_id)

    initial_file = folder / "initial.npz"
    state_file = folder / record["state_file"]
    if record["cfg_sha256"] != engine.config_digest(cfg):
        raise RuntimeError(f"Config digest mismatch: {record_file}")
    if record["initial_sha256"] != engine.sha(initial_file):
        raise RuntimeError(f"Initial-state hash mismatch: {record_file}")
    if record["state_sha256"] != engine.sha(state_file):
        raise RuntimeError(f"Final-state hash mismatch: {record_file}")

    g0, r0, _ = engine.load_state(initial_file)
    for name in ("endpoints", "tx_positions", "rx_positions"):
        # Trigonometric values can differ at roughly machine precision across
        # NumPy/platform builds.  This is the same geometry, so use the
        # engine's 1e-12 initialization tolerance instead of bitwise equality.
        if not np.allclose(
            getattr(g0, name), getattr(expected_g, name), rtol=0, atol=1e-12
        ):
            raise RuntimeError(f"Initial geometry mismatch ({name}): {record_file}")
    # 已保存初态已通过上方 SHA256 校验，继续检查其可行性。
    engine.check_state(g0, r0, cfg, protocol, task["scheme"])
    if not np.allclose(np.abs(r0.q_ul) ** 2, 0.025, rtol=0, atol=1e-14):
        raise RuntimeError(f"UL initial power is not 25 mW/user: {record_file}")

    g, r, history = engine.load_state(state_file)
    engine.check_state(g, r, cfg, protocol, task["scheme"], expected_g)
    wsr = engine.metrics_values(g, r, cfg)["weighted_sum_rate"]
    if not np.isclose(wsr, float(row["weighted_sum_rate"]), rtol=1e-9, atol=1e-9):
        raise RuntimeError(f"Final WSR mismatch: {record_file}")
    if not bool(row["converged"]):
        raise RuntimeError(f"Checkpoint is not converged: {record_file}")

    required = 2 if task["scheme"] in ("proposed", "b6") else 1
    tolerance = cfg.tol_outer if required == 2 else cfg.tol_inner
    delta = np.diff(history)
    if (
        len(delta) < required
        or np.min(delta) < -1e-8
        or np.max(np.abs(delta[-required:])) > tolerance
    ):
        raise RuntimeError(f"Stopping-history check failed: {record_file}")
    return {**row, "start_id": start_id, "origin": "verified_checkpoint"}


def load_verified_base_rows(base: Path, protocol: dict) -> list[dict]:
    """Load the 30 verified rows at 20/25/30 dBm from their state records."""
    completion = engine.read_json(base / "completion.json")
    if not completion.get("complete") or completion.get("returned_records") != 30:
        raise RuntimeError("The 20/25/30 dBm base run is not complete.")

    manifest = engine.read_json(base / "manifest.json")
    if manifest.get("settings") != HIGH_SETTINGS:
        raise RuntimeError("The base run does not use the agreed 25 mW initialization.")
    if manifest.get("script_sha256") != engine.sha(Path(high25.__file__)):
        raise RuntimeError("The base runner differs from the runner recorded in its manifest.")
    for filename, digest in manifest.get("source_sha256", {}).items():
        if engine.sha(ROOT / filename) != digest:
            raise RuntimeError(f"Solver source changed since the base run: {filename}")

    rows = []
    for dbm in BASE_POWERS_DBM:
        for scheme, starts in SCHEMES:
            task = make_task(dbm, scheme)
            for start_id in range(starts):
                folder = (
                    base
                    / f"candidate_{start_id}"
                    / "tasks"
                    / engine.task_key(task)
                )
                rows.append(
                    verify_converged_record(folder, protocol, task, start_id)
                )
    if len(rows) != 30:
        raise RuntimeError("Expected exactly 30 verified base rows.")
    return rows


def geometry_signature(geometry) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(
        getattr(geometry, name).copy()
        for name in ("endpoints", "tx_positions", "rx_positions")
    )


def verify_new_initializations(protocol: dict, base: Path) -> None:
    """Check that 35/40 dBm use the same geometry candidates as the base run."""
    for scheme, starts in SCHEMES:
        reference_task = make_task(BASE_POWERS_DBM[0], scheme)
        reference_cfg = engine.config_for_task(ROOT, protocol, reference_task)
        for start_id in range(starts):
            reference_initial = (
                base
                / f"candidate_{start_id}"
                / "tasks"
                / engine.task_key(reference_task)
                / "initial.npz"
            )
            reference_g, _, _ = engine.load_state(reference_initial)
            reference_signature = geometry_signature(reference_g)
            for dbm in NEW_POWERS_DBM:
                task = make_task(dbm, scheme)
                cfg = engine.config_for_task(ROOT, protocol, task)
                g, r = matched_pair(cfg, protocol, scheme, start_id)
                engine.check_state(g, r, cfg, protocol, scheme)
                if not all(
                    np.allclose(a, b, rtol=0, atol=1e-12)
                    for a, b in zip(reference_signature, geometry_signature(g))
                ):
                    raise RuntimeError(
                        f"Geometry candidate changed: {scheme}, start {start_id}, {dbm} dBm"
                    )
                if not np.allclose(np.abs(r.q_ul) ** 2, 0.025, rtol=0, atol=1e-14):
                    raise RuntimeError("New task does not start at 25 mW/user.")


def write_outputs(output: Path, rows: list[dict]) -> None:
    """Write the 50 start records, 20 winners and the five-point figure."""
    rows.sort(key=lambda r: (float(r["parameter_value"]), r["scheme"], r["start_id"]))
    fields = list(rows[0])
    engine.csv_write(output / "all_starts_20to40.csv", rows, fields)

    best = []
    for dbm in ALL_POWERS_DBM:
        value = make_task(dbm, "proposed")["parameter_value"]
        for scheme, starts in SCHEMES:
            group = [
                r
                for r in rows
                if r["scheme"] == scheme
                and np.isclose(float(r["parameter_value"]), value)
            ]
            if len(group) != starts or not all(bool(r["converged"]) for r in group):
                raise RuntimeError(f"Incomplete result cell: {scheme}, {dbm} dBm")
            best.append(max(group, key=lambda r: float(r["weighted_sum_rate"])))
    engine.csv_write(output / "best_starts_20to40.csv", best, fields)

    compact = []
    for row in best:
        compact.append(
            {
                "p_dl_dbm": round(
                    10 * np.log10(float(row["parameter_value"])) + 30
                ),
                "scheme": row["scheme"],
                "weighted_sum_rate": row["weighted_sum_rate"],
                "best_start_id": row["start_id"],
            }
        )
    engine.csv_write(
        output / "five_point_summary.csv",
        compact,
        ["p_dl_dbm", "scheme", "weighted_sum_rate", "best_start_id"],
    )

    plt = engine.plotting()
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for scheme, _ in SCHEMES:
        style = engine.STYLES[scheme]
        selected = sorted(
            [r for r in compact if r["scheme"] == scheme],
            key=lambda r: r["p_dl_dbm"],
        )
        ax.plot(
            [r["p_dl_dbm"] for r in selected],
            [float(r["weighted_sum_rate"]) for r in selected],
            color=style["color"],
            marker=style["marker"],
            linewidth=1.8,
            markersize=5.5,
            label=style["label"],
        )
    ax.set(
        xlabel="Maximum downlink power (dBm)",
        ylabel="WSR (bit/s/Hz)",
        xticks=ALL_POWERS_DBM,
        title="Scene 8001 | matched multi-start | UL init 25 mW/user",
    )
    ax.grid(alpha=0.23)
    ax.legend(fontsize=9)
    fig.tight_layout()
    for extension in ("png", "pdf", "svg"):
        fig.savefig(
            output / f"scene8001_regular25_multistart_20to40.{extension}",
            dpi=240,
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-calls", type=int, default=2)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--base-results", default="results/regular25_multistart_pilot"
    )
    parser.add_argument(
        "--output", default="results/regular25_scene8001_20to40"
    )
    args = parser.parse_args()
    if args.workers < 1 or args.max_calls < 1:
        parser.error("workers and max-calls must be positive")

    base = (ROOT / args.base_results).resolve()
    output = (ROOT / args.output).resolve()
    protocol = high25.protocol()

    base_rows = load_verified_base_rows(base, protocol)
    verify_new_initializations(protocol, base)
    print(
        "CHECK PASSED: 30 old checkpoints verified; 20 new starts use the "
        "same geometries, 25 mW/user UL initialization and 1e-4 tolerances.",
        flush=True,
    )
    if args.check_only:
        return

    output.mkdir(parents=True, exist_ok=True)
    source_hashes = {
        filename: engine.sha(ROOT / filename)
        for filename in (
            "algorithm.py",
            "baselines.py",
            "ma_baseline.py",
            "initialization.py",
            "inner_solver.py",
            "outer_solver.py",
            "channels.py",
            "metrics.py",
            "geometry.py",
            "config.py",
            "utils.py",
            "state.py",
        )
    }
    manifest = {
        "scene_seed": SCENE_SEED,
        "base_powers_dbm": BASE_POWERS_DBM,
        "new_powers_dbm": NEW_POWERS_DBM,
        "initial_ul_power_W_per_user": 0.025,
        "tol_inner": 1e-4,
        "tol_outer": 1e-4,
        "starts": dict(SCHEMES),
        "base_results": os.path.relpath(base, ROOT),
        "base_manifest_sha256": engine.sha(base / "manifest.json"),
        "source_sha256": source_hashes,
        "script_sha256": engine.sha(Path(__file__)),
    }
    manifest_file = output / "manifest.json"
    if manifest_file.exists() and engine.read_json(manifest_file) != manifest:
        raise RuntimeError("Manifest changed; select a new output directory.")
    engine.json_write(manifest_file, manifest)

    jobs = []
    for dbm in NEW_POWERS_DBM:
        for scheme, starts in SCHEMES:
            task = make_task(dbm, scheme)
            for start_id in range(starts):
                jobs.append(
                    (str(output), protocol, task, start_id, args.max_calls)
                )

    new_rows = []
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                new_rows.append(future.result())
            except Exception as exc:  # keep all independent jobs running
                errors.append(
                    {
                        "task": job[2],
                        "start_id": job[3],
                        "error": repr(exc),
                    }
                )
                print("ERROR", repr(exc), flush=True)

    all_rows = base_rows + new_rows
    complete = (
        not errors
        and len(new_rows) == 20
        and len(all_rows) == 50
        and all(bool(row["converged"]) for row in all_rows)
    )
    engine.json_write(
        output / "completion.json",
        {
            "complete": complete,
            "reused_verified_records": len(base_rows),
            "expected_new_optimizations": 20,
            "returned_new_records": len(new_rows),
            "converged_total_records": sum(
                bool(row["converged"]) for row in all_rows
            ),
            "errors": errors,
        },
    )
    if not complete:
        raise SystemExit("Incomplete: rerun the same command to continue checkpoints.")

    write_outputs(output, all_rows)
    print(f"COMPLETE: five-point results written to {output}", flush=True)


if __name__ == "__main__":
    main()
