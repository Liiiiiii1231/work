"""Stage 9F：正式 algorithm.py 接口下比较 Original 与 FP-aware-2。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm import run_joint_algorithm
from config import load_config
from stage8 import make_scene_config


CONFIG_PATH = ROOT / "configs" / "stage9_outer_debug.yaml"
RESULT_PATH = ROOT / "results" / "stage9" / "stage9f_integrated_ab.csv"
POSITION_JITTER = np.array([1.5, 1.5, 0.25], dtype=float)


def _build_scene_cfg(scene_seed: int):
    cfg = load_config(CONFIG_PATH)
    return make_scene_config(
        cfg,
        scene_seed=scene_seed,
        position_jitter=POSITION_JITTER,
        randomize_fading=False,
    )


def _run(cfg, scene_seed: int, start_seed: int, mode: str, max_outer: int | None):
    t0 = perf_counter()
    result = run_joint_algorithm(
        cfg,
        rng=np.random.default_rng(start_seed),
        verbose=False,
        max_outer_iterations=max_outer,
        position_update_mode=mode,
        fp_aware_candidate_iters=2,
    )
    runtime = perf_counter() - t0

    return {
        "mode": mode,
        "scene_seed": int(scene_seed),
        "start_seed": int(start_seed),
        "outer_iterations": int(result.outer_iterations),
        "converged": bool(result.converged),
        "stop_reason": result.stop_reason,
        "final_wsr": float(result.performance.weighted_sum_rate),
        "runtime_s": float(runtime),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-seed", type=int, default=8001)
    parser.add_argument("--start-seed", type=int, default=2029)
    parser.add_argument("--max-outer", type=int, default=None)
    args = parser.parse_args()

    cfg = _build_scene_cfg(args.scene_seed)

    rows = []
    for mode in ("original", "fp_aware"):
        print(f"Running {mode} ...")
        row = _run(
            cfg,
            args.scene_seed,
            args.start_seed,
            mode,
            args.max_outer,
        )
        rows.append(row)
        print(
            f"  done: outer={row['outer_iterations']}, "
            f"WSR={row['final_wsr']:.10f}, "
            f"runtime={row['runtime_s']:.2f}s, "
            f"stop={row['stop_reason']}"
        )

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nStage 9F integrated A/B")
    for row in rows:
        print(
            f"  {row['mode']:8s}: "
            f"outer={row['outer_iterations']:3d}, "
            f"WSR={row['final_wsr']:.10f}, "
            f"runtime={row['runtime_s']:.2f}s"
        )
    print(f"CSV -> {RESULT_PATH}")


if __name__ == "__main__":
    main()
