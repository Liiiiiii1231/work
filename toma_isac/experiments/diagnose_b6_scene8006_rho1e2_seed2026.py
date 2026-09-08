"""诊断 B6 MA+FP 在 scene 8006 / rho_si=1e-2 / seed=2026 的外层收敛。

用途：
- 精确复现 residual-SI sweep 中报错的那个 MA 起点；
- 不修改 multi-start 逻辑；
- 单独把该起点最多跑到 1500 次，判断 1000 次是否只是上限不够；
- 输出第 1000 轮附近和最终阶段的 WSR 变化。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter
import argparse
import csv
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from ma_baseline import run_ma_fp
from stage8 import load_stage8_experiment_config, make_scene_config


CONFIG_PATH = ROOT / "configs" / "stage8_monte_carlo.yaml"
OUTPUT_PATH = ROOT / "results" / "stage8" / "diagnostics" / "b6_scene8006_rho1e2_seed2026.csv"

SCENE_SEED = 8006
RHO_SI = 1.0e-2          # -20 dB
START_SEED = 2026        # 报错中对应的 B6 start
DEFAULT_MAX_OUTER = 1500 # 诊断时给 1000 之后留出观察区间


def relative_change(new: float, old: float) -> float:
    return abs(new - old) / max(1.0, abs(old))


def window_gain(history: np.ndarray, end_iter: int, window: int) -> float:
    """返回以 end_iter 为终点、过去 window 轮的相对 WSR 增益。"""
    if end_iter <= 0 or history.size <= end_iter:
        return float("nan")

    start_iter = max(0, end_iter - window)
    old = float(history[start_iter])
    new = float(history[end_iter])
    return (new - old) / max(1.0, abs(old))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-outer",
        type=int,
        default=DEFAULT_MAX_OUTER,
        help="诊断最大外层迭代数，默认 1500。",
    )
    args = parser.parse_args()

    base_cfg = load_config(CONFIG_PATH)
    exp_cfg = load_stage8_experiment_config(CONFIG_PATH)

    rho_cfg = replace(base_cfg, rho_si=RHO_SI)
    scene_cfg = make_scene_config(
        rho_cfg,
        scene_seed=SCENE_SEED,
        position_jitter=exp_cfg.position_jitter,
        randomize_fading=exp_cfg.fading_enabled,
        fading_model=exp_cfg.fading_model,
    )

    randomize_initial_geometry = (
        int(START_SEED) != int(exp_cfg.fixed_upa_seed)
    )

    print("B6 difficult-start convergence diagnostic")
    print(f"  scene_seed       = {SCENE_SEED}")
    print(f"  rho_si           = {RHO_SI:.1e} (-20 dB)")
    print(f"  start_seed       = {START_SEED}")
    print(f"  fixed_upa_seed   = {exp_cfg.fixed_upa_seed}")
    print(f"  random_init      = {randomize_initial_geometry}")
    print(f"  fading_enabled   = {exp_cfg.fading_enabled}")
    print(f"  MA region_size   = {exp_cfg.ma_region_size:.6f} m")
    print(f"  MA min_spacing   = {exp_cfg.ma_min_spacing:.6f} m")
    print(f"  MA initial_step  = {exp_cfg.ma_initial_step:.6f} m")
    print(f"  tol_inner        = {scene_cfg.tol_inner:.3e}")
    print(f"  tol_outer        = {scene_cfg.tol_outer:.3e}")
    print(f"  cfg max_inner    = {scene_cfg.max_inner_iter}")
    print(f"  plateau_window   = {scene_cfg.plateau_window}")
    print(f"  tol_window       = {scene_cfg.tol_window:.3e}")
    print(f"  diagnostic outer = {args.max_outer}")

    if abs(exp_cfg.ma_region_size - 0.03) > 1e-12:
        print("\n[WARNING] 当前 MA region_size 不是 0.03 m，请先确认 3λ 配置。")

    start = perf_counter()

    result = run_ma_fp(
        scene_cfg,
        rng=np.random.default_rng(START_SEED),
        verbose=False,
        region_size=exp_cfg.ma_region_size,
        min_spacing=exp_cfg.ma_min_spacing,
        initial_step=exp_cfg.ma_initial_step,
        max_outer_iterations=args.max_outer,
        randomize_initial_geometry=randomize_initial_geometry,
        initial_jitter_fraction=exp_cfg.ma_initial_jitter_fraction,
    )

    runtime = perf_counter() - start
    history = np.asarray(result.wsr_history, dtype=float)

    final_iter = int(result.iterations)
    final_rel = (
        relative_change(history[-1], history[-2])
        if history.size >= 2
        else float("nan")
    )

    print("\n[Final result]")
    print(f"  iterations       = {result.iterations}")
    print(f"  converged        = {result.converged}")
    print(f"  stop_reason      = {result.stop_reason}")
    print(f"  initial WSR      = {history[0]:.10f}")
    print(f"  final WSR        = {history[-1]:.10f}")
    print(f"  final rel change = {final_rel:.3e}")
    print(f"  last-20 gain     = {window_gain(history, final_iter, 20):.3e}")
    print(f"  last-50 gain     = {window_gain(history, final_iter, 50):.3e}")
    print(f"  runtime          = {runtime:.2f} s")

    if history.size > 1000:
        rel_1000 = relative_change(history[1000], history[999])
        gain20_1000 = window_gain(history, 1000, 20)
        gain50_1000 = window_gain(history, 1000, 50)

        print("\n[State at outer iteration 1000]")
        print(f"  WSR@1000         = {history[1000]:.10f}")
        print(f"  rel change@1000  = {rel_1000:.3e}")
        print(f"  previous-20 gain = {gain20_1000:.3e}")
        print(f"  previous-50 gain = {gain50_1000:.3e}")
    else:
        rel_1000 = gain20_1000 = gain50_1000 = float("nan")
        print("\n[State at outer iteration 1000]")
        if result.converged:
            print("  This run converged before iteration 1000.")
        else:
            print("  No iteration-1000 state is available in this run.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "scene_seed": SCENE_SEED,
        "rho_si": RHO_SI,
        "start_seed": START_SEED,
        "max_outer_diagnostic": args.max_outer,
        "iterations": result.iterations,
        "converged": result.converged,
        "stop_reason": result.stop_reason,
        "initial_wsr": history[0],
        "final_wsr": history[-1],
        "final_relative_change": final_rel,
        "final_last20_gain": window_gain(history, final_iter, 20),
        "final_last50_gain": window_gain(history, final_iter, 50),
        "wsr_at_1000": history[1000] if history.size > 1000 else float("nan"),
        "relative_change_at_1000": rel_1000,
        "previous20_gain_at_1000": gain20_1000,
        "previous50_gain_at_1000": gain50_1000,
        "runtime_s": runtime,
    }

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    print(f"\nCSV saved = {OUTPUT_PATH}")

    print("\n[Decision guide]")
    if result.converged and result.iterations <= 1000:
        print("  This isolated run converged within 1000; keep max_outer_iter=1000.")
    elif result.converged and result.iterations <= 1500:
        print("  It converged only after 1000; max_outer_iter=1500 is justified.")
    elif result.converged:
        print("  It converged, but needed >1500; use the observed iterations to set a margin.")
    else:
        print("  It still did not converge. Do not change multi-start logic; inspect the")
        print("  1000/final WSR gains before deciding whether to test max_outer_iter=2000.")


if __name__ == "__main__":
    main()
