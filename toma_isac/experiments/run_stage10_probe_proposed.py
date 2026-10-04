"""Stage 10 Probe：仅用 Proposed 定位 Communication-Sensing trade-off 敏感区。"""

from __future__ import annotations

import csv
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm import run_joint_algorithm
from config import SystemConfig, load_config
from stage8 import load_stage8_experiment_config, make_scene_config


CONFIG_PATH = ROOT / "configs" / "stage10_tradeoff.yaml"

RESULT_DIR = ROOT / "results" / "stage10_probe_proposed"
CSV_PATH = RESULT_DIR / "stage10_probe_proposed.csv"


# ============================================================
# 第一轮非均匀权重扫描
# 感知占优区域大量加密
# ============================================================
OMEGA_COMM = (
    0.0,
    1.0e-10,
    3.0e-10,
    1.0e-9,
    3.0e-9,
    1.0e-8,
    2.0e-8,
    3.0e-8,
    5.0e-8,
    1.0e-7,
    3.0e-7,
    1.0e-6,
    1.0e-5,
    1.0e-4,
    1.0e-3,
    1.0e-2,
    5.0e-2,
    1.0e-1,
    5.0e-1,
    1.0,
)

SCENE_SEED = 8001
BASE_SEED = 2026


def make_group_weights(
    cfg: SystemConfig,
    omega_comm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构造原始 WSR 权重，不做归一化。"""

    omega_comm = float(omega_comm)

    if not 0.0 <= omega_comm <= 1.0:
        raise ValueError("omega_comm must lie in [0, 1].")

    omega_sensing = 1.0 - omega_comm

    # 通信总权重平均分给 DL / UL 两组，
    # 再在各组内部平均分配。
    weight_dl = np.full(
        cfg.k_dl,
        omega_comm / (2.0 * cfg.k_dl),
        dtype=float,
    )

    weight_ul = np.full(
        cfg.j_ul,
        omega_comm / (2.0 * cfg.j_ul),
        dtype=float,
    )

    # 感知总权重平均分给 L 个目标。
    weight_s = np.full(
        cfg.l_target,
        omega_sensing / cfg.l_target,
        dtype=float,
    )

    total = float(
        weight_dl.sum()
        + weight_ul.sum()
        + weight_s.sum()
    )

    if not np.isclose(total, 1.0, atol=1.0e-12):
        raise RuntimeError(
            f"Weight sum is {total}, expected 1."
        )

    return weight_dl, weight_ul, weight_s


def build_weighted_cfg(
    base_cfg: SystemConfig,
    omega_comm: float,
) -> SystemConfig:
    """生成当前 omega_C 对应的系统配置。"""

    weight_dl, weight_ul, weight_s = make_group_weights(
        base_cfg,
        omega_comm,
    )

    cfg = replace(
        base_cfg,
        weight_dl=weight_dl,
        weight_ul=weight_ul,
        weight_s=weight_s,
    )

    cfg.validate()

    return cfg


def extract_rates(result) -> dict[str, float]:
    """提取当前解的通信/感知速率。"""

    perf = result.performance

    sum_rate_dl = float(np.sum(perf.rate_dl))
    sum_rate_ul = float(np.sum(perf.rate_ul))

    return {
        "sum_rate_dl": sum_rate_dl,
        "sum_rate_ul": sum_rate_ul,
        "sum_rate_comm": sum_rate_dl + sum_rate_ul,
        "sum_rate_sensing": float(np.sum(perf.rate_s)),
        "weighted_sum_rate": float(perf.weighted_sum_rate),
    }


def print_transition_diagnostic(rows: list[dict]) -> None:
    """根据归一化 RC-RS 距离寻找真正的 trade-off 转折区。"""

    if len(rows) < 2:
        return

    rc = np.asarray(
        [row["sum_rate_comm"] for row in rows],
        dtype=float,
    )

    rs = np.asarray(
        [row["sum_rate_sensing"] for row in rows],
        dtype=float,
    )

    rc_span = max(float(np.max(rc) - np.min(rc)), 1.0e-15)
    rs_span = max(float(np.max(rs) - np.min(rs)), 1.0e-30)

    rc_norm = (rc - np.min(rc)) / rc_span
    rs_norm = (rs - np.min(rs)) / rs_span

    jumps = []

    for index in range(len(rows) - 1):
        delta_rc = rc_norm[index + 1] - rc_norm[index]
        delta_rs = rs_norm[index + 1] - rs_norm[index]

        jump = float(
            np.hypot(delta_rc, delta_rs)
        )

        jumps.append(
            (
                jump,
                rows[index]["omega_comm"],
                rows[index + 1]["omega_comm"],
            )
        )

    jumps.sort(reverse=True)

    print("\nLargest normalized trade-off transitions:")

    for jump, left, right in jumps[:6]:
        print(
            f"  omega_C: {left:.3e} -> {right:.3e}, "
            f"normalized jump = {jump:.6f}"
        )


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    base_cfg = load_config(CONFIG_PATH)
    experiment_cfg = load_stage8_experiment_config(CONFIG_PATH)

    if (
        base_cfg.k_dl,
        base_cfg.j_ul,
        base_cfg.l_target,
        base_cfg.r_sensing,
    ) != (5, 5, 5, 4):
        raise RuntimeError(
            "Probe requires K=J=L=5 and R_s=4."
        )

    if experiment_cfg.fading_enabled:
        raise RuntimeError(
            "Probe requires fading.enabled=false."
        )

    # --------------------------------------------------------
    # 所有权重点使用完全相同的物理场景
    # --------------------------------------------------------
    physical_scene_cfg = make_scene_config(
        base_cfg,
        scene_seed=SCENE_SEED,
        position_jitter=experiment_cfg.position_jitter,
        randomize_fading=False,
        fading_model=experiment_cfg.fading_model,
    )

    rows: list[dict] = []

    previous_result = None

    total_start = perf_counter()

    print("=" * 78)
    print("Stage 10 Proposed-only nonuniform weight probe")
    print("=" * 78)

    print(f"scene_seed = {SCENE_SEED}")
    print(f"number of weights = {len(OMEGA_COMM)}")
    print("method = original weighted-sum WSR")
    print("normalization = OFF")
    print("scheme = FP-aware RCG-ToMA + FP (Proposed)")
    print("continuation = previous omega solution warm-start")

    for index, omega_comm in enumerate(OMEGA_COMM):

        cfg = build_weighted_cfg(
            physical_scene_cfg,
            omega_comm,
        )

        print("\n" + "-" * 78)
        print(
            f"[{index + 1:02d}/{len(OMEGA_COMM)}] "
            f"omega_C = {omega_comm:.3e}, "
            f"omega_S = {1.0 - omega_comm:.12f}"
        )
        print("-" * 78)

        run_start = perf_counter()

        # ----------------------------------------------------
        # 第一个权重点使用固定初始随机种子；
        # 后续权重点使用前一个最优解 warm-start。
        # ----------------------------------------------------
        if previous_result is None:

            result = run_joint_algorithm(
                cfg,
                rng=np.random.default_rng(BASE_SEED),
                verbose=False,
                position_update_mode="fp_aware",
                fp_aware_candidate_iters=2,
            )

            initialization = "base"

        else:

            result = run_joint_algorithm(
                cfg,
                rng=np.random.default_rng(BASE_SEED + index),
                verbose=False,
                position_update_mode="fp_aware",
                fp_aware_candidate_iters=2,
                initial_endpoints=previous_result.geometry.endpoints,
                initial_resources=previous_result.resources,
            )

            initialization = "warm_start"

        runtime_s = perf_counter() - run_start

        rates = extract_rates(result)

        row = {
            "omega_comm": float(omega_comm),
            "omega_sensing": float(1.0 - omega_comm),
            "sum_rate_dl": rates["sum_rate_dl"],
            "sum_rate_ul": rates["sum_rate_ul"],
            "sum_rate_comm": rates["sum_rate_comm"],
            "sum_rate_sensing": rates["sum_rate_sensing"],
            "weighted_sum_rate": rates["weighted_sum_rate"],
            "outer_iterations": int(result.outer_iterations),
            "runtime_s": float(runtime_s),
            "converged": bool(result.converged),
            "initialization": initialization,
        }

        rows.append(row)

        print(
            f"R_C = {row['sum_rate_comm']:.10e}"
        )
        print(
            f"R_S = {row['sum_rate_sensing']:.10e}"
        )
        print(
            f"WSR = {row['weighted_sum_rate']:.10e}"
        )
        print(
            f"outer iterations = {row['outer_iterations']}"
        )
        print(
            f"converged = {row['converged']}"
        )

        # 下一权重点从当前解继续。
        previous_result = result

    # --------------------------------------------------------
    # 保存 CSV
    # --------------------------------------------------------
    fieldnames = [
        "omega_comm",
        "omega_sensing",
        "sum_rate_dl",
        "sum_rate_ul",
        "sum_rate_comm",
        "sum_rate_sensing",
        "weighted_sum_rate",
        "outer_iterations",
        "runtime_s",
        "converged",
        "initialization",
    ]

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    elapsed = perf_counter() - total_start

    print("\n" + "=" * 78)
    print("Probe completed")
    print("=" * 78)

    print(f"CSV saved to: {CSV_PATH}")
    print(f"total runtime = {elapsed:.2f} s")

    print_transition_diagnostic(rows)


if __name__ == "__main__":
    main()