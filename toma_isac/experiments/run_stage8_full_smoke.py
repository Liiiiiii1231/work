"""Stage 8 正式大规模仿真入口（不再运行 cable-length sweep）。

依次执行：
1. Monte Carlo 七方案；
2. rho_si 参数扫描；
3. P_DL,max 参数扫描；
4. K=J=L=3/4/5/6 system-load 扫描；
5. 检查新 CSV 是否包含 7 个方案且数值有限；
6. 重画正式 Stage 8 结果图。

Cable-length sweep 已从正式主图与大规模重跑流程中移除。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "stage8"
CONFIG_PATH = ROOT / "configs" / "stage8_monte_carlo.yaml"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from stage8 import SCHEME_ORDER, load_stage8_experiment_config


EXPECTED_SCHEMES = (
    "b1",
    "b2",
    "b3",
    "b4",
    "b5",
    "b6",
    "proposed",
)

EXPECTED_RHO = np.asarray(
    [1.0e-3, 1.0e-2, 1.0e-1, 1.0],
    dtype=float,
)

EXPECTED_LOADS = (3, 4, 5, 6)


def _run(label: str, *args: str) -> None:
    """运行一个子实验；失败时立即停止整个 smoke test。"""

    command = [sys.executable, *args]

    print("\n" + "=" * 92)
    print(f"[Stage 8B-5] {label}")
    print("Command:", " ".join(command))
    print("=" * 92)

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing result CSV: {path}")

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    if len(rows) == 0:
        raise RuntimeError(f"CSV has no data rows: {path}")

    return rows


def _assert_fresh(path: Path, start_time: float) -> None:
    """避免误把 Stage 8B 之前的旧 CSV 当成本次结果。"""

    if not path.exists():
        raise FileNotFoundError(path)

    # Windows/不同文件系统的时间戳精度留 2 秒余量。
    if path.stat().st_mtime < start_time - 2.0:
        raise RuntimeError(
            "Result file was not regenerated in this smoke run: "
            f"{path}"
        )


def _assert_seven_scheme_columns(
    rows: list[dict[str, str]],
    path: Path,
) -> None:
    required = [f"{key}_wsr" for key in EXPECTED_SCHEMES]
    available = set(rows[0])

    missing = [column for column in required if column not in available]

    if missing:
        raise RuntimeError(
            f"{path.name} is missing 7-scheme columns: {missing}"
        )

    for row_index, row in enumerate(rows, start=1):
        for column in required:
            value = float(row[column])
            if not np.isfinite(value):
                raise RuntimeError(
                    f"Non-finite {column} at row {row_index} "
                    f"in {path.name}: {value}"
                )


def _validate_monte_carlo(
    n_mc: int,
    start_time: float,
) -> None:
    path = RESULT_DIR / "stage8_monte_carlo.csv"
    _assert_fresh(path, start_time)
    rows = _read_csv(path)
    _assert_seven_scheme_columns(rows, path)

    if len(rows) != n_mc:
        raise RuntimeError(
            f"Monte Carlo expected {n_mc} rows, got {len(rows)}."
        )


def _validate_parameter_sweep(
    parameter: str,
    expected_values: np.ndarray,
    n_mc: int,
    start_time: float,
) -> None:
    path = RESULT_DIR / f"stage8_sweep_{parameter}.csv"
    _assert_fresh(path, start_time)
    rows = _read_csv(path)
    _assert_seven_scheme_columns(rows, path)

    values = np.asarray(
        sorted({float(row["parameter_value"]) for row in rows}),
        dtype=float,
    )

    if values.shape != expected_values.shape or not np.allclose(
        values,
        expected_values,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            f"Unexpected {parameter} values: {values.tolist()}"
        )

    expected_rows = len(expected_values) * n_mc

    if len(rows) != expected_rows:
        raise RuntimeError(
            f"{parameter} sweep expected {expected_rows} rows, "
            f"got {len(rows)}."
        )


def _validate_system_load(
    n_mc: int,
    start_time: float,
) -> None:
    path = RESULT_DIR / "stage8_sweep_system_load.csv"
    _assert_fresh(path, start_time)
    rows = _read_csv(path)
    _assert_seven_scheme_columns(rows, path)

    loads = tuple(
        sorted({int(row["system_load"]) for row in rows})
    )

    if loads != EXPECTED_LOADS:
        raise RuntimeError(
            f"Unexpected system loads: {loads}; "
            f"expected {EXPECTED_LOADS}."
        )

    expected_rows = len(EXPECTED_LOADS) * n_mc

    if len(rows) != expected_rows:
        raise RuntimeError(
            f"System-load sweep expected {expected_rows} rows, "
            f"got {len(rows)}."
        )

    # 检查 R_s=min(L,N_T)，N_T 从各行本身不能直接恢复；
    # 当前 N_T=4 的正式配置对应 3,4,4,4。
    expected_r_s = {3: 3, 4: 4, 5: 4, 6: 4}

    for row in rows:
        load = int(row["system_load"])
        r_s = int(row["R_s"])
        if r_s != expected_r_s[load]:
            raise RuntimeError(
                f"load={load}: expected R_s={expected_r_s[load]}, "
                f"got {r_s}."
            )


def _preflight() -> np.ndarray:
    """在长时间运行前先检查正式设置是否已接入。"""

    if tuple(SCHEME_ORDER) != EXPECTED_SCHEMES:
        raise RuntimeError(
            "SCHEME_ORDER is not the expected 7-scheme order.\n"
            f"Current: {tuple(SCHEME_ORDER)}\n"
            f"Expected: {EXPECTED_SCHEMES}"
        )

    cfg = load_config(CONFIG_PATH)
    experiment_cfg = load_stage8_experiment_config(CONFIG_PATH)

    if not experiment_cfg.fading_enabled:
        raise RuntimeError(
            "Stage 8 random fading is disabled in "
            "configs/stage8_monte_carlo.yaml."
        )

    if experiment_cfg.fading_model != "scalar_rayleigh":
        raise RuntimeError(
            "Unexpected fading model: "
            f"{experiment_cfg.fading_model}"
        )

    if not np.isclose(cfg.finite_diff_delta, 1.0e-6):
        raise RuntimeError(
            "finite_diff_delta is not the frozen Stage 8A value 1e-6: "
            f"{cfg.finite_diff_delta}"
        )

    rho_values = np.asarray(
        experiment_cfg.rho_si_values,
        dtype=float,
    )

    if rho_values.shape != EXPECTED_RHO.shape or not np.allclose(
        rho_values,
        EXPECTED_RHO,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "rho_si sweep is not [1e-3, 1e-2, 1e-1, 1]. "
            f"Current: {rho_values.tolist()}"
        )

    p_values = np.asarray(
        experiment_cfg.p_dl_max_values,
        dtype=float,
    )
    for load in EXPECTED_LOADS:
        path = ROOT / "configs" / f"stage8_load{load}.yaml"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing system-load config: {path}"
            )

    print("[Preflight passed]")
    print(f"  schemes       = {EXPECTED_SCHEMES}")
    print(
        "  fading        = "
        f"{experiment_cfg.fading_model} "
        f"(enabled={experiment_cfg.fading_enabled})"
    )
    print(f"  delta         = {cfg.finite_diff_delta:.1e}")
    print(f"  rho_si        = {rho_values.tolist()}")
    print(f"  p_dl_max      = {p_values.tolist()}")
    print(f"  system loads  = {EXPECTED_LOADS}")

    return p_values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-mc",
        type=int,
        default=2,
        help="Monte Carlo size for the formal Stage 8 experiments.",
    )
    args = parser.parse_args()

    if args.n_mc < 1:
        raise ValueError("--n-mc must be >= 1.")

    n_mc = int(args.n_mc)

    p_values = _preflight()

    start_time = time.time()

    _run(
        "1/5 Monte Carlo: random fading + 7 schemes",
        "experiments/run_stage8_monte_carlo.py",
        "--n-mc",
        str(n_mc),
    )

    _run(
        "2/5 rho_si sweep",
        "experiments/run_stage8_sweeps.py",
        "--parameter",
        "rho_si",
        "--n-mc",
        str(n_mc),
    )

    _run(
        "3/5 DL-power sweep",
        "experiments/run_stage8_sweeps.py",
        "--parameter",
        "p_dl_max",
        "--n-mc",
        str(n_mc),
    )

    _run(
        "4/5 system-load sweep",
        "experiments/run_stage8_load_sweep.py",
        "--loads",
        "3",
        "4",
        "5",
        "6",
        "--n-mc",
        str(n_mc),
    )

    print("\n" + "=" * 92)
    print("[Stage 8B-5] Validating newly generated CSV files")
    print("=" * 92)

    _validate_monte_carlo(n_mc, start_time)
    _validate_parameter_sweep(
        "rho_si",
        EXPECTED_RHO,
        n_mc,
        start_time,
    )
    _validate_parameter_sweep(
        "p_dl_max",
        p_values,
        n_mc,
        start_time,
    )
    _validate_system_load(n_mc, start_time)

    print("CSV validation passed: all new files contain 7 schemes and finite WSRs.")

    _run(
        "5/5 Regenerate Stage 8 figures",
        "experiments/plot_stage8_results.py",
    )

    print("\n" + "=" * 92)
    print("Stage 8 FINAL RUN PASSED")
    print("=" * 92)
    print(f"n_mc = {n_mc}")
    print("Checked:")
    print("  - scene-level random fading")
    print("  - B1~B6 + Proposed")
    print("  - rho_si = 1e-3, 1e-2, 1e-1, 1")
    print("  - P_DL,max sweep")
    print("  - K=J=L = 3,4,5,6")
    print("  - regenerated result figures")
    print(f"Results: {RESULT_DIR}")
    print(f"Figures: {RESULT_DIR / 'figures'}")


if __name__ == "__main__":
    main()
