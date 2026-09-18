"""K=J=L=5 后重新运行 Stage 8 正式实验。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from stage8 import load_stage8_experiment_config


MAIN_CONFIG = ROOT / "configs" / "stage8_monte_carlo.yaml"
LOAD_CONFIGS = {
    load: ROOT / "configs" / f"stage8_load{load}.yaml"
    for load in (3, 4, 5, 6, 7)
}

TASKS = {
    "tests": [sys.executable, "-m", "pytest", "-q"],
    "convergence": [
        sys.executable,
        "experiments/plot_proposed_convergence.py",
    ],
    "monte_carlo": [
        sys.executable,
        "experiments/run_stage8_monte_carlo.py",
    ],
    "p_dl_max": [
        sys.executable,
        "experiments/run_stage8_sweeps.py",
        "--parameter",
        "p_dl_max",
    ],
    "rho_si": [
        sys.executable,
        "experiments/run_stage8_sweeps.py",
        "--parameter",
        "rho_si",
    ],
    "system_load": [
        sys.executable,
        "experiments/run_stage8_load_sweep.py",
    ],
    "plots": [
        sys.executable,
        "experiments/plot_stage8_results.py",
    ],
}

FORMAL_ORDER = (
    "tests",
    "convergence",
    "monte_carlo",
    "p_dl_max",
    "rho_si",
    "system_load",
    "plots",
)


def _check_close(value: float, expected: float, name: str) -> None:
    if not np.isclose(value, expected, rtol=0.0, atol=1.0e-12):
        raise RuntimeError(
            f"{name} must be {expected}, but got {value}."
        )


def validate_configs() -> None:
    """运行前检查 K=J=L=5 与正式 Stage 8 参数。"""

    cfg = load_config(MAIN_CONFIG)
    exp = load_stage8_experiment_config(MAIN_CONFIG)

    if (cfg.k_dl, cfg.j_ul, cfg.l_target, cfg.r_sensing) != (5, 5, 5, 4):
        raise RuntimeError(
            "stage8_monte_carlo.yaml must use K=J=L=5 and R_s=4."
        )

    if cfg.weight_dl.shape != (5,):
        raise RuntimeError("DL weight vector must have length 5.")
    if cfg.weight_ul.shape != (5,):
        raise RuntimeError("UL weight vector must have length 5.")
    if cfg.weight_s.shape != (5,):
        raise RuntimeError("Sensing weight vector must have length 5.")

    weight_sum = float(
        np.sum(cfg.weight_dl)
        + np.sum(cfg.weight_ul)
        + np.sum(cfg.weight_s)
    )
    _check_close(weight_sum, 1.0, "Total WSR weight")

    _check_close(cfg.rho_si, 1.0e-10, "Nominal rho_si")

    if exp.fading_enabled:
        raise RuntimeError("Formal Stage 8 must use fading.enabled=false.")

    _check_close(
        exp.ma_region_size,
        30.0 * cfg.wavelength,
        "MA region_size",
    )
    _check_close(
        exp.ma_min_spacing,
        0.5 * cfg.wavelength,
        "MA min_spacing",
    )
    _check_close(exp.ma_initial_step, 0.03, "MA initial_step")

    if exp.ma_n_starts != 3:
        raise RuntimeError("MA baseline must use n_starts=3.")

    if cfg.plateau_window != 0 or cfg.tol_window != 0.0:
        raise RuntimeError(
            "Formal main config must disable window plateau stopping."
        )

    for load, path in LOAD_CONFIGS.items():
        load_cfg = load_config(path)
        load_exp = load_stage8_experiment_config(path)

        expected_r_s = min(load, load_cfg.n_tx)
        actual = (
            load_cfg.k_dl,
            load_cfg.j_ul,
            load_cfg.l_target,
            load_cfg.r_sensing,
        )
        expected = (load, load, load, expected_r_s)

        if actual != expected:
            raise RuntimeError(
                f"stage8_load{load}.yaml expected {expected}, got {actual}."
            )

        _check_close(
            load_cfg.rho_si,
            1.0e-10,
            f"load={load} nominal rho_si",
        )

        if load_exp.fading_enabled:
            raise RuntimeError(
                f"stage8_load{load}.yaml must use fading.enabled=false."
            )

        if load_cfg.plateau_window != 0 or load_cfg.tol_window != 0.0:
            raise RuntimeError(
                f"stage8_load{load}.yaml must disable window plateau stopping."
            )

    print("[PASS] Stage 8 K=5 configuration check")
    print("  nominal: K=J=L=5, R_s=4")
    print("  rho_si = 1e-10")
    print("  fading = OFF")
    print("  MA region = 30 lambda x 30 lambda")
    print("  MA starts = 3")
    print("  load sweep = 3, 4, 5, 6, 7")
    print("  window plateau stop = OFF")


def run_task(name: str, dry_run: bool = False) -> None:
    command = TASKS[name]
    printable = " ".join(command)

    print("\n" + "=" * 88)
    print(f"[{name}] {printable}")
    print("=" * 88)

    if dry_run:
        return

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        nargs="+",
        choices=tuple(TASKS),
        default=None,
        help="Only run selected tasks in the specified order.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip pytest in a full formal rerun.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configs and print commands without running experiments.",
    )
    args = parser.parse_args()

    validate_configs()

    if args.only is not None:
        order = tuple(args.only)
    else:
        order = FORMAL_ORDER
        if args.skip_tests:
            order = tuple(name for name in order if name != "tests")

    for name in order:
        run_task(name, dry_run=args.dry_run)

    if args.dry_run:
        print("\nDry run passed. No experiment was executed.")
    else:
        print("\nAll requested Stage 8 tasks completed successfully.")


if __name__ == "__main__":
    main()
