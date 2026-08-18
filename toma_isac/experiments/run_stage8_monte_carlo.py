"""Stage 8：六方案 Monte Carlo 主实验。"""

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from stage8 import (
    SCHEME_ORDER,
    load_stage8_experiment_config,
    run_monte_carlo,
    save_monte_carlo_results,
    summarize_monte_carlo,
)


CONFIG_PATH = ROOT / "configs" / "stage8_monte_carlo.yaml"
OUTPUT_DIR = ROOT / "results" / "stage8"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-mc",
        type=int,
        default=None,
        help="Override experiment.n_mc for a quick/formal run.",
    )
    args = parser.parse_args()

    cfg = load_config(CONFIG_PATH)
    experiment_cfg = load_stage8_experiment_config(CONFIG_PATH)

    print("=" * 88)
    print("Stage 8: Monte Carlo Six-scheme Comparison")
    print("=" * 88)
    print(
        f"K={cfg.k_dl}, J={cfg.j_ul}, "
        f"L={cfg.l_target}, R_S={cfg.r_sensing}"
    )
    print(f"Multi-start seeds = {experiment_cfg.multi_start_seeds}")

    scenes = run_monte_carlo(
        cfg,
        experiment_cfg,
        n_mc=args.n_mc,
        require_convergence=True,
        verbose=True,
    )

    summary = summarize_monte_carlo(scenes)

    print("\n[Monte Carlo Summary]")
    for key in SCHEME_ORDER:
        mean, std = summary[key]
        print(
            f"{key.upper():>8}: "
            f"mean WSR={mean:.10f}, std={std:.10f}"
        )

    csv_path, npz_path = save_monte_carlo_results(
        scenes,
        OUTPUT_DIR,
    )

    print(f"\nCSV saved to: {csv_path}")
    print(f"NPZ saved to: {npz_path}")
    print("Stage 8 Monte Carlo passed.")


if __name__ == "__main__":
    main()
