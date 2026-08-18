"""Stage 8：rho_si、P_DL,max 与 cable length 参数扫描。"""

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
    run_parameter_sweep,
    save_sweep_results,
)


CONFIG_PATH = ROOT / "configs" / "stage8_monte_carlo.yaml"
OUTPUT_DIR = ROOT / "results" / "stage8"


def _values_for(parameter: str, experiment_cfg):
    if parameter == "rho_si":
        return experiment_cfg.rho_si_values
    if parameter == "p_dl_max":
        return experiment_cfg.p_dl_max_values
    if parameter == "cable_length":
        return experiment_cfg.cable_length_values
    raise ValueError(f"Unknown parameter: {parameter}")


def _print_sweep_summary(records) -> None:
    parameter_name = records[0].parameter_name
    values = sorted({record.parameter_value for record in records})

    print(f"\n[{parameter_name} Sweep Summary]")

    for value in values:
        subset = [
            record
            for record in records
            if record.parameter_value == value
        ]

        text = []
        for key in SCHEME_ORDER:
            mean = sum(
                record.scene.schemes[key].wsr
                for record in subset
            ) / len(subset)
            text.append(f"{key.upper()}={mean:.6f}")

        print(f"{parameter_name}={value:.6g}: " + "  ".join(text))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parameter",
        choices=("rho_si", "p_dl_max", "cable_length"),
        required=True,
    )
    parser.add_argument(
        "--n-mc",
        type=int,
        default=None,
        help="Override experiment.n_mc.",
    )
    args = parser.parse_args()

    cfg = load_config(CONFIG_PATH)
    experiment_cfg = load_stage8_experiment_config(CONFIG_PATH)
    values = _values_for(args.parameter, experiment_cfg)

    records = run_parameter_sweep(
        cfg,
        experiment_cfg,
        parameter_name=args.parameter,
        values=values,
        n_mc=args.n_mc,
        require_convergence=True,
        verbose=True,
    )

    _print_sweep_summary(records)

    stem = f"stage8_sweep_{args.parameter}"
    csv_path, npz_path = save_sweep_results(
        records,
        OUTPUT_DIR,
        stem=stem,
    )

    print(f"\nCSV saved to: {csv_path}")
    print(f"NPZ saved to: {npz_path}")
    print("Stage 8 parameter sweep passed.")


if __name__ == "__main__":
    main()
