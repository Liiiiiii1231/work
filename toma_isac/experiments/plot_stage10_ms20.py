r"""Redraw only the new Stage 10 20-start / tighter-tolerance experiment."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.plot_stage10_cold20 import load_data, render

RUN_DIR = ROOT / "results/stage10/cold20_ms20_in1e6_out1e5"


def main():
    data = load_data(RUN_DIR / "stage10_tradeoff.csv")
    output = RUN_DIR / "figures"
    output.mkdir(parents=True, exist_ok=True)
    render(data, output)


if __name__ == "__main__":
    main()
