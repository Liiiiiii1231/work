"""Run only p_dl_max; use the shared models and solvers in the parent directory."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from run_figures import main as run_shared


def main():
    args = sys.argv[1:]
    if any(arg == '--figures' or arg.startswith('--figures=') for arg in args):
        raise SystemExit('This entry point runs p_dl_max only.')
    sys.argv = [str(ROOT / 'run_figures.py'),
                '--figures', 'p_dl_max',
                '--config', str(ROOT / 'simulation_config.json'),
                '--output', str(HERE / 'results' / 'manual'), *args]
    run_shared()


if __name__ == '__main__':
    main()
