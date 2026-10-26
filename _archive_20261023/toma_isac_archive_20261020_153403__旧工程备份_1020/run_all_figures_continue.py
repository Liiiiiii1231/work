"""ToMA-ISAC: sequentially rerun five paper figures; one failure never blocks later figures.

Place this file beside config.py in the project root, then run:
    python -u run_all_figures_continue.py

Results: results/stage8/figures and results/stage10/cold_wide14_tol1e4/figures.
Note: Stage 10 job uses the independent cold-start runner, not the multiround pool runner.
Logs, backups, status: results/auto_figure_runs/<timestamp>/
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parent
STAGE8 = ROOT / 'results' / 'stage8'
STAGE10 = ROOT / 'results' / 'stage10'
PYTHON = sys.executable


# Each entry is a self-contained figure job: regenerate data, then plot only on success.
JOBS = [
    {
        'name': '01_proposed_convergence',
        'title': 'Proposed outer WSR convergence',
        'simulate': ['experiments/plot_proposed_convergence.py'],  # This also plots.
        'plot': None,
        'data': ['results/stage8/proposed_convergence_scene_8001_with_inset.csv'],
        'figures': ['results/stage8/figures/proposed_convergence_scene_8001_with_inset.png',
                    'results/stage8/figures/proposed_convergence_scene_8001_with_inset.pdf'],
    },
    {
        'name': '02_downlink_power',
        'title': 'Average WSR vs max downlink transmit power',
        'simulate': ['experiments/run_stage8_sweeps.py', '--parameter', 'p_dl_max'],
        'plot': ['-c',
                 'import numpy as np; from experiments.plot_stage8_results import plot_sweep,_watt_to_dbm; '
                 'plot_sweep("p_dl_max", "Maximum DL transmit power (dBm)", '
                 '_watt_to_dbm, np.array([20.,25.,30.,35.,40.]))'],
        'data': ['results/stage8/stage8_sweep_p_dl_max.csv',
                 'results/stage8/stage8_sweep_p_dl_max.npz'],
        'figures': ['results/stage8/figures/wsr_vs_p_dl_max.png',
                    'results/stage8/figures/wsr_vs_p_dl_max.pdf'],
    },
    {
        'name': '03_residual_SI',
        'title': 'Average WSR vs residual self-interference',
        'simulate': ['experiments/run_stage8_sweeps.py', '--parameter', 'rho_si'],
        'plot': ['-c',
                 'import numpy as np; from experiments.plot_stage8_results import plot_sweep,_rho_to_db; '
                 'plot_sweep("rho_si", "Residual SI level (dB)", '
                 '_rho_to_db, np.array([-50.,-40.,-30.,-20.,-10.]))'],
        'data': ['results/stage8/stage8_sweep_rho_si.csv',
                 'results/stage8/stage8_sweep_rho_si.npz'],
        'figures': ['results/stage8/figures/wsr_vs_rho_si.png',
                    'results/stage8/figures/wsr_vs_rho_si.pdf'],
    },
    {
        'name': '04_system_load',
        'title': 'Average WSR vs system load (K=J=L=3..7)',
        'simulate': ['experiments/run_stage8_load_sweep.py', '--loads', '3', '4', '5', '6', '7'],
        'plot': ['-c',
                 'from experiments.plot_stage8_results import plot_system_load; '
                 'plot_system_load()'],
        'data': ['results/stage8/stage8_sweep_system_load.csv',
                 'results/stage8/stage8_sweep_system_load.npz'],
        'figures': ['results/stage8/figures/wsr_vs_system_load.png',
                    'results/stage8/figures/wsr_vs_system_load.pdf'],
    },
    {
        'name': '05_communication_sensing_tradeoff',
        'title': 'Stage 10 communication-sensing tradeoff',
        'simulate': ['experiments/run_stage10_tradeoff.py'],
        'plot': ['experiments/plot_stage10_tradeoff.py'],
        'data': ['results/stage10/cold_wide14_tol1e4/stage10_tradeoff.csv',
                 'results/stage10/cold_wide14_tol1e4/stage10_tradeoff.npz'],
        'figures': ['results/stage10/cold_wide14_tol1e4/figures/communication_sensing_tradeoff.png',
                    'results/stage10/cold_wide14_tol1e4/figures/communication_sensing_tradeoff.pdf'],
    },
]


def check_environment() -> None:
    missing = [str(p) for p in ('config.py', 'experiments/plot_stage8_results.py')
               if not (ROOT / p).is_file()]
    for job in JOBS:
        script = job['simulate'][0]
        if script.endswith('.py') and not (ROOT / script).is_file():
            missing.append(script)
        plot = job['plot']
        if plot and plot[0].endswith('.py') and not (ROOT / plot[0]).is_file():
            missing.append(plot[0])
    if missing:
        raise RuntimeError('Script must be placed in PROJECT ROOT (beside config.py). '
                           'Missing: ' + ', '.join(sorted(set(missing))))

    # Fail fast if the current project still uses the OLD channel/noise configuration.
    try:
        from config import load_config
    except Exception as exc:
        raise RuntimeError('Cannot load project configuration (check Python env): ' + str(exc)) from exc
    for filename in ('stage8_monte_carlo.yaml', 'stage8_load3.yaml', 'stage8_load4.yaml',
                     'stage8_load5.yaml', 'stage8_load6.yaml', 'stage8_load7.yaml',
                     'stage10_tradeoff.yaml'):
        cfg = load_config(ROOT / 'configs' / filename)
        expected_noise = 3.981071705534969e-15
        if not math.isclose(cfg.sigma_bs2, expected_noise, rel_tol=1e-7):
            raise RuntimeError(f'{filename}: sigma_bs2={cfg.sigma_bs2}; '
                               f'expected {expected_noise} W (-114 dBm).')
        xi = tuple(complex(x) for x in cfg.xi_s)
        expected_xi = math.sqrt(4 * math.pi * 10.0) / cfg.wavelength
        if not xi or any(not math.isclose(x.real, expected_xi, rel_tol=1e-7)
                         or not math.isclose(x.imag, 0.0, abs_tol=1e-10) for x in xi):
            raise RuntimeError(f'{filename}: scattering coefficient does not match '
                               'RCS=10 m^2. Apply the latest patch first.')
        if any(not math.isclose(float(x), expected_noise, rel_tol=1e-7)
               for x in cfg.sigma_dl2):
            raise RuntimeError(f'{filename}: DL user noise is not -114 dBm.')


def save_status(folder: Path, rows: list[dict]) -> None:
    path = folder / 'STATUS.csv'
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle,
                                fieldnames=['figure', 'status', 'seconds', 'reason', 'log'])
        writer.writeheader()
        writer.writerows(rows)


def relocate(paths: list[str], to_dir: Path) -> None:
    for relative in paths:
        src = ROOT / relative
        if src.is_file():
            dest = to_dir / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))


def run_command(args: list[str], log: Path, timeout_seconds: float | None) -> tuple[bool, str]:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w', encoding='utf-8', errors='replace') as output:
        output.write('Working directory: ' + str(ROOT) + '\n')
        output.write('Command: ' + ' '.join([PYTHON, '-u', *args]) + '\n\n')
        output.flush()
        try:
            p = subprocess.run([PYTHON, '-u', *args], cwd=str(ROOT),
                               stdout=output, stderr=subprocess.STDOUT,
                               timeout=timeout_seconds, check=False,
                               env={**os.environ, 'MPLBACKEND': 'Agg', 'PYTHONUNBUFFERED': '1'})
            return p.returncode == 0, f'exit code {p.returncode}'
        except subprocess.TimeoutExpired:
            output.write('\n[TIMEOUT] Process exceeded per-figure limit.\n')
            return False, 'timeout'
        except Exception as exc:
            output.write('\n[EXCEPTION] ' + repr(exc) + '\n')
            return False, str(exc)


def execute(jobs: list[dict], log_dir: Path, timeout_hours: float) -> list[dict]:
    log_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    timeout = timeout_hours * 3600 if timeout_hours > 0 else None
    print('Project:', ROOT, flush=True)
    print('Logs:', log_dir, flush=True)
    print('Mode: each figure is independent; failure moves to the next.\n', flush=True)
    for index, job in enumerate(jobs, start=1):
        name, title = job['name'], job['title']
        print(f'[{index}/{len(jobs)}] START {name} - {title}', flush=True)
        started = time.monotonic()
        outputs = job['data'] + job['figures']
        relocate(outputs, log_dir / 'old_results' / name)  # Old plots remain recoverable.
        sim_log = log_dir / f'{name}_simulation.log'
        ok, reason = run_command(job['simulate'], sim_log, timeout)
        final_log = sim_log
        if ok and not all((ROOT / p).is_file() for p in job['data']):
            ok, reason = False, 'simulation returned success but expected new CSV/NPZ is missing'
        if ok and job['plot']:
            plot_log = log_dir / f'{name}_plot.log'
            ok, reason = run_command(job['plot'], plot_log, timeout)
            final_log = plot_log if not ok else sim_log
        if ok and not all((ROOT / p).is_file() for p in job['figures']):
            ok, reason = False, 'expected new PNG/PDF was not produced'
        if not ok:
            # Never leave failed/incomplete outputs in the official results directory.
            relocate(outputs, log_dir / 'incomplete_results' / name)
        status = 'SUCCESS' if ok else 'FAILED_SKIPPED'
        elapsed = round(time.monotonic() - started, 2)
        rows.append({'figure': name, 'status': status, 'seconds': elapsed,
                     'reason': reason, 'log': str(final_log.relative_to(ROOT))})
        save_status(log_dir, rows)
        print(f'[{index}/{len(jobs)}] {status} ({elapsed:.0f}s) - {reason}', flush=True)
        if not ok:
            print(f'    Error details: {sim_log}'
                  + (f' / {final_log}' if final_log != sim_log else ''), flush=True)
        else:
            print('    New figures: ' + ', '.join(job['figures']), flush=True)
    print('\nALL JOBS ATTEMPTED. See:', log_dir / 'STATUS.csv', flush=True)
    print(f'Success: {sum(row["status"] == "SUCCESS" for row in rows)}; '
          f'Failed: {sum(row["status"] != "SUCCESS" for row in rows)}', flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout-hours', type=float, default=12.,
                        help='Max hours for each simulation/plot, default 12; 0=unlimited.')
    parser.add_argument('--check-only', action='store_true',
                        help='Only verify files and RCS/noise configs, do not run simulations.')
    parser.add_argument('--self-test', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.self_test:
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as path:
            log = Path(path) / 'run.log'
            ok1, _ = run_command(['-c', 'import sys;sys.exit(5)'], log, 20)
            ok2, _ = run_command(['-c', 'print("second job still runs")'], log, 20)
            assert not ok1 and ok2
            print('SELF TEST PASSED: first failure does not block the second.')
        return
    check_environment()
    if args.check_only:
        print('CHECK PASSED: correct project entrypoints, RCS=10 m^2, BS=-114 dBm, DL=-114 dBm.')
        return
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = ROOT / 'results' / 'auto_figure_runs' / stamp
    execute(JOBS, log_dir, args.timeout_hours)


if __name__ == '__main__':
    main()
