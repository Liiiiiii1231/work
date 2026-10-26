"""Add only the -50 dB residual-SI point to an existing Stage 8 sweep.

Place this file under toma_isac/experiments/ and run from project root:
    python -u .\\experiments\\extend_stage8_rho_si_minus50.py --activate

The original sweep is validated and backed up before --activate replaces the
canonical CSV/NPZ. Neither the other 4 points nor any other Stage 8 figure is
recomputed. With no --activate, only separate merged CSV/NPZ are written.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_config
from stage8 import (
    SCHEME_ORDER, load_stage8_experiment_config,
    run_parameter_sweep, save_sweep_results,
)

CONFIG = ROOT / 'configs' / 'stage8_monte_carlo.yaml'
OUT = ROOT / 'results' / 'stage8'
OLD_CSV = OUT / 'stage8_sweep_rho_si.csv'
OLD_NPZ = OUT / 'stage8_sweep_rho_si.npz'
OLD_VALUES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
KEEP_VALUES = OLD_VALUES[:-1]
NEW_VALUE = 1e-5
SCENES = tuple(range(8001, 8011))
COLUMNS = ('parameter_name', 'parameter_value', 'scene_seed',
           *(f'{key}_wsr' for key in SCHEME_ORDER))


def _key(value: float) -> float:
    for allowed in (NEW_VALUE, *OLD_VALUES):
        if abs(value - allowed) < 1e-14:
            return allowed
    raise ValueError(f'Unexpected residual-SI value: {value!r}')


def _validate_input():
    if not OLD_CSV.is_file() or not OLD_NPZ.is_file():
        raise FileNotFoundError('Original Stage 8 sweep CSV/NPZ not found.')
    with OLD_CSV.open('r', newline='', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(f'CSV columns differ from expected format: {reader.fieldnames}')
        rows = list(reader)
    if len(rows) != 50:
        raise ValueError(f'Expected 50 original rows (5 levels × 10 seeds), got {len(rows)}.')
    counts = {v: set() for v in OLD_VALUES}
    seen = set()
    for row in rows:
        if row['parameter_name'] != 'rho_si':
            raise ValueError('Old CSV contains another sweep parameter.')
        val = _key(float(row['parameter_value']))
        if val == NEW_VALUE:
            raise ValueError('Old CSV already contains the -50 dB point.')
        seed = int(row['scene_seed'])
        if (val, seed) in seen:
            raise ValueError(f'Duplicate old row: {val}, {seed}')
        seen.add((val, seed))
        counts[val].add(seed)
    for val, seeds in counts.items():
        if seeds != set(SCENES):
            raise ValueError(f'Old rho={val}: unexpected scene seeds {sorted(seeds)}.')

    with np.load(OLD_NPZ, allow_pickle=False) as data:
        if str(data['parameter_name']) != 'rho_si':
            raise ValueError('Old NPZ parameter does not match CSV.')
        if list(data['scheme_order']) != list(SCHEME_ORDER):
            raise ValueError('Old NPZ scheme order differs.')
        if data['wsr'].shape != (50, len(SCHEME_ORDER)):
            raise ValueError('Old NPZ WSR dimensions differ.')
        for i, row in enumerate(rows):
            if _key(float(data['parameter_value'][i])) != _key(float(row['parameter_value'])):
                raise ValueError(f'Old CSV/NPZ value mismatch at row {i}.')
            if int(data['scene_seed'][i]) != int(row['scene_seed']):
                raise ValueError(f'Old CSV/NPZ scene mismatch at row {i}.')
            expected = np.array([float(row[f'{k}_wsr']) for k in SCHEME_ORDER])
            if not np.allclose(data['wsr'][i], expected, rtol=1e-12, atol=1e-12):
                raise ValueError(f'Old CSV/NPZ WSR mismatch at row {i}.')
    return [r for r in rows if _key(float(r['parameter_value'])) in KEEP_VALUES]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activate', action='store_true',
                        help='Back up and activate the merged file for plotting.')
    args = parser.parse_args()
    try:
        cfg = load_config(CONFIG)
        exp = load_stage8_experiment_config(CONFIG)
    except Exception as exc:
        raise RuntimeError(f'Unable to read {CONFIG}; fix YAML indentation/syntax first.') from exc
    if exp.n_mc != 10 or exp.scene_seed_start != SCENES[0] or exp.fading_enabled:
        raise ValueError('The experiment must use n_mc=10, scene_seed_start=8001, LoS fading OFF.')
    if (cfg.k_dl, cfg.j_ul, cfg.l_target, cfg.r_sensing) != (5,5,5,4):
        raise ValueError('Unexpected K/J/L/R_s; cannot reuse old results.')
    target_noise = 10 ** ((-114.0 - 30) / 10)
    if not np.allclose(cfg.sigma_dl2, target_noise, rtol=1e-9, atol=0) or \
       not np.isclose(cfg.sigma_bs2, target_noise, rtol=1e-9, atol=0):
        raise ValueError('DL/BS noise must both be -114 dBm.')
    target_xi = (4 * np.pi * 10.0) ** 0.5 / cfg.wavelength
    if not np.allclose(cfg.xi_s, target_xi, rtol=1e-9, atol=0):
        raise ValueError('Expected RCS=10 m^2 for all 5 sensing targets.')
    kept = _validate_input()
    print('[PASS] Original 50 rows/CSV/NPZ verified; retaining 40 rows for -40..-10 dB.')
    print('[RUN] Only -50 dB, rho_si=1e-5, 10 Monte Carlo scenes, all 7 algorithms.')
    records = run_parameter_sweep(
        cfg, exp, parameter_name='rho_si', values=[NEW_VALUE],
        n_mc=10, require_convergence=True, verbose=True,
    )
    if len(records) != 10 or {r.scene.scene_seed for r in records} != set(SCENES):
        raise ValueError('The new -50 dB data is incomplete; original files are unchanged.')
    save_sweep_results(records, OUT, stem='stage8_sweep_rho_si_minus50_only')
    new = [{
        'parameter_name': 'rho_si',
        'parameter_value': repr(float(record.parameter_value)),
        'scene_seed': str(record.scene.scene_seed),
        **{f'{scheme}_wsr': repr(float(record.scene.schemes[scheme].wsr))
           for scheme in SCHEME_ORDER},
    } for record in records]
    merged = sorted(new + kept, key=lambda r: (_key(float(r['parameter_value'])), int(r['scene_seed'])))
    if len(merged) != 50:
        raise ValueError('Merged sweep must have exactly 50 rows.')
    merged_csv = OUT / 'stage8_sweep_rho_si_merged_minus50_minus10.csv'
    merged_npz = OUT / 'stage8_sweep_rho_si_merged_minus50_minus10.npz'
    with merged_csv.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(merged)
    np.savez_compressed(
        merged_npz,
        parameter_name=np.asarray('rho_si'),
        parameter_value=np.asarray([float(r['parameter_value']) for r in merged], dtype=float),
        scene_seed=np.asarray([int(r['scene_seed']) for r in merged], dtype=int),
        scheme_order=np.asarray(SCHEME_ORDER),
        wsr=np.asarray([[float(r[f'{s}_wsr']) for s in SCHEME_ORDER] for r in merged], dtype=float),
    )
    print(f'[PASS] Merged CSV: {merged_csv}')
    print(f'[PASS] Merged NPZ: {merged_npz}')
    if args.activate:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        for original, merged_path in ((OLD_CSV, merged_csv), (OLD_NPZ, merged_npz)):
            backup = original.with_name(f'{original.stem}_backup_{stamp}{original.suffix}')
            shutil.copy2(original, backup)
            tmp = original.with_name(original.name + '.new.tmp')
            shutil.copy2(merged_path, tmp)
            os.replace(tmp, original)
            print(f'[BACKUP] {backup}')
        print('[DONE] Canonical stage8_sweep_rho_si.csv and .npz now use -50..-10 dB.')
    else:
        print('[INFO] Originals unchanged. Rerun once with --activate to copy merged files;')
        print('       NOTE: that rerun would repeat -50 dB calculation. Or manually copy both merged files.')

if __name__ == '__main__':
    main()
