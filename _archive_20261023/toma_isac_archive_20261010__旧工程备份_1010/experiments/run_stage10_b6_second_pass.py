"""Stage 10: B6-only second-pass warm starts from selected omega_C=0.5 state.

Run at project root:
    python -u experiments/run_stage10_b6_second_pass.py --dry-run
    python -u experiments/run_stage10_b6_second_pass.py

Reads the *selected* B6 omega_C=0.5 NPZ from the earlier 4-weight pool run;
uses that full geometry/resource state to warm start omega_C=0.005 and 0.3.
Does not alter or overwrite previous results, solver, or tolerances.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from stage8 import load_stage8_experiment_config, make_scene_config
from experiments.run_stage10_tradeoff import load_tradeoff_config
from experiments import run_stage10_pool_refinement as pool

DEFAULT_POOL = (ROOT / 'results' / 'stage10' / 'pool_wide14_tol1e4'
                / 'subset_test_0p005_0p3_0p5_0p6')
DEFAULT_OUT = ROOT / 'results' / 'stage10' / 'b6_second_pass_from_0p5'
SOURCE_WEIGHT = 0.5
TARGET_WEIGHTS = (0.005, 0.3)


def read_csv(path: Path) -> list[dict]:
    with path.open('r', newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def find_row(rows: list[dict], w: float) -> dict:
    matches = [r for r in rows if r.get('scheme') == 'b6'
               and abs(float(r.get('omega_comm', 'nan')) - w) < 1e-10]
    if len(matches) != 1:
        raise RuntimeError(f'Expected exactly 1 B6 row for omega={w:g}; found {len(matches)}.')
    return matches[0]


def state_path(out: Path, row: dict) -> Path:
    raw = row.get('state_file', '')
    if not raw:
        raise RuntimeError('Source row has no saved state_file; full geometry/resources required.')
    rel = Path(raw.replace('\\', '/'))
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError(f'Unsafe state path: {raw!r}')
    p = out / rel
    if not p.is_file():
        raise FileNotFoundError(f'Saved NPZ missing: {p}\n'
                                'Keep cold_states/ and warm_states/ from the previous pool run.')
    return p


def wsr_and_rates(result):
    p = result.performance
    return (float(p.weighted_sum_rate),
            float(np.sum(p.rate_dl) + np.sum(p.rate_ul)),
            float(np.sum(p.rate_s)))


def save_csv(out: Path, rows: list[dict]):
    fields = ['target_omega', 'source_omega', 'source_seed',
              'prior_selected_wsr', 'source_reweighted_wsr',
              'new_wsr', 'new_rate_comm', 'new_rate_sensing',
              'new_converged', 'new_iterations', 'new_stop_reason',
              'new_runtime_s', 'best_selected_wsr', 'best_selected_from',
              'improved_vs_prior', 'state_file']
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / 'comparison.csv.tmp'
    with tmp.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(out / 'comparison.csv')


def main():
    p = argparse.ArgumentParser(description='B6 0.5 -> {0.005, 0.3} targeted warm-start')
    p.add_argument('--pool-dir', type=Path, default=DEFAULT_POOL,
                   help='Folder with stage10_tradeoff.csv and cold_states/warm_states NPZ')
    p.add_argument('--output', type=Path, default=DEFAULT_OUT)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    pool_dir = args.pool_dir.resolve()
    out = args.output.resolve()

    cfg_path = ROOT / 'configs' / 'stage10_tradeoff.yaml'
    base_cfg = load_config(cfg_path)
    exp = load_stage8_experiment_config(cfg_path)
    tr = load_tradeoff_config(cfg_path)
    if not (np.isclose(base_cfg.tol_inner, 1e-4) and np.isclose(base_cfg.tol_outer, 1e-4)):
        raise RuntimeError('Expected tol_inner=tol_outer=1e-4; do not change them for this comparison.')
    scene = make_scene_config(base_cfg, scene_seed=tr.scene_seed,
                              position_jitter=exp.position_jitter,
                              randomize_fading=exp.fading_enabled,
                              fading_model=exp.fading_model)
    csv_file = pool_dir / 'stage10_tradeoff.csv'
    if not csv_file.is_file():
        raise FileNotFoundError(f'Previous pool CSV not found: {csv_file}\n'
                                'Specify --pool-dir if you used a custom --output folder.')
    rows = read_csv(csv_file)
    source = find_row(rows, SOURCE_WEIGHT)
    if source['converged'].strip().lower() != 'true':
        raise RuntimeError('Source omega=0.5 must be converged.')
    source_file = state_path(pool_dir, source)

    # The previous pool uses a manifest with the solver/config source hashes.
    manifest_file = pool_dir / 'manifest.json'
    if not manifest_file.is_file():
        raise FileNotFoundError(f'Missing run manifest: {manifest_file}')
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    hashes = manifest.get('parameters', {}).get('source_hashes', {})
    for item in ('algorithm.py', 'ma_baseline.py', 'inner_solver.py', 'metrics.py',
                 'config.py', 'configs/stage10_tradeoff.yaml'):
        if item in hashes:
            now = hashlib.sha256((ROOT / item).read_bytes()).hexdigest()
            if now != hashes[item]:
                raise RuntimeError(f'{item} changed since the previous pool run. '
                                   'Do not mix solver/config versions; restore the same version or rerun source.')
    source_cfg = pool.target_cfg(scene, SOURCE_WEIGHT)
    source_wsr = pool.verify_state(source_file, source_cfg,
                                   float(source['weighted_sum_rate']))
    geometry, resources, _ = pool.read_state(source_file)
    source_seed = int(source['best_seed'])
    print(f'Source: B6 omega={SOURCE_WEIGHT:g}, seed={source_seed}, '
          f'WSR={source_wsr:.8f}, NPZ={source_file}')
    if source.get('selection_kind') != 'warm_converged':
        print('Note: source selection_kind is', source.get('selection_kind'),
              '(will use the saved selected state nonetheless).')

    prior = {}
    for weight in TARGET_WEIGHTS:
        row = find_row(rows, weight)
        if row['converged'].strip().lower() != 'true':
            raise RuntimeError(f'Prior B6 omega={weight:g} was not converged.')
        prior[weight] = row
        target_cfg = pool.target_cfg(scene, weight)
        value = pool.verify_state(source_file, target_cfg)
        print(f'Target omega={weight:g}: prior selected WSR={float(row["weighted_sum_rate"]):.8f}; '
              f'reweighted source WSR={value:.8f}')
    if args.dry_run:
        print('[OK] All source/target states and configuration checks passed. No optimization performed.')
        return

    out.mkdir(parents=True, exist_ok=True)
    results = []
    for weight in TARGET_WEIGHTS:
        label = str(weight).replace('.', 'p')
        dest = out / f'b6_target_{label}.json'
        target_cfg = pool.target_cfg(scene, weight)
        start_score = pool.verify_state(source_file, target_cfg)
        old = float(prior[weight]['weighted_sum_rate'])
        if dest.is_file():
            record = json.loads(dest.read_text(encoding='utf-8'))
            if record.get('source_hash') != hashlib.sha256(source_file.read_bytes()).hexdigest():
                raise RuntimeError(f'Checkpoint is from a different source state: {dest}')
            if record['new_converged'] and record['state_file']:
                pool.verify_state(out / record['state_file'], target_cfg, record['new_wsr'])
            print(f'[RESUME] target={weight:g} WSR={record["new_wsr"]:.8f}')
        else:
            print(f'[RUN] B6 omega=0.5 -> omega={weight:g}', flush=True)
            begin = perf_counter()
            ans = pool.run_solver('b6', target_cfg, exp, source_seed,
                                  warm=(geometry, resources))
            value, rc, rs = wsr_and_rates(ans)
            conv = bool(ans.converged)
            state_rel = ''
            if conv:
                state_rel = f'b6_target_{label}.npz'
                pool.save_state(out / state_rel, ans)
                pool.verify_state(out / state_rel, target_cfg, value)
            picked = ('warm_converged' if conv and value > old else 'prior_converged')
            record = dict(target_omega=weight, source_omega=SOURCE_WEIGHT,
                          source_seed=source_seed,
                          prior_selected_wsr=old, source_reweighted_wsr=start_score,
                          new_wsr=value, new_rate_comm=rc, new_rate_sensing=rs,
                          new_converged=conv, new_iterations=pool.iterations(ans),
                          new_stop_reason=str(getattr(ans, 'stop_reason', 'not_recorded')),
                          new_runtime_s=round(perf_counter()-begin, 3),
                          best_selected_wsr=value if picked=='warm_converged' else old,
                          best_selected_from=picked,
                          improved_vs_prior=(value-old) if conv else '',
                          state_file=state_rel,
                          source_hash=hashlib.sha256(source_file.read_bytes()).hexdigest())
            pool.json_write(dest, record)
            print(f'  final WSR={value:.8f} converged={conv}, '
                  f'R_C={rc:.5f}, R_S={rs:.5f} selected={picked}')
        results.append(record)
        save_csv(out, results)
    print('Completed. Comparison:', out / 'comparison.csv')


if __name__ == '__main__':
    main()
