"""Add seven weights to the saved scene8001/seed2042 pilot; no MC averaging.

Run from the project root. --check-only performs validation without optimization.
Repeated commands resume compatible saved states at the SAME weight only.
"""
from __future__ import annotations

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch
import numpy as np
import yaml

from stage10_seed2042_support import FIELDS, LABELS, csv_read, csv_write, json_write, plot, sha, values

NEW_WEIGHTS = [0.02, 0.04, 0.075, 0.125, 0.15, 0.175, 0.25]
PLOT_WEIGHTS = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10,
                0.125, 0.15, 0.175, 0.20, 0.25, 0.30, 0.90]
SCHEMES = ['b2', 'b3', 'b6', 'proposed']
ASSETS = Path(__file__).resolve().parent / 'stage10_seed2042_assets'
TOL = 1e-4


def abs_change(a, b):
    return abs(float(a) - float(b))


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_files(root):
    manifest = read_json(ASSETS / 'manifest.json')
    for name, digest in manifest['assets_sha256'].items():
        if sha(ASSETS / name) != digest:
            raise RuntimeError('Packaged data checksum mismatch: ' + name)
    for name, digest in manifest['solver_lf_sha256'].items():
        path = root / name
        if not path.is_file():
            raise RuntimeError('Project root must contain original solver: ' + str(path))
        # Accept Windows CRLF line endings; require the same actual source.
        actual = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        if actual != digest:
            raise RuntimeError('Solver differs from the saved-result version: ' + name
                               + '. Use the original version; do not mix incompatible results.')
    return manifest


def load_scene(root):
    sys.path.insert(0, str(root))
    from config import load_config
    from stage8 import make_scene_config
    raw = read_json(ASSETS / 'config.json')
    with TemporaryDirectory() as temp:
        p = Path(temp) / 'config.yaml'
        p.write_text(yaml.safe_dump(raw), encoding='utf-8')
        base = load_config(p)
    scene = make_scene_config(base, 8001, np.asarray(raw['experiment']['position_jitter']),
                              randomize_fading=False)
    assert scene.tol_inner == TOL and scene.tol_outer == TOL
    assert scene.max_inner_iter == 1200 and scene.max_outer_iter == 1500
    assert scene.plateau_window == 0
    return scene, raw['experiment']['ma']


def cfg_for(scene, w):
    # Exactly the original objective: w*(R_DL+R_UL)/10 + (1-w)*R_S/5.
    return replace(scene, weight_dl=np.full(5, w / 10),
                   weight_ul=np.full(5, w / 10), weight_s=np.full(5, (1-w) / 5))


def load_state(path):
    from state import GeometryState, ResourceState
    with np.load(path, allow_pickle=False) as s:
        g = GeometryState(**{n: s[n].copy() for n in ('endpoints', 'tx_positions', 'rx_positions')})
        r = ResourceState(**{n: s[n].copy() for n in ('q_matrix', 'q_ul', 'b_ul', 'u_s')})
        h = s['wsr_history'].astype(float).tolist() if 'wsr_history' in s else []
    return g, r, h


def initial_file(sc):
    return ASSETS / ('initial_toma.npz' if sc in ('proposed', 'b2') else
                     'initial_ma.npz' if sc == 'b6' else 'initial_upa.npz')


def preflight(root):
    verify_files(root)
    scene, opts = load_scene(root)
    from channels import build_channels
    from geometry import build_geometry
    from metrics import compute_performance
    from utils import check_resource_constraints
    import baselines
    import ma_baseline as ma
    base = baselines.build_fpa_upa_geometry(scene)
    for sc in SCHEMES:
        g, r, _ = load_state(initial_file(sc))
        check_resource_constraints(r, scene)
        if sc in ('proposed', 'b2'):
            rebuilt = build_geometry(g.endpoints, scene)
            assert np.allclose(rebuilt.tx_positions, g.tx_positions, rtol=0, atol=1e-12)
            assert np.allclose(rebuilt.rx_positions, g.rx_positions, rtol=0, atol=1e-12)
        elif sc == 'b3':
            assert np.array_equal(base.tx_positions, g.tx_positions)
            assert np.array_equal(base.rx_positions, g.rx_positions)
        else:
            assert ma._ma_geometry_is_feasible(g.tx_positions, g.rx_positions,
                np.mean(base.tx_positions, axis=0), np.mean(base.rx_positions, axis=0),
                opts['region_size'], opts['min_spacing'], scene)
        for w in NEW_WEIGHTS:
            v = values(compute_performance(build_channels(g, cfg_for(scene, w)), r, cfg_for(scene, w)))
            assert all(np.isfinite(x) for x in v.values())
    old = csv_read((ASSETS / 'existing_80_points.csv').read_bytes())
    assert len(old) == 80
    assert len({(r['scheme'], float(r['omega_comm'])) for r in old}) == 80
    assert all(int(r['best_seed']) == 2042 and str(r['converged']).lower() == 'true' for r in old)
    print('Preflight OK: original solvers, frozen seed2042 initial states, 80 existing points.', flush=True)


def run_job(root_string, output_string, sc, w, max_calls):
    root, out = Path(root_string), Path(output_string)
    scene, opts = load_scene(root)
    cfg = cfg_for(scene, w)
    import algorithm
    import baselines
    import inner_solver
    import outer_solver
    import ma_baseline as ma
    from channels import build_channels
    from metrics import compute_performance
    from utils import check_resource_constraints
    record = out / 'records' / f'{sc}_omega_{w:g}.json'
    g, resources, _ = load_state(initial_file(sc))
    initial_g = deepcopy(g)
    history, iterations, seconds, attempt = [], 0, 0., 0
    if record.exists():
        d = read_json(record)
        if sha(out / d['state_file']) != d['state_sha256']:
            raise RuntimeError('Checkpoint checksum mismatch: ' + str(record))
        if d['row']['converged']:
            print(f'SKIP {sc} w={w:g}: already converged', flush=True)
            return d['row']
        g, resources, history = load_state(out / d['state_file'])
        iterations = int(d['row']['iterations'])
        seconds = float(d['row']['runtime_s'])
        attempt = int(d['attempt'])
    for _ in range(max_calls):
        attempt += 1
        tick = perf_counter()
        expected = float(compute_performance(build_channels(g, cfg), resources, cfg).weighted_sum_rate)
        print(f'START {sc} w={w:g}, call={attempt}, previous_iterations={iterations}, WSR={expected:.9f}', flush=True)
        # Patching is process-local and temporary; no project source is modified.
        with ExitStack() as stack:
            for mod in (algorithm, baselines, inner_solver, outer_solver, ma):
                stack.enter_context(patch.object(mod, 'relative_change', abs_change))
            target = algorithm if sc == 'proposed' else ma if sc == 'b6' else baselines
            native_inner = target.solve_inner_problem
            calls = 0
            def monitored(*a, **kw):
                nonlocal calls
                calls += 1
                if calls % 50 == 0:
                    print(f'PROGRESS {sc} w={w:g}: FP calls={calls}, current_call_seconds={perf_counter()-tick:.1f}', flush=True)
                return native_inner(*a, **kw)
            stack.enter_context(patch.object(target, 'solve_inner_problem', monitored))
            rng = np.random.default_rng(2042)
            if sc == 'proposed':
                result = algorithm.run_joint_algorithm(cfg, rng=rng, verbose=False,
                    position_update_mode='fp_aware', fp_aware_candidate_iters=2,
                    initial_endpoints=g.endpoints.copy(), initial_resources=deepcopy(resources))
                new = list(map(float, result.joint_wsr_history))
                count = result.outer_iterations
            elif sc == 'b6':
                result = ma.run_ma_fp(cfg, rng=rng, verbose=False,
                    region_size=opts['region_size'], min_spacing=opts['min_spacing'],
                    initial_step=opts['initial_step'], initial_tx_positions=g.tx_positions.copy(),
                    initial_rx_positions=g.rx_positions.copy(), initial_resources=deepcopy(resources))
                new = list(map(float, result.wsr_history))
                count = result.iterations
            elif sc == 'b2':
                result = baselines.run_fixed_toma_fp(cfg, rng=rng, verbose=False,
                    initial_endpoints=g.endpoints.copy(), initial_resources=deepcopy(resources))
                new = list(map(float, result.wsr_history))
                count = result.iterations
            else:
                result = baselines.run_fpa_upa_fp(cfg, rng=rng, verbose=False,
                    initial_resources=deepcopy(resources))
                new = list(map(float, result.wsr_history))
                count = result.iterations
        seconds += perf_counter() - tick
        if len(new) < 2 or not np.isclose(new[0], expected, rtol=1e-9, atol=1e-9):
            raise RuntimeError('Initial/continuation WSR mismatch.')
        history = history + new[1:] if history else new
        iterations += int(count)
        if any(b < a - 1e-8*max(1, abs(a)) for a, b in zip(history, history[1:])):
            raise RuntimeError('True WSR decreased.')
        check_resource_constraints(result.resources, cfg)
        perf = compute_performance(build_channels(result.geometry, cfg), result.resources, cfg)
        v = values(perf)
        if not all(np.isfinite(x) for x in v.values()):
            raise RuntimeError('Nonfinite final performance.')
        if not np.isclose(history[-1], v['weighted_sum_rate'], rtol=1e-9, atol=1e-9):
            raise RuntimeError('Final WSR does not match saved history.')
        converged = bool(result.converged)
        required = 2 if sc in ('proposed', 'b6') else 1
        if converged and (len(new) < required+1 or
                not np.all(np.abs(np.diff(new[-required-1:])) <= TOL)):
            raise RuntimeError('Absolute stopping criterion mismatch.')
        if sc in ('b2', 'b3'):
            assert np.array_equal(initial_g.tx_positions, result.geometry.tx_positions)
            assert np.array_equal(initial_g.rx_positions, result.geometry.rx_positions)
        if sc == 'b6':
            base = baselines.build_fpa_upa_geometry(cfg)
            assert ma._ma_geometry_is_feasible(result.geometry.tx_positions, result.geometry.rx_positions,
                np.mean(base.tx_positions, axis=0), np.mean(base.rx_positions, axis=0),
                opts['region_size'], opts['min_spacing'], cfg)
        state_name = f'states/{sc}_omega_{w:g}_call_{attempt:03d}.npz'
        temp = (out / state_name).with_suffix('.tmp')
        with temp.open('wb') as f:
            np.savez_compressed(f, endpoints=result.geometry.endpoints,
                tx_positions=result.geometry.tx_positions, rx_positions=result.geometry.rx_positions,
                q_matrix=result.resources.q_matrix, q_ul=result.resources.q_ul,
                b_ul=result.resources.b_ul, u_s=result.resources.u_s,
                wsr_history=np.asarray(history), omega_comm=w, init_seed=2042, scene_seed=8001)
        os.replace(temp, out / state_name)
        row = dict(omega_comm=w, omega_sensing=1-w, scheme=sc, scheme_label=LABELS[sc], **v,
            best_seed=2042, iterations=iterations, runtime_s=seconds, n_starts=1,
            converged=converged, source_seed=2042, provenance='new_weight_same_seed_cold_start')
        json_write(record, dict(row=row, state_file=state_name, state_sha256=sha(out/state_name), attempt=attempt,
            initialization_file_sha256=sha(initial_file(sc)), continuation_same_weight_only=True))
        print(f'DONE {sc} w={w:g}: WSR={v["weighted_sum_rate"]:.9f}, iterations={iterations}, '
              f'converged={converged}, total_seconds={seconds:.1f}', flush=True)
        if converged:
            return row
        g, resources = result.geometry, result.resources
    print(f'PAUSED {sc} w={w:g}: saved; repeat the command for same-weight continuation.', flush=True)
    return row


def aggregate(out):
    old = csv_read((ASSETS / 'existing_80_points.csv').read_bytes())
    records = [read_json(p)['row'] for p in sorted((out/'records').glob('*.json'))]
    done = [r for r in records if r['converged']]
    rows = sorted(old + done, key=lambda r: (float(r['omega_comm']), SCHEMES.index(r['scheme'])))
    assert len({(r['scheme'], float(r['omega_comm'])) for r in rows}) == len(rows)
    csv_write(out/'new_points.csv', records, FIELDS)
    csv_write(out/'all_27_weights.csv', rows, FIELDS)
    selected = [r for r in rows if float(r['omega_comm']) in PLOT_WEIGHTS]
    csv_write(out/'plot_15_weights.csv', selected, FIELDS)
    complete = len(done) == 28
    json_write(out/'completion.json', dict(complete=complete, existing_points=80,
        completed_new_points=len(done), expected_new_points=28, total_available_points=len(rows),
        new_weights=NEW_WEIGHTS, plot_weights=PLOT_WEIGHTS, init_seed=2042, scene_seed=8001,
        single_scene_preview=True, n_starts_per_scheme=1, cross_weight_warm_start=False))
    if complete:
        plot(selected, out/'figures_15_weights')
        plot(rows, out/'figures_all_27_weights')
        print('COMPLETE: 28 new + 80 existing = 108 points; 15-weight and full 27-weight figures saved.', flush=True)
    else:
        print(f'Saved progress: {len(done)}/28 new points converged. No incomplete final figure generated.', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--output', type=Path, default=Path('results/stage10/seed2042_refined_weights'))
    ap.add_argument('--workers', type=int, default=1, help='Independent processes; 1 or 2 recommended.')
    ap.add_argument('--weights', type=float, nargs='+', default=NEW_WEIGHTS, choices=NEW_WEIGHTS)
    ap.add_argument('--schemes', nargs='+', default=SCHEMES, choices=SCHEMES)
    ap.add_argument('--max-calls', type=int, default=20, help='Per job per invocation; repeat to continue if needed.')
    ap.add_argument('--check-only', action='store_true', help='Validate inputs only; NEVER run optimization.')
    ap.add_argument('--plot-only', action='store_true', help='Merge saved results and plot if all 28 points are ready.')
    args = ap.parse_args()
    if args.workers < 1 or args.max_calls < 1:
        ap.error('--workers and --max-calls must be positive')
    root = args.project_root.resolve()
    out = (root/args.output).resolve()
    preflight(root)
    if args.check_only:
        print('CHECK ONLY finished. No optimizer called; no simulation result produced.', flush=True)
        return
    manifest = read_json(ASSETS/'manifest.json')
    identity = dict(script_sha256=sha(Path(__file__)), support_sha256=sha(Path(__file__).with_name('stage10_seed2042_support.py')),
                    manifest_sha256=sha(ASSETS/'manifest.json'), project_root=str(root))
    if out.exists():
        if not (out/'run_metadata.json').exists() or read_json(out/'run_metadata.json')['identity'] != identity:
            raise RuntimeError('Output identity differs. Choose a new --output directory.')
    else:
        if args.plot_only:
            raise RuntimeError('No saved output exists for --plot-only.')
        out.mkdir(parents=True)
        (out/'records').mkdir()
        (out/'states').mkdir()
        json_write(out/'run_metadata.json', dict(identity=identity, protocol=manifest['protocol']))
    if args.plot_only:
        aggregate(out)
        return
    jobs = [(str(root), str(out), sc, w, args.max_calls)
            for w in dict.fromkeys(args.weights) for sc in dict.fromkeys(args.schemes)]
    print(f'Jobs requested: {len(jobs)}; workers={args.workers}; saved converged jobs will be skipped.', flush=True)
    try:
        if args.workers == 1:
            for job in jobs:
                run_job(*job)
                aggregate(out)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(run_job, *job) for job in jobs]
                for f in as_completed(futures):
                    f.result()
                    aggregate(out)
    finally:
        aggregate(out)
    print('Output:', out, flush=True)


if __name__ == '__main__':
    main()
