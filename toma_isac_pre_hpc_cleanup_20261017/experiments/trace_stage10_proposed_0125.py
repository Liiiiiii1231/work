"""Trace ONE Proposed cold start at w=0.125 without modifying its algorithm.

Default: one complete run, same original settings as the uploaded result.
--check-only validates inputs without invoking any optimizer.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import argparse
from contextlib import ExitStack
from copy import deepcopy
import csv
from dataclasses import asdict
import inspect
import json
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch
import numpy as np

import run_stage10_refine_seed2042 as common
from stage10_seed2042_support import csv_read, json_write, sha, values

REFS = Path(__file__).resolve().parent / 'stage10_seed2042_diagnostic_refs'


class Table:
    def __init__(self, path):
        self.path = path
        self.file = path.open('w', encoding='utf-8-sig', newline='')
        self.writer = None
        self.rows = []

    def append(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.file.flush()
        self.rows.append(row)

    def close(self):
        self.file.close()


def stats(perf, resources, cfg):
    row = values(perf)
    row['comm_objective_term'] = 0.125 * row['sum_rate_comm'] / 10
    row['sensing_objective_term'] = 0.875 * row['sum_rate_sensing'] / 5
    powers = np.abs(resources.q_ul)**2
    for j, power in enumerate(powers, 1):
        row[f'ul_power_{j}_W'] = float(power)
    row['zero_ul_users'] = int(np.count_nonzero(powers <= 1e-14))
    stream_power = np.sum(np.abs(resources.q_matrix)**2, axis=0)
    row['dl_tx_power_W'] = float(np.sum(stream_power[:cfg.k_dl]))
    row['sensing_tx_power_W'] = float(np.sum(stream_power[cfg.k_dl:]))
    for tag, rates in [('dl', perf.rate_dl), ('ul', perf.rate_ul), ('sensing', perf.rate_s)]:
        for j, rate in enumerate(rates, 1):
            row[f'{tag}_rate_{j}'] = float(rate)
    return row


def save_state(path, geometry, resources):
    with path.open('wb') as f:
        np.savez_compressed(f, endpoints=geometry.endpoints,
            tx_positions=geometry.tx_positions, rx_positions=geometry.rx_positions,
            q_matrix=resources.q_matrix, q_ul=resources.q_ul,
            b_ul=resources.b_ul, u_s=resources.u_s)


def check_references(cfg):
    from channels import build_channels
    from metrics import compute_performance
    from utils import check_resource_constraints
    manifest = json.loads((REFS/'manifest.json').read_text(encoding='utf-8'))
    for name, digest in manifest['sha256'].items():
        if sha(REFS/name) != digest:
            raise RuntimeError('Diagnostic reference checksum mismatch: ' + name)
    result = []
    for w in [0.075, 0.125]:
        g, r, _ = common.load_state(REFS/f'proposed_omega_{w:g}.npz')
        check_resource_constraints(r, cfg)
        v = values(compute_performance(build_channels(g, cfg), r, cfg))
        result.append(dict(source_weight=w, **v))
    return result


def make_plot(out, phases, reference):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    last = [r for r in phases if r['phase'] in ('initial', 'after_position_and_candidate_fp')]
    inner = [r for r in phases if r['phase'] == 'after_inner']
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    axs[0].plot([r['outer_iteration'] for r in last], [r['weighted_sum_rate'] for r in last], label='After joint iteration')
    axs[0].scatter([r['outer_iteration'] for r in inner], [r['weighted_sum_rate'] for r in inner], s=8, label='After main inner FP')
    axs[0].axhline(reference[0]['weighted_sum_rate'], color='gray', ls='--', label='Saved w=0.075 state, scored at w=0.125')
    axs[0].set_ylabel('True WSR at w=0.125')
    axs[1].plot([r['outer_iteration'] for r in last], [r['sum_rate_comm'] for r in last], label='Communication sum rate')
    axs[1].plot([r['outer_iteration'] for r in last], [r['sum_rate_sensing'] for r in last], label='Sensing sum rate')
    axs[1].set_ylabel('Rate (bit/s/Hz)')
    for ax in axs:
        ax.set_xlabel('Outer iteration')
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle('Proposed | scene8001 | seed2042 | w=0.125 | Original update rules')
    fig.tight_layout()
    fig.savefig(out/'trace_summary.png', dpi=200)
    plt.close(fig)


def run_trace(root, out, cfg, max_outer):
    import algorithm
    import inner_solver
    import outer_solver
    import baselines
    import ma_baseline
    from channels import build_channels
    from metrics import compute_performance
    from utils import check_resource_constraints
    references = check_references(cfg)
    geometry, resources, _ = common.load_state(common.initial_file('proposed'))
    out.mkdir(parents=True)
    (out/'states').mkdir()
    tables = {name: Table(out/f'{name}.csv') for name in ('phases', 'inner_iterations', 'inner_calls', 'endpoints')}
    start = perf_counter()
    context = {'outer': 0, 'kind': None, 'fp_call': 0, 'main_geometry': geometry,
               'main_iteration': 0, 'last_inner_wsr': None}
    native_inner = inner_solver.solve_inner_problem
    native_inner_metric = inner_solver.compute_performance
    native_position = algorithm.update_positions_once_fp_aware
    original_hashes = {n: sha(root/n) for n in ('algorithm.py', 'inner_solver.py', 'outer_solver.py')}

    def phase(label, g, r, **extras):
        p = compute_performance(build_channels(g, cfg), r, cfg)
        row = dict(outer_iteration=context['outer'], phase=label,
                   elapsed_s=perf_counter()-start, **stats(p, r, cfg))
        previous = tables['phases'].rows[-1] if tables['phases'].rows else None
        row['wsr_gain_from_previous_phase'] = row['weighted_sum_rate'] - previous['weighted_sum_rate'] if previous else 0.
        row['accepted_endpoints'] = extras.get('accepted_endpoints', 0)
        row['endpoint_displacement_norm_m'] = extras.get('displacement', 0.)
        tables['phases'].append(row)
        save_state(out/'states'/f'outer_{context["outer"]:04d}_{label}.npz', g, r)
        return row

    def logged_metric(channels, r, local_cfg):
        p = native_inner_metric(channels, r, local_cfg)
        # The same metric function is also called by LDT updates. Record only
        # the solver's true iteration-0 and end-of-iteration evaluations.
        frame = inspect.currentframe().f_back
        try:
            if context['kind'] == 'main' and frame.f_code.co_name == 'solve_inner_problem':
                i = int(frame.f_locals.get('inner_iter', 0))
                previous = context['last_inner_wsr']
                row = dict(outer_iteration=context['outer'], inner_iteration=i,
                    fp_call=context['fp_call'], elapsed_s=perf_counter()-start,
                    **stats(p, r, cfg))
                row['wsr_gain_from_previous_inner_iteration'] = float(p.weighted_sum_rate)-previous if previous is not None else 0.
                tables['inner_iterations'].append(row)
                context['last_inner_wsr'] = float(p.weighted_sum_rate)
        finally:
            del frame
        return p

    def traced_inner(kind, channels, r, local_cfg, *a, **kw):
        previous_kind = context['kind']
        context['kind'] = kind
        context['fp_call'] += 1
        call_id = context['fp_call']
        if kind == 'main':
            context['outer'] += 1
            context['last_inner_wsr'] = None
        before = compute_performance(channels, r, local_cfg)
        tick = perf_counter()
        try:
            new_r, history = native_inner(channels, r, local_cfg, *a, **kw)
        finally:
            context['kind'] = previous_kind
        after = compute_performance(channels, new_r, local_cfg)
        tables['inner_calls'].append(dict(outer_iteration=context['outer'], kind=kind,
            fp_call=call_id, iterations=len(history)-1, iteration_cap=local_cfg.max_inner_iter,
            stopped_on_tolerance=bool(len(history)>=2 and abs(float(history[-1])-float(history[-2]))<=local_cfg.tol_inner),
            wsr_before=float(before.weighted_sum_rate), wsr_after=float(after.weighted_sum_rate),
            wsr_gain=float(after.weighted_sum_rate-before.weighted_sum_rate), elapsed_s=perf_counter()-tick))
        if kind == 'main':
            phase('after_inner', context['main_geometry'], new_r)
        return new_r, history

    def main_inner(*a, **kw):
        return traced_inner('main', *a, **kw)

    def candidate_inner(*a, **kw):
        return traced_inner('candidate', *a, **kw)

    def position(g, r, local_cfg, *a, **kw):
        records = []
        if 'diagnostic_records' in kw:
            raise RuntimeError('Unexpected preexisting endpoint logger.')
        result = native_position(g, r, local_cfg, *a, diagnostic_records=records, **kw)
        new_g, channels, new_r, wsr, rcg, accepted = result
        for entry in records:
            tables['endpoints'].append(dict(outer_iteration=context['outer'], **asdict(entry)))
        movement = float(np.linalg.norm(new_g.endpoints-g.endpoints))
        final = phase('after_position_and_candidate_fp', new_g, new_r,
                      accepted_endpoints=int(accepted), displacement=movement)
        context['main_geometry'] = new_g
        print(f'outer={context["outer"]:4d} WSR={final["weighted_sum_rate"]:.9f} '
              f'Rc={final["sum_rate_comm"]:.5f} Rs={final["sum_rate_sensing"]:.5f} '
              f'accepted={accepted} zero_UL={final["zero_ul_users"]}', flush=True)
        return result

    json_write(out/'run_metadata.json', dict(scene_seed=8001, initialization_seed=2042,
        omega_comm=0.125, position_update_mode='fp_aware', candidate_fp_iterations=2,
        tol_inner=cfg.tol_inner, tol_outer=cfg.tol_outer, inner_consecutive=1, outer_consecutive=2,
        max_inner_iter=cfg.max_inner_iter, max_outer_iterations=max_outer,
        original_solver_hashes=original_hashes, script_sha256=sha(Path(__file__)),
        original_update_rules_unchanged=True, cold_start=True, cross_weight_warm_start=False))
    try:
        phase('initial', geometry, resources)
        with ExitStack() as stack:
            for mod in (algorithm, inner_solver, outer_solver, baselines, ma_baseline):
                stack.enter_context(patch.object(mod, 'relative_change', common.abs_change))
            stack.enter_context(patch.object(inner_solver, 'compute_performance', logged_metric))
            stack.enter_context(patch.object(algorithm, 'solve_inner_problem', main_inner))
            stack.enter_context(patch.object(outer_solver, 'solve_inner_problem', candidate_inner))
            stack.enter_context(patch.object(algorithm, 'update_positions_once_fp_aware', position))
            result = algorithm.run_joint_algorithm(cfg, rng=np.random.default_rng(2042),
                verbose=False, max_outer_iterations=max_outer, position_update_mode='fp_aware',
                fp_aware_candidate_iters=2, initial_endpoints=geometry.endpoints.copy(),
                initial_resources=deepcopy(resources))
        check_resource_constraints(result.resources, cfg)
        final = values(compute_performance(build_channels(result.geometry, cfg), result.resources, cfg))
        saved_same_weight = references[1]['weighted_sum_rate']
        history = list(map(float, result.joint_wsr_history))
        phase_history = [r['weighted_sum_rate'] for r in tables['phases'].rows
                         if r['phase'] in ('initial', 'after_position_and_candidate_fp')]
        if not np.allclose(history, phase_history, rtol=1e-9, atol=1e-9):
            raise RuntimeError('Trace and original algorithm history disagree.')
        if result.converged and not np.all(np.abs(np.diff(history[-3:]))<=cfg.tol_outer):
            raise RuntimeError('Absolute outer stopping verification failed.')
        # Store the actual algorithm result separately from phase snapshots.
        save_state(out/'final_state.npz', result.geometry, result.resources)
        np.save(out/'joint_wsr_history.npy', np.asarray(history))
        main_calls = [r for r in tables['inner_calls'].rows if r['kind']=='main']
        report = dict(run_finished=True, converged=bool(result.converged), stop_reason=result.stop_reason,
            outer_iterations=int(result.outer_iterations), **final,
            saved_same_weight_WSR=saved_same_weight,
            difference_from_saved_same_weight=final['weighted_sum_rate']-saved_same_weight,
            reproduces_saved_final_WSR=bool(np.isclose(final['weighted_sum_rate'], saved_same_weight, rtol=1e-7, atol=1e-7)),
            reference_states_rescored_at_0125=references,
            first_main_inner=main_calls[0] if main_calls else None,
            main_inner_calls_not_stopped_on_tolerance=sum(not r['stopped_on_tolerance'] for r in main_calls),
            runtime_s=perf_counter()-start,
            position_phase_includes_accepted_candidate_FP=True,
            power_is_abs_q_ul_squared_in_W=True)
        json_write(out/'summary.json', report)
        make_plot(out, tables['phases'].rows, references)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    except BaseException as error:
        json_write(out/'interrupted.json', dict(error_type=type(error).__name__, message=str(error),
            last_outer_iteration=context['outer'], trace_rows_flushed=True))
        raise
    finally:
        for table in tables.values():
            table.close()
        for name, digest in original_hashes.items():
            if sha(root/name) != digest:
                raise RuntimeError('Original source changed during diagnosis: '+name)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--output', type=Path, default=Path('results/stage10/proposed_0125_trace'))
    ap.add_argument('--max-outer', type=int, default=1500,
                    help='1500 reproduces the prior cap; a smaller number produces a partial diagnostic only.')
    ap.add_argument('--check-only', action='store_true')
    args = ap.parse_args()
    if args.max_outer < 1:
        ap.error('--max-outer must be positive')
    root = args.project_root.resolve()
    common.preflight(root)
    scene, _ = common.load_scene(root)
    cfg = common.cfg_for(scene, .125)
    refs = check_references(cfg)
    print('Reference WSR at w=0.125:', refs, flush=True)
    if args.check_only:
        print('CHECK ONLY: no optimizer invoked; no diagnostic simulation run.', flush=True)
        return
    out = (root/args.output).resolve()
    if out.exists():
        raise RuntimeError('Diagnostic output already exists. Choose a new --output directory; previous traces are preserved.')
    print('Running ONE Proposed point: w=0.125, scene8001, seed2042, cold start.', flush=True)
    run_trace(root, out, cfg, args.max_outer)
    print('Diagnostic output:', out, flush=True)


if __name__ == '__main__':
    main()
