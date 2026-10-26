"""Isolated controlled experiments; never writes production modules/configurations.

Every run starts independently, with empty RCG memory. Saved final states are
local restart tests, not continuations of the original RCG trajectory.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import sys, json, time, zipfile, hashlib, argparse
import io
from pathlib import Path
from copy import deepcopy
from dataclasses import replace, asdict
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / '_archive_20261023/convergence_33774_audit_20261024__收敛长尾诊断'
sys.path.insert(0, str(OLD))
import check_larger_step as source
import inner_solver as inner, outer_solver as outer, algorithms as algo
from channel_model import build_channels, compute_performance
from system_model import build_geometry

def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def production_hashes():
    names = ['outer_solver.py', 'inner_solver.py', 'algorithms.py', 'system_model.py',
             'channel_model.py', 'final_support_i1200_o1500_s8005/final_config.json',
             'tradeoff_support_i1200_o1500_s8005/tradeoff_final_config.json']
    return {n: hashlib.sha256((source.TOMA/n).read_bytes()).hexdigest() for n in names}

class Counter:
    def __init__(self):
        self.calls = dict(main=0, candidate=0, probe=0)
        self.steps = dict(main=0, candidate=0, probe=0)
        self.fixed_evaluations = 0
    def solve(self, channels, resources, cfg, kind, exact=False):
        if exact:
            with patch.object(inner, 'relative_change', lambda a,b: float('inf')):
                rr, hh = inner.solve_inner_problem(channels, deepcopy(resources), cfg, verbose=False)
        else:
            rr, hh = inner.solve_inner_problem(channels, deepcopy(resources), cfg, verbose=False)
        self.calls[kind] += 1
        self.steps[kind] += len(hh)-1
        return rr, hh

def basis_at(ep):
    bases = []
    for point in ep:
        normal = point / np.linalg.norm(point)
        axis = np.eye(3)[np.argmin(abs(normal))]
        u = np.cross(normal, axis); u /= np.linalg.norm(u)
        bases.append([u, np.cross(normal, u)])
    return np.asarray(bases)

def retract(ep, bases, x, cfg):
    return np.asarray([outer.sphere_retraction(p, b.T @ xx, 1., cfg)
                       for p,b,xx in zip(ep, bases, x.reshape(-1,2))])

def curvature_step(g, r, cfg, counter, mode, cached):
    ep = g.endpoints; bases = basis_at(ep); n = 2*len(ep)
    f = outer.evaluate_fixed_resource_wsr(ep, r, cfg)
    probe_cfg = replace(cfg, max_inner_iter=2)
    def value(x):
        pp = retract(ep, bases, x, cfg)
        ch = build_channels(build_geometry(pp, cfg, check_constraints=False), cfg)
        return counter.solve(ch, r, probe_cfg, 'probe', exact=True)[1][-1]
    # Gradient uses the same exactly-two-FP map as the curvature probes.
    h = 1e-6; zero = np.zeros(n); grad = np.zeros(n)
    refresh = cached is None
    f0 = value(zero) if refresh else None
    H = np.zeros((n,n)) if refresh else None
    for i in range(n):
        x = zero.copy(); x[i] = h
        fp, fm = value(x), value(-x)
        grad[i] = (fp-fm)/(2*h)
        if refresh:
            H[i,i] = (fp-2*f0+fm)/h**2
            for j in range(i):
                if mode == 'block' and i//2 != j//2:
                    continue
                y = zero.copy(); y[j] = h
                H[i,j] = H[j,i] = (value(x+y)-value(x-y)-value(-x+y)+value(-x-y))/(4*h*h)
    if refresh:
        # Store the positive curvature operator in ambient coordinates so that
        # subsequent tangent bases can change without reusing wrong coordinates.
        ev, U = np.linalg.eigh(-H)
        floor = max(1., np.max(abs(ev))*1e-3)
        A = (U * np.maximum(ev, floor)) @ U.T
        B = np.zeros((3*len(ep),n))
        for i in range(len(ep)): B[3*i:3*i+3,2*i:2*i+2] = bases[i].T
        cached = B @ A @ B.T
    B = np.zeros((3*len(ep),n))
    for i in range(len(ep)): B[3*i:3*i+3,2*i:2*i+2] = bases[i].T
    A = B.T @ cached @ B
    direction = np.linalg.solve(A + np.eye(n)*1e-8, grad)
    # A common scale preserves joint direction and bounds endpoint trial length.
    largest = np.max(np.linalg.norm(direction.reshape(-1,2), axis=1))
    if largest > .002: direction *= .002/largest
    deriv = float(grad @ direction)
    assert deriv >= 0
    alpha = 1.; backtracks = 0; accepted = False
    for backtracks in range(cfg.max_backtracking_iter):
        pp = retract(ep, bases, alpha*direction, cfg)
        dist = np.linalg.norm(pp[:,None]-pp[None,:], axis=2)
        np.fill_diagonal(dist, np.inf)
        if np.min(dist) >= cfg.min_uav_distance:
            gg = build_geometry(pp, cfg)
            ch = build_channels(gg, cfg)
            rr, hh = counter.solve(ch, r, probe_cfg, 'candidate')
            if hh[-1] >= f + cfg.armijo_mu*alpha*deriv:
                accepted = True
                break
        alpha *= cfg.backtracking_factor
    if not accepted:
        gg, rr = g, r
        alpha = 0.
    displacement = np.linalg.norm(gg.endpoints-ep, axis=1)
    ff = source.e.metrics_values(gg, rr, cfg)['weighted_sum_rate']
    return gg, rr, cached, dict(position_gain=ff-f, alpha=alpha,
        backtracks=backtracks, accepted=accepted, gradient_norm=float(np.linalg.norm(grad)),
        mean_displacement_mm=float(displacement.mean()*1000),
        max_displacement_mm=float(displacement.max()*1000), curvature_refreshed=refresh)

def run(cfg, g0, r0, method, iterations, tag, refresh_period=10, fixed_budget=False):
    cfg = replace(cfg, plateau_window=0, tol_window=0.)
    if method.endswith('_tight'):
        cfg = replace(cfg, tol_inner=1e-6)
    g, r = deepcopy(g0), deepcopy(r0)
    counter = Counter(); history = [source.e.metrics_values(g,r,cfg)['weighted_sum_rate']]
    rows=[]; cached=None; rcg=None; small=0; first_stop=None; tick=time.perf_counter()
    for k in range(iterations):
        r, hh = counter.solve(build_channels(g,cfg),r,cfg,'main')
        after_inner = hh[-1]
        if method.startswith('baseline'):
            diagnostics=[]; before=g.endpoints.copy()
            original = outer.solve_inner_problem
            def candidate(ch, rr, cc, verbose=False):
                # Keep the production candidate FP behavior exactly intact.
                counter.calls['candidate'] += 1
                result = original(ch,rr,cc,verbose=verbose)
                counter.steps['candidate'] += len(result[1])-1
                return result
            with patch.object(outer,'solve_inner_problem',candidate):
                g,ch,r,ff,rcg,accepted = outer.update_positions_once_fp_aware(
                    g,r,cfg,rcg_state=rcg,candidate_fp_iterations=2,diagnostic_records=diagnostics)
            d=np.linalg.norm(g.endpoints-before,axis=1)*1000
            previous=after_inner; geometry_gain=0.; adaptation_gain=0.; accepted_fp=0
            for dd in diagnostics:
                if dd.accepted:
                    geometry_gain += dd.fixed_wsr_at_candidate-previous
                    adaptation_gain += dd.aware_wsr_at_candidate-dd.fixed_wsr_at_candidate
                    previous=dd.aware_wsr_at_candidate
                    accepted_fp += dd.candidate_fp_iterations
            detail=dict(position_gain=ff-after_inner, accepted=accepted,
                fixed_resource_geometry_gain=geometry_gain,candidate_adaptation_gain=adaptation_gain,
                accepted_candidate_fp_iterations=accepted_fp,
                backtracks=sum(x.backtracking_steps for x in diagnostics),
                mean_displacement_mm=float(d.mean()),max_displacement_mm=float(d.max()))
        elif method == 'resource_only':
            # Baseline pilot accepts eight endpoints with two FP updates each.
            # No trial geometry or cross-parameter initialization is used here.
            r, extra = counter.solve(build_channels(g,cfg),r,replace(cfg,max_inner_iter=16),
                'candidate',exact=True)
            detail=dict(position_gain=extra[-1]-after_inner,accepted=0,backtracks=0,
                mean_displacement_mm=0.,max_displacement_mm=0.)
        else:
            if k % refresh_period == 0: cached=None
            g,r,cached,detail=curvature_step(g,r,cfg,counter,method.split('_')[0],cached)
        source.e.check_state(g,r,cfg,{},'proposed')
        ff=source.e.metrics_values(g,r,cfg)['weighted_sum_rate']
        assert np.isfinite(ff) and ff >= history[-1]-1e-8
        delta=abs(ff-history[-1]); small=small+1 if delta<=cfg.tol_outer else 0
        if small>=2 and first_stop is None: first_stop=k+1
        rows.append(dict(iteration=k+1,wsr=ff,seconds=time.perf_counter()-tick,
            resource_gain=after_inner-history[-1],inner_iterations=len(hh)-1,
            calls=counter.calls.copy(),fp_iterations=counter.steps.copy(),**detail))
        history.append(ff)
        if (k+1)%10 == 0 or small>=2:
            print(json.dumps(dict(tag=tag,method=method,iteration=k+1,gain=ff-history[0],
                seconds=time.perf_counter()-tick,calls=counter.calls)),flush=True)
        if small>=2 and not fixed_budget: break
    result=dict(tag=tag,method=method,iterations=len(rows),converged=small>=2,
        initial_score=history[0],final_score=history[-1],gain=history[-1]-history[0],
        seconds=time.perf_counter()-tick,history=history,rows=rows,
        first_tolerance_iteration=first_stop,fixed_budget=fixed_budget,inner_tolerance=cfg.tol_inner,
        calls=counter.calls,fp_iterations=counter.steps,refresh_period=refresh_period,
        tolerance=cfg.tol_outer,minimum_increment=float(np.min(np.diff(history))))
    save(ROOT/f'{tag}_{method}.json',result)
    np.savez(ROOT/f'{tag}_{method}_final.npz',endpoints=g.endpoints,
        **{n:getattr(r,n) for n in ('q_matrix','q_ul','b_ul','u_s')})
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--powers',nargs='+',default=['1'])
    p.add_argument('--seeds',nargs='+',type=int,default=[8003])
    p.add_argument('--methods',nargs='+',default=['baseline','block','joint'])
    p.add_argument('--iterations',type=int,default=30)
    p.add_argument('--refresh',type=int,default=10)
    p.add_argument('--fixed-budget',action='store_true')
    p.add_argument('--initial',action='store_true')
    args=p.parse_args(); hashes=production_hashes()
    save(ROOT/'production_hashes.json',hashes)
    with zipfile.ZipFile(source.ZIP) as z, ExitStack() as stack:
        for mod in (inner,outer,algo):
            stack.enter_context(patch.object(mod,'relative_change',source.e.abs_change))
        for power in args.powers:
            for seed in args.seeds:
                cfg,g,r=source.load(z,power,seed)
                if args.initial:
                    from system_model import GeometryState, ResourceState
                    folder=f'final_33774/candidate_0/tasks/convergence/{power}/scene_{seed}/proposed/'
                    record=json.loads(z.read(folder+'record.json'));data=z.read(folder+'initial.npz')
                    assert hashlib.sha256(data).hexdigest()==record['initial_sha256']
                    with np.load(io.BytesIO(data)) as a:
                        g=GeometryState(**{n:a[n].copy() for n in ('endpoints','tx_positions','rx_positions')})
                        r=ResourceState(**{n:a[n].copy() for n in ('q_matrix','q_ul','b_ul','u_s')})
                for method in args.methods:
                    tag=f's{seed}_p{power}_n{args.iterations}'+('_budget' if args.fixed_budget else '')
                    if args.initial: tag+='_initial'
                    result=run(cfg,g,r,method,args.iterations,tag,args.refresh,args.fixed_budget)
                    print(json.dumps({k:v for k,v in result.items() if k not in ('history','rows')}),flush=True)
    assert hashes==production_hashes(), 'Production files changed during experiment'

if __name__=='__main__': main()
