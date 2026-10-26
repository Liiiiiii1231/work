"""Compare UL resource starts at ONE weight w=0.18, Proposed and MA equally.

Reuse the two existing 25mW full runs; compute only two schemes x two new powers.
This is a diagnostic comparison, not a replacement tradeoff curve.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import stage10_planar_pm_common as common
from stage10_planar_pm_support import csv_read,csv_write,json_write,sha,FIELDS
from run_stage10_regular_planar_pm import run_one

INPUTS=Path(__file__).resolve().parent/'stage10_ul_compare_inputs'
LEVELS=[('p2p5mW',.0025),('p0p25mW',.00025)]
SCHEMES=['proposed','b6']
WEIGHT=.18


def validate(root):
    common.verify_files(root)
    scene,_=common.load_scene(root)
    from utils import check_resource_constraints
    from geometry import check_geometry_constraints
    from planar_pm_initialization import make_initials
    m=common.read_json(INPUTS/'manifest.json')
    for n,h in m['sha256'].items():
        if sha(INPUTS/n)!=h:raise RuntimeError('Input checksum mismatch: '+n)
    baseline=csv_read((INPUTS/'baseline_018.csv').read_bytes())
    if len(baseline)!=2 or {r['scheme'] for r in baseline}!=set(SCHEMES):
        raise RuntimeError('Expected two full-run reference results.')
    originals={};expected=make_initials(scene,2042)
    for sc in SCHEMES:
        g,r,_=common.load_state(INPUTS/f'{sc}_original_initial.npz')
        if sc=='proposed':check_geometry_constraints(g,scene)
        for n in ('endpoints','tx_positions','rx_positions'):
            if not np.allclose(getattr(g,n),getattr(expected[sc][0],n),rtol=0,atol=1e-12):
                raise RuntimeError('Initial geometry differs from regular-planar experiment.')
        check_resource_constraints(r,scene)
        if not np.allclose(np.abs(r.q_ul)**2,.025,rtol=0,atol=1e-14):
            raise RuntimeError('Expected original 25mW UL initialization.')
        originals[sc]=(g,r)
    for row in baseline:
        if float(row['omega_comm'])!=WEIGHT or str(row['converged']).lower()!='true':
            raise RuntimeError('Reference weight/convergence mismatch.')
        score=WEIGHT*float(row['sum_rate_comm'])/10+(1-WEIGHT)*float(row['sum_rate_sensing'])/5
        if not np.isclose(score,float(row['weighted_sum_rate']),rtol=1e-10,atol=1e-10):
            raise RuntimeError('Reference WSR arithmetic mismatch.')
    print('Inputs verified: regular initial geometry, original resource arrays, two existing 25mW full runs.',flush=True)
    return originals,baseline


def save_initial(path,g,r):
    with path.open('wb') as f:
        np.savez_compressed(f,endpoints=g.endpoints,tx_positions=g.tx_positions,rx_positions=g.rx_positions,
            q_matrix=r.q_matrix,q_ul=r.q_ul,b_ul=r.b_ul,u_s=r.u_s)


def aggregate(out,baseline):
    rows=[]
    for r in baseline:
        rows.append(dict(initial_UL_power_mW=25.,result_kind='existing_full_run',**r))
    done=0
    for label,power in LEVELS:
        for sc in SCHEMES:
            path=out/label/'records'/f'{sc}_omega_0.18.json'
            if path.exists():
                r=common.read_json(path)['row']
                done+=int(bool(r['converged']))
                rows.append(dict(initial_UL_power_mW=power*1000,result_kind='new_full_run',**r))
    rows.sort(key=lambda r:(SCHEMES.index(r['scheme']),-float(r['initial_UL_power_mW'])))
    csv_write(out/'comparison.csv',rows,['initial_UL_power_mW','result_kind']+FIELDS)
    json_write(out/'completion.json',dict(complete=done==4,expected_new_full_runs=4,
        converged_new_full_runs=done,reused_baseline_runs=2,weight=WEIGHT,
        schemes=SCHEMES,initial_UL_powers_mW=[25.,2.5,.25],
        fixed_regular_initial_geometry=True,original_objective_unchanged=True,
        cross_weight_warm_start=False,diagnostic_only=True))
    print(f'Comparison: {done}/4 new full runs converged.',flush=True)
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--output',type=Path,default=Path('results/stage10/regular_ul_init_compare018'))
    ap.add_argument('--workers',type=int,default=1)
    ap.add_argument('--max-calls',type=int,default=20)
    ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args()
    if args.workers<1 or args.max_calls<1:ap.error('Positive workers/max-calls required.')
    root=args.project_root.resolve();out=(root/args.output).resolve()
    originals,baseline=validate(root)
    if args.check_only:
        print('CHECK ONLY: no optimizer invoked.',flush=True);return
    base=Path(__file__).resolve().parent
    identity=dict(entry_sha256=sha(Path(__file__)),inputs_manifest_sha256=sha(INPUTS/'manifest.json'),
        dependencies={n:sha(base/n) for n in ['run_stage10_regular_planar_pm.py','stage10_planar_pm_common.py',
            'stage10_planar_pm_support.py','planar_pm_initialization.py']},
        assets_manifest_sha256=sha(common.ASSETS/'manifest.json'),project_root=str(root))
    if out.exists():
        if not (out/'run_metadata.json').exists() or common.read_json(out/'run_metadata.json')['identity']!=identity:
            raise RuntimeError('Output identity differs; choose a new --output directory.')
    else:
        out.mkdir(parents=True)
        for label,power in LEVELS:
            folder=out/label;folder.mkdir()
            for name in ['initial_states','records','states']:(folder/name).mkdir()
            for sc,(g,r0) in originals.items():
                r=deepcopy(r0);r.q_ul*=np.sqrt(power/.025)
                save_initial(folder/'initial_states'/f'{sc}.npz',g,r)
        json_write(out/'run_metadata.json',dict(identity=identity,protocol=dict(weight=WEIGHT,
            scene_seed=8001,resource_seed=2042,levels_mW=[25.,2.5,.25],
            changed_variable='Only initial UL amplitudes; Q and initial receiver arrays copied unchanged',
            max_allowed_UL_power_W=.1,not_a_power_constraint_change=True,
            max_inner_iter_per_call=1200,max_outer_iter_per_call=1500,
            tol_inner=1e-4,tol_outer=1e-4,stopping='absolute WSR change; inner once, outer twice',
            no_cross_weight_warm_start=True,full_inner_outer_optimization=True,
            baseline_note='Existing full-run CSV references; their new-weight final NPZ states were not uploaded')))
    # Ensure saved cold starts are still the intended one-variable perturbations.
    for label,power in LEVELS:
        for sc,(g0,r0) in originals.items():
            g,r,_=common.load_state(out/label/'initial_states'/f'{sc}.npz')
            for n in ['endpoints','tx_positions','rx_positions']:
                if not np.array_equal(getattr(g,n),getattr(g0,n)):raise RuntimeError('Initial geometry changed.')
            for n in ['q_matrix','b_ul','u_s']:
                if not np.array_equal(getattr(r,n),getattr(r0,n)):raise RuntimeError('Other initial resources changed.')
            if not np.allclose(np.abs(r.q_ul)**2,power,rtol=1e-12,atol=1e-14):raise RuntimeError('UL initial power mismatch.')
    jobs=[(str(root),str(out/label),sc,WEIGHT,args.max_calls) for label,_ in LEVELS for sc in SCHEMES]
    print('Computing ONLY four new complete runs at w=0.18. Existing 25mW runs are reused.',flush=True)
    try:
        if args.workers==1:
            for job in jobs:run_one(*job);aggregate(out,baseline)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures=[pool.submit(run_one,*job) for job in jobs]
                for f in as_completed(futures):f.result();aggregate(out,baseline)
    finally:aggregate(out,baseline)
    print('Output:',out,flush=True)


if __name__=='__main__':main()
