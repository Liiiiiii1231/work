"""New regular-planar initialization experiment; no reuse of old random results.

Default: Proposed and MA only, all 27 weights. --pilot uses 0.075, 0.125, 0.175.
Repeated matching commands skip converged jobs and resume only at the same weight.
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
from unittest.mock import patch
import numpy as np
import stage10_planar_pm_common as common
from stage10_planar_pm_support import FIELDS,LABELS,csv_write,json_write,sha,plot,geom_sha,res_sha
from planar_pm_initialization import make_initials,save_initials,plot_initials

PILOT_WEIGHTS=[.075,.125,.175]
FULL_WEIGHTS=[.001,.005,.01,.02,.03,.04,.05,.075,.1,.125,.15,.175,.2,.25,.3,.35,.4,.42,.44,.46,.48,.5,.55,.6,.7,.8,.9]
PM_SCHEMES=['proposed','b6']


def validated_initials(root):
    common.verify_files(root)
    scene,opts=common.load_scene(root)
    from geometry import check_geometry_constraints
    from utils import check_resource_constraints
    from baselines import build_fpa_upa_geometry
    import ma_baseline as ma
    data=make_initials(scene,2042)
    check_geometry_constraints(data['proposed'][0],scene)
    base=build_fpa_upa_geometry(scene)
    assert ma._ma_geometry_is_feasible(data['b6'][0].tx_positions,data['b6'][0].rx_positions,
        np.mean(base.tx_positions,axis=0),np.mean(base.rx_positions,axis=0),
        opts['region_size'],opts['min_spacing'],scene)
    for sc,(g,r) in data.items():
        check_resource_constraints(r,scene)
    for a,b in [('proposed','b2'),('b6','b3')]:
        assert geom_sha(data[a][0])==geom_sha(data[b][0])
        assert res_sha(data[a][1])==res_sha(data[b][1])
    assert np.all(data['proposed'][0].endpoints[:,0]==0)
    # In the current no-fading Rs<L configuration, resources are deterministic
    # given geometry; changing ONLY the objective weight does not change them.
    again=make_initials(common.cfg_for(scene,.125),2042)
    for sc in common.SCHEMES:
        for name in ('q_matrix','q_ul','b_ul','u_s'):
            assert np.array_equal(getattr(data[sc][1],name),getattr(again[sc][1],name))
    return {sc:data[sc] for sc in PM_SCHEMES}


def run_one(root,out,sc,w,max_calls):
    out=Path(out)
    def custom_initial(scheme):return out/'initial_states'/f'{scheme}.npz'
    # Reuse the previously checked continuation driver with this experiment's
    # frozen initial states. No original solver module/file is rewritten.
    with patch.object(common,'initial_file',custom_initial):
        row=common.run_job(root,str(out),sc,w,max_calls)
    record=out/'records'/f'{sc}_omega_{w:g}.json'
    d=common.read_json(record)
    row['provenance']='regular_planar_toma_or_regular_upa_cold_start'
    d['row']=row;d['initialization_protocol']='regular_planar_v1'
    json_write(record,d)
    return row


def aggregate(out,weights,schemes):
    expected={(sc,float(w)) for sc in schemes for w in weights}
    records=[common.read_json(p)['row'] for p in sorted((out/'records').glob('*.json'))]
    actual={(r['scheme'],float(r['omega_comm'])) for r in records}
    assert len(actual)==len(records) and actual<=expected
    done=[r for r in records if r['converged']]
    done.sort(key=lambda r:(float(r['omega_comm']),common.SCHEMES.index(r['scheme'])))
    csv_write(out/'current_points.csv',records,FIELDS)
    csv_write(out/'stage10_tradeoff.csv',done,FIELDS)
    complete=len(done)==len(expected)
    json_write(out/'completion.json',dict(complete=complete,expected_points=len(expected),
        converged_points=len(done),weights=weights,schemes=schemes,scene_seed=8001,
        resource_seed=2042,n_starts_per_scheme=1,geometry_initialization='deterministic_regular_planar',
        cross_weight_warm_start=False,old_random_start_results_reused=False,
        full_27_weight_grid=weights==FULL_WEIGHTS))
    if complete and set(schemes)==set(PM_SCHEMES):
        plot(done,out/'figures')
    print(f'Converged {len(done)}/{len(expected)}; output={out}',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--output',type=Path,default=Path('results/stage10/regular_planar_pm_full27'))
    grid=ap.add_mutually_exclusive_group()
    grid.add_argument('--weights',type=float,nargs='+')
    grid.add_argument('--full-grid',action='store_true',help='All 27 weights (also the default).')
    grid.add_argument('--pilot',action='store_true',help='Only 0.075, 0.125, 0.175.')
    ap.add_argument('--schemes',nargs='+',choices=PM_SCHEMES,default=PM_SCHEMES)
    ap.add_argument('--workers',type=int,default=1)
    ap.add_argument('--max-calls',type=int,default=20)
    ap.add_argument('--check-only',action='store_true')
    ap.add_argument('--plot-only',action='store_true')
    args=ap.parse_args()
    if args.workers<1 or args.max_calls<1:ap.error('Positive --workers and --max-calls required.')
    weights=list(dict.fromkeys(args.weights if args.weights is not None else PILOT_WEIGHTS if args.pilot else FULL_WEIGHTS))
    if not weights or any(not 0<w<1 for w in weights):ap.error('Weights must lie strictly between 0 and 1.')
    weights=sorted(weights);schemes=list(dict.fromkeys(args.schemes))
    root=args.project_root.resolve();out=(root/args.output).resolve()
    data=validated_initials(root)
    print('Regular ToMA endpoints (m):',data['proposed'][0].endpoints.tolist(),flush=True)
    print('Shared original UL initial powers (W):',(np.abs(data['proposed'][1].q_ul)**2).tolist(),flush=True)
    print('Checks passed: deterministic ToMA/MA initial layouts, geometry and resource constraints. Only Proposed and MA are selected.',flush=True)
    identity=dict(entry_sha256=sha(Path(__file__)),initializer_sha256=sha(Path(__file__).with_name('planar_pm_initialization.py')),
        common_sha256=sha(Path(common.__file__)),support_sha256=sha(Path(__file__).with_name('stage10_planar_pm_support.py')),
        assets_manifest_sha256=sha(common.ASSETS/'manifest.json'),weights=weights,schemes=schemes,project_root=str(root))
    if args.check_only:
        print('CHECK ONLY: no optimizer invoked and no new output written.',flush=True)
        return
    if out.exists():
        if not (out/'run_metadata.json').exists() or common.read_json(out/'run_metadata.json')['identity']!=identity:
            raise RuntimeError('Output identity differs. Choose a new --output directory.')
        for sc,(g,r) in data.items():
            sg,sr,_=common.load_state(out/'initial_states'/f'{sc}.npz')
            assert geom_sha(g)==geom_sha(sg) and res_sha(r)==res_sha(sr)
    else:
        if args.plot_only:raise RuntimeError('No output found for --plot-only.')
        out.mkdir(parents=True);(out/'records').mkdir();(out/'states').mkdir()
        save_initials(out/'initial_states',data)
        plot_initials(data,out/'initial_layout.png')
        json_write(out/'run_metadata.json',dict(identity=identity,protocol=dict(
            toma_plane='x=0 (y-z plane through cable anchor)',endpoint_phase_rad=float(np.pi/4),
            endpoint_angles='pi/4 + 2*pi*m/M',toma_optimization_domain='original 3D sphere',
            ma_initialization='regular UPA, no random position jitter',ma_optimization_domain='original 0.3m y-z regions',
            resource_initialization='original MRT/steering/kappa_ul=0.5; 25mW per UL user',
            original_objective_unchanged=True,original_constraints_unchanged=True,
            stopping='absolute WSR change',tol_inner=1e-4,tol_outer=1e-4,
            inner_consecutive=1,outer_consecutive=2,max_inner_iter_per_call=1200,max_outer_iter_per_call=1500,schemes=schemes,weights=weights,
            single_scene_preview=True,n_starts_per_scheme=1,cross_weight_warm_start=False),
            initial_states={sc:dict(geometry_sha256=geom_sha(g),resources_sha256=res_sha(r)) for sc,(g,r) in data.items()}))
    if args.plot_only:aggregate(out,weights,schemes);return
    jobs=[(str(root),str(out),sc,w,args.max_calls) for w in weights for sc in schemes]
    try:
        if args.workers==1:
            for job in jobs:run_one(*job);aggregate(out,weights,schemes)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures=[pool.submit(run_one,*job) for job in jobs]
                for f in as_completed(futures):f.result();aggregate(out,weights,schemes)
    finally:aggregate(out,weights,schemes)


if __name__=='__main__':main()
