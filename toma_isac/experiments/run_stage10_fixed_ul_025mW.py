"""Fixed 0.25mW UL initialization; 20 weights, two schemes, reuse six runs."""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import argparse,shutil
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import stage10_planar_pm_common as common
from stage10_planar_pm_support import sha,json_write,csv_write,FIELDS,plot
from run_stage10_regular_planar_pm import run_one

WEIGHTS=[.001,.005,.015,.03,.10,.11,.12,.125,.13,.14,.145,.15,.16,.175,.18,.20,.225,.25,.42,.46]
SCHEMES=['proposed','b6']
ASSETS=Path(__file__).resolve().parent/'stage10_fixed_ul_assets'


def validate(root):
    common.verify_files(root)
    scene,opts=common.load_scene(root)
    from planar_pm_initialization import make_initials
    from geometry import check_geometry_constraints
    from utils import check_resource_constraints
    from channels import build_channels
    from metrics import compute_performance
    import baselines,ma_baseline as ma
    manifest=common.read_json(ASSETS/'manifest.json')
    for name,digest in manifest['sha256'].items():
        if sha(ASSETS/name)!=digest:raise RuntimeError('Frozen input checksum mismatch: '+name)
    expected=make_initials(scene,2042)
    for sc in SCHEMES:
        g,r,_=common.load_state(ASSETS/'initial_states'/f'{sc}.npz')
        for n in ['endpoints','tx_positions','rx_positions']:
            if not np.allclose(getattr(g,n),getattr(expected[sc][0],n),rtol=0,atol=1e-12):raise RuntimeError('Initial geometry mismatch.')
        for n in ['q_matrix','b_ul','u_s']:
            if not np.allclose(getattr(r,n),getattr(expected[sc][1],n),rtol=1e-12,atol=1e-12):raise RuntimeError('Other initial resource mismatch.')
        if not np.allclose(abs(r.q_ul)**2,.00025,rtol=0,atol=1e-14):raise RuntimeError('Initial UL power mismatch.')
        check_resource_constraints(r,scene)
    for path in (ASSETS/'records').glob('*.json'):
        d=common.read_json(path);row=d['row'];cfg=common.cfg_for(scene,float(row['omega_comm']))
        if sha(ASSETS/d['state_file'])!=d['state_sha256']:raise RuntimeError('Reused state hash mismatch.')
        if sha(ASSETS/'initial_states'/f"{row['scheme']}.npz")!=d['initialization_file_sha256']:raise RuntimeError('Reused initial state mismatch.')
        g,r,h=common.load_state(ASSETS/d['state_file']);check_resource_constraints(r,cfg)
        if row['scheme']=='proposed':check_geometry_constraints(g,cfg)
        else:
            base=baselines.build_fpa_upa_geometry(cfg)
            if not ma._ma_geometry_is_feasible(g.tx_positions,g.rx_positions,np.mean(base.tx_positions,axis=0),np.mean(base.rx_positions,axis=0),opts['region_size'],opts['min_spacing'],cfg):raise RuntimeError('Reused MA geometry infeasible.')
        score=compute_performance(build_channels(g,cfg),r,cfg).weighted_sum_rate
        if abs(score-float(row['weighted_sum_rate']))>1e-9:raise RuntimeError('Reused WSR mismatch.')
        delta=np.diff(h)
        if not row['converged'] or len(delta)<2 or np.max(abs(delta[-2:]))>1e-4 or np.min(delta)<-1e-8:raise RuntimeError('Reused history/stopping mismatch.')
    print('Verified fixed 0.25mW initial states and six completed full runs.',flush=True)


def aggregate(out,draw=False):
    expected={(sc,w) for sc in SCHEMES for w in WEIGHTS}
    rows=[common.read_json(p)['row'] for p in (out/'records').glob('*.json')]
    keys={(r['scheme'],float(r['omega_comm'])) for r in rows}
    if len(keys)!=len(rows) or not keys<=expected:raise RuntimeError('Unexpected output records.')
    rows.sort(key=lambda r:(float(r['omega_comm']),SCHEMES.index(r['scheme'])))
    done=[r for r in rows if r['converged']]
    csv_write(out/'current_points.csv',rows,FIELDS)
    csv_write(out/'stage10_tradeoff.csv',done,FIELDS)
    complete=len(done)==40
    json_write(out/'completion.json',dict(complete=complete,expected_points=40,converged_points=len(done),
        reused_points=6,expected_new_points=34,weights=WEIGHTS,schemes=SCHEMES,
        initial_UL_power_per_user_mW=.25,n_starts_per_scheme=1,
        cross_weight_warm_start=False,original_objective_unchanged=True))
    if complete and draw:plot(done,out/'figures')
    print(f'Converged {len(done)}/40 (six reused); fixed initial UL power 0.25mW.',flush=True)
    return complete


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--output',type=Path,default=Path('results/stage10/regular_planar_pm_fixed_ul025mW_20'))
    ap.add_argument('--workers',type=int,default=1)
    ap.add_argument('--max-calls',type=int,default=20)
    ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args()
    if args.workers<1 or args.max_calls<1:ap.error('Positive workers/max-calls required.')
    root=args.project_root.resolve();out=(root/args.output).resolve();validate(root)
    if args.check_only:print('CHECK ONLY: no optimizer invoked.');return
    base=Path(__file__).resolve().parent
    identity=dict(entry_sha256=sha(Path(__file__)),assets_sha256=sha(ASSETS/'manifest.json'),
        dependencies={n:sha(base/n) for n in ['run_stage10_regular_planar_pm.py','stage10_planar_pm_common.py','stage10_planar_pm_support.py','planar_pm_initialization.py']},
        solver_assets_sha256=sha(common.ASSETS/'manifest.json'),project_root=str(root))
    if out.exists():
        if not (out/'run_metadata.json').exists() or common.read_json(out/'run_metadata.json')['identity']!=identity:raise RuntimeError('Output identity differs; choose a new --output directory.')
    else:
        out.mkdir(parents=True)
        for name in ['initial_states','records','states']:shutil.copytree(ASSETS/name,out/name)
        json_write(out/'run_metadata.json',dict(identity=identity,protocol=dict(weights=WEIGHTS,
            initial_UL_power_W=.00025,max_UL_power_W=.1,scene_seed=8001,resource_seed=2042,
            geometric_start='frozen regular ToMA / regular UPA',resource_starts=1,
            tol_inner=1e-4,tol_outer=1e-4,stopping='absolute WSR change; inner once, outer twice',
            max_inner_iter_per_call=1200,max_outer_iter_per_call=1500,
            same_weight_continuation_only=True,original_objective='w*(R_DL+R_UL)/10+(1-w)*R_S/5',
            reused_weights=[.1,.18,.46])))
    for sc in SCHEMES:
        if sha(out/'initial_states'/f'{sc}.npz')!=sha(ASSETS/'initial_states'/f'{sc}.npz'):raise RuntimeError('Frozen initial state changed.')
    jobs=[]
    for sc in SCHEMES:
        for w in WEIGHTS:
            path=out/'records'/f'{sc}_omega_{w:g}.json'
            if not path.exists() or not common.read_json(path)['row']['converged']:jobs.append((str(root),str(out),sc,w,args.max_calls))
    print(f'{len(jobs)} incomplete/new full optimization tasks; no per-weight power selection.',flush=True)
    aggregate(out)
    try:
        if args.workers==1:
            for job in jobs:run_one(*job);aggregate(out)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures=[pool.submit(run_one,*job) for job in jobs]
                for f in as_completed(futures):f.result();aggregate(out)
    finally:aggregate(out,draw=True)
    print('Output:',out,flush=True)

if __name__=='__main__':main()
