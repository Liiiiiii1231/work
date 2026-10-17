"""MA only: weights0.016,0.017,0.018,0.019, full inner/outer, frozen regular start and0.25mW."""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
from unittest.mock import patch
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,shutil
import numpy as np
import ma_add4_common as common
from ma_add4_support import csv_read,csv_write,json_write,sha,FIELDS,nondominated
from ma_add4_styles import STYLES
INPUTS=Path(__file__).resolve().parent/'ma_transition_inputs'
WEIGHTS=[.016,.017,.018,.019]
OLD_WEIGHTS=[.001,.015,.16,.18,.2,.42,.46,.5,.55,.6,.65,.7,.75,.8,.85,.9]
SCHEMES=['proposed','b6','b2','b3']


def validate(root):
    common.verify_files(root);scene,opts=common.load_scene(root)
    from baselines import build_fpa_upa_geometry
    from utils import check_resource_constraints
    import ma_baseline as ma
    for n,h in common.read_json(INPUTS/'manifest.json').items():
        if sha(INPUTS/n)!=h:raise RuntimeError('Frozen input checksum mismatch: '+n)
    rows=csv_read((INPUTS/'existing_four_schemes.csv').read_bytes())
    expected={(s,w) for s in SCHEMES for w in OLD_WEIGHTS}|{('b6',w) for w in [.02,.04,.06,.08]}
    if len(rows)!=68 or {(r['scheme'],float(r['omega_comm'])) for r in rows}!=expected:raise RuntimeError('Expected exact accepted68-point baseline.')
    for row in rows:
        w=float(row['omega_comm']);score=w*float(row['sum_rate_comm'])/10+(1-w)*float(row['sum_rate_sensing'])/5
        if str(row['converged']).lower()!='true' or abs(score-float(row['weighted_sum_rate']))>1e-9:raise RuntimeError('Baseline convergence/objective mismatch.')
    g,r,_=common.load_state(INPUTS/'initial_ma.npz');check_resource_constraints(r,scene);base=build_fpa_upa_geometry(scene)
    for n in ['tx_positions','rx_positions']:
        if not np.allclose(getattr(g,n),getattr(base,n),rtol=0,atol=1e-12):raise RuntimeError('Expected regular UPA, no random jitter.')
    if not ma._ma_geometry_is_feasible(g.tx_positions,g.rx_positions,np.mean(base.tx_positions,axis=0),np.mean(base.rx_positions,axis=0),opts['region_size'],opts['min_spacing'],scene):raise RuntimeError('Initial MA geometry infeasible.')
    if not np.allclose(abs(r.q_ul)**2,.00025,rtol=0,atol=1e-14):raise RuntimeError('Expected0.25mW per-user initial UL power.')
    print('CHECK PASSED: exact68 existing rows, original solver, regular MA start,0.25mW UL.',flush=True)
    return rows


def run_one(root,out,w,max_calls):
    out=Path(out)
    with patch.object(common,'initial_file',lambda scheme:out/'initial_states/b6.npz'):
        return common.run_job(root,str(out),'b6',w,max_calls)


def redraw(out,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,ax=plt.subplots(figsize=(8.4,6.1));fronts=[]
    for sc in SCHEMES:
        a=[r for r in rows if r['scheme']==sc];front=nondominated(a);fronts+=front;style=STYLES[sc]
        ax.scatter([float(r['sum_rate_comm']) for r in a],[float(r['sum_rate_sensing']) for r in a],marker=style['marker'],color=style['color'],alpha=.18,s=20)
        ax.plot([float(r['sum_rate_comm']) for r in front],[float(r['sum_rate_sensing']) for r in front],marker=style['marker'],color=style['color'],lw=1.8,ms=5,label=style['label'])
    ax.set_xlabel('Sum communication rate (bit/s/Hz)');ax.set_ylabel('Sum sensing rate (bit/s/Hz)');ax.grid(alpha=.22);ax.legend(loc='upper right',fontsize=9)
    ax.set_title('Communication-sensing tradeoff | scene8001')
    fig.text(.11,.065,'16 common weights +8 additional MA weights; each scheme retains its own nondominated samples.',fontsize=8)
    fig.text(.11,.039,'Faint markers show all results. No smoothing; lines guide the eye.',fontsize=8)
    fig.text(.11,.014,'No claim of feasible interpolation or a global Pareto boundary.',fontsize=8)
    fig.subplots_adjust(left=.11,right=.98,bottom=.18,top=.92)
    folder=out/'figures';folder.mkdir(exist_ok=True)
    for ext in ['png','pdf','svg']:fig.savefig(folder/f'05_communication_sensing_tradeoff_MA_transition4.{ext}',dpi=240,facecolor='white')
    plt.close(fig);csv_write(out/'nondominated_points.csv',fronts,FIELDS)


def aggregate(out,baseline,draw=False):
    extra=[]
    for w in WEIGHTS:
        p=out/'records'/f'b6_omega_{w:g}.json'
        if p.exists():extra.append(common.read_json(p)['row'])
    done=[r for r in extra if r['converged']]
    csv_write(out/'new_MA_points.csv',extra,FIELDS)
    merged=baseline+done;merged.sort(key=lambda r:(float(r['omega_comm']),SCHEMES.index(r['scheme'])))
    keys={(r['scheme'],float(r['omega_comm'])) for r in merged}
    if len(keys)!=len(merged):raise RuntimeError('Duplicate points during merge.')
    csv_write(out/'all_four_schemes.csv',merged,FIELDS)
    oldfront=nondominated([r for r in baseline if r['scheme']=='b6'])
    newfront=nondominated([r for r in merged if r['scheme']=='b6'])
    front_weights={float(r['omega_comm']) for r in newfront}
    diagnostics=[dict(omega_comm=float(r['omega_comm']),weighted_sum_rate=r['weighted_sum_rate'],sum_rate_comm=r['sum_rate_comm'],sum_rate_sensing=r['sum_rate_sensing'],on_updated_MA_nondominated_front=float(r['omega_comm']) in front_weights) for r in done]
    csv_write(out/'new_MA_sampling_status.csv',diagnostics,['omega_comm','weighted_sum_rate','sum_rate_comm','sum_rate_sensing','on_updated_MA_nondominated_front'])
    complete=len(done)==4
    json_write(out/'completion.json',dict(complete=complete,expected_new_runs=4,converged_new_runs=len(done),reused_points=68,total_converged_points=len(merged),new_MA_weights=WEIGHTS,unchanged_other_three_schemes=True,old_MA_nondominated_count=len(oldfront),updated_MA_nondominated_count=len(newfront),single_regular_initial_geometry=True,initial_UL_power_mW=.25,cross_weight_warm_start=False,original_objective_unchanged=True))
    if complete and draw:redraw(out,merged)
    print(f'MA new runs converged {len(done)}/4; front samples {len(oldfront)} -> {len(newfront)}.',flush=True)
    return complete


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--output',type=Path,default=Path('results/stage10/fixed_ul025mW_MA_transition4'))
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--max-calls',type=int,default=20)
    ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args()
    if args.workers<1 or args.max_calls<1:ap.error('Positive workers/max-calls required.')
    root=args.project_root.resolve();out=(root/args.output).resolve();baseline=validate(root)
    if args.check_only:print('CHECK ONLY: no optimizer invoked.');return
    base=Path(__file__).resolve().parent
    identity=dict(entry_sha256=sha(Path(__file__)),inputs_sha256=sha(INPUTS/'manifest.json'),solver_assets_sha256=sha(common.ASSETS/'manifest.json'),dependencies={n:sha(base/n) for n in ['ma_add4_common.py','ma_add4_support.py','ma_add4_styles.py']},project_root=str(root))
    if out.exists():
        if not (out/'run_metadata.json').exists() or common.read_json(out/'run_metadata.json')['identity']!=identity:raise RuntimeError('Output identity changed; choose a new --output directory.')
    else:
        out.mkdir(parents=True)
        for n in ['initial_states','records','states']:(out/n).mkdir()
        shutil.copy2(INPUTS/'initial_ma.npz',out/'initial_states/b6.npz')
        json_write(out/'run_metadata.json',dict(identity=identity,protocol=dict(scene_seed=8001,resource_seed=2042,schemes_computed=['b6'],weights=WEIGHTS,initial_UL_power_W=.00025,tol_inner=1e-4,tol_outer=1e-4,stopping='absolute WSR change; inner once, outer twice',max_inner_iter_per_call=1200,max_outer_iter_per_call=1500,same_weight_continuation_only=True,full_inner_outer_optimization=True,initial_geometry='same frozen regular UPA, no random jitter',resource_starts=1,objective='w*(R_DL+R_UL)/10+(1-w)*R_S/5',reuse_note='68 accepted existing CSV rows; no other scheme is reoptimized')))
    if sha(out/'initial_states/b6.npz')!=sha(INPUTS/'initial_ma.npz'):raise RuntimeError('Initial MA state changed.')
    jobs=[(str(root),str(out),w,args.max_calls) for w in WEIGHTS]
    print('Only4 new MA full inner/outer runs. No resource or geometry multistart.',flush=True)
    try:
        if args.workers==1:
            for job in jobs:run_one(*job);aggregate(out,baseline)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures=[pool.submit(run_one,*job) for job in jobs]
                for f in as_completed(futures):f.result();aggregate(out,baseline)
    finally:complete=aggregate(out,baseline,draw=True)
    print('Output:',out,flush=True)
    if not complete:raise SystemExit(1)

if __name__=='__main__':main()
