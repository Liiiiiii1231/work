"""Controlled saved-state tests; no production source edits or cross-weight warm starts."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
from pathlib import Path
import sys,zipfile,json,io,time,hashlib,csv
from copy import deepcopy
from dataclasses import replace,asdict
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
TOMA=Path(r'D:\Vs codeProjects\research work1\toma_isac_hpc_by_figure\toma')
sys.path[:0]=[str(TOMA/'final_support_i1200_o1500_s8005'),str(TOMA)]
import figure_engine as e,inner_solver as inner,outer_solver as outer,algorithms as algo
from system_model import GeometryState,ResourceState,SystemConfig
from channel_model import build_channels
OUT=Path(__file__).resolve().parent
ZIP=Path(r'C:\Users\27809\Downloads\final_33774_bundle.zip')

def load(z,power,seed):
    folder=f'final_33774/candidate_0/tasks/convergence/{power}/scene_{seed}/proposed/'
    d=json.loads(z.read(folder+'record.json'));raw=d['effective_config'].copy()
    for k,v in list(raw.items()):
        if isinstance(v,list):
            a=np.asarray(v);raw[k]=a[:,0]+1j*a[:,1] if k in ('xi_s','fading_dl','fading_ul','fading_s') else a
    cfg=SystemConfig(**raw)
    assert e.config_digest(cfg)==d['cfg_sha256']
    state=z.read(folder+d['state_file']);assert hashlib.sha256(state).hexdigest()==d['state_sha256']
    with np.load(io.BytesIO(state)) as a:
        g=GeometryState(**{n:a[n].copy() for n in ('endpoints','tx_positions','rx_positions')})
        r=ResourceState(**{n:a[n].copy() for n in ('q_matrix','q_ul','b_ul','u_s')})
    return cfg,g,r

def main():
    rows=[];details=[]
    with zipfile.ZipFile(ZIP) as z,ExitStack() as stack:
        for name in ('algorithms.py','channel_model.py','inner_solver.py','outer_solver.py','system_model.py'):
            assert z.read('final_33774/provenance/shared_modules/'+name).replace(b'\r\n',b'\n')==(TOMA/name).read_bytes().replace(b'\r\n',b'\n')
        for mod in (inner,outer,algo):stack.enter_context(patch.object(mod,'relative_change',e.abs_change))
        for power,seed in [('0.1',8003),('1',8003),('10',8003)]:
            cfg,g,r=load(z,power,seed)
            before=e.metrics_values(g,r,cfg)['weighted_sum_rate']
            r,h=inner.solve_inner_problem(build_channels(g,cfg),r,cfg,verbose=False)
            baseline=None
            for alpha in [.025,.1,.2,.8,1.]:
                test=replace(cfg,initial_outer_step=alpha);diagnostics=[];tick=time.perf_counter()
                ng,nch,nr,score,rcg,accepted=outer.update_positions_once_fp_aware(
                    deepcopy(g),deepcopy(r),test,candidate_fp_iterations=2,diagnostic_records=diagnostics)
                runtime=time.perf_counter()-tick
                e.check_state(ng,nr,test,{},'proposed')
                assert score>=h[-1]-1e-8
                if alpha==.1:baseline=ng
                disp=np.linalg.norm(ng.endpoints-g.endpoints,axis=1)
                row=dict(power_W=float(power),seed=seed,initial_alpha=alpha,initial_score=before,
                    after_inner=h[-1],inner_iterations=len(h)-1,after_positions=score,
                    position_gain=score-h[-1],accepted=accepted,
                    backtracks=sum(d.backtracking_steps for d in diagnostics),
                    candidate_fp_calls=sum(d.candidate_fp_calls for d in diagnostics),
                    alpha_min=min(d.step_size for d in diagnostics),alpha_max=max(d.step_size for d in diagnostics),
                    mean_displacement_mm=float(disp.mean()*1000),max_displacement_mm=float(disp.max()*1000),seconds=runtime,
                    max_difference_from_alpha01_m=(float(np.max(abs(ng.endpoints-baseline.endpoints))) if baseline is not None else None))
                rows.append(row);details.append(dict(row=row,endpoints=ng.endpoints.tolist(),diagnostics=[asdict(d) for d in diagnostics]))
                print(json.dumps(row),flush=True)
                e.json_write(OUT/'larger_step_results.json',dict(note='Three saved final states. RCG memory reset consistently for every test. Local sweeps cannot establish full convergence counts.',rows=rows,details=details))
    e.csv_write(OUT/'larger_step_results.csv',rows,list(rows[0]))

if __name__=='__main__':main()
