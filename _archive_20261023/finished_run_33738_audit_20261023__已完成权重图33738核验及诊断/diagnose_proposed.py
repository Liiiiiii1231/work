import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import sys,json,zipfile,io,time
from dataclasses import replace,asdict
from unittest.mock import patch
import numpy as np
BASE=Path(r'D:\Vs codeProjects\research work1')
sys.path.insert(0,str(BASE/'toma_isac_hpc_by_figure'/'toma'/'test_pilot7_56core'))
import run_figures
import figure_engine as e
import inner_solver as inner
import outer_solver as outer
from system_model import SystemConfig,GeometryState,ResourceState
from channel_model import build_channels

Z=zipfile.ZipFile(Path(r'C:\Users\27809\Downloads\tradeoff_mc10_33738_bundle.zip'))
records={}
for n in Z.namelist():
    if '/candidate_0/' in n and n.endswith('/record.json'):
        d=json.loads(Z.read(n));r=d['row']
        if r['scheme']=='proposed':records[(r['parameter_value'],r['scene_seed'])]=(d,n.rsplit('/',1)[0])


def load(w,seed):
    d,folder=records[(w,seed)];raw=d['effective_config'].copy()
    for k,v in list(raw.items()):
        if isinstance(v,list):
            a=np.array(v);raw[k]=a[:,0]+1j*a[:,1] if k in ('xi_s','fading_dl','fading_ul','fading_s') else a
    cfg=SystemConfig(**raw)
    with np.load(io.BytesIO(Z.read(folder+'/'+d['state_file']))) as a:
        g=GeometryState(**{k:a[k].copy() for k in ('endpoints','tx_positions','rx_positions')})
        r=ResourceState(**{k:a[k].copy() for k in ('q_matrix','q_ul','b_ul','u_s')})
        history=a['wsr_history'].copy()
    return cfg,g,r,history,d


means=[];targets=[]
for w in (.001,.015,.05,.2):
    gaps=[];orig=[];reference=[]
    for seed in range(8001,8011):
        d,_=records[(w,seed)];zero,_=records[(0,seed)]
        a=d['effective_config'].copy();b=zero['effective_config'].copy()
        for k in ('weight_dl','weight_ul','weight_s'):a.pop(k);b.pop(k)
        assert a==b
        value=w*zero['row']['sum_rate_comm']/10+(1-w)*zero['row']['sum_rate_sensing']/5
        actual=d['row']['weighted_sum_rate'];gaps.append((value-actual,seed));orig.append(actual);reference.append(value)
    means.append(dict(weight=w,current_mean=float(np.mean(orig)),pure_sensing_reference_mean=float(np.mean(reference)),reference_better_scenes=sum(g>1e-6 for g,s in gaps)))
    targets.append((w,max(gaps)[1]))

cases=[]
for w,seed in targets:
    cfg,g,r,h,d=load(w,seed);case=dict(weight=w,scene=seed,current=d['row'],tail_deltas=np.diff(h[-11:]).tolist(),initial_state='saved final state; diagnostic restart of RCG history')
    ch=build_channels(g,cfg)
    with patch.object(inner,'relative_change',e.abs_change):
        tick=time.perf_counter();rr,hh=inner.solve_inner_problem(ch,r,cfg,verbose=False)
    case['fixed_geometry_inner']=dict(iterations=len(hh)-1,gain=float(hh[-1]-hh[0]),seconds=time.perf_counter()-tick,last_gain=float(hh[-1]-hh[-2]))
    case['uplink_power_W']=(r.q_ul**2).tolist()
    case['dl_beam_power_W']=np.sum(abs(r.q_matrix[:,:cfg.k_dl])**2,axis=0).tolist()
    case['sensing_factor_power_W']=np.sum(abs(r.q_matrix[:,cfg.k_dl:])**2,axis=0).tolist()
    gradients=[]
    for delta in (1e-6,3e-7,1e-7):
        dc=replace(cfg,finite_diff_delta=delta)
        gg=np.array([outer.project_to_tangent(g.endpoints[i],outer.central_difference_gradient(g.endpoints,i,rr,dc),dc) for i in range(cfg.m_uav)])
        gradients.append(gg)
    case['gradient_check']=dict(norm=float(np.linalg.norm(gradients[0])),relative_change_delta3e7=float(np.linalg.norm(gradients[1]-gradients[0])/max(1e-15,np.linalg.norm(gradients[0]))),relative_change_delta1e7=float(np.linalg.norm(gradients[2]-gradients[0])/max(1e-15,np.linalg.norm(gradients[0]))))
    sweeps=[]
    for count in (2,10):
        diagnostics=[]
        with patch.object(inner,'relative_change',e.abs_change):
            tick=time.perf_counter();gg,cc,rnew,wsr,rcg,accepted=outer.update_positions_once_fp_aware(g,rr,cfg,candidate_fp_iterations=count,diagnostic_records=diagnostics)
        sweeps.append(dict(candidate_FP_limit=count,gain=float(wsr-hh[-1]),seconds=time.perf_counter()-tick,max_displacement_m=float(np.linalg.norm(gg.endpoints-g.endpoints,axis=1).max()),accepted=accepted,endpoints=[asdict(x) for x in diagnostics]))
    case['position_sweeps']=sweeps;cases.append(case)
    print('CASE',w,seed,json.dumps({k:v for k,v in case.items() if k in ('fixed_geometry_inner','gradient_check')}),flush=True)
report=dict(mean_comparisons=means,cases=cases,note='Read-only reference evaluation and isolated fixed-geometry FP/one position sweep; no formal checkpoint changes, no cross-weight warm-start optimization')
Path(__file__).with_name('proposed_low_weight_diagnosis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('FINISHED',json.dumps(means),flush=True)
