from __future__ import annotations


# ===== paper_figure_styles: shared implementation =====
"""One stable scheme -> marker/color mapping for every paper figure."""
STYLES={
 'proposed':dict(label='Proposed: ToMA + FP',marker='D',color='#df2228'),
 'b6':dict(label='MA + FP',marker='^',color='#279b3b'),
 'b2':dict(label='Fixed-ToMA + FP',marker='o',color='#e99a21'),
 'b3':dict(label='Fixed UPA + FP',marker='s',color='#2675c1'),
}
POWER_LINESTYLES=['-','--','-.',':']
POWER_COLORS=['#2675c1','#df2228','#8250a8','#9d6813']


# ===== paper_figures_unified: shared implementation =====
"""Shared experiment engine: original solvers, fixed starts, absolute stopping."""

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
from pathlib import Path
from dataclasses import replace,asdict
from contextlib import ExitStack
from copy import deepcopy
from unittest.mock import patch
import csv,json,hashlib,time,shutil
import numpy as np
pass

FIGURES=['convergence','p_dl_max','rho_si','system_load','tradeoff']
FIELDS=['figure','parameter_value','scene_seed','scheme','sum_rate_dl','sum_rate_ul','sum_rate_comm','sum_rate_sensing','weighted_sum_rate','iterations','runtime_s','converged','calls']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def json_write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,p)
def csv_write(p,rows,fields):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    with tmp.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    os.replace(tmp,p)
def serial(v):
    if isinstance(v,np.ndarray):return serial(v.tolist())
    if isinstance(v,complex):return [v.real,v.imag]
    if isinstance(v,np.generic):return serial(v.item())
    if isinstance(v,dict):return {str(k):serial(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [serial(x) for x in v]
    return v
def config_digest(cfg):return hashlib.sha256(json.dumps(serial(asdict(cfg)),sort_keys=True).encode()).hexdigest()
def abs_change(a,b):return abs(float(a)-float(b))

def save_state(path,g,r,h):
    path=Path(path);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f,**{k:getattr(g,k) for k in ['endpoints','tx_positions','rx_positions']},
            **{k:getattr(r,k) for k in ['q_matrix','q_ul','b_ul','u_s']},wsr_history=np.asarray(h))
    os.replace(tmp,path)
def load_state(path):
    from system_model import GeometryState, ResourceState
    with np.load(path,allow_pickle=False) as a:
        g=GeometryState(**{n:a[n].copy() for n in ['endpoints','tx_positions','rx_positions']})
        r=ResourceState(**{n:a[n].copy() for n in ['q_matrix','q_ul','b_ul','u_s']})
        h=a['wsr_history'].astype(float).tolist() if 'wsr_history' in a else []
    return g,r,h

def config_for_task(root,p,task):
    from system_model import load_config
    from system_model import make_scene_config
    source=p['load_configs'][str(int(task['parameter_value']))] if task['figure']=='system_load' else p['base_config']
    raw=p['_models'][source];cfg=load_config(raw)
    proto=p['protocol']
    # Preserve the original objective arrays: only algorithm protocol/initial amplitude changes.
    cfg=replace(cfg,p_dl_max=proto['base_p_dl_W'],rho_si=proto['base_rho_si'],max_inner_iter=proto['max_inner_iter'],max_outer_iter=proto['max_outer_iter'],
        tol_inner=proto['tol_inner'],tol_outer=proto['tol_outer'],plateau_window=0,tol_window=0.,
        random_seed=proto['resource_seed'],kappa_ul=float(np.sqrt(proto['initial_ul_power_W']/cfg.p_ul_max[0])))
    if task['figure'] in ['p_dl_max','convergence']:cfg=replace(cfg,p_dl_max=float(task['parameter_value']))
    if task['figure']=='rho_si':cfg=replace(cfg,rho_si=float(task['parameter_value']))
    if task['figure']=='tradeoff':
        w=float(task['parameter_value'])
        cfg=replace(cfg,weight_dl=np.full(cfg.k_dl,w/(cfg.k_dl+cfg.j_ul)),weight_ul=np.full(cfg.j_ul,w/(cfg.k_dl+cfg.j_ul)),weight_s=np.full(cfg.l_target,(1-w)/cfg.l_target))
    exp=raw['experiment'];fading=exp.get('fading',{})
    return make_scene_config(cfg,int(task['scene_seed']),np.asarray(exp['position_jitter'],dtype=float),
        randomize_fading=bool(fading.get('enabled',False)),fading_model=fading.get('model','scalar_rayleigh'))

def initial_pair(cfg,p,scheme):
    from system_model import build_geometry
    from algorithms import build_fpa_upa_geometry
    from channel_model import build_channels
    from channel_model import initialize_resources
    if scheme in ['proposed','b2']:
        theta=np.pi/4+2*np.pi*np.arange(cfg.m_uav)/cfg.m_uav
        e=np.column_stack([np.zeros(cfg.m_uav),cfg.cable_length*np.cos(theta),cfg.cable_length*np.sin(theta)])
        g=build_geometry(e,cfg,check_constraints=True)
    else:g=build_fpa_upa_geometry(cfg)
    r=initialize_resources(g,build_channels(g,cfg),cfg,np.random.default_rng(p['protocol']['resource_seed']))
    r.q_ul[:]=np.sqrt(float(p['protocol']['initial_ul_power_W']))
    return g,r

def metrics_values(g,r,cfg):
    from channel_model import build_channels
    from channel_model import compute_performance
    perf=compute_performance(build_channels(g,cfg),r,cfg)
    dl,ul,s=map(float,[sum(perf.rate_dl),sum(perf.rate_ul),sum(perf.rate_s)])
    return dict(sum_rate_dl=dl,sum_rate_ul=ul,sum_rate_comm=dl+ul,sum_rate_sensing=s,weighted_sum_rate=float(perf.weighted_sum_rate))

def check_state(g,r,cfg,p,scheme,g0=None):
    from system_model import check_resource_constraints
    from system_model import check_geometry_constraints
    from algorithms import build_fpa_upa_geometry
    import algorithms as ma
    check_resource_constraints(r,cfg)
    if scheme in ['proposed','b2']:check_geometry_constraints(g,cfg)
    elif scheme=='b6':
        base=build_fpa_upa_geometry(cfg);proto=p['protocol']
        if not ma._ma_geometry_is_feasible(g.tx_positions,g.rx_positions,np.mean(base.tx_positions,axis=0),np.mean(base.rx_positions,axis=0),proto['ma_region_size_m'],proto['ma_min_spacing_m'],cfg):raise RuntimeError('MA geometry infeasible.')
    if g0 is not None and scheme in ['b2','b3']:
        for n in ['tx_positions','rx_positions']:
            if not np.array_equal(getattr(g,n),getattr(g0,n)):raise RuntimeError('Fixed geometry changed.')


def task_key(task):return f"{task['figure']}/{float(task['parameter_value']):.12g}/scene_{task['scene_seed']}/{task['scheme']}"

def build_tasks(root,p,figure,n_mc=None):
    
    count=int(p['n_mc'] if n_mc is None else n_mc)
    if count<1:raise ValueError('n_mc must be positive.')
    seeds=list(range(int(p['scene_seed_start']),int(p['scene_seed_start'])+count))
    if figure=='convergence':values=list(map(float,p['convergence']['p_dl_W']));schemes=['proposed']
    elif figure=='tradeoff':values=list(map(float,p['tradeoff']['weights']));schemes=p['schemes']
    else:
        values=([10**((float(x)-30)/10) for x in p['sweeps']['p_dl_dbm']] if figure=='p_dl_max' else
                [10**(float(x)/10) for x in p['sweeps']['rho_si_db']] if figure=='rho_si' else list(map(float,p['sweeps']['loads'])))
        schemes=p['schemes']
    tasks=[dict(figure=figure,parameter_value=v,scene_seed=s,scheme=sc) for v in values for s in seeds for sc in schemes]
    if len({task_key(t) for t in tasks})!=len(tasks):raise ValueError('Duplicate task keys.')
    return tasks


def run_solver_call(cfg,p,task,g,r):
    import algorithms as algorithm, algorithms as baselines, inner_solver as inner_solver, outer_solver as outer_solver, algorithms as ma
    sc=task['scheme'];rng=np.random.default_rng(p['protocol']['resource_seed']);proto=p['protocol']
    with ExitStack() as stack:
        for mod in [algorithm,baselines,inner_solver,outer_solver,ma]:stack.enter_context(patch.object(mod,'relative_change',abs_change))
        if sc=='proposed':
            result=algorithm.run_joint_algorithm(cfg,rng=rng,verbose=False,position_update_mode='fp_aware',fp_aware_candidate_iters=2,initial_endpoints=g.endpoints.copy(),initial_resources=deepcopy(r))
            history=list(map(float,result.joint_wsr_history));iterations=result.outer_iterations
        elif sc=='b6':
            result=ma.run_ma_fp(cfg,rng=rng,verbose=False,region_size=proto['ma_region_size_m'],min_spacing=proto['ma_min_spacing_m'],initial_step=proto['ma_initial_step_m'],initial_tx_positions=g.tx_positions.copy(),initial_rx_positions=g.rx_positions.copy(),initial_resources=deepcopy(r))
            history=list(map(float,result.wsr_history));iterations=result.iterations
        elif sc=='b2':
            result=baselines.run_fixed_toma_fp(cfg,rng=rng,initial_endpoints=g.endpoints.copy(),initial_resources=deepcopy(r));history=list(map(float,result.wsr_history));iterations=result.iterations
        elif sc=='b3':
            result=baselines.run_fpa_upa_fp(cfg,rng=rng,initial_resources=deepcopy(r));history=list(map(float,result.wsr_history));iterations=result.iterations
        else:raise ValueError('Unsupported scheme.')
    return result,history,int(iterations)


def run_task(root_string,out_string,p,task,max_calls):
    root,out=Path(root_string),Path(out_string);cfg=config_for_task(root,p,task);folder=out/'tasks'/task_key(task);folder.mkdir(parents=True,exist_ok=True)
    g0,r0=initial_pair(cfg,p,task['scheme']);check_state(g0,r0,cfg,p,task['scheme']);fingerprint=config_digest(cfg)
    if not np.allclose(np.abs(r0.q_ul)**2,float(p['protocol']['initial_ul_power_W']),rtol=0,atol=1e-12):raise RuntimeError('Generated initial UL power differs before optimization.')
    initial=folder/'initial.npz';record=folder/'record.json'
    if not initial.exists():save_state(initial,g0,r0,[])
    gi,ri,_=load_state(initial)
    for n in ['endpoints','tx_positions','rx_positions']:
        if not np.allclose(getattr(gi,n),getattr(g0,n),rtol=0,atol=1e-12):raise RuntimeError('Saved initial geometry differs.')
    if p['protocol'].get('saved_initialization_authoritative', False) and record.exists():
        check_state(gi,ri,cfg,p,task['scheme'])
        if not np.allclose(np.abs(ri.q_ul)**2, float(p['protocol']['initial_ul_power_W']), rtol=0, atol=1e-12):
            raise RuntimeError('Saved initial UL power differs.')
    else:
        for n in ['q_matrix','q_ul','b_ul','u_s']:
            if not np.allclose(getattr(ri,n),getattr(r0,n),rtol=1e-12,atol=1e-12):raise RuntimeError('Saved initial resource differs.')
    g,r,h=gi,ri,[];iterations=0;seconds=0.;call=0
    if record.exists():
        d=read_json(record)
        if d['cfg_sha256']!=fingerprint or d['initial_sha256']!=sha(initial) or d['state_sha256']!=sha(folder/d['state_file']):raise RuntimeError('Checkpoint identity mismatch.')
        g,r,h=load_state(folder/d['state_file']);check_state(g,r,cfg,p,task['scheme'],g0)
        if abs(metrics_values(g,r,cfg)['weighted_sum_rate']-float(d['row']['weighted_sum_rate']))>1e-9:raise RuntimeError('Checkpoint WSR mismatch.')
        if d['row']['converged']:
            required=2 if task['scheme'] in ['proposed','b6'] else 1
            delta=np.diff(h);tol=cfg.tol_outer if required==2 else cfg.tol_inner
            if len(delta)<required or np.min(delta)<-1e-8 or np.max(abs(delta[-required:]))>tol:raise RuntimeError('Cached stopping/history mismatch.')
            return d['row']
        iterations=int(d['row']['iterations']);seconds=float(d['row']['runtime_s']);call=int(d['row']['calls'])
    for _ in range(max_calls):
        call+=1;expected=metrics_values(g,r,cfg)['weighted_sum_rate'];tick=time.perf_counter()
        print('START',task_key(task),'call',call,'previous_steps',iterations,flush=True)
        result,new,count=run_solver_call(cfg,p,task,g,r);seconds+=time.perf_counter()-tick
        if len(new)<2 or not np.isclose(new[0],expected,rtol=1e-9,atol=1e-9):raise RuntimeError('Initial/continuation WSR mismatch.')
        h=h+new[1:] if h else new;iterations+=count
        if np.min(np.diff(h))<-1e-8*max(1.,max(abs(x) for x in h)):raise RuntimeError('True WSR decreased.')
        g,r=result.geometry,result.resources;check_state(g,r,cfg,p,task['scheme'],g0);v=metrics_values(g,r,cfg)
        if not all(np.isfinite(x) for x in v.values()) or not np.isclose(h[-1],v['weighted_sum_rate'],rtol=1e-9,atol=1e-9):raise RuntimeError('Final metrics mismatch.')
        converged=bool(result.converged);required=2 if task['scheme'] in ['proposed','b6'] else 1;tol=cfg.tol_outer if required==2 else cfg.tol_inner
        if converged and (len(new)<required+1 or np.max(abs(np.diff(new[-required-1:])))>tol):raise RuntimeError('Absolute stopping check failed.')
        state_name=f'call_{call:03d}.npz';save_state(folder/state_name,g,r,h)
        row=dict(**task,**v,iterations=iterations,runtime_s=seconds,converged=converged,calls=call)
        json_write(record,dict(row=row,cfg_sha256=fingerprint,effective_config=serial(asdict(cfg)),initial_sha256=sha(initial),state_file=state_name,state_sha256=sha(folder/state_name),absolute_stopping=True,same_parameter_scene_weight_only=True,convergence_measure="absolute_difference"))
        print('DONE',task_key(task),'WSR',v['weighted_sum_rate'],'converged',converged,'steps',iterations,flush=True)
        if converged:return row
    return row


def collect_rows(out,tasks):
    rows=[]
    for task in tasks:
        f=Path(out)/'tasks'/task_key(task)/'record.json'
        if f.exists():rows.append(read_json(f)['row'])
    return rows

def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.grid':True,'grid.alpha':.22})
    return plt

def save_figure(fig,out,name):
    folder=Path(out)/'figures';folder.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf','svg']:fig.savefig(folder/f'{name}.{ext}',dpi=240,facecolor='white',bbox_inches='tight')


def summarize_sweep(rows,tasks,schemes):
    summary=[]
    for value in sorted({float(t['parameter_value']) for t in tasks}):
        for sc in schemes:
            expected=[t for t in tasks if float(t['parameter_value'])==value and t['scheme']==sc]
            selected=[r for r in rows if float(r['parameter_value'])==value and r['scheme']==sc and r['converged']]
            if len(selected)!=len(expected):raise RuntimeError('Incomplete Monte Carlo cell: never average partial scenes.')
            data=np.asarray([float(r['weighted_sum_rate']) for r in selected]);summary.append(dict(parameter_value=value,scheme=sc,n_scenes=len(data),mean_wsr=float(data.mean()),std_wsr=float(data.std(ddof=1)) if len(data)>1 else 0.))
    return summary

def plot_sweep(out,p,figure,tasks,rows):
    summary=summarize_sweep(rows,tasks,p['schemes']);csv_write(Path(out)/'tables'/f'{figure}_mean.csv',summary,['parameter_value','scheme','n_scenes','mean_wsr','std_wsr'])
    plt=plotting();fig,ax=plt.subplots(figsize=(7.5,5.1))
    for sc in p['schemes']:
        a=sorted([r for r in summary if r['scheme']==sc],key=lambda r:r['parameter_value']);x=np.asarray([r['parameter_value'] for r in a]);style=STYLES[sc]
        if figure=='p_dl_max':x=10*np.log10(x)+30
        if figure=='rho_si':x=10*np.log10(x)
        ax.plot(x,[r['mean_wsr'] for r in a],color=style['color'],marker=style['marker'],label=style['label'],lw=1.7,ms=5)
    xlabel={'p_dl_max':'Maximum DL transmit power (dBm)','rho_si':'Residual SI level (dB)','system_load':'System load ($K=J=L$)'}[figure]
    ax.set_xlabel(xlabel);ax.set_ylabel('Average WSR (bit/s/Hz)');ax.legend(fontsize=9);ax.set_title(f"{summary[0]['n_scenes']} shared Monte Carlo scenes | regular initialization")
    if figure=='system_load':ax.set_xticks(p['sweeps']['loads'])
    elif figure=='p_dl_max':ax.set_xticks(p['sweeps']['p_dl_dbm'])
    else:ax.set_xticks(p['sweeps']['rho_si_db'])
    fig.tight_layout();name={'p_dl_max':'02_wsr_vs_p_dl_max','rho_si':'03_wsr_vs_rho_si','system_load':'04_wsr_vs_system_load'}[figure]
    save_figure(fig,out,name);plt.close(fig)




# ===== final_protocol: shared implementation =====
"""Frozen 100-scene protocol and deterministic multi-start initialization."""
from pathlib import Path
from unittest.mock import patch
import numpy as np
import sys
engine = sys.modules[__name__]

ROOT = Path(__file__).resolve().parent
ANCHOR = engine.initial_pair
STARTS = {'proposed': 5, 'b6': 3, 'b2': 1, 'b3': 1}


def n_starts(task):
    return 1 if task['figure'] == 'tradeoff' else STARTS[task['scheme']]


def pair(cfg, protocol, scheme, start):
    """Start 0: uniform planar anchor; remaining starts: feasible perturbations.

    Geometry streams do not depend on scene, weight, or sweep value. All weights
    use exactly the same anchor; channel-dependent MRT is rebuilt per scene.
    """
    from system_model import build_geometry
    from channel_model import build_channels
    from channel_model import initialize_resources
    import algorithms as ma
    g, r = ANCHOR(cfg, protocol, scheme)
    if start == 0:
        return g, r
    seeds = [2026, 2027, 2028, 2029] if scheme == 'proposed' else [2026, 2027]
    rng = np.random.default_rng(seeds[start-1])
    if scheme == 'proposed':
        anchor = g.endpoints/cfg.cable_length
        for _ in range(cfg.max_geometry_trials):
            directions = anchor + .15*rng.normal(size=anchor.shape)
            endpoints = cfg.cable_length*directions/np.linalg.norm(directions, axis=1, keepdims=True)
            try:
                g = build_geometry(endpoints, cfg, check_constraints=True)
                break
            except ValueError:
                continue
        else:
            raise RuntimeError('No feasible perturbed ToMA geometry')
    elif scheme == 'b6':
        p = protocol['protocol']
        g = ma._randomize_initial_ma_geometry(g, np.mean(g.tx_positions, axis=0),
            np.mean(g.rx_positions, axis=0), p['ma_region_size_m'], p['ma_min_spacing_m'], .5, cfg, rng)
    else:
        raise ValueError('Fixed arrays have one geometry start')
    r = initialize_resources(g, build_channels(g, cfg), cfg,
                             np.random.default_rng(protocol['protocol']['resource_seed']))
    r.q_ul[:] = np.sqrt(protocol['protocol']['initial_ul_power_W'])
    return g, r


def run_job(job):
    out, protocol, task, start, max_calls = job
    with patch.object(engine, 'initial_pair', lambda cfg, p, sc: pair(cfg, p, sc, start)):
        row = engine.run_task(str(ROOT), str(Path(out)/f'candidate_{start}'),
                              protocol, task, max_calls)
    return dict(row, start_id=start)


def read_verified(out, protocol, task, start):
    folder = Path(out)/f'candidate_{start}'/'tasks'/engine.task_key(task)
    record = folder/'record.json'
    if not record.exists():
        return None
    d = engine.read_json(record)
    cfg = engine.config_for_task(ROOT, protocol, task)
    if any(d['row'].get(k) != v for k, v in task.items()):
        raise RuntimeError('Checkpoint task mismatch')
    if d['cfg_sha256'] != engine.config_digest(cfg):
        raise RuntimeError('Checkpoint config mismatch')
    if d['initial_sha256'] != engine.sha(folder/'initial.npz') or d['state_sha256'] != engine.sha(folder/d['state_file']):
        raise RuntimeError('Checkpoint checksum mismatch')
    g0, r0, _ = engine.load_state(folder/'initial.npz')
    expected, resources = pair(cfg, protocol, task['scheme'], start)
    for name in ('endpoints', 'tx_positions', 'rx_positions'):
        if not np.allclose(getattr(g0, name), getattr(expected, name), rtol=0, atol=1e-12):
            raise RuntimeError('Checkpoint initial geometry mismatch')
    for name in ('q_matrix', 'q_ul', 'b_ul', 'u_s'):
        if not np.allclose(getattr(r0, name), getattr(resources, name), rtol=1e-12, atol=1e-12):
            raise RuntimeError('Checkpoint initial resource mismatch')
    if not np.allclose(abs(r0.q_ul)**2, .025, rtol=0, atol=1e-12):
        raise RuntimeError('Initial UL power must be 25 mW')
    g, r, h = engine.load_state(folder/d['state_file'])
    engine.check_state(g, r, cfg, protocol, task['scheme'], g0)
    if abs(engine.metrics_values(g, r, cfg)['weighted_sum_rate']-d['row']['weighted_sum_rate']) > 1e-8:
        raise RuntimeError('Checkpoint metrics mismatch')
    if d['row']['converged']:
        required = 2 if task['scheme'] in ('proposed', 'b6') else 1
        delta = np.diff(h)
        if len(delta) < required or not np.all(np.isfinite(h)) or np.min(delta) < -1e-8*max(1., max(abs(x) for x in h)) or max(abs(delta[-required:])) > 1e-4:
            raise RuntimeError('Absolute convergence verification failed')
    return dict(d['row'], start_id=start)


def plot_convergence(out, protocol, tasks, winners):
    plt = engine.plotting()
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    histories = []
    for i, power in enumerate(protocol['convergence']['p_dl_W']):
        group = [r for r in winners if r['parameter_value'] == power]
        hs = []
        for row in group:
            task = {k:row[k] for k in ('figure','parameter_value','scene_seed','scheme')}
            folder = Path(out)/f"candidate_{row['start_id']}"/'tasks'/engine.task_key(task)
            d = engine.read_json(folder/'record.json')
            _, _, h = engine.load_state(folder/d['state_file'])
            hs.append(h)
            histories.extend(dict(p_dl_W=power,scene_seed=row['scene_seed'],start_id=row['start_id'],iteration=j,weighted_sum_rate=v) for j,v in enumerate(h))
        # A converged scene keeps its terminal value when other scenes continue.
        matrix = np.asarray([np.pad(h,(0,max(map(len,hs))-len(h)),mode='edge') for h in hs])
        ax.plot(matrix.mean(axis=0),color=engine.POWER_COLORS[i],linestyle=engine.POWER_LINESTYLES[i],label=f'{power:g} W')
    ax.set(xlabel='Joint AO iteration',ylabel='Mean WSR (bit/s/Hz)',title=f"{len(group)} Monte Carlo scenes | best start per scene")
    ax.legend(title='Maximum DL power')
    fig.text(.1,.015,'Stopped scenes retain their terminal WSR when computing the mean trajectory.',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1))
    engine.save_figure(fig,out,'01_proposed_convergence')
    plt.close(fig)
    engine.csv_write(Path(out)/'tables/convergence_histories.csv',histories,list(histories[0]))


def plot_tradeoff(out, protocol, tasks, rows):
    summary = []
    for w in protocol['tradeoff']['weights']:
        for sc in protocol['schemes']:
            group = [r for r in rows if r['parameter_value']==w and r['scheme']==sc]
            summary.append(dict(omega_comm=w,scheme=sc,n_scenes=len(group),
                mean_comm=float(np.mean([r['sum_rate_comm'] for r in group])),
                mean_sensing=float(np.mean([r['sum_rate_sensing'] for r in group]))))
    engine.csv_write(Path(out)/'tables/tradeoff_mean.csv',summary,list(summary[0]))
    plt = engine.plotting()
    fig,ax = plt.subplots(figsize=(8.4,6.1))
    for sc in protocol['schemes']:
        group = [r for r in summary if r['scheme']==sc]
        style = engine.STYLES[sc]
        ax.plot([r['mean_comm'] for r in group],[r['mean_sensing'] for r in group],
            color=style['color'],marker=style['marker'],label=style['label'])
    ax.set(xlabel='Mean communication rate (bit/s/Hz)',ylabel='Mean sensing rate (bit/s/Hz)',
           title=f"{group[0]['n_scenes']} Monte Carlo scenes | common initial geometry across weights")
    ax.legend()
    fig.tight_layout()
    engine.save_figure(fig,out,'05_communication_sensing_tradeoff')
    plt.close(fig)
