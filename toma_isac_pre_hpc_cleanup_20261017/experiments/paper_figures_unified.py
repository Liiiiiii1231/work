"""Shared experiment engine: original solvers, fixed starts, absolute stopping."""
from __future__ import annotations
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
import yaml
from paper_figure_styles import STYLES,POWER_LINESTYLES,POWER_COLORS
ASSETS=Path(__file__).resolve().parent/'paper_figures_assets'
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
    from state import GeometryState,ResourceState
    with np.load(path,allow_pickle=False) as a:
        g=GeometryState(**{n:a[n].copy() for n in ['endpoints','tx_positions','rx_positions']})
        r=ResourceState(**{n:a[n].copy() for n in ['q_matrix','q_ul','b_ul','u_s']})
        h=a['wsr_history'].astype(float).tolist() if 'wsr_history' in a else []
    return g,r,h

def load_protocol(path):
    d=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if d['schemes']!=list(STYLES):raise ValueError('Use the four agreed schemes in the supplied order.')
    p=d['protocol']
    if float(p['initial_ul_power_W'])!=.00025:raise ValueError('Frozen tradeoff requires initial UL power 0.25mW.')
    expected={'base_p_dl_W':1.,'base_rho_si':1e-10,'resource_seed':2042,'max_inner_iter':1200,'max_outer_iter':1500,'tol_inner':1e-4,'tol_outer':1e-4,'plateau_window':0,'ma_region_size_m':.3,'ma_min_spacing_m':.005,'ma_initial_step_m':.03}
    if any(p[k]!=v for k,v in expected.items()):raise ValueError('Protocol differs from frozen tradeoff. Use a separate experiment if changing it.')
    if not d['tradeoff']['reuse_verified_results']:raise ValueError('This package reuses the verified tradeoff; it does not rerun joint optimization.')
    powers=d['convergence']['p_dl_W']
    if len(powers)!=3 or len(set(powers))!=3 or min(powers)<=0:raise ValueError('Exactly three distinct positive convergence powers required.')
    return d

def input_config_paths(root,p):
    return [Path(root)/p['base_config']]+[Path(root)/p['load_configs'][int(n)] for n in p['sweeps']['loads']]

def config_for_task(root,p,task):
    from config import load_config
    from stage8 import make_scene_config
    source=p['load_configs'][int(task['parameter_value'])] if task['figure']=='system_load' else p['base_config']
    raw=yaml.safe_load((Path(root)/source).read_text(encoding='utf-8'));cfg=load_config(Path(root)/source)
    proto=p['protocol']
    # Preserve the original objective arrays: only algorithm protocol/initial amplitude changes.
    cfg=replace(cfg,p_dl_max=proto['base_p_dl_W'],rho_si=proto['base_rho_si'],max_inner_iter=proto['max_inner_iter'],max_outer_iter=proto['max_outer_iter'],
        tol_inner=proto['tol_inner'],tol_outer=proto['tol_outer'],plateau_window=0,tol_window=0.,
        random_seed=proto['resource_seed'],kappa_ul=float(np.sqrt(proto['initial_ul_power_W']/cfg.p_ul_max[0])))
    if task['figure'] in ['p_dl_max','convergence']:cfg=replace(cfg,p_dl_max=float(task['parameter_value']))
    if task['figure']=='rho_si':cfg=replace(cfg,rho_si=float(task['parameter_value']))
    exp=raw['experiment'];fading=exp.get('fading',{})
    return make_scene_config(cfg,int(task['scene_seed']),np.asarray(exp['position_jitter'],dtype=float),
        randomize_fading=bool(fading.get('enabled',False)),fading_model=fading.get('model','scalar_rayleigh'))

def initial_pair(cfg,p,scheme):
    from geometry import build_geometry
    from baselines import build_fpa_upa_geometry
    from channels import build_channels
    from initialization import initialize_resources
    if scheme in ['proposed','b2']:
        theta=np.pi/4+2*np.pi*np.arange(cfg.m_uav)/cfg.m_uav
        e=np.column_stack([np.zeros(cfg.m_uav),cfg.cable_length*np.cos(theta),cfg.cable_length*np.sin(theta)])
        g=build_geometry(e,cfg,check_constraints=True)
    else:g=build_fpa_upa_geometry(cfg)
    r=initialize_resources(g,build_channels(g,cfg),cfg,np.random.default_rng(p['protocol']['resource_seed']))
    r.q_ul[:]=np.sqrt(float(p['protocol']['initial_ul_power_W']))
    return g,r

def metrics_values(g,r,cfg):
    from channels import build_channels
    from metrics import compute_performance
    perf=compute_performance(build_channels(g,cfg),r,cfg)
    dl,ul,s=map(float,[sum(perf.rate_dl),sum(perf.rate_ul),sum(perf.rate_s)])
    return dict(sum_rate_dl=dl,sum_rate_ul=ul,sum_rate_comm=dl+ul,sum_rate_sensing=s,weighted_sum_rate=float(perf.weighted_sum_rate))

def check_state(g,r,cfg,p,scheme,g0=None):
    from utils import check_resource_constraints
    from geometry import check_geometry_constraints
    from baselines import build_fpa_upa_geometry
    import ma_baseline as ma
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
    if figure=='tradeoff':return []
    raw=yaml.safe_load((Path(root)/p['base_config']).read_text(encoding='utf-8'));exp=raw['experiment']
    count=int(exp['n_mc'] if n_mc is None else n_mc)
    if count<1:raise ValueError('n_mc must be positive.')
    seeds=list(range(int(exp['scene_seed_start']),int(exp['scene_seed_start'])+count))
    if figure=='convergence':seeds=[int(p['convergence']['scene_seed'])];values=list(map(float,p['convergence']['p_dl_W']));schemes=['proposed']
    else:
        values=([10**((float(x)-30)/10) for x in p['sweeps']['p_dl_dbm']] if figure=='p_dl_max' else
                [10**(float(x)/10) for x in p['sweeps']['rho_si_db']] if figure=='rho_si' else list(map(float,p['sweeps']['loads'])))
        schemes=p['schemes']
    tasks=[dict(figure=figure,parameter_value=v,scene_seed=s,scheme=sc) for v in values for s in seeds for sc in schemes]
    if len({task_key(t) for t in tasks})!=len(tasks):raise ValueError('Duplicate task keys.')
    return tasks


def validate_inputs(root,p,figures):
    from config import load_config
    from tempfile import TemporaryDirectory
    manifest=read_json(ASSETS/'solver_manifest.json')
    for name,digest in manifest.items():
        f=Path(root)/name
        if not f.exists() or hashlib.sha256(f.read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=digest:raise RuntimeError('Solver missing/different from verified version: '+str(f))
    for name,digest in read_json(ASSETS/'inputs_manifest.json').items():
        if sha(ASSETS/name)!=digest:raise RuntimeError('Packaged input changed: '+name)
    needed=[]
    if any(x!='tradeoff' for x in figures):needed=[Path(root)/p['base_config']]
    if 'system_load' in figures:needed+=input_config_paths(root,p)[1:]
    for f in needed:
        if not f.exists():raise FileNotFoundError('Required ORIGINAL project config missing: '+str(f))
        cfg=load_config(f);raw=yaml.safe_load(f.read_text(encoding='utf-8'))
        if (cfg.m_uav,cfg.n_cable)!=(4,2) or not np.isclose(cfg.wavelength,.01):raise ValueError(str(f)+': expected M4/Nc2/wavelength0.01.')
        if not np.allclose(cfg.p_ul_max,.1) or cfg.cable_length!=1.:raise ValueError(str(f)+': expected ULmax0.1W and cable1m.')
        noise=3.981071705534969e-15
        if not np.isclose(cfg.sigma_bs2,noise,rtol=1e-7,atol=0) or not np.allclose(cfg.sigma_dl2,noise,rtol=1e-7,atol=0):raise ValueError(str(f)+': expected -114dBm noise.')
        xi=np.sqrt(4*np.pi*10)/cfg.wavelength
        if not np.allclose(cfg.xi_s,xi,rtol=1e-7,atol=0):raise ValueError(str(f)+': expected RCS10m2 definition.')
        if raw['experiment'].get('fading',{}).get('enabled',False):raise ValueError(str(f)+': expected pure LoS with fading disabled.')
        if f==Path(root)/p['base_config'] and (cfg.k_dl,cfg.j_ul,cfg.l_target,cfg.r_sensing)!=(5,5,5,4):raise ValueError('Base config must use K=J=L=5 and Rs=4.')
    if 'system_load' in figures:
        for n in p['sweeps']['loads']:
            cfg=load_config(Path(root)/p['load_configs'][int(n)])
            if (cfg.k_dl,cfg.j_ul,cfg.l_target)!=(int(n),)*3:raise ValueError('Load config count mismatch.')
    # Exact accepted tradeoff config + saved-state verification, no optimization.
    if 'tradeoff' in figures:
        with TemporaryDirectory() as t:
            f=Path(t)/'cfg.yaml';f.write_text(yaml.safe_dump(read_json(ASSETS/'tradeoff_config.json')))
            base=load_config(f)
        from stage8 import make_scene_config
        scene=make_scene_config(base,8001,np.asarray([.5,.5,0.]),randomize_fading=False)
        rows=tradeoff_rows(p)
        for row in rows:
            d=read_json(ASSETS/'tradeoff/records'/f"{row['scheme']}_omega_{float(row['omega_comm']):g}.json")
            g,r,h=load_state(ASSETS/'tradeoff'/d['state_file']);g0,r0,_=load_state(ASSETS/'tradeoff/initial_states'/f"{row['scheme']}.npz")
            if sha(ASSETS/'tradeoff'/d['state_file'])!=d['state_sha256'] or sha(ASSETS/'tradeoff/initial_states'/f"{row['scheme']}.npz")!=d['initialization_file_sha256']:raise RuntimeError('Frozen state linkage mismatch.')
            w=float(row['omega_comm']);cfg=replace(scene,weight_dl=np.full(5,w/10),weight_ul=np.full(5,w/10),weight_s=np.full(5,(1-w)/5))
            check_state(g,r,cfg,p,row['scheme'],g0)
            if not np.allclose(abs(r0.q_ul)**2,.00025,atol=1e-14,rtol=0):raise RuntimeError('Frozen initial power mismatch.')
            if abs(metrics_values(g,r,cfg)['weighted_sum_rate']-float(row['weighted_sum_rate']))>1e-9:raise RuntimeError('Frozen WSR mismatch.')
            delta=np.diff(h);required=2 if row['scheme'] in ['proposed','b6'] else 1
            if len(delta)<required or np.min(delta)<-1e-8 or np.max(abs(delta[-required:]))>1e-4 or not row['converged']:raise RuntimeError('Frozen history mismatch.')
    print('INPUT CHECK PASSED. Original objective arrays are preserved for each Stage8/load config.',flush=True)


def run_solver_call(cfg,p,task,g,r):
    import algorithm,baselines,inner_solver,outer_solver,ma_baseline as ma
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
        json_write(record,dict(row=row,cfg_sha256=fingerprint,effective_config=serial(asdict(cfg)),initial_sha256=sha(initial),state_file=state_name,state_sha256=sha(folder/state_name),absolute_stopping=True,same_parameter_scene_weight_only=True))
        print('DONE',task_key(task),'WSR',v['weighted_sum_rate'],'converged',converged,'steps',iterations,flush=True)
        if converged:return row
    return row


def collect_rows(out,tasks):
    rows=[]
    for task in tasks:
        f=Path(out)/'tasks'/task_key(task)/'record.json'
        if f.exists():rows.append(read_json(f)['row'])
    return rows

def tradeoff_rows(p):
    rows=[read_json(f)['row'] for f in (ASSETS/'tradeoff/records').glob('*.json')]
    expected={(sc,float(w)) for sc in p['schemes'] for w in p['tradeoff']['weights']}
    keys={(r['scheme'],float(r['omega_comm'])) for r in rows}
    if len(rows)!=len(expected) or keys!=expected:raise ValueError('Frozen tradeoff grid differs; do not silently reuse another grid.')
    return sorted(rows,key=lambda r:(float(r['omega_comm']),p['schemes'].index(r['scheme'])))

def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.grid':True,'grid.alpha':.22})
    return plt

def save_figure(fig,out,name):
    folder=Path(out)/'figures';folder.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf','svg']:fig.savefig(folder/f'{name}.{ext}',dpi=240,facecolor='white',bbox_inches='tight')


def plot_convergence(out,p,tasks):
    plt=plotting();fig,ax=plt.subplots(figsize=(8.4,5.6));allh=[]
    for i,task in enumerate(tasks):
        folder=Path(out)/'tasks'/task_key(task);d=read_json(folder/'record.json');_,_,h=load_state(folder/d['state_file']);allh.append(h)
        stop=len(h)-1;power=float(task['parameter_value']);style=STYLES['proposed']
        ax.plot(np.arange(len(h)),h,marker=style['marker'],color=POWER_COLORS[i],linestyle=POWER_LINESTYLES[i],markevery=max(1,len(h)//14),ms=4,lw=1.6,label=f'Proposed, $P_{{DL}}^{{max}}$={power:g} W (stop: {stop})')
        ax.scatter([stop],[h[-1]],color=POWER_COLORS[i],marker=style['marker'],s=32)
    ax.set_xlabel('Joint AO iteration');ax.set_ylabel('WSR (bit/s/Hz)');ax.legend(fontsize=9,loc='lower center',bbox_to_anchor=(.5,1.02));fig.suptitle('Proposed convergence at three DL power limits',y=.995)
    if p['convergence'].get('inset',True):
        # Same WSR units; actual last up-to-ten steps per curve, no extended flat tails.
        inset=ax.inset_axes([.51,.14,.46,.30])
        for i,h in enumerate(allh):
            start=max(0,len(h)-11);inset.plot(np.arange(start,len(h)),h[start:],color=POWER_COLORS[i],linestyle=POWER_LINESTYLES[i],marker=STYLES['proposed']['marker'],ms=2.5,markevery=max(1,(len(h)-start)//4))
        inset.set_title('Last 10 iterations of each curve',fontsize=8);inset.tick_params(labelsize=7)
    fig.tight_layout(rect=(0,0,1,.94));save_figure(fig,out,'01_proposed_convergence');plt.close(fig)
    history_rows=[]
    for task,h in zip(tasks,allh):history_rows.extend(dict(p_dl_W=task['parameter_value'],iteration=i,weighted_sum_rate=v) for i,v in enumerate(h))
    csv_write(Path(out)/'tables/convergence_histories.csv',history_rows,['p_dl_W','iteration','weighted_sum_rate'])


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


def plot_tradeoff(out,p):
    rows=tradeoff_rows(p);dest=Path(out)/'tradeoff'
    if not dest.exists():shutil.copytree(ASSETS/'tradeoff',dest)
    fields=list(rows[0]);csv_write(Path(out)/'tables/tradeoff_all_points.csv',rows,fields)
    plt=plotting();fig,ax=plt.subplots(figsize=(8.4,6.1));selected=[]
    for sc in p['schemes']:
        a=[r for r in rows if r['scheme']==sc];front=[]
        for r in a:
            x,y=float(r['sum_rate_comm']),float(r['sum_rate_sensing'])
            if not any(float(q['sum_rate_comm'])>=x-1e-10 and float(q['sum_rate_sensing'])>=y-1e-10 and (float(q['sum_rate_comm'])>x+1e-10 or float(q['sum_rate_sensing'])>y+1e-10) for q in a):front.append(r)
        front.sort(key=lambda r:float(r['sum_rate_comm']));selected+=front;style=STYLES[sc]
        ax.scatter([float(r['sum_rate_comm']) for r in a],[float(r['sum_rate_sensing']) for r in a],marker=style['marker'],color=style['color'],s=20,alpha=.18)
        ax.plot([float(r['sum_rate_comm']) for r in front],[float(r['sum_rate_sensing']) for r in front],marker=style['marker'],color=style['color'],lw=1.8,ms=5,label=style['label'])
    ax.set_xlabel('Sum communication rate (bit/s/Hz)');ax.set_ylabel('Sum sensing rate (bit/s/Hz)');ax.legend(loc='upper right',fontsize=9)
    ax.set_title('Communication-sensing tradeoff | scene 8001')
    fig.text(.11,.035,'Each scheme: own nondominated samples. Faint markers: all results. No smoothing.',fontsize=8)
    fig.text(.11,.012,'Lines guide the eye; no claim of feasible interpolation or a global Pareto boundary.',fontsize=8)
    fig.subplots_adjust(left=.11,right=.98,bottom=.16,top=.92);save_figure(fig,out,'05_communication_sensing_tradeoff');plt.close(fig)
    csv_write(Path(out)/'tables/tradeoff_nondominated.csv',selected,fields)
