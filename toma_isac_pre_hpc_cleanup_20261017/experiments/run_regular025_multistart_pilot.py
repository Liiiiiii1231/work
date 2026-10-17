"""规则初始化作为第1起点，0.25mW上行初值，绝对容差1e-4的小范围多起点。"""
from __future__ import annotations
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'1')
import sys,json,csv,hashlib,argparse,shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from unittest.mock import patch
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
import paper_figures_unified as engine
ASSETS=Path(__file__).resolve().parent/'regular025_multistart_assets'
SETTINGS={'scene_seed':8001,'power_dbm':[20,25,30],'proposed_seeds':[2026,2027,2028,2029],'ma_seeds':[2026,2027],'toma_direction_noise_std':.15,'ma_jitter_fraction':.5,'initial_ul_power_W':.00025,'tol_inner':1e-4,'tol_outer':1e-4,'resource_seed':2042}

def protocol():
 p=yaml.safe_load((ASSETS/'protocol.yaml').read_text(encoding='utf-8'))
 p['base_config']=str(ASSETS/'base.yaml');return p

def pair(cfg,p,scheme,start):
 from geometry import build_geometry
 from channels import build_channels
 from initialization import initialize_resources
 import ma_baseline as ma
 # 第一个起点严格保持已冻结的规则ToMA/UPA初值。
 g,r=engine.initial_pair(cfg,p,scheme)
 if start==0:return g,r
 seeds=SETTINGS['proposed_seeds'] if scheme=='proposed' else SETTINGS['ma_seeds']
 rng=np.random.default_rng(seeds[start-1])
 if scheme=='proposed':
  anchor=g.endpoints/cfg.cable_length
  for _ in range(cfg.max_geometry_trials):
   directions=anchor+SETTINGS['toma_direction_noise_std']*rng.normal(size=anchor.shape)
   e=cfg.cable_length*directions/np.linalg.norm(directions,axis=1,keepdims=True)
   try:g=build_geometry(e,cfg,check_constraints=True);break
   except ValueError:continue
  else:raise RuntimeError('Cannot generate feasible perturbed ToMA')
 elif scheme=='b6':
  g=ma._randomize_initial_ma_geometry(g,np.mean(g.tx_positions,axis=0),np.mean(g.rx_positions,axis=0),p['protocol']['ma_region_size_m'],p['protocol']['ma_min_spacing_m'],SETTINGS['ma_jitter_fraction'],cfg,rng)
 else:raise ValueError('Fixed arrays do not vary geometry')
 r=initialize_resources(g,build_channels(g,cfg),cfg,np.random.default_rng(p['protocol']['resource_seed']))
 r.q_ul[:]=np.sqrt(float(p['protocol']['initial_ul_power_W']))
 return g,r

def run_job(args):
 out,p,task,start,max_calls=args
 # The protocol is pickled with every job, including on Windows spawn.
 # 保存原函数，避免对自身递归调用。
 original=engine.initial_pair
 def initial(cfg,protocol,scheme):
  with patch.object(engine,'initial_pair',original):return pair(cfg,protocol,scheme,start)
 with patch.object(engine,'initial_pair',initial):
  row=engine.run_task(str(ROOT),str(Path(out)/f'candidate_{start}'),p,task,max_calls)
 return dict(row,start_id=start,origin='new_optimization')

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--workers',type=int,default=2);ap.add_argument('--max-calls',type=int,default=2);ap.add_argument('--check-only',action='store_true');ap.add_argument('--smoke',action='store_true');ap.add_argument('--output',default='results/regular025_multistart_pilot')
 a=ap.parse_args();p=protocol();out=(ROOT/a.output).resolve()
 if a.workers<1 or a.max_calls<1:ap.error('workers/max-calls must be positive')
 if a.smoke:
  out=out.with_name(out.name+'_smoke');p['protocol'].update(max_inner_iter=2,max_outer_iter=2)
 refs=[]
 with (ASSETS/'regular_single_start_scene8001.csv').open(encoding='utf-8-sig') as f:
  for raw in csv.DictReader(f):
   r={k:(v if k in ('figure','scheme') else v=='True' if k=='converged' else float(v)) for k,v in raw.items()}
   r['scene_seed']=int(r['scene_seed']);r.update(start_id=0,origin='existing_single_start_table');refs.append(r)
 assert len(refs)==12
 jobs=[]
 for dbm in SETTINGS['power_dbm']:
  for sc,n in [('proposed',5),('b6',3),('b2',1),('b3',1)]:
   task=dict(figure='p_dl_max',parameter_value=10**((dbm-30)/10),scene_seed=8001,scheme=sc)
   cfg=engine.config_for_task(ROOT,p,task)
   for start in range(n):
    g,r=pair(cfg,p,sc,start);engine.check_state(g,r,cfg,p,sc)
    assert np.allclose(abs(r.q_ul)**2,.00025,rtol=0,atol=1e-14)
    if start:jobs.append((str(out),p,task,start,a.max_calls))
 print('INITIALIZATION CHECK PASSED: 30 total starts, 12 existing table records + 18 new optimizations.',flush=True)
 if a.check_only:return
 if a.smoke:jobs=[next(j for j in jobs if j[2]['scheme']==s) for s in ('proposed','b6')]
 out.mkdir(parents=True,exist_ok=True)
 sources={name:engine.sha(ROOT/name) for name in ['algorithm.py','baselines.py','ma_baseline.py','initialization.py','inner_solver.py','outer_solver.py','channels.py','metrics.py','geometry.py','config.py','utils.py','state.py']}
 manifest=dict(settings=SETTINGS,base_config_sha256=engine.sha(ASSETS/'base.yaml'),protocol_sha256=engine.sha(ASSETS/'protocol.yaml'),reference_sha256=engine.sha(ASSETS/'regular_single_start_scene8001.csv'),script_sha256=engine.sha(Path(__file__)),source_sha256=sources,smoke=a.smoke)
 mp=out/'manifest.json'
 if mp.exists() and engine.read_json(mp)!=manifest:raise RuntimeError('Manifest changed: select a new output directory')
 engine.json_write(mp,manifest)
 rows=refs.copy();errors=[]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  fs={pool.submit(run_job,j):j for j in jobs}
  for f in as_completed(fs):
   try:rows.append(f.result())
   except Exception as e:errors.append(dict(task=fs[f][2],start_id=fs[f][3],error=repr(e)));print('ERROR',repr(e),flush=True)
 rows.sort(key=lambda r:(r['parameter_value'],r['scheme'],r['start_id']))
 engine.csv_write(out/'all_starts.csv',rows,list(rows[0]))
 best=[];comparison=[]
 for dbm in SETTINGS['power_dbm']:
  v=10**((dbm-30)/10)
  for sc,n in [('proposed',5),('b6',3),('b2',1),('b3',1)]:
   group=[r for r in rows if r['scheme']==sc and np.isclose(r['parameter_value'],v)]
   if len(group)!=n or not all(r['converged'] for r in group):continue
   winner=max(group,key=lambda r:r['weighted_sum_rate']);base=next(r for r in group if r['start_id']==0)
   best.append(winner);comparison.append(dict(scheme=sc,p_dl_dbm=dbm,single_start_wsr=base['weighted_sum_rate'],multistart_wsr=winner['weighted_sum_rate'],gain=winner['weighted_sum_rate']-base['weighted_sum_rate'],best_start_id=winner['start_id']))
 engine.csv_write(out/'best_starts.csv',best,list(rows[0]));engine.csv_write(out/'comparison.csv',comparison,list(comparison[0]))
 complete=not a.smoke and not errors and len(rows)==30 and all(r['converged'] for r in rows)
 engine.json_write(out/'completion.json',dict(complete=complete,smoke=a.smoke,expected_starts=30,new_optimizations=18,returned_records=len(rows),converged_starts=sum(r['converged'] for r in rows),errors=errors))
 if complete:
  plt=engine.plotting();fig,ax=plt.subplots(figsize=(7.5,4.8))
  for sc,style in engine.STYLES.items():
   rr=sorted([r for r in comparison if r['scheme']==sc],key=lambda r:r['p_dl_dbm']);x=[r['p_dl_dbm'] for r in rr]
   ax.plot(x,[r['multistart_wsr'] for r in rr],color=style['color'],marker=style['marker'],label=style['label']+' (multi-start)' if sc in ['proposed','b6'] else style['label'])
   if sc in ['proposed','b6']:ax.plot(x,[r['single_start_wsr'] for r in rr],color=style['color'],marker=style['marker'],ls='--',alpha=.45,label=style['label']+' (single start)')
  ax.set(xlabel='Maximum downlink power (dBm)',ylabel='WSR (bit/s/Hz)',xticks=[20,25,30],title='Scene 8001 | regular anchor + perturbations | UL init 0.25 mW')
  ax.grid(alpha=.25);ax.legend(fontsize=8,loc='best');fig.tight_layout()
  for ext in ['png','pdf']:fig.savefig(out/f'regular025_multistart_comparison.{ext}',dpi=220)
  plt.close(fig)
 if not complete and not a.smoke:raise SystemExit('Incomplete: rerun same command to continue.')
if __name__=='__main__':main()
