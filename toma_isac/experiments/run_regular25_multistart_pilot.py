"""同一组规则/扰动几何起点，仅将初始UL功率改为每用户25mW。"""
from __future__ import annotations
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'1')
import argparse,csv,json,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(Path(__file__).resolve().parent)]
import run_regular025_multistart_pilot as pilot
engine=pilot.engine
LOW_SETTINGS=pilot.SETTINGS.copy()
pilot.SETTINGS={**LOW_SETTINGS,'initial_ul_power_W':.025}
SCHEMES=[('proposed',5),('b6',3),('b2',1),('b3',1)]

def protocol():
 p=pilot.protocol();p['protocol']['initial_ul_power_W']=.025;return p

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--workers',type=int,default=2);ap.add_argument('--max-calls',type=int,default=2);ap.add_argument('--check-only',action='store_true');ap.add_argument('--output',default='results/regular25_multistart_pilot')
 a=ap.parse_args();p=protocol();out=(ROOT/a.output).resolve()
 if a.workers<1 or a.max_calls<1:ap.error('workers/max-calls must be positive')
 low=pilot.protocol();jobs=[]
 for dbm in [20,25,30]:
  for sc,n in SCHEMES:
   task=dict(figure='p_dl_max',parameter_value=10**((dbm-30)/10),scene_seed=8001,scheme=sc)
   cfg=engine.config_for_task(ROOT,p,task);lc=engine.config_for_task(ROOT,low,task)
   for start in range(n):
    g,r=pilot.pair(cfg,p,sc,start)
    with patch.object(pilot,'SETTINGS',LOW_SETTINGS):lg,lr=pilot.pair(lc,low,sc,start)
    for name in ['endpoints','tx_positions','rx_positions']:assert np.array_equal(getattr(g,name),getattr(lg,name))
    for name in ['q_matrix','b_ul','u_s']:assert np.allclose(getattr(r,name),getattr(lr,name),rtol=0,atol=1e-14)
    assert np.allclose(r.q_ul,10*lr.q_ul,rtol=0,atol=1e-14)
    engine.check_state(g,r,cfg,p,sc)
    jobs.append((str(out),p,task,start,a.max_calls))
 print('MATCHED INITIALIZATION CHECK PASSED: 30 starts, only UL initial amplitude differs by x10 (power x100).',flush=True)
 if a.check_only:return
 out.mkdir(parents=True,exist_ok=True)
 sources={name:engine.sha(ROOT/name) for name in ['algorithm.py','baselines.py','ma_baseline.py','initialization.py','inner_solver.py','outer_solver.py','channels.py','metrics.py','geometry.py','config.py','utils.py','state.py']}
 manifest=dict(settings=pilot.SETTINGS,base_config_sha256=engine.sha(pilot.ASSETS/'base.yaml'),source_sha256=sources,script_sha256=engine.sha(Path(__file__)),low_protocol_script_sha256=engine.sha(Path(pilot.__file__)))
 mp=out/'manifest.json'
 if mp.exists() and engine.read_json(mp)!=manifest:raise RuntimeError('Manifest changed; choose a new output directory')
 engine.json_write(mp,manifest);rows=[];errors=[]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  fs={pool.submit(pilot.run_job,j):j for j in jobs}
  for f in as_completed(fs):
   try:rows.append(f.result())
   except Exception as e:errors.append(dict(task=fs[f][2],start_id=fs[f][3],error=repr(e)));print('ERROR',repr(e),flush=True)
 rows.sort(key=lambda r:(r['parameter_value'],r['scheme'],r['start_id']))
 if rows:engine.csv_write(out/'all_starts.csv',rows,list(rows[0]))
 best=[]
 for dbm in [20,25,30]:
  v=10**((dbm-30)/10)
  for sc,n in SCHEMES:
   group=[r for r in rows if r['scheme']==sc and np.isclose(r['parameter_value'],v)]
   if len(group)==n and all(r['converged'] for r in group):best.append(max(group,key=lambda r:r['weighted_sum_rate']))
 if best:engine.csv_write(out/'best_starts.csv',best,list(best[0]))
 complete=not errors and len(rows)==30 and all(r['converged'] for r in rows)
 engine.json_write(out/'completion.json',dict(complete=complete,expected_starts=30,returned_records=len(rows),converged_starts=sum(r['converged'] for r in rows),errors=errors))
 if not complete:raise SystemExit('Incomplete: rerun same command to continue.')
if __name__=='__main__':main()
