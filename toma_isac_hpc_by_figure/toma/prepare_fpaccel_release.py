"""Prepare the isolated paper-figure release; never edits shared/old solvers."""
from pathlib import Path
from copy import deepcopy
import json,shutil,hashlib,sys,importlib.util

ROOT=Path(__file__).resolve().parent
FINAL='final_support_fpaccel_o4_i5_s8005'
TRADE='tradeoff_support_fpaccel_o4_i5_s8005'
SELECTION='final_initial_geometry_fpaccel_o4_i5_s8005'
POLICY=dict(method='projected_squared_extrapolation',max_extrapolation=64.,
    consecutive_small_blocks=2,scope='main_resource_solves_all_schemes',candidate_solver='plain_fp_max_2')

def write(path,text):path.write_text(text,encoding='utf-8',newline='\n')
def json_write(path,data):write(path,json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def digest(path):return hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
def update_tolerances(data):
    if isinstance(data,dict):
        for k,v in data.items():
            if k=='tol_inner':data[k]=1e-5
            elif k=='tol_outer':data[k]=1e-4
            else:update_tolerances(v)
    elif isinstance(data,list):
        for v in data:update_tolerances(v)

def main():
    current=(ROOT/'README_CURRENT_FPACCEL.md').read_text(encoding='utf-8')
    for support in [FINAL,TRADE]:write(ROOT/support/'README.md',current)
    write(ROOT/'05_communication_sensing_tradeoff'/'README_FINAL_TRADEOFF.md',current)
    write(ROOT/'README.md','# ToMA/MA 论文仿真工程\n\n当前参数、上传及提交说明见 [README_CURRENT_FPACCEL.md](README_CURRENT_FPACCEL.md)。\n\n当前：主资源 FP 外推，内层 1e-5 连续两块，外层绝对 1e-4 连续两轮。先提交 01 生成新版冻结初始几何，再提交其他图。旧支持目录及旧结果保留；不自动续算预算截止任务。\n')
    for name in ['README_CURRENT_S8005.md','README_FINAL_1200_1500.md']:
        path=ROOT/name;text=path.read_text(encoding='utf-8')
        if not text.startswith('> 历史版本说明。'):
            write(path,'> 历史版本说明。当前请阅读 README_CURRENT_FPACCEL.md。\n\n'+text)
    write(ROOT/'submit_all_other_figures56.sh','''#!/bin/bash
# Run with bash on the login node after the new geometry is frozen.
set -euo pipefail
TOMA_ROOT="${TOMA_ROOT:-/home/lyq00008339/Lihuang_Li/toma}"
SELECTION="${HPC_SELECTION:-$TOMA_ROOT/results/final_initial_geometry_fpaccel_o4_i5_s8005}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo 'Use bash on the login node; each figure requests its own 56-core job.' >&2
    exit 1
fi
if [[ ! -f "$SELECTION/frozen_initial_geometry.json" ]]; then
    echo "Wait for 01 FROZEN GEOMETRY READY: $SELECTION/frozen_initial_geometry.json" >&2
    exit 1
fi
for script in 02_wsr_vs_p_dl_max/submit_p_dl_max.sh 03_wsr_vs_rho_si/submit_rho_si.sh 04_wsr_vs_system_load/submit_system_load.sh 05_communication_sensing_tradeoff/submit_tradeoff.sh; do
    sbatch "$TOMA_ROOT/$script"
done
''')
    for support,name in [(FINAL,'final_config.json'),(TRADE,'tradeoff_final_config.json')]:
        path=ROOT/support/name;b=json.loads(path.read_text(encoding='utf-8'))
        update_tolerances(b)
        b['protocol']['protocol']['resource_solver']=POLICY.copy()
        b['protocol']['protocol']['stopping_measure']='absolute_wsr_increment'
        b['protocol']['protocol']['outer_consecutive']=2
        b['protocol']['protocol']['inner_consecutive']=2
        json_write(path,b)
    for name in ['figure_engine.py','fp_acceleration.py','accelerated_adapter.py']:
        shutil.copy2(ROOT/FINAL/name,ROOT/TRADE/name)
    for support,entry in [(FINAL,'run_final.py'),(TRADE,'run_tradeoff_final.py')]:
        path=ROOT/support/entry;text=path.read_text(encoding='utf-8')
        text=text.replace('final_initial_geometry_i1200_o1500_s8005',SELECTION)
        write(path,text)
    # Entrypoints and submissions now select the new immutable version paths.
    for folder in sorted(ROOT.glob('0[1-5]_*')):
        for path in folder.glob('*'):
            if path.suffix not in ('.py','.sh'):continue
            text=path.read_text(encoding='utf-8')
            text=text.replace('final_support_i1200_o1500_s8005',FINAL)
            text=text.replace('tradeoff_support_i1200_o1500_s8005',TRADE)
            text=text.replace('final_initial_geometry_i1200_o1500_s8005',SELECTION)
            text=text.replace('manual_final','manual_fpaccel_o4_i5')
            text=text.replace('results/final_${SLURM_JOB_ID}','results/final_fpaccel_o4_i5_${SLURM_JOB_ID}')
            text=text.replace('results/final_tradeoff_${SLURM_JOB_ID}','results/final_tradeoff_fpaccel_o4_i5_${SLURM_JOB_ID}')
            text=text.replace('TOMA_ROOT="/home/lyq00008339/Lihuang_Li/toma"',
                'TOMA_ROOT="${TOMA_ROOT:-/home/lyq00008339/Lihuang_Li/toma}"')
            write(path,text)
    shutil.copy2(ROOT/TRADE/'tradeoff_final_config.json',ROOT/'05_communication_sensing_tradeoff'/'tradeoff_config.json')
    # Freeze identities must be regenerated after the runtime files are final.
    sys.path[:0]=[str(ROOT/FINAL),str(ROOT)]
    spec=importlib.util.spec_from_file_location('fpaccel_release_runner',ROOT/FINAL/'run_final.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    b=json.loads((ROOT/FINAL/'final_config.json').read_text(encoding='utf-8'))
    p=deepcopy(b['protocol']);p['_models']=b['models']
    json_write(ROOT/TRADE/'reference_screening_identity.json',runner.source_identity(p))
    folders=[p.name for p in ROOT.glob('0[1-5]_*')]+[FINAL,TRADE]
    files={}
    for folder in folders:
        for path in (ROOT/folder).iterdir():
            if path.is_file() and path.suffix in ('.py','.sh','.json','.md','.csv'):
                files[path.relative_to(ROOT).as_posix()]=digest(path)
    for name in ['README_CURRENT_FPACCEL.md','README.md','submit_all_other_figures56.sh']:
        if (ROOT/name).is_file():files[name]=digest(ROOT/name)
    json_write(ROOT/'UPLOAD_FPACCEL_O4_I5.json',dict(platform_root='/home/lyq00008339/Lihuang_Li/toma',
        folders=folders,files=files,shared_modules_unchanged=True,
        selection_directory='results/'+SELECTION,
        note='Upload listed source/config files only. Preserve old results/support versions; run 01 to screen new initial geometries.'))
    print('Prepared',FINAL,TRADE,'new selection',SELECTION)

if __name__=='__main__':main()
