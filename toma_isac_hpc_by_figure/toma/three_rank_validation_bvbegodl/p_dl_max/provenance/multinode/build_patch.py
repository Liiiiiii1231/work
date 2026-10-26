"""Generate the four three-node submission scripts and an upload-only patch."""
from pathlib import Path
import hashlib,json,zipfile

HERE=Path(__file__).resolve().parent
TOMA=HERE.parent
FIGURES=[('02_wsr_vs_p_dl_max','p_dl_max','submit_p_dl_max.sh'),
    ('03_wsr_vs_rho_si','rho_si','submit_rho_si.sh'),
    ('04_wsr_vs_system_load','system_load','submit_system_load.sh'),
    ('05_communication_sensing_tradeoff','tradeoff','submit_tradeoff.sh')]

TEMPLATE='''#!/bin/bash
#SBATCH --job-name=three_{figure}
#SBATCH -p cpu
#SBATCH -N 3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH -o %j.out
#SBATCH -e %j.err
set -euo pipefail
TOMA_ROOT="${{TOMA_ROOT:-/home/lyq00008339/Lihuang_Li/toma}}"
SUPPORT="$TOMA_ROOT/{support}"
MULTINODE="$TOMA_ROOT/multinode_support_fpaccel_o4_i5"
FIGURE_DIR="$TOMA_ROOT/{directory}"
SELECTION="${{HPC_SELECTION:-$TOMA_ROOT/results/final_initial_geometry_fpaccel_o4_i5_s8005}}"
for required in "$MULTINODE/run_multinode.py" "$SUPPORT/finalize_run.py" "$SELECTION/frozen_initial_geometry.json"; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: missing $required; upload the patch and wait for 01 FROZEN GEOMETRY READY" >&2
        exit 1
    fi
done
. "${{CONDA_SH:-/home/lyq00008339/miniconda3/etc/profile.d/conda.sh}}"
conda activate "${{TARGET_CONDA_ENV:-JM_CVX}}"
cd "$FIGURE_DIR"
PYTHON_BIN="${{HPC_PYTHON:-python}}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${{TMPDIR:-/tmp}}/toma_3nodes_${{SLURM_JOB_ID}}"
export SLURM_EXPORT_ENV=ALL
if [[ "${{SLURM_JOB_NUM_NODES:?Submit through Slurm}}" != 3 || "${{SLURM_NTASKS:?}}" != 3 || "${{SLURM_CPUS_PER_TASK:?}}" != 56 ]]; then
    echo 'ERROR: expected 3 nodes, one Slurm task per node and 56 CPUs per task' >&2
    exit 1
fi
OUTPUT_DIR="${{HPC_OUTPUT:-$FIGURE_DIR/results/final_3nodes_fpaccel_o4_i5_${{SLURM_JOB_ID}}}}"
if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo 'ERROR: output is not empty; use a fresh job/output directory' >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR/logs"
finalize() {{
    local run_status=$?
    trap - EXIT
    set +e
    for ext in out err; do
        if [[ -f "$FIGURE_DIR/${{SLURM_JOB_ID}}.$ext" ]]; then
            cp "$FIGURE_DIR/${{SLURM_JOB_ID}}.$ext" "$OUTPUT_DIR/logs/slurm_snapshot.$ext"
        fi
    done
    "$PYTHON_BIN" -u "$SUPPORT/finalize_run.py" --output "$OUTPUT_DIR" --project "$SUPPORT" --run-exit-code "$run_status"
    local pack_status=$?
{combine}
    if ((run_status)); then exit "$run_status"; fi
    exit "$pack_status"
}}
trap finalize EXIT
"$PYTHON_BIN" -u "$MULTINODE/run_multinode.py" --figure {figure} --output "$OUTPUT_DIR" --selection-dir "$SELECTION" --workers-per-node 56 --check-only
# One leader is launched ON EACH NODE; each owns a local pool of 56 processes.
srun --mpi=none --nodes=3 --ntasks=3 --ntasks-per-node=1 --cpus-per-task=56 --cpu-bind=cores --kill-on-bad-exit=1 --label --export=ALL \\
    "$PYTHON_BIN" -u "$MULTINODE/run_multinode.py" --figure {figure} --output "$OUTPUT_DIR" --selection-dir "$SELECTION" --workers-per-node 56 \\
    2>&1 | tee "$OUTPUT_DIR/logs/console.log"
'''

def main():
    for directory,figure,name in FIGURES:
        support='tradeoff_support_fpaccel_o4_i5_s8005' if figure=='tradeoff' else 'final_support_fpaccel_o4_i5_s8005'
        combine='' if figure=='tradeoff' else '''    if ((run_status == 0 && pack_status == 0)); then
        "$PYTHON_BIN" -u "$SUPPORT/collect_four.py" --toma-root "$TOMA_ROOT" --selection-dir "$SELECTION"
        pack_status=$?
    fi'''
        text=TEMPLATE.format(figure=figure,support=support,directory=directory,combine=combine)
        (TOMA/directory/name).write_text(text,encoding='utf-8',newline='\n')
    current=TOMA/'README_CURRENT_FPACCEL.md'
    text=current.read_text(encoding='utf-8')
    text=text.replace('每图独立申请 1 节点、56 核、48GB、48 小时，进程内 BLAS 单线程；新输出目录带 `fpaccel_o4_i5` 标识。',
        '01 保持 1 节点、56 CPU；02–05 每图改为 3 节点、每节点 56 CPU 和 48GB、48 小时，通过 srun 跨节点执行。进程内 BLAS 单线程，输出目录带 `3nodes_fpaccel_o4_i5` 标识。详见 multinode_support_fpaccel_o4_i5/README.md。')
    current.write_text(text,encoding='utf-8',newline='\n')
    changed=[f'{d}/{name}' for d,_,name in FIGURES]+['README_CURRENT_FPACCEL.md']
    changed.extend(p.relative_to(TOMA).as_posix() for p in HERE.iterdir()
        if p.is_file() and p.suffix in ('.py','.json','.md') and p.name!='patch_manifest.json')
    hashes={rel:hashlib.sha256((TOMA/rel).read_bytes()).hexdigest() for rel in sorted(changed)}
    manifest=dict(files=hashes,replace_figures=['02','03','04','05'],nodes_per_figure=3,
        cpus_per_node=56,total_workers_per_figure=168,
        original_solver_and_screening_sources_unchanged=True,
        required_existing_supports=['final_support_fpaccel_o4_i5_s8005','tradeoff_support_fpaccel_o4_i5_s8005'])
    (HERE/'patch_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    archive=TOMA.parent/'toma_02_05_three_nodes_patch.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for rel in sorted(hashes):z.write(TOMA/rel,'toma/'+rel)
        z.write(HERE/'patch_manifest.json','toma/multinode_support_fpaccel_o4_i5/patch_manifest.json')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert not any('/results/' in n or '__pycache__' in n or '01_proposed_convergence/' in n for n in z.namelist())
    # Keep the optional full-upload manifest consistent with these current entries.
    full=TOMA/'UPLOAD_FPACCEL_O4_I5.json'
    if full.exists():
        d=json.loads(full.read_text(encoding='utf-8'))
        if HERE.name not in d['folders']:d['folders'].append(HERE.name)
        for rel in list(hashes)+['multinode_support_fpaccel_o4_i5/patch_manifest.json']:
            d['files'][rel]=hashlib.sha256((TOMA/rel).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
        full.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(archive,'files',len(hashes)+1,'bytes',archive.stat().st_size)

if __name__=='__main__':main()
