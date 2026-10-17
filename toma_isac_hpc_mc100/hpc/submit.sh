#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${HPC_PYTHON:?Set an absolute executable such as /home/account/miniconda3/envs/env/bin/python}"
[[ "$HPC_PYTHON" = /* && -x "$HPC_PYTHON" ]] || { echo 'HPC_PYTHON must be an existing absolute executable'; exit 1; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
export HPC_OUTPUT="${HPC_OUTPUT:-results/mc100_$(date +%Y%m%d_%H%M%S)}"
mkdir -p logs
"$HPC_PYTHON" run_all_figures_unified.py --check-only --n-mc 100
job_id=$(sbatch --parsable --export=ALL hpc/run_array.sh)
job_id=${job_id%%;*}
printf 'Array job: %s\nOutput: %s\n' "$job_id" "$HPC_OUTPUT"
sbatch --dependency="afterok:$job_id" --export=ALL hpc/collect.sh
