#!/bin/bash
#SBATCH --job-name=toma_saved_preview
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH -o %j.out
#SBATCH -e %j.err
set -euo pipefail
. "${CONDA_SH:-/home/lyq00008339/miniconda3/etc/profile.d/conda.sh}"
conda activate "${TARGET_CONDA_ENV:-JM_CVX}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
TOOLS="/home/lyq00008339/Lihuang_Li/toma/preview_tools"
SOURCE="${HPC_SOURCE:-/home/lyq00008339/Lihuang_Li/toma/test/results/tradeoff_mc10_33614}"
OUTPUT="${HPC_PREVIEW_OUTPUT:-${SOURCE}_preview_${SLURM_JOB_ID}}"
python -u "$TOOLS/plot_saved_tradeoff.py" --source "$SOURCE" --output "$OUTPUT"
