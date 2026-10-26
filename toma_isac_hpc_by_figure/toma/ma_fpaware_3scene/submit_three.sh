#!/bin/bash
#SBATCH --job-name=ma_fpaware_three
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH -o %j.out
#SBATCH -e %j.err
set -euo pipefail
. "${CONDA_SH:-/home/lyq00008339/miniconda3/etc/profile.d/conda.sh}"
conda activate "${TARGET_CONDA_ENV:-JM_CVX}"
cd /home/lyq00008339/Lihuang_Li/toma/ma_fpaware_3scene
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
OUT="${HPC_OUTPUT:-$PWD/results/three_${SLURM_JOB_ID}}"
mkdir -p "$OUT/logs"
python -u run_three.py --output "$OUT" --check-only
python -u run_three.py --output "$OUT" --workers 3 2>&1 | tee -a "$OUT/logs/console.log"
