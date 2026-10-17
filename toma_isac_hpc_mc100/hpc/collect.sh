#!/usr/bin/env bash
#SBATCH --job-name=toma_collect
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${HPC_PYTHON:?}" "${HPC_OUTPUT:?}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/toma_mpl_${SLURM_JOB_ID}"
"$HPC_PYTHON" -u run_all_figures_unified.py --n-mc 100 --collect-only --output "$HPC_OUTPUT"
