#!/bin/bash
#SBATCH --job-name=toma_neighbor_scene
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=50
#SBATCH --mem=48G
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --time=48:00:00
set -euo pipefail

PROJECT_ROOT="/home/lyq00008339/Lihuang_Li/toma"
FIGURE_DIR="${PROJECT_ROOT}/05_communication_sensing_tradeoff"
CONDA_SH="/home/lyq00008339/miniconda3/etc/profile.d/conda.sh"
if [[ ! -f "$CONDA_SH" ]]; then echo "Missing conda init: $CONDA_SH" >&2; exit 1; fi
. "$CONDA_SH"
conda activate "${TARGET_CONDA_ENV:-JM_CVX}"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/toma_neighbor_${SLURM_JOB_ID:-check}"
PYTHON_BIN="${HPC_PYTHON:-python}"
SOURCE_RUN="${HPC_SOURCE:-${FIGURE_DIR}/results/run_33570}"
OUTPUT_RUN="${HPC_REFINE_OUTPUT:-${FIGURE_DIR}/results/neighbor_refine_scene8001_${SLURM_JOB_ID}}"
ENTRY="${FIGURE_DIR}/refine_neighbor_scene.py"
"$PYTHON_BIN" -u "$ENTRY" --project-root "$PROJECT_ROOT" --source "$SOURCE_RUN" --output "$OUTPUT_RUN" --check-only
"$PYTHON_BIN" -u "$ENTRY" --project-root "$PROJECT_ROOT" --source "$SOURCE_RUN" --output "$OUTPUT_RUN" --workers 50 --max-calls 20
