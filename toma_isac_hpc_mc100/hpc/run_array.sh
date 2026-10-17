#!/usr/bin/env bash
#SBATCH --job-name=toma_mc100
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-9%10
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
: "${HPC_PYTHON:?Set HPC_PYTHON to the absolute cluster Python executable}"
: "${HPC_OUTPUT:?Use hpc/submit.sh to set an output directory}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/toma_mpl_${SLURM_JOB_ID}"
batch_index="${SLURM_ARRAY_TASK_ID:?Missing array batch index}"
[[ "$batch_index" =~ ^[0-9]$ ]] || { echo 'Batch index must be 0 through 9'; exit 1; }
first_scene=$((batch_index * 10))
last_scene=$((first_scene + 9))
failed=0
for ((scene_index=first_scene; scene_index<=last_scene; scene_index++)); do
  printf 'Batch %s: starting scene index %s (seed %s)\n' "$batch_index" "$scene_index" "$((8001 + scene_index))"
  if "$HPC_PYTHON" -u run_all_figures_unified.py --n-mc 100 --workers 1 \
      --scene-index "$scene_index" --output "$HPC_OUTPUT"; then
    printf 'Scene index %s complete\n' "$scene_index"
  else
    printf 'Scene index %s failed or incomplete; continuing remaining scenes\n' "$scene_index" >&2
    failed=1
  fi
done
exit "$failed"
