#!/bin/bash
#SBATCH --job-name=final_tradeoff20
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH -o %j.out
#SBATCH -e %j.err
set -euo pipefail
TOMA_ROOT="${TOMA_ROOT:-/home/lyq00008339/Lihuang_Li/toma}"
SUPPORT="$TOMA_ROOT/tradeoff_support_i1200_o1500"
FIGURE_DIR="$TOMA_ROOT/05_communication_sensing_tradeoff"
SELECTION="${HPC_SELECTION:-$TOMA_ROOT/results/final_initial_geometry_i1200_o1500}"
for required in "$SUPPORT/run_tradeoff_final.py" "$SUPPORT/finalize_run.py" "$SELECTION/frozen_initial_geometry.json"; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: missing $required; upload the complete support folder and wait for 01 FROZEN GEOMETRY READY" >&2
        exit 1
    fi
done
. "${CONDA_SH:-/home/lyq00008339/miniconda3/etc/profile.d/conda.sh}"
conda activate "${TARGET_CONDA_ENV:-JM_CVX}"
cd "$FIGURE_DIR"
PYTHON_BIN="${HPC_PYTHON:-python}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/toma_tradeoff_${SLURM_JOB_ID}"
WORKERS="${SLURM_CPUS_PER_TASK:?Submit through Slurm}"
OUTPUT_DIR="${HPC_OUTPUT:-$FIGURE_DIR/results/final_tradeoff_${SLURM_JOB_ID}}"
# Reject used output directories before installing the export trap.
if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "ERROR: output is not empty; use a new job/output directory" >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR/logs"
finalize() {
    local run_status=$?
    trap - EXIT
    set +e
    for ext in out err; do
        if [[ -f "$FIGURE_DIR/${SLURM_JOB_ID}.$ext" ]]; then cp "$FIGURE_DIR/${SLURM_JOB_ID}.$ext" "$OUTPUT_DIR/logs/slurm_snapshot.$ext"; fi
    done
    "$PYTHON_BIN" -u "$SUPPORT/finalize_run.py" --output "$OUTPUT_DIR" --project "$SUPPORT" --run-exit-code "$run_status"
    local pack_status=$?
    if ((run_status)); then exit "$run_status"; fi
    exit "$pack_status"
}
trap finalize EXIT
"$PYTHON_BIN" -u "$SUPPORT/run_tradeoff_final.py" --output "$OUTPUT_DIR" --selection-dir "$SELECTION" --workers "$WORKERS" --check-only
"$PYTHON_BIN" -u "$SUPPORT/run_tradeoff_final.py" --output "$OUTPUT_DIR" --selection-dir "$SELECTION" --workers "$WORKERS" 2>&1 | tee "$OUTPUT_DIR/logs/console.log"
