#!/bin/bash
#SBATCH --job-name=final_p_dl_max
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH -o %j.out
#SBATCH -e %j.err
set -euo pipefail
TOMA_ROOT="/home/lyq00008339/Lihuang_Li/toma"
SUPPORT="$TOMA_ROOT/final_support_i1200_o1500"
FIGURE_DIR="$TOMA_ROOT/02_wsr_vs_p_dl_max"
SELECTION="${HPC_SELECTION:-$TOMA_ROOT/results/final_initial_geometry_i1200_o1500}"
if [[ ! -f "$SELECTION/frozen_initial_geometry.json" ]]; then
    echo "ERROR: first run 01 and wait for FROZEN GEOMETRY READY; expected $SELECTION/frozen_initial_geometry.json" >&2
    exit 1
fi
. "${CONDA_SH:-/home/lyq00008339/miniconda3/etc/profile.d/conda.sh}"
conda activate "${TARGET_CONDA_ENV:-JM_CVX}"
cd "$FIGURE_DIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/toma_final_${SLURM_JOB_ID}"
WORKERS="${SLURM_CPUS_PER_TASK:?Submit through Slurm}"
OUTPUT_DIR="${HPC_OUTPUT:-$FIGURE_DIR/results/final_${SLURM_JOB_ID}}"
mkdir -p "$OUTPUT_DIR/logs"
finalize() {
    local run_status=$?
    trap - EXIT
    set +e
    for ext in out err; do
        if [[ -f "$FIGURE_DIR/${SLURM_JOB_ID}.$ext" ]]; then cp "$FIGURE_DIR/${SLURM_JOB_ID}.$ext" "$OUTPUT_DIR/logs/slurm_snapshot.$ext"; fi
    done
    python -u "$SUPPORT/finalize_run.py" --output "$OUTPUT_DIR" --project "$SUPPORT" --run-exit-code "$run_status"
    local pack_status=$?
    if ((run_status == 0 && pack_status == 0)); then
        python -u "$SUPPORT/collect_four.py" --toma-root "$TOMA_ROOT" --selection-dir "$SELECTION"
        pack_status=$?
    fi
    if ((run_status)); then exit "$run_status"; fi
    exit "$pack_status"
}
trap finalize EXIT
python -u "$SUPPORT/run_final.py" --figure p_dl_max --output "$OUTPUT_DIR" --selection-dir "$SELECTION" --workers "$WORKERS" --check-only
python -u "$SUPPORT/run_final.py" --figure p_dl_max --output "$OUTPUT_DIR" --selection-dir "$SELECTION" --workers "$WORKERS" 2>&1 | tee -a "$OUTPUT_DIR/logs/console.log"
