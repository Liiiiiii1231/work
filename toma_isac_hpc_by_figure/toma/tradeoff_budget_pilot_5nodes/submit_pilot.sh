#!/usr/bin/env bash
#SBATCH --job-name=tradeoff_budget_pilot5
#SBATCH --partition=cpu
#SBATCH --nodes=5
#SBATCH --ntasks=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=/home/lyq00008339/logs/%x_%j.out
#SBATCH --error=/home/lyq00008339/logs/%x_%j.err
set -euo pipefail
# Slurm copies batch scripts to its spool; resolve from submission directory
# unless the ZIP bootstrap explicitly provides the extracted pilot directory.
PILOT_ROOT="${PILOT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}}"
source "${CONDA_SH:-/home/lyq00008339/miniconda3/etc/profile.d/conda.sh}"
conda activate "${TARGET_CONDA_ENV:-JM_CVX}"
cd "$PILOT_ROOT"
PYTHON_BIN="${HPC_PYTHON:-python}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/toma_budget_pilot_${SLURM_JOB_ID}"
export SLURM_EXPORT_ENV=ALL
if [[ "${SLURM_JOB_NUM_NODES:?Submit with sbatch}" != 5 || "${SLURM_NTASKS:?}" != 5 || "${SLURM_CPUS_PER_TASK:?}" != 56 ]]; then
    echo 'ERROR: expected 5 nodes, one task per node, 56 CPUs per task' >&2
    exit 1
fi
OUTPUT="${PILOT_OUTPUT:-$PILOT_ROOT/results/pilot_budget5_${SLURM_JOB_ID}}"
if [[ "$OUTPUT" == /tmp || "$OUTPUT" == /tmp/* ]]; then
    echo 'ERROR: pilot output must be on shared /home storage' >&2
    exit 1
fi
if [[ -d "$OUTPUT" ]] && [[ -n "$(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo 'ERROR: use a fresh pilot output directory; continuation is disabled' >&2
    exit 1
fi
mkdir -p "$OUTPUT/logs"
snapshot_logs() {
    local status=$?
    trap - EXIT
    for ext in out err; do
        local log="${PILOT_LOG_DIR:-/home/lyq00008339/logs}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.$ext"
        if [[ -f "$log" ]]; then cp "$log" "$OUTPUT/logs/slurm_snapshot.$ext" || true; fi
    done
    exit "$status"
}
trap snapshot_logs EXIT
"$PYTHON_BIN" -u "$PILOT_ROOT/run_pilot.py" --output "$OUTPUT" --workers-per-node 56 --check-only
srun --mpi=none --nodes=5 --ntasks=5 --ntasks-per-node=1 --cpus-per-task=56 --cpu-bind=cores --kill-on-bad-exit=1 --label --export=ALL \
    "$PYTHON_BIN" -u "$PILOT_ROOT/run_pilot.py" --output "$OUTPUT" --workers-per-node 56 \
    2>&1 | tee "$OUTPUT/logs/console.log"
