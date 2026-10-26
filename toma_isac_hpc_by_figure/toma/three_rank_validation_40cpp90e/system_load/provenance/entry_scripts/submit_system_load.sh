#!/usr/bin/env bash
#SBATCH --job-name=three_system_load
#SBATCH --partition=cpu
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=/home/lyq00008339/logs/%x_%j.out
#SBATCH --error=/home/lyq00008339/logs/%x_%j.err
set -euo pipefail
TOMA_ROOT="${TOMA_ROOT:-/home/lyq00008339/Lihuang_Li/toma}"
SUPPORT="$TOMA_ROOT/final_support_plainfp_o4_i5_s8005"
MULTINODE="$TOMA_ROOT/multinode_support_plainfp_o4_i5"
FIGURE_DIR="$TOMA_ROOT/04_wsr_vs_system_load"
SELECTION="${HPC_SELECTION:-$TOMA_ROOT/results/final_initial_geometry_plainfp_o4_i5_mafp_s8005}"
for required in "$MULTINODE/run_multinode.py" "$SUPPORT/finalize_run.py" "$SELECTION/frozen_initial_geometry.json"; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: missing $required; upload the patch and wait for 01 FROZEN GEOMETRY READY" >&2
        exit 1
    fi
done
source "${CONDA_SH:-/home/lyq00008339/miniconda3/etc/profile.d/conda.sh}"
conda activate "${TARGET_CONDA_ENV:-JM_CVX}"
cd "$FIGURE_DIR"
PYTHON_BIN="${HPC_PYTHON:-python}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/toma_3nodes_${SLURM_JOB_ID}"
export SLURM_EXPORT_ENV=ALL
if [[ "${SLURM_JOB_NUM_NODES:?Submit through Slurm}" != 3 || "${SLURM_NTASKS:?}" != 3 || "${SLURM_CPUS_PER_TASK:?}" != 56 ]]; then
    echo 'ERROR: expected 3 nodes, one Slurm task per node and 56 CPUs per task' >&2
    exit 1
fi
OUTPUT_DIR="${HPC_OUTPUT:-$FIGURE_DIR/results/final_3nodes_plainfp_mafp_o4_i5_${SLURM_JOB_ID}}"
if [[ "$OUTPUT_DIR" == /tmp || "$OUTPUT_DIR" == /tmp/* ]]; then
    echo 'ERROR: results must use shared /home storage, not node-local /tmp' >&2
    exit 1
fi
if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo 'ERROR: output is not empty; use a fresh job/output directory' >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR/logs"
finalize() {
    local run_status=$?
    trap - EXIT
    set +e
    for ext in out err; do
        local slurm_log="/home/lyq00008339/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.$ext"
        if [[ -f "$slurm_log" ]]; then
            cp "$slurm_log" "$OUTPUT_DIR/logs/slurm_snapshot.$ext"
        fi
    done
    "$PYTHON_BIN" -u "$SUPPORT/finalize_run.py" --output "$OUTPUT_DIR" --project "$SUPPORT" --run-exit-code "$run_status"
    local pack_status=$?
    if ((run_status == 0 && pack_status == 0)); then
        "$PYTHON_BIN" -u "$SUPPORT/collect_four.py" --toma-root "$TOMA_ROOT" --selection-dir "$SELECTION"
        pack_status=$?
    fi
    if ((run_status)); then exit "$run_status"; fi
    exit "$pack_status"
}
trap finalize EXIT
"$PYTHON_BIN" -u "$MULTINODE/run_multinode.py" --figure system_load --output "$OUTPUT_DIR" --selection-dir "$SELECTION" --workers-per-node 56 --check-only
# One leader is launched ON EACH NODE; each owns a local pool of 56 processes.
srun --mpi=none --nodes=3 --ntasks=3 --ntasks-per-node=1 --cpus-per-task=56 --cpu-bind=cores --kill-on-bad-exit=1 --label --export=ALL \
    "$PYTHON_BIN" -u "$MULTINODE/run_multinode.py" --figure system_load --output "$OUTPUT_DIR" --selection-dir "$SELECTION" --workers-per-node 56 \
    2>&1 | tee "$OUTPUT_DIR/logs/console.log"
