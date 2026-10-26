#!/bin/bash
#SBATCH --job-name=toma_p_dl_max50
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=50
#SBATCH --mem=48G
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --time=48:00:00
set -euo pipefail

PROJECT_ROOT="/home/lyq00008339/Lihuang_Li/toma"
FIGURE_DIR="${PROJECT_ROOT}/02_wsr_vs_p_dl_max"
ENTRY_SCRIPT="${FIGURE_DIR}/run_p_dl_max.py"
CONFIG_FILE="${PROJECT_ROOT}/simulation_config.json"
TARGET_CONDA_ENV="${TARGET_CONDA_ENV:-JM_CVX}"
CONDA_SH="${CONDA_SH:-/home/lyq00008339/miniconda3/etc/profile.d/conda.sh}"

if [ ! -f "${CONDA_SH}" ]; then
    echo "ERROR: conda init script was not found at ${CONDA_SH}." >&2
    exit 1
fi
. "${CONDA_SH}"
conda activate "${TARGET_CONDA_ENV}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${HPC_PYTHON:-python}"
WORKERS=50
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/toma_mpl_${SLURM_JOB_ID:-check}"
if [[ -z "${SLURM_JOB_ID:-}" || "${SLURM_CPUS_PER_TASK:-0}" -lt "$WORKERS" ]]; then
    echo 'ERROR: submit this script through Slurm with 50 allocated CPUs.' >&2
    exit 1
fi
missing=0
for required_file in "$ENTRY_SCRIPT" "$CONFIG_FILE" \
    "$PROJECT_ROOT/run_figures.py" "$PROJECT_ROOT/system_model.py" \
    "$PROJECT_ROOT/channel_model.py" "$PROJECT_ROOT/inner_solver.py" \
    "$PROJECT_ROOT/outer_solver.py" "$PROJECT_ROOT/algorithms.py" "$PROJECT_ROOT/figure_engine.py"; do
    if [[ -f "$required_file" ]]; then
        echo "FOUND: $required_file"
    else
        echo "MISSING FILE: $required_file" >&2
        missing=1
    fi
done
if ((missing)); then
    echo 'Upload the missing files to the exact paths above, then resubmit.' >&2
    exit 1
fi
command -v xargs >/dev/null
OUTPUT_DIR="${HPC_OUTPUT:-${FIGURE_DIR}/results/run_${SLURM_JOB_ID}}"
mkdir -p "${OUTPUT_DIR}/logs"
echo "Figure=p_dl_max; CPU workers=50; scenes=100; output=$OUTPUT_DIR"
"$PYTHON_BIN" -u "$ENTRY_SCRIPT" --config "$CONFIG_FILE" --n-mc 100 --check-only

# Dynamic scene queue: one scene per CPU; all schemes/weights within it run serially.
run_scene() {
    local scene_index="$1"
    export MPLCONFIGDIR="${TMPDIR:-/tmp}/toma_mpl_${SLURM_JOB_ID}_scene_${scene_index}"
    if "$PYTHON_BIN" -u "$ENTRY_SCRIPT" --config "$CONFIG_FILE" --n-mc 100 --workers 1 \
        --scene-index "$scene_index" --output "$OUTPUT_DIR" \
        >"${OUTPUT_DIR}/logs/scene_${scene_index}.out" 2>"${OUTPUT_DIR}/logs/scene_${scene_index}.err"; then
        echo "Scene ${scene_index} complete"
    else
        echo "Scene ${scene_index} incomplete; see ${OUTPUT_DIR}/logs/scene_${scene_index}.err" >&2
        return 1
    fi
}
export -f run_scene
export PYTHON_BIN ENTRY_SCRIPT CONFIG_FILE OUTPUT_DIR
if printf '%s\n' {0..99} | xargs -n 1 -P "$WORKERS" bash -c 'run_scene "$1"' _; then
    echo 'All scenes finished. Collecting this figure.'
else
    echo "ERROR: incomplete scenes; preserve this directory and set HPC_OUTPUT=$OUTPUT_DIR to resume." >&2
    exit 1
fi
"$PYTHON_BIN" -u "$ENTRY_SCRIPT" --config "$CONFIG_FILE" --n-mc 100 --collect-only --output "$OUTPUT_DIR"
echo "Completed: $OUTPUT_DIR"
