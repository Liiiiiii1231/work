#!/bin/bash
#SBATCH --job-name=toma16x16_mc10
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH -o %j.out
#SBATCH -e %j.err
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/lyq00008339/Lihuang_Li/toma/test_singlepass_1500_2000}"
CONDA_SH="${CONDA_SH:-/home/lyq00008339/miniconda3/etc/profile.d/conda.sh}"
. "$CONDA_SH"
conda activate "${TARGET_CONDA_ENV:-JM_CVX}"
cd "$PROJECT_ROOT"
PYTHON_BIN="${HPC_PYTHON:-python}"
WORKERS="${SLURM_CPUS_PER_TASK:?Submit through sbatch}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/toma_annulus_mpl_${SLURM_JOB_ID}"
OUTPUT_DIR="${HPC_OUTPUT:-${PROJECT_ROOT}/results/tradeoff_mc10_${SLURM_JOB_ID}}"
mkdir -p "$OUTPUT_DIR/logs"
# Finalize after Python has exited, including a normal nonzero exit.
finalize() {
    local run_status=$?
    trap - EXIT
    set +e
    for suffix in out err; do
        if [[ -f "$PROJECT_ROOT/${SLURM_JOB_ID}.$suffix" ]]; then
            cp "$PROJECT_ROOT/${SLURM_JOB_ID}.$suffix" "$OUTPUT_DIR/logs/slurm_${SLURM_JOB_ID}_snapshot.$suffix"
        fi
    done
    "$PYTHON_BIN" -u "$PROJECT_ROOT/finalize_run.py" --output "$OUTPUT_DIR" \
        --project "$PROJECT_ROOT" --run-exit-code "$run_status"
    local export_status=$?
    if (( run_status != 0 )); then exit "$run_status"; fi
    exit "$export_status"
}
trap finalize EXIT
"$PYTHON_BIN" -c 'import json; b=json.load(open("tradeoff_config.json")); assert all("ground_annulus" in m["experiment"] for m in b["models"].values()), "Ground geometry is not configured: run configure_trial.py first"'
echo "CPUs=$WORKERS; dynamic independent tasks=800; scenes=8001..8010; output=$OUTPUT_DIR"
"$PYTHON_BIN" -u run_figures.py --config tradeoff_config.json --figures tradeoff --n-mc 10 --check-only
"$PYTHON_BIN" -u run_figures.py --config tradeoff_config.json --figures tradeoff --n-mc 10 \
    --workers "$WORKERS" --output "$OUTPUT_DIR" 2>&1 | tee -a "$OUTPUT_DIR/logs/console.log"
