#!/bin/bash
#SBATCH --job-name=toma_tradeoff56
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=56
#SBATCH --mem=48G
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --time=72:00:00
set -euo pipefail

PROJECT_ROOT="/home/lyq00008339/Lihuang_Li"
CODE_DIR="/home/lyq00008339/Lihuang_Li/toma"
TARGET_CONDA_ENV="${TARGET_CONDA_ENV:-JM_CVX}"
CONDA_SH="${CONDA_SH:-/home/lyq00008339/miniconda3/etc/profile.d/conda.sh}"
FIGURE="${1:-tradeoff}"  # 默认先跑权重图；其他图通过参数单独选择

if [ ! -f "${CONDA_SH}" ]; then
    echo "Error: conda init script was not found at ${CONDA_SH}." >&2
    exit 1
fi
. "${CONDA_SH}"
conda activate "${TARGET_CONDA_ENV}"
export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${CODE_DIR}"

PYTHON_BIN="${HPC_PYTHON:-python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/toma_mpl_${SLURM_JOB_ID:-check}"
WORKERS=56
if [[ -z "${SLURM_JOB_ID:-}" || "${SLURM_CPUS_PER_TASK:-0}" -lt "$WORKERS" ]]; then
    echo 'ERROR: submit through Slurm with 56 allocated CPUs.' >&2
    exit 1
fi
case "$FIGURE" in
    tradeoff)
        CONFIG_FILE="${CODE_DIR}/tradeoff_config.json"
        OUTPUT_DIR="${HPC_OUTPUT:-results/tradeoff_endpoints_mc100_${SLURM_JOB_ID}}"
        ;;
    convergence|p_dl_max|rho_si|system_load)
        CONFIG_FILE="${CODE_DIR}/simulation_config.json"
        OUTPUT_DIR="${HPC_OUTPUT:-results/${FIGURE}_mc100_${SLURM_JOB_ID}}"
        ;;
    *) echo "ERROR: select exactly one figure; unknown figure ${FIGURE}" >&2; exit 1;;
esac
if [[ ! -f run_figures.py || ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: simulation code/config not found in ${CODE_DIR}: ${CONFIG_FILE}" >&2
    exit 1
fi
command -v xargs >/dev/null
mkdir -p "${OUTPUT_DIR}/logs"
echo "Figure=${FIGURE}; scenes=100; workers=${WORKERS}; output=${OUTPUT_DIR}"
"$PYTHON_BIN" -u run_figures.py --config "$CONFIG_FILE" --figures "$FIGURE" --n-mc 100 --check-only

# 动态场景队列：一个场景一个进程，空闲 CPU 立即领取下一个场景。
# 场景内串行执行权重和四方案；每个进程/BLAS 都只使用一个 CPU。
run_scene() {
    local scene_index="$1"
    export MPLCONFIGDIR="${TMPDIR:-/tmp}/toma_mpl_${SLURM_JOB_ID}_scene_${scene_index}"
    if "$PYTHON_BIN" -u run_figures.py --config "$CONFIG_FILE" --figures "$FIGURE" \
        --n-mc 100 --workers 1 --scene-index "$scene_index" --output "$OUTPUT_DIR" \
        >"${OUTPUT_DIR}/logs/scene_${scene_index}.out" 2>"${OUTPUT_DIR}/logs/scene_${scene_index}.err"; then
        echo "Scene ${scene_index} completed"
    else
        echo "Scene ${scene_index} incomplete; check ${OUTPUT_DIR}/logs/scene_${scene_index}.err" >&2
        return 1
    fi
}
export -f run_scene
export PYTHON_BIN CONFIG_FILE FIGURE OUTPUT_DIR
if printf '%s\n' {0..99} | xargs -n 1 -P "$WORKERS" bash -c 'run_scene "$1"' _; then
    echo 'All scenes completed. Collecting the requested figure.'
else
    echo "ERROR: some scenes failed or remain unfinished. Resume with HPC_OUTPUT=${OUTPUT_DIR}." >&2
    exit 1
fi
"$PYTHON_BIN" -u run_figures.py --config "$CONFIG_FILE" --figures "$FIGURE" \
    --n-mc 100 --collect-only --output "$OUTPUT_DIR"
echo "Completed: ${OUTPUT_DIR}"
