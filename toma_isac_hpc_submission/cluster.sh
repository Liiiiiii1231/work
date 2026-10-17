#!/bin/bash
#SBATCH --job-name=toma_mc100
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --time=72:00:00
set -euo pipefail

# 路径和环境配置：直接使用用户确认的绝对路径。
PROJECT_ROOT="/home/lyq00008339/Lihuang_Li"
CODE_DIR="/home/lyq00008339/Lihuang_Li/toma"
TARGET_CONDA_ENV="${TARGET_CONDA_ENV:-JM_CVX}"
CONDA_SH="${CONDA_SH:-/home/lyq00008339/miniconda3/etc/profile.d/conda.sh}"
FIGURE="${1:-${HPC_FIGURE:-all}}"  # all 跑全部；单图可改为 p_dl_max 等

if [ ! -f "${CONDA_SH}" ]; then
    echo "Error: conda init script was not found at ${CONDA_SH}." >&2
    exit 1
fi
. "${CONDA_SH}"
conda activate "${TARGET_CONDA_ENV}"
export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${CODE_DIR}"

# 本仿真使用 NumPy/SciPy，不调用 CVX、CVXPY 或 MOSEK，无需许可证。
PYTHON_BIN="${HPC_PYTHON:-python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/toma_mpl_${SLURM_JOB_ID:-check}"
if [[ -z "${SLURM_JOB_ID:-}" || "${SLURM_CPUS_PER_TASK:-0}" -lt 10 ]]; then
    echo 'ERROR: submit through Slurm with 10 CPUs.' >&2
    exit 1
fi
if [[ ! -f run_figures.py || ! -f simulation_config.json ]]; then
    echo "ERROR: CODE_DIR must be the directory containing the 9 simulation files: ${CODE_DIR}" >&2
    exit 1
fi
case "$FIGURE" in
    all|convergence|p_dl_max|rho_si|system_load|tradeoff) ;;
    *) echo "ERROR: unknown figure ${FIGURE}" >&2; exit 1;;
esac
figure_args=()
if [[ "$FIGURE" != all ]]; then figure_args=(--figures "$FIGURE"); fi
OUTPUT_DIR="${HPC_OUTPUT:-results/${FIGURE}_mc100_${SLURM_JOB_ID}}"
mkdir -p "${OUTPUT_DIR}/logs"
echo "开始仿真：${FIGURE}；100 个场景，10 个 CPU；结果目录：${OUTPUT_DIR}"
"$PYTHON_BIN" -u run_figures.py --n-mc 100 "${figure_args[@]}" --check-only

# 10 个 CPU 同时工作，每个 CPU 顺序处理 10 个场景。
run_batch() {
    local batch="$1" scene_index failed=0
    export MPLCONFIGDIR="${TMPDIR:-/tmp}/toma_mpl_${SLURM_JOB_ID}_batch_${batch}"
    for ((scene_index=batch*10; scene_index<batch*10+10; scene_index++)); do
        echo "批次 ${batch}：场景索引 ${scene_index}"
        if "$PYTHON_BIN" -u run_figures.py --n-mc 100 --workers 1 "${figure_args[@]}" \
            --scene-index "$scene_index" --output "$OUTPUT_DIR"; then
            echo "场景 ${scene_index} 执行完成"
        else
            echo "场景 ${scene_index} 未完成，继续其余场景" >&2
            failed=1
        fi
    done
    return "$failed"
}
pids=()
for ((batch=0; batch<10; batch++)); do
    run_batch "$batch" >"${OUTPUT_DIR}/logs/batch_${batch}.out" 2>"${OUTPUT_DIR}/logs/batch_${batch}.err" &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
    if wait "$pid"; then :; else failed=1; fi
done
if ((failed)); then
    echo "部分场景未完成，请查看 ${OUTPUT_DIR}/logs；设置 HPC_OUTPUT 为原目录可续跑。" >&2
    exit 1
fi

# 所有场景成功后，统一汇总并生成仿真图。
"$PYTHON_BIN" -u run_figures.py --n-mc 100 "${figure_args[@]}" --collect-only --output "$OUTPUT_DIR"
echo "全部仿真和绘图执行完毕：${OUTPUT_DIR}"
