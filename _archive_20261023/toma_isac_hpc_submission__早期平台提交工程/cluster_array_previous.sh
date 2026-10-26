#!/usr/bin/env bash
#SBATCH --job-name=toma_mc100
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir"
mode="${1:-submit}"
if (($#)); then shift; fi
case "$mode" in
  all|convergence|p_dl_max|rho_si|system_load|tradeoff)
    export HPC_FIGURE="$mode"
    mode=submit
    ;;
  submit|worker|collect|check) ;;
  *) echo 'Usage: bash cluster.sh [all|convergence|p_dl_max|rho_si|system_load|tradeoff|check]' >&2; exit 2;;
esac
: "${HPC_PYTHON:?Set HPC_PYTHON to the absolute Python executable in your cluster environment}"
export HPC_FIGURE="${HPC_FIGURE:-all}"
export HPC_CONFIG="${HPC_CONFIG:-$script_dir/simulation_config.json}"
case "$HPC_FIGURE" in
  all|convergence|p_dl_max|rho_si|system_load|tradeoff) ;;
  *) echo "Unknown HPC_FIGURE: $HPC_FIGURE" >&2; exit 2;;
esac
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/toma_mpl_${SLURM_JOB_ID:-check}_${SLURM_ARRAY_TASK_ID:-none}"
figure_args=()
if [[ "$HPC_FIGURE" != all ]]; then figure_args=(--figures "$HPC_FIGURE"); fi
common_args=(--config "$HPC_CONFIG" --n-mc 100 "${figure_args[@]}")

case "$mode" in
  check)
    "$HPC_PYTHON" -u run_figures.py "${common_args[@]}" --check-only
    ;;
  submit)
    [[ "$HPC_PYTHON" = /* && -x "$HPC_PYTHON" ]] || { echo 'HPC_PYTHON must be an existing absolute executable' >&2; exit 1; }
    export HPC_OUTPUT="${HPC_OUTPUT:-results/${HPC_FIGURE}_mc100_$(date +%Y%m%d_%H%M%S)}"
    mkdir -p logs
    "$HPC_PYTHON" -u run_figures.py "${common_args[@]}" --check-only
    job_id=$(sbatch --parsable --array=0-9%10 --job-name="toma_${HPC_FIGURE}" --export=ALL "$script_dir/cluster.sh" worker)
    job_id=${job_id%%;*}
    printf 'Figure: %s\nArray job: %s\nOutput: %s\n' "$HPC_FIGURE" "$job_id" "$HPC_OUTPUT"
    sbatch --job-name="collect_${HPC_FIGURE}" --time=04:00:00 --dependency="afterok:$job_id" \
      --export=ALL "$script_dir/cluster.sh" collect
    ;;
  worker)
    : "${HPC_OUTPUT:?Set HPC_OUTPUT or submit with bash cluster.sh}"
    batch_index="${SLURM_ARRAY_TASK_ID:?Missing array batch index}"
    [[ "$batch_index" =~ ^[0-9]$ ]] || { echo 'Batch index must be 0 through 9' >&2; exit 2; }
    first_scene=$((batch_index * 10))
    failed=0
    for ((scene_index=first_scene; scene_index<first_scene+10; scene_index++)); do
      printf 'Figure %s | batch %s | scene index %s\n' "$HPC_FIGURE" "$batch_index" "$scene_index"
      if "$HPC_PYTHON" -u run_figures.py "${common_args[@]}" --workers 1 \
          --scene-index "$scene_index" --output "$HPC_OUTPUT"; then
        printf 'Scene index %s complete\n' "$scene_index"
      else
        printf 'Scene index %s failed or incomplete; continuing remaining scenes\n' "$scene_index" >&2
        failed=1
      fi
    done
    exit "$failed"
    ;;
  collect)
    : "${HPC_OUTPUT:?Set the original HPC_OUTPUT directory}"
    "$HPC_PYTHON" -u run_figures.py "${common_args[@]}" --collect-only --output "$HPC_OUTPUT"
    ;;
esac
