#!/bin/bash
# Run with bash on the login node after the new geometry is frozen.
set -euo pipefail
TOMA_ROOT="${TOMA_ROOT:-/home/lyq00008339/Lihuang_Li/toma}"
SELECTION="${HPC_SELECTION:-$TOMA_ROOT/results/final_initial_geometry_plainfp_o4_i5_mafp_final_s8005}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo 'Use bash on the login node; each figure requests 3 nodes x 56 CPUs.' >&2
    exit 1
fi
if [[ ! -f "$SELECTION/frozen_initial_geometry.json" ]]; then
    echo "Wait for 01 FROZEN GEOMETRY READY: $SELECTION/frozen_initial_geometry.json" >&2
    exit 1
fi
for script in 02_wsr_vs_p_dl_max/submit_p_dl_max.sh 03_wsr_vs_rho_si/submit_rho_si.sh 04_wsr_vs_system_load/submit_system_load.sh 05_communication_sensing_tradeoff/submit_tradeoff.sh; do
    sbatch "$TOMA_ROOT/$script"
done
