#!/bin/bash
# Experiment 4: per-SNR sweep. Each data/rff/radar/snr_tasks/snr_<v>db folder
# holds the full task set at a single SNR. One job per (mode, SNR, model);
# each job sweeps all six seeds at 1e.
#
# The SNR folders name the UCLResM files uclresm_task_0X (not the
# uclresm_degraded_task_0X in base.yaml), so the task order is overridden
# here, keeping base.yaml's dataset ordering.
#
# Usage:
#   scripts/pipeline_60h/exp4_snr_sweep.sh          # run (MAX_JOBS=4)
#   scripts/pipeline_60h/exp4_snr_sweep.sh --list   # print job specs

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_queue.sh"

SNR_ROOT="data/rff/radar/snr_tasks"
SNR_VALUES="-10 -8 -6 -4 -2 0 2 4 6 8 10"
SNR_TASK_ORDER="deeprad_task_01,deeprad_task_02,deeprad_task_03,deeprad_task_04,radchar_task_01,radnist_task_01,radnist_task_02,uclresm_task_01,uclresm_task_02,uclresm_task_03"
TIL_MODELS="gem cmaml ctn er_ring lwf eralg4 iid2 ft"
CIL_MODELS="eralg4 bcl_dual cmaml er_ring iid2 ft"

emit_snr_sweep_jobs() {
    local mode models snr_value snr_tag model
    for mode in til cil; do
        models="$TIL_MODELS"
        [ "$mode" = "cil" ] && models="$CIL_MODELS"
        for snr_value in $SNR_VALUES; do
            snr_tag="${snr_value/-/m}"
            for model in $models; do
                echo "snr_${snr_tag}db_${mode}_${model}|${mode}|${model}|--seeds ${ALL_SEEDS} --data_path ${SNR_ROOT}/snr_${snr_value}db --task-order-files ${SNR_TASK_ORDER} ${EPOCHS_1E} --log_dir $(group_log_dir "snr_${mode^^}/${snr_tag}db")"
            done
        done
    done
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    experiment_main emit_snr_sweep_jobs "$@"
fi
