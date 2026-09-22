#!/bin/bash
# Experiment 3: isolate task-order sensitivity from training noise.
# Training seed is fixed at 0 and only --task-order-seed varies. The existing
# seed-0 runs (task-order seed derived from --seed, i.e. 0) supply the sixth
# order. --seeds cannot sweep task order, so each order is its own job.
#
# Usage:
#   scripts/pipeline_60h/exp3_task_order.sh          # run (MAX_JOBS=4)
#   scripts/pipeline_60h/exp3_task_order.sh --list   # print job specs

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_queue.sh"

TRAINING_SEED=0
TASK_ORDER_SEEDS="39 55 100 390 550"
TIL_MODELS="gem er_ring ctn packnet cmaml bcl_dual lwf"
CIL_MODELS="eralg4 bcl_dual er_ring cmaml"

emit_task_order_jobs() {
    local mode models task_order_seed model
    for mode in til cil; do
        models="$TIL_MODELS"
        [ "$mode" = "cil" ] && models="$CIL_MODELS"
        for task_order_seed in $TASK_ORDER_SEEDS; do
            for model in $models; do
                echo "taskorder_to${task_order_seed}_${mode}_${model}|${mode}|${model}|--seeds ${TRAINING_SEED} --task-order-seed ${task_order_seed} ${EPOCHS_1E} --log_dir $(group_log_dir "taskorder_${mode^^}/to${task_order_seed}")"
            done
        done
    done
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    experiment_main emit_task_order_jobs "$@"
fi
