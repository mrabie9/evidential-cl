#!/bin/bash
# Experiment 1: extend every current experiment with seeds 100,390,550.
# One job per (epochs, mode, model); each job sweeps the three new seeds via
# main.py --seeds. Covers 1e (single-pass) and 5e (base.yaml default).
#
# Usage:
#   scripts/pipeline_60h/exp1_seed_extension.sh          # run (MAX_JOBS=4)
#   scripts/pipeline_60h/exp1_seed_extension.sh --list   # print job specs

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_queue.sh"

NEW_SEEDS="${NEW_SEEDS:-100,390,550}"
TIL_MODELS="agem bcl_dual cmaml ctn eralg4 er_ring ewc ft gem hat icarl iid2 la-er lamaml lwf packnet rwalk si smaml ucl"
CIL_MODELS="agem bcl_dual cmaml eralg4 er_ring ewc ft gem icarl iid2 la-er lamaml lwf rwalk si smaml ucl"

emit_seed_extension_jobs() {
    local epochs epoch_flags mode models model
    for epochs in 1e 5e; do
        epoch_flags="$EPOCHS_1E"
        [ "$epochs" = "5e" ] && epoch_flags="$EPOCHS_5E"
        for mode in til cil; do
            models="$TIL_MODELS"
            [ "$mode" = "cil" ] && models="$CIL_MODELS"
            for model in $models; do
                echo "seedext_${epochs}_${mode}_${model}|${mode}|${model}|--seeds ${NEW_SEEDS} ${epoch_flags} --log_dir $(group_log_dir "${epochs}_${mode^^}")"
            done
        done
    done
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    experiment_main emit_seed_extension_jobs "$@"
fi
