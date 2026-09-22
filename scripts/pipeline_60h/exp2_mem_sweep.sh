#!/bin/bash
# Experiment 2: replay-buffer sweep for single-pass (1e) replay models.
# One job per (mode, buffer size, model); each job sweeps all six seeds.
# The same capacity goes to both --memories and --n_memories, as in
# scripts/full_experiments_mem_sweep.sh.
#
# Usage:
#   scripts/pipeline_60h/exp2_mem_sweep.sh          # run (MAX_JOBS=4)
#   scripts/pipeline_60h/exp2_mem_sweep.sh --list   # print job specs

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_queue.sh"

MEMORY_SIZES="1024 2048 8192 16384"
TIL_MODELS="gem er_ring eralg4 cmaml bcl_dual ctn agem"
CIL_MODELS="cmaml er_ring eralg4 bcl_dual"

emit_mem_sweep_jobs() {
    local mode models memory_size model
    for mode in til cil; do
        models="$TIL_MODELS"
        [ "$mode" = "cil" ] && models="$CIL_MODELS"
        for memory_size in $MEMORY_SIZES; do
            for model in $models; do
                echo "memsweep_m${memory_size}_${mode}_${model}|${mode}|${model}|--seeds ${ALL_SEEDS} --memories ${memory_size} --n_memories ${memory_size} ${EPOCHS_1E} --log_dir $(group_log_dir "memsweep_${mode^^}/m${memory_size}")"
            done
        done
    done
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    experiment_main emit_mem_sweep_jobs "$@"
fi
