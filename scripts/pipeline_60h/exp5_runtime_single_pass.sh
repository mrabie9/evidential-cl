#!/bin/bash
# Experiment 5: relative runtime of every TIL algorithm, single-pass (1e),
# seed 0 only. Jobs run strictly one at a time (MAX_JOBS=1) so the timings
# are not distorted by GPU contention, matching the earlier
# "computational_cost" runs (n_epochs=1, inner_steps=1, seed 0).
#
# Usage:
#   scripts/pipeline_60h/exp5_runtime_single_pass.sh          # run serially
#   scripts/pipeline_60h/exp5_runtime_single_pass.sh --list   # print job specs

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_queue.sh"

MAX_JOBS=1
DONE_FILE="${PIPELINE_LOG_ROOT}/done_runtime.txt"
RUNTIME_SEED=0
TIL_MODELS="agem bcl_dual cmaml ctn eralg4 er_ring ewc ft gem hat icarl iid2 la-er lamaml lwf packnet rwalk si smaml ucl"

emit_runtime_jobs() {
    local model
    for model in $TIL_MODELS; do
        echo "runtime_1e_til_${model}|til|${model}|--seeds ${RUNTIME_SEED} ${EPOCHS_1E}"
    done
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    experiment_main emit_runtime_jobs "$@"
fi
