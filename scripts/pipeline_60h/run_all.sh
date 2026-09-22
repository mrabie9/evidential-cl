#!/bin/bash
# Run the whole 60h pipeline through one shared queue (MAX_JOBS=4 by default),
# so the concurrency cap spans all experiments. Order: seed extension, memory
# sweep, task order, SNR sweep. Jobs recorded in DONE_FILE are skipped, so a
# rerun after a crash resumes where it stopped.
#
# Usage:
#   nohup scripts/pipeline_60h/run_all.sh > logs/pipeline_60h/nohup.out 2>&1 &
#   scripts/pipeline_60h/run_all.sh --list   # print every job spec
#   MAX_JOBS=3 scripts/pipeline_60h/run_all.sh

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PIPELINE_DIR}/lib_queue.sh"

EXPERIMENT_SCRIPTS=(
    exp1_seed_extension.sh
    exp2_mem_sweep.sh
    exp3_task_order.sh
    exp4_snr_sweep.sh
)

emit_all_jobs() {
    local script
    for script in "${EXPERIMENT_SCRIPTS[@]}"; do
        bash "${PIPELINE_DIR}/${script}" --list
    done
}

mkdir -p "$PIPELINE_LOG_ROOT"
experiment_main emit_all_jobs "$@"
