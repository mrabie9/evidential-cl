#!/bin/bash
# Re-run every bcl_dual job of the 60h pipeline after two fixes that change all of
# BCL-Dual's numbers:
#   * bcl_dual.observe clears the inner-step gradient before the outer step (it used to be
#     re-applied, so the "validation" step was mostly a second current-batch step);
#   * AdcIqAdapter mixes the three ADCs with softmax(w) instead of w / sum(w), which was
#     singular at sum(w) = 0 (collapsed CIL 1-epoch runs in the ablation campaign).
#
# Jobs: seed extension 1e/5e TIL/CIL over all six seeds (matching logs/00_sync), memory
# sweep, task-order sweep and SNR sweep (exp1-exp4), then the single-pass runtime job
# alone (MAX_JOBS=1, as in exp5). Everything writes to its own tree and done-files, so
# the pre-fix results and logs/pipeline_60h/done*.txt are untouched:
#   logs/${PIPELINE_DATE}_full-pipeline/<group>/saved_models/bcl_dual/...
#   logs/pipeline_bcl_fixed/{done.txt,done_runtime.txt,run_*/}
#
# Usage:
#   nohup bash scripts/pipeline_60h/rerun_bcl_dual_fixed.sh > logs/pipeline_bcl_fixed/nohup.out 2>&1 &
#   bash scripts/pipeline_60h/rerun_bcl_dual_fixed.sh --list

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PIPELINE_DATE="${PIPELINE_DATE:-20260926bclfix}"
export PIPELINE_LOG_ROOT="${PIPELINE_LOG_ROOT:-$(cd "$PIPELINE_DIR/../.." && pwd)/logs/pipeline_bcl_fixed}"
export NEW_SEEDS="${NEW_SEEDS:-0,39,55,100,390,550}"
source "${PIPELINE_DIR}/lib_queue.sh"

emit_bcl_jobs() {
    local script
    for script in exp1_seed_extension.sh exp2_mem_sweep.sh exp3_task_order.sh exp4_snr_sweep.sh; do
        bash "${PIPELINE_DIR}/${script}" --list | grep -F "|bcl_dual|"
    done
}

emit_bcl_runtime_job() {
    TIL_MODELS=bcl_dual bash "${PIPELINE_DIR}/exp5_runtime_single_pass.sh" --list
}

if [ "${1:-}" = "--list" ]; then
    emit_bcl_jobs
    emit_bcl_runtime_job
    exit 0
fi

mkdir -p "$PIPELINE_LOG_ROOT"
MAX_JOBS="${MAX_JOBS:-3}"
mapfile -t specs < <(emit_bcl_jobs)
run_job_queue "${specs[@]}"
rc=$?

# Timing job last and alone, so GPU contention does not distort it.
MAX_JOBS=1
DONE_FILE="${PIPELINE_LOG_ROOT}/done_runtime.txt"
mapfile -t specs < <(emit_bcl_runtime_job)
run_job_queue "${specs[@]}" || rc=1
exit $rc
