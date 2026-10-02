#!/bin/bash
# Re-run the 1e CIL seed-extension jobs of the replay learners after switching
# their CIL training to one joint BatchNorm forward over replay + current rows
# (model.task_bn.replay_forward_is_joint). Before the fix the current task was
# normalized with single-task batch statistics during training but scored inside
# mixed pooled batches, so each new task looked unlearned until replayed and BWT
# came out large and positive.
#
# Jobs: exp1 1e CIL for lamaml, cmaml, smaml, eralg4, la-er, er_ring, all six
# seeds. Writes to its own tree and done-file:
#   logs/${PIPELINE_DATE}_full-pipeline/1e_CIL/saved_models/<model>/...
#   logs/pipeline_cil_jointbn/{done.txt,run_*/}
#
# Usage:
#   nohup bash scripts/pipeline_60h/rerun_cil_joint_bn.sh > logs/pipeline_cil_jointbn/nohup.out 2>&1 &
#   bash scripts/pipeline_60h/rerun_cil_joint_bn.sh --list

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PIPELINE_DATE="${PIPELINE_DATE:-20261002cilbn}"
export PIPELINE_LOG_ROOT="${PIPELINE_LOG_ROOT:-$(cd "$PIPELINE_DIR/../.." && pwd)/logs/pipeline_cil_jointbn}"
export NEW_SEEDS="${NEW_SEEDS:-0,39,55,100,390,550}"
source "${PIPELINE_DIR}/lib_queue.sh"

TARGET_RE='^seedext_1e_cil_[^|]*\|cil\|(lamaml|cmaml|smaml|eralg4|la-er|er_ring)\|'

emit_jobs() {
    bash "${PIPELINE_DIR}/exp1_seed_extension.sh" --list | grep -E "${TARGET_RE}"
}

if [ "${1:-}" = "--list" ]; then
    emit_jobs
    exit 0
fi

mkdir -p "$PIPELINE_LOG_ROOT"
MAX_JOBS="${MAX_JOBS:-2}"
mapfile -t specs < <(emit_jobs)
run_job_queue "${specs[@]}"
