#!/bin/bash
# Shared job-queue helpers for the 60h experiment pipeline. Source this file;
# do not execute it.
#
# A job spec is one line of the form:
#   label|mode|model|extra main.py args
# where mode is til or cil and model is a stem under configs/models/<mode>/.
# Each job runs:
#   python main.py --config configs/base.yaml --config configs/models/<mode>/<model>.yaml \
#       --expt_name <label> <extra>
# The label doubles as expt_name so every job writes to its own log dir.
#
# Environment knobs:
#   MAX_JOBS     concurrent jobs (default 4)
#   DONE_FILE    labels of jobs that exited 0; these are skipped on rerun
#                (default logs/pipeline_60h/done.txt)
#   FORCE=1      ignore DONE_FILE and run every job
#   PIPELINE_LOG_ROOT  where per-run log dirs go (default logs/pipeline_60h)

PIPELINE_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$PIPELINE_LIB_DIR/../.." && pwd)"
export PYTHONUNBUFFERED=1

EPOCHS_1E="--n_epochs 1 --inner_steps 1"
EPOCHS_5E=""
ALL_SEEDS="0,39,55,100,390,550"

# Every job writes under logs/<PIPELINE_DATE>_full-pipeline/<group>/saved_models,
# mirroring the logs/00_sync/<group>/saved_models layout directly (no manual
# sync step needed). PIPELINE_DATE defaults to today but can be pinned via env
# so a run split across multiple servers/invocations lands under one shared
# date stamp.
PIPELINE_DATE="${PIPELINE_DATE:-$(date +%Y%m%d)}"
FULL_PIPELINE_ROOT="logs/${PIPELINE_DATE}_full-pipeline"

# group_log_dir <group> -> logs/<date>_full-pipeline/<group>/saved_models
# <group> matches the 00_sync naming convention, e.g. "5e_TIL",
# "memsweep_TIL/m1024", "taskorder_CIL/to39", "snr_TIL/m10db".
group_log_dir() {
    echo "${FULL_PIPELINE_ROOT}/$1/saved_models"
}

PIPELINE_LOG_ROOT="${PIPELINE_LOG_ROOT:-${REPO_ROOT}/logs/pipeline_60h}"
DONE_FILE="${DONE_FILE:-${PIPELINE_LOG_ROOT}/done.txt}"
FORCE="${FORCE:-0}"

activate_env_once() {
    if [ -z "${VIRTUAL_ENV:-}" ] && [ -d "${REPO_ROOT}/la-maml_env" ]; then
        # shellcheck disable=SC1091
        source "${REPO_ROOT}/la-maml_env/bin/activate"
    fi
}

format_duration() {
    local total_seconds="$1"
    printf '%02d:%02d:%02d' $((total_seconds / 3600)) $(((total_seconds % 3600) / 60)) $((total_seconds % 60))
}

pipeline_log() {
    echo "[$(date -Iseconds)] $*" | tee -a "$DRIVER_LOG"
}

is_done() {
    local label="$1"
    [ "$FORCE" = "1" ] && return 1
    [ -f "$DONE_FILE" ] && grep -qxF "$label" "$DONE_FILE"
}

launch_job() {
    local label="$1" mode="$2" model="$3" extra="$4"
    local model_yaml="configs/models/${mode}/${model}.yaml"
    local job_log="${RUN_LOG_DIR}/${label}.log"
    if [ ! -f "${REPO_ROOT}/${model_yaml}" ]; then
        pipeline_log "ERROR: ${label}: missing ${model_yaml}"
        return 1
    fi
    pipeline_log "START ${label} (${model_yaml} ${extra}) -> ${job_log}"
    (
        cd "$REPO_ROOT" || exit 1
        # shellcheck disable=SC2086
        python main.py --config configs/base.yaml --config "$model_yaml" \
            --expt_name "$label" $extra >"$job_log" 2>&1
    ) &
    JOB_LABEL[$!]="$label"
    JOB_START[$!]="$(date +%s)"
    return 0
}

reap_one_job() {
    local finished_pid="" rc label elapsed
    while :; do
        wait -n -p finished_pid
        rc=$?
        if [ -z "$finished_pid" ] && [ "$rc" -eq 127 ]; then
            pipeline_log "WARN: wait -n found no child to reap"
            return 1
        fi
        [ -n "$finished_pid" ] && [ -n "${JOB_LABEL[$finished_pid]:-}" ] && break
    done
    label="${JOB_LABEL[$finished_pid]}"
    elapsed=$(($(date +%s) - JOB_START[$finished_pid]))
    if [ "$rc" -eq 0 ]; then
        echo "$label" >>"$DONE_FILE"
        successful_jobs=$((successful_jobs + 1))
        pipeline_log "DONE ${label} (exit 0, $(format_duration "$elapsed"))"
    else
        failed_jobs=$((failed_jobs + 1))
        pipeline_log "FAIL ${label} (exit ${rc}, $(format_duration "$elapsed"))"
    fi
    unset "JOB_LABEL[$finished_pid]" "JOB_START[$finished_pid]"
}

# Run the job specs given as arguments with at most MAX_JOBS in flight.
# Specs are passed as arguments rather than on stdin so that no
# process-substitution child exists for `wait -n` to reap by mistake.
# Returns 0 when every dispatched job succeeded, 1 otherwise.
run_job_queue() {
    local max_jobs="${MAX_JOBS:-4}"
    local specs=("$@") spec label mode model extra running=0 skipped=0
    RUN_LOG_DIR="${PIPELINE_LOG_ROOT}/run_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$RUN_LOG_DIR"
    touch "$DONE_FILE"
    DRIVER_LOG="${RUN_LOG_DIR}/driver.log"
    declare -gA JOB_LABEL=() JOB_START=()
    successful_jobs=0
    failed_jobs=0
    local queue_start
    queue_start="$(date +%s)"
    activate_env_once

    pipeline_log "Queue: ${#specs[@]} jobs, MAX_JOBS=${max_jobs}, DONE_FILE=${DONE_FILE}, FORCE=${FORCE}"
    for spec in "${specs[@]}"; do
        [ -z "$spec" ] && continue
        IFS='|' read -r label mode model extra <<<"$spec"
        if is_done "$label"; then
            skipped=$((skipped + 1))
            continue
        fi
        if [ "$running" -ge "$max_jobs" ]; then
            reap_one_job
            running=$((running - 1))
        fi
        if launch_job "$label" "$mode" "$model" "$extra"; then
            running=$((running + 1))
        else
            failed_jobs=$((failed_jobs + 1))
        fi
        # misc_utils.log_dir() stamps run dirs at 1s resolution.
        sleep 2
    done
    while [ "$running" -gt 0 ]; do
        reap_one_job
        running=$((running - 1))
    done

    local total_seconds=$(($(date +%s) - queue_start))
    pipeline_log "Finished: ok=${successful_jobs} failed=${failed_jobs} skipped_done=${skipped} runtime=$(format_duration "$total_seconds")"
    [ "$failed_jobs" -eq 0 ]
}

# Shared entrypoint for the expN scripts: `--list` prints specs, otherwise
# the specs are run through the queue. $1 is the name of a function that
# prints the job specs.
experiment_main() {
    local emit_function="$1"
    shift
    if [ "${1:-}" = "--list" ]; then
        "$emit_function"
        return 0
    fi
    local specs=()
    mapfile -t specs < <("$emit_function")
    run_job_queue "${specs[@]}"
}
