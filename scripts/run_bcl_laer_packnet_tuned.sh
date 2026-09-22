#!/bin/bash
# Run the tuned-hyperparam matrix for bcl_dual, la-er and packnet concurrently
# (up to MAX_JOBS at once), using the already-updated model configs:
#   - configs/models/{cil,til}/bcl_dual.yaml  (memory_strength/beta)
#   - configs/models/til/la-er.yaml           (memory_loss_lambda: 1.0)
#   - configs/models/til/packnet.yaml         (post_prune_epochs: 1)
#
# Jobs (8 total):
#   bcl_dual : one-shot (single-pass) + 5e(full), for both til and cil
#   la-er    : one-shot (single-pass) + 5e(full), til only
#   packnet  : one-shot (single-pass) + 5e(full), til only
#
# Each job is dispatched via scripts/full_experiments.sh --models <model>
# (which itself handles config chaining + env activation). Single-pass jobs
# pass --n_epochs 1 --inner_steps 1 explicitly (NOT the --one-shot flag,
# which forces inner_steps=2); the plain (5e) job relies on the n_epochs: 5 /
# inner_steps: 1 defaults in configs/base.yaml.
#
# Set SKIP_LABELS="label1,label2" to skip dispatching jobs already running/
# completed elsewhere (comma-separated, matching the labels in JOBS below).

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1

MAX_JOBS="${MAX_JOBS:-4}"
SKIP_LABELS="${SKIP_LABELS:-}"

is_skipped() {
    local label="$1" item
    IFS=',' read -r -a skip_arr <<<"$SKIP_LABELS"
    for item in "${skip_arr[@]}"; do
        [ "$item" = "$label" ] && return 0
    done
    return 1
}

STAMP="$(date +%Y%m%d_%H%M%S)"
DRIVER_LOG_DIR="${REPO_ROOT}/logs/batch_runs/run_${STAMP}"
mkdir -p "$DRIVER_LOG_DIR"
DRIVER_LOG="${DRIVER_LOG_DIR}/driver.log"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$DRIVER_LOG"; }

format_duration() {
    local total_seconds="$1"
    printf '%02d:%02d:%02d' $((total_seconds / 3600)) $(((total_seconds % 3600) / 60)) $((total_seconds % 60))
}

# label|model|mode|extra_flags
JOBS=(
    "bcl_dual_til_oneshot|bcl_dual|til|--n_epochs 1 --inner_steps 1"
    "bcl_dual_til_5e|bcl_dual|til|"
    "bcl_dual_cil_oneshot|bcl_dual|cil|--n_epochs 1 --inner_steps 1"
    "bcl_dual_cil_5e|bcl_dual|cil|"
    "la-er_til_oneshot|la-er|til|--n_epochs 1 --inner_steps 1"
    "la-er_til_5e|la-er|til|"
    "packnet_til_oneshot|packnet|til|--n_epochs 1 --inner_steps 1"
    "packnet_til_5e|packnet|til|"
)

log "=== run_bcl_laer_packnet_tuned.sh started ==="
log "REPO_ROOT=$REPO_ROOT"
log "DRIVER_LOG_DIR=$DRIVER_LOG_DIR"
log "MAX_JOBS=$MAX_JOBS"
log "Jobs: ${#JOBS[@]} total (bcl_dual x4, la-er x2, packnet x2)"

declare -A JOB_PID_LABEL=()
SCRIPT_START_EPOCH="$(date +%s)"
overall_exit=0
successful_runs=0
unsuccessful_runs=0

launch_job() {
    local label="$1" model="$2" mode="$3" extra="$4"
    local job_log="${DRIVER_LOG_DIR}/${label}.log"
    log "Dispatching ${label} (model=${model} mode=${mode} extra='${extra:-none}') -> ${job_log}"
    (
        # shellcheck disable=SC2086
        bash scripts/full_experiments.sh --models "$model" --mode "$mode" $extra -d "$label" >"$job_log" 2>&1
    ) &
    local pid=$!
    JOB_PID_LABEL[$pid]="$label"
}

job_index=0
running=0
total=${#JOBS[@]}

while [ "$job_index" -lt "$total" ] || [ "$running" -gt 0 ]; do
    while [ "$running" -lt "$MAX_JOBS" ] && [ "$job_index" -lt "$total" ]; do
        IFS='|' read -r label model mode extra <<<"${JOBS[$job_index]}"
        if is_skipped "$label"; then
            log "Skipping ${label} (in SKIP_LABELS)"
            job_index=$((job_index + 1))
            continue
        fi
        launch_job "$label" "$model" "$mode" "$extra"
        job_index=$((job_index + 1))
        running=$((running + 1))
        # Stagger launches by >1s: misc_utils.log_dir() names each run's output
        # dir as logs/<config-stem>/<timestamp>_<expt_name>/<seed>/, using a
        # 1-second-resolution timestamp and an expt_name that does NOT encode
        # til/cil mode. Two same-model jobs starting in the same second collide
        # on that path and silently overwrite each other's seed outputs.
        sleep 2
    done

    if [ "$running" -gt 0 ]; then
        wait -n -p finished_pid
        rc=$?
        finished_label="${JOB_PID_LABEL[$finished_pid]:-unknown_pid_${finished_pid}}"
        if [ "$rc" -eq 0 ]; then
            log "Completed: ${finished_label} (exit 0)"
            successful_runs=$((successful_runs + 1))
        else
            log "ERROR: ${finished_label} failed with exit code ${rc}"
            unsuccessful_runs=$((unsuccessful_runs + 1))
            overall_exit=1
        fi
        running=$((running - 1))
    fi
done

script_end_epoch="$(date +%s)"
total_runtime_seconds=$((script_end_epoch - SCRIPT_START_EPOCH))
log "=== run_bcl_laer_packnet_tuned.sh finished: successful_runs=${successful_runs} unsuccessful_runs=${unsuccessful_runs} total_runtime=$(format_duration "$total_runtime_seconds") (${total_runtime_seconds}s) ==="

exit "$overall_exit"
