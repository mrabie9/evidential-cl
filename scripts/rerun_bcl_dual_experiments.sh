#!/bin/bash
# Re-run the bcl_dual experiment matrix ONLY (no tuning), picking up the corrected
# task_order_files from configs/base.yaml and the already-tuned beta=1.0 configs.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

source la-maml_env/bin/activate
export PYTHONPATH="$REPO_ROOT"
export PYTHONUNBUFFERED=1

STAMP="$(date +%Y%m%d_%H%M%S)"
DRIVER_LOG_DIR="${REPO_ROOT}/logs/bcl_dual_beta_rerun/exp_rerun_${STAMP}"
mkdir -p "$DRIVER_LOG_DIR"
DRIVER_LOG="${DRIVER_LOG_DIR}/driver.log"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$DRIVER_LOG"; }

run_step() {
    local label="$1"; shift
    log "START ${label}"
    printf 'CMD:%s\n' "$(printf ' %q' "$@")" | tee -a "$DRIVER_LOG"
    if "$@" >>"$DRIVER_LOG" 2>&1; then
        log "OK ${label}"
    else
        local rc=$?
        log "FAIL ${label} (exit ${rc})"
        return "$rc"
    fi
}

log "=== bcl_dual experiment rerun (corrected task_order_files) started ==="
log "task_order_files: $(grep '^task_order_files' configs/base.yaml)"

for seed in 0 39 55; do
    run_step "oneshot-til-seed${seed}" bash scripts/full_experiments.sh \
        --models bcl_dual --mode til --one-shot --seed "$seed" \
        -d "bcl_dual_oneshot_til_seed${seed}"
done

for seed in 0 39 55; do
    run_step "oneshot-cil-seed${seed}" bash scripts/full_experiments.sh \
        --models bcl_dual --mode cil --one-shot --seed "$seed" \
        -d "bcl_dual_oneshot_cil_seed${seed}"
done

run_step "full-til-10ep" bash scripts/full_experiments.sh \
    --models bcl_dual --mode til --seed 0 -d "bcl_dual_full_til_10ep"

run_step "full-cil-10ep" bash scripts/full_experiments.sh \
    --models bcl_dual --mode cil --seed 0 -d "bcl_dual_full_cil_10ep"

log "=== bcl_dual experiment rerun finished ==="
