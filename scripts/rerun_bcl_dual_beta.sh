#!/bin/bash
# Driver: retune bcl_dual beta (TIL + CIL) then re-run bcl_dual experiments.
# Phase 1 retunes beta (the parameter that the parser bug previously froze) and
# writes the best value back into the TIL/CIL model configs. Phase 2 re-runs the
# bcl_dual experiment matrix using those freshly tuned configs.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

source la-maml_env/bin/activate
export PYTHONPATH="$REPO_ROOT"
export PYTHONUNBUFFERED=1

STAMP="$(date +%Y%m%d_%H%M%S)"
DRIVER_LOG_DIR="${REPO_ROOT}/logs/bcl_dual_beta_rerun/run_${STAMP}"
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

log "=== bcl_dual beta retune + rerun started ==="

# ---- Phase 1: beta-only tuning (writes best beta into the model configs) ----
run_step "tune-til" python tuning/Alpha/tune_bcl.py \
    --config configs/tuning_defaults.yaml \
    --config configs/models/til/bcl_dual.yaml \
    --tune-only beta || exit 1
log "TIL config after tuning:"; sed -n '1,20p' configs/models/til/bcl_dual.yaml | tee -a "$DRIVER_LOG"

run_step "tune-cil" python tuning/Alpha/tune_bcl.py \
    --config configs/tuning_defaults.yaml \
    --config configs/models/cil/bcl_dual.yaml \
    --tune-only beta || exit 1
log "CIL config after tuning:"; sed -n '1,20p' configs/models/cil/bcl_dual.yaml | tee -a "$DRIVER_LOG"

# ---- Phase 2: experiment matrix (bcl_dual only) ----
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

log "=== bcl_dual beta retune + rerun finished ==="
