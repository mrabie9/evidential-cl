#!/bin/bash
# GOAL PROBE: is C-MAML's TIL advantage over Res-ER an optimisation-strength /
# tuning artefact rather than an algorithmic one?
#
# Post-fix TIL, n=6: M0 63.83+-0.36 vs E0 62.41+-0.80, +1.42 (p=0.031, 6/6 seeds),
# all of it plasticity (+2.58 diagonal). Every C-MAML meta mechanism is inert or
# borderline, and M4 (alpha=0) reduces to the SAME update rule as eralg4's
# ER_joint -- yet still leads by +1.04. So the lead is not the meta machinery.
#
# The per-task breakdown says the gap is largest exactly where Res-ER learns
# worst (task 9: E0 diag 52.1 vs M0 59.2; task 4: 53.1 vs 58.5) and smallest on
# the tasks both find easy. That is the signature of Res-ER being under-optimised
# at its configured step size, not of a missing mechanism.
#
# E0 runs at --lr 0.01, the config-matched point inherited from the budget-match
# study; it has never been tuned post-fix. This sweeps it. If any rate reaches
# M0's 63.8, the "advantage" is an HP artefact and the honest claim changes.
#
# All rows: eralg4.yaml, --eralg4_joint_er (as E0), single-epoch TIL,
# --inner_steps 2, seeds 0,39,55 to pair against the existing E0 pool.
set -uo pipefail
REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
BASE="$REPO/configs/base.yaml"
CFG="$REPO/configs/models/til"
SEEDS="${SEEDS:-0,39,55}"
LOGDIR="${LOGDIR:-$REPO/scripts/logs/e0_lr_sweep}"
mkdir -p "$LOGDIR"

run_job() {
  local lr="$1" tag="$2"
  echo "[e0lr] START lr=$lr $(date '+%H:%M:%S')"
  "$PY" "$REPO/main.py" \
    --config "$BASE" --config "$CFG/eralg4.yaml" \
    --n_epochs 1 --inner_steps 2 --no-save_checkpoints \
    --eralg4_joint_er --lr "$lr" \
    --expt_name "e0_lr${tag}_joint_se_til" --seeds "$SEEDS" \
    > "$LOGDIR/e0_lr${tag}.log" 2>&1
  echo "[e0lr] END   lr=$lr exit=$? $(date '+%H:%M:%S')"
}

echo "[e0lr] ===== START seeds=$SEEDS $(date) ====="
run_job 0.003 "003" &
run_job 0.02  "02" &
run_job 0.03  "03" &
wait
echo "[e0lr] ===== ALL DONE $(date) ====="
