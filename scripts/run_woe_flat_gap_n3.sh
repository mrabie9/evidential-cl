#!/bin/bash
# Close the one n=1 claim section F rests on: that WoE-SI's measured-vs-uniform
# gap is FLAT across epochs (+0.0567 -> +0.0566) while SI's widens x1.68 and
# EWC's x1.42.
#
# Single-run sd on this host is ~0.004 and a gap is a difference of two grid
# maxima, so gap sd is ~0.006-0.008. SI's +0.045 change clears that; WoE-SI's
# 0.0001 null does not distinguish "flat" from "lucky" at n=1. Only the WoE-SI
# 3-epoch pair is re-run -- at its own peak lambdas, not a grid, since the
# brackets are already closed (measured 1e5, uniform 1).
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/multiepoch
mkdir -p "$LOGS"
# shellcheck source=/dev/null
. "$REPO/scripts/lib_joblimit.sh"

launch () {
  local name=$1; shift
  job_slot
  echo "[launch] $name"
  PYTHONUNBUFFERED=1 $PY $REPO/main.py \
    --config configs/base.yaml \
    --config configs/models/til/woe_si_lc.yaml \
    --lr 0.003 --n_epochs 3 --inner_steps 2 \
    --single-seed --no-save_checkpoints \
    --expt_name "$name" "$@" > "$LOGS/$name.log" 2>&1 &
  PIDS+=($!)
}

for seed in 39 55; do
  launch "me3n3_woe_meas_s${seed}" --seed "$seed" --woe_lambda 100000
  launch "me3n3_woe_unif_s${seed}" --seed "$seed" \
    --woe_omega_transform uniform --woe_lambda 1
done
wait
echo "[chain] woe flat-gap n=3 complete"
