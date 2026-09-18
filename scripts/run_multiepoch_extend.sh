#!/bin/bash
# Close the two 3-epoch brackets that ran to their grid edge.
#
# EWC uniform: 0.4/1.2/4 -> 0.3425/0.3678/0.3748, rising to the edge. Its
#   one-shot counterpart peaked at lambda 12, so the 3-epoch grid simply started
#   too low; 0.3748 is a LOWER bound on the uniform peak and the +0.1476 gap is
#   an upper bound on what EWC's Fisher is worth.
# WoE-SI measured: 3e5/1e6/3e6 -> 0.4889/0.4735/0.4819, best at the LOW edge, so
#   the peak is below 3e5 -- consistent with Omega growing ~3x with the epochs.
#
# SI needs no extension: both its arms peak in the interior (measured 300 of
# 100/300/1000, uniform 3 of 1/3/10).
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
    --n_epochs 3 --inner_steps 2 \
    --single-seed --seed 0 --no-save_checkpoints \
    --expt_name "$name" "$@" > "$LOGS/$name.log" 2>&1 &
  PIDS+=($!)
}

for lam in 12 40 120; do
  launch "me3_ewc_unif_lam${lam}" --config configs/models/til/ewc.yaml \
    --anchor_mode proximal --anchor_omega_uniform --lamb "$lam"
done
for lam in 30000 100000; do
  launch "me3_woe_meas_lam${lam}" --config configs/models/til/woe_si_lc.yaml \
    --lr 0.003 --woe_lambda "$lam"
done
# WoE-SI uniform also peaked at its low edge (1/3/10 -> 0.4463/0.4238/0.4190),
# so its baseline is a lower bound too and the +0.0426 gap is unsettled in BOTH
# directions. SI's uniform peaked interior at 3, so this is specific to the
# 3-epoch WoE grid starting too high.
for lam in 0.1 0.3; do
  launch "me3_woe_unif_lam${lam}" --config configs/models/til/woe_si_lc.yaml \
    --lr 0.003 --woe_omega_transform uniform --woe_lambda "$lam"
done
wait
echo "[chain] multi-epoch brackets closed"
