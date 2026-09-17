#!/bin/bash
# Which factor of omega = sum_steps h . Delta supplies the ranking?
#
# A9 (`uniform`) removed the whole of Omega and showed per-parameter importance
# is worth +0.066. It did not say *which* factor earns that. `displacement` sets
# h = dScalar/dtheta to a constant, so the path integral telescopes to the task's
# net displacement |Delta^t| -- the non-DS, non-gradient half of the product.
#
# Prediction, from the dumps: Spearman(Omega_abs, Omega_|Delta|) = -0.152 and
# Spearman(Omega, total |Delta|) = -0.060, so displacement carries almost none of
# the ranking and this arm should land near or below `uniform` (0.4345). If it
# instead matches `abs` (0.5008), the gradient of the tracked scalar is doing
# nothing and A9's +0.066 belongs to displacement alone.
#
# Lambda: total Omega is 1.52e4 against abs's 1.23e2, a ratio of 124, so matched
# lambda ~1.9e3. Grid spans a decade either side (Omega ratios place grids only).
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/displacement
mkdir -p "$LOGS"
# shellcheck source=/dev/null
. "$REPO/scripts/lib_joblimit.sh"

for lam in 200 600 1900 6000 19000; do
  name="disp_lam${lam}"
  job_slot
  echo "[launch] $name"
  WOE_LC_DEBUG=1 PYTHONUNBUFFERED=1 $PY $REPO/main.py \
    --config configs/base.yaml \
    --config configs/models/til/woe_si_lc.yaml \
    --n_epochs 1 --inner_steps 2 --lr 0.003 \
    --woe_omega_transform displacement --woe_anchor_mode proximal \
    --woe_omega_accum sum --woe_lambda "$lam" \
    --single-seed --seed 0 --no-save_checkpoints \
    --expt_name "$name" > "$LOGS/$name.log" 2>&1 &
  PIDS+=($!)
done
wait
echo "[displacement control] done"
