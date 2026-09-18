#!/bin/bash
# n=3 on the two cells that carry the refined B6 claim on the anchor-only host:
# `ce` at its peak (which ties i2, and a tie needs replication to be claimable)
# and `z2` at its peak (a 0.0176 loss, ~5.7x sd, the marginal one). `phi2` loses
# 0.0743 -- 24x sd -- and does not need seeds to be believed.
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/b6_anchor_only
mkdir -p "$LOGS"
# shellcheck source=/dev/null
. "$REPO/scripts/lib_joblimit.sh"

cell () {
  local scalar=$1 lam=$2 seed=$3
  local name="b6ao_${scalar}_lam${lam}_s${seed}"
  job_slot
  echo "[launch] $name"
  WOE_LC_DEBUG=1 PYTHONUNBUFFERED=1 $PY $REPO/main.py \
    --config configs/base.yaml \
    --config configs/models/til/woe_si_lc.yaml \
    --n_epochs 1 --inner_steps 2 --lr 0.003 \
    --woe_omega_transform abs --woe_anchor_mode proximal \
    --woe_omega_accum sum --woe_importance_scalar "$scalar" \
    --woe_lambda "$lam" \
    --single-seed --seed "$seed" --no-save_checkpoints \
    --expt_name "$name" > "$LOGS/$name.log" 2>&1 &
  PIDS+=($!)
}

for seed in 39 55; do cell ce 75 "$seed"; done
for seed in 39 55; do cell z2 9 "$seed"; done
wait
echo "[b6ao seeds] done"
