#!/bin/bash
# Re-tune WoE-SI's lambda at p=0. The standing value 2.4e5 was tuned at p=0.2 and
# carried over, which makes the current tie with proximal si unfair in si's
# favour: si was swept to its own peak at p=0, WoE-SI was not.
#
# Direction is genuinely open. Removing dropout raised the Omega *mass* 2.55x with
# the anchored count unchanged, so at a fixed lambda the anchor already runs ~2.5x
# stronger than the p=0.2 tuning intended -- which argues the peak moved DOWN,
# toward the mass-matched 9.6e4. But the wider engagement (25% of parameters in
# b in [0.1,1) vs 7%) is part of why the arm won, and scaling lambda down undoes
# it. The grid spans both sides rather than assuming either.
#
# 2.4e5 is not re-run: it is the existing n=3 arm (0.5823 +/- 0.0080,
# scripts/logs/dropout_none), and seed 0 there scored 0.5750.
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/woesi_lambda_p0
mkdir -p "$LOGS"
MAXJOBS=3
PIDS=()

slot () {
  while true; do
    local alive=()
    for pid in ${PIDS[@]+"${PIDS[@]}"}; do
      kill -0 "$pid" 2>/dev/null && alive+=("$pid")
    done
    PIDS=(${alive[@]+"${alive[@]}"})
    [ "${#PIDS[@]}" -lt "$MAXJOBS" ] && break
    sleep 20
  done
  sleep 20
}

run_woe () {
  local lam=$1 seed=$2
  local name="woesip0_lam${lam}_s${seed}"
  slot
  echo "[$(date +%H:%M:%S)] launch $name"
  RESNET1D_DROPOUT=0 WOE_LC_DEBUG=1 PYTHONUNBUFFERED=1 $PY $REPO/main.py \
    --config configs/base.yaml \
    --config configs/models/til/woe_si_lc.yaml \
    --n_epochs 1 --inner_steps 2 --lr 0.003 \
    --woe_omega_transform abs --woe_anchor_mode proximal \
    --woe_omega_accum sum --woe_importance_scalar i2 \
    --woe_lambda "$lam" \
    --single-seed --seed "$seed" --no-save_checkpoints \
    --expt_name "$name" > "$LOGS/$name.log" 2>&1 &
  PIDS+=($!)
}

for lam in 30000 100000 600000 1500000; do run_woe "$lam" 0; done
wait
echo "[$(date +%H:%M:%S)] woe-si lambda bracket done"
