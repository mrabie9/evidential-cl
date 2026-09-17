#!/bin/bash
# A7 follow-up, batch 3. The sum arm is still *improving* as lambda falls
# (0.5008 at 2.4e5, 0.5036 at 1.2e5), so A6's recorded peak of 2.4e5 is not the
# sum optimum either. Bracket it, otherwise every sum-vs-max comparison is being
# made against a sum arm that is itself mis-tuned -- which would be the same
# criticism levelled at the max arm, applied in reverse.
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/cautious_lowlam
mkdir -p "$LOGS"

launch () {
  local name=$1 accum=$2 lam=$3
  echo "[launch] $name"
  WOE_LC_DEBUG=1 PYTHONUNBUFFERED=1 $PY $REPO/main.py \
    --config configs/base.yaml \
    --config configs/models/til/woe_si_lc.yaml \
    --n_epochs 1 --inner_steps 2 --lr 0.003 \
    --woe_omega_transform abs --woe_anchor_mode proximal \
    --woe_omega_accum "$accum" --woe_lambda "$lam" \
    --single-seed --seed 0 --no-save_checkpoints \
    --expt_name "$name" > "$LOGS/$name.log" 2>&1
}

launch caut_sum_lam60000_s0 sum 60000.0 &
launch caut_sum_lam30000_s0 sum 30000.0 &
wait
echo "[batch3] done"
