#!/bin/bash
# A7 follow-up, batch 2. Two additions the challenge to the first sweep needs.
#
# (a) Seeds 39 and 55 on the headline cells, so sum-vs-max is finally at the
#     project's n=3 bar rather than seed 0 alone. Note the pipeline is
#     *deterministic* at fixed seed -- three independent sum/2.4e5 runs from
#     three different campaigns returned bit-identical 0.5206/0.5008/-0.0198 --
#     so all scatter here is between seeds, and any same-seed difference is a
#     real effect of the intervention.
#
# (b) The sum-side lambda curve. The first sweep varied lambda on `max` only and
#     compared against a single `sum` cell, which cannot separate "max is a
#     worse rule" from "max at 2.4e5 is simply a weaker anchor" (its Omega is
#     1.33x smaller). Two lower sum cells give a curve to compare against at
#     matched mean, which is what the minimax reading of the cautious rule
#     actually needs.
set -u

REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/cautious_lowlam
mkdir -p "$LOGS"

launch () {
  local name=$1 accum=$2 lam=$3 seed=$4
  echo "[launch] $name"
  WOE_LC_DEBUG=1 PYTHONUNBUFFERED=1 $PY $REPO/main.py \
    --config configs/base.yaml \
    --config configs/models/til/woe_si_lc.yaml \
    --n_epochs 1 --inner_steps 2 \
    --lr 0.003 \
    --woe_omega_transform abs \
    --woe_anchor_mode proximal \
    --woe_omega_accum "$accum" \
    --woe_lambda "$lam" \
    --single-seed --seed "$seed" --no-save_checkpoints \
    --expt_name "$name" > "$LOGS/$name.log" 2>&1
}

launch caut_sum_lam240000_s39 sum 240000.0 39 &
launch caut_sum_lam240000_s55 sum 240000.0 55 &
launch caut_max_lam240000_s39 max 240000.0 39 &
launch caut_max_lam240000_s55 max 240000.0 55 &
launch caut_sum_lam120000_s0  sum 120000.0 0 &
wait
echo "[batch2] done"
