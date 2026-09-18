#!/bin/bash
# Is the conflict share an *actionable* signal or only a post-hoc one?
#
# The task-end conflict share tracks task difficulty tightly (pooled
# corr(share, diagonal) = -0.85 over 9 runs, 81 task-observations). That alone is
# not useful: by the time it is available so is the diagonal F1 it correlates
# with, and the capacity has already been spent. The allocation-relevant question
# is whether the share over the *first* WOE_SPLIT_EARLY steps predicts the task's
# eventual diagonal -- a signal available before the decision, and computable
# without labels.
#
# Three seeds, sum/2.4e5 (the n=3 control), so the early-vs-late comparison is
# made on the arm every other number in A7 is quoted from.
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/conflict_early
mkdir -p "$LOGS"

for seed in 0 39 55; do
  name="confearly_s${seed}"
  echo "[launch] $name"
  WOE_LC_DEBUG=1 WOE_SPLIT_EARLY=32 PYTHONUNBUFFERED=1 $PY $REPO/main.py \
    --config configs/base.yaml \
    --config configs/models/til/woe_si_lc.yaml \
    --n_epochs 1 --inner_steps 2 --lr 0.003 \
    --woe_omega_transform abs --woe_anchor_mode proximal \
    --woe_omega_accum sum --woe_lambda 240000.0 \
    --single-seed --seed "$seed" --no-save_checkpoints \
    --expt_name "$name" > "$LOGS/$name.log" 2>&1 &
done
wait
echo "[early probe] done"
