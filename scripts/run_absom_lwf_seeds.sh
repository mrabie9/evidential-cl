#!/bin/bash
# A6 x B4: does the `abs` Omega transform hold up on top of LwF at n=3?
#
# B4's headline cell (anchor + LwF, 0.5223 +- 0.0067) predates the
# woe_omega_transform flag, so it is `relu` by construction. The `abs` variant
# has only ever been run on seed 0 on top of LwF: 0.5234 at lambda=7.6e4 against
# relu's 0.5249 at 1e6 -- a tie on one seed, which this project has been misled
# by four times. Seeds 39 and 55 at that same cell complete the paired n=3.
#
# lambda=7.6e4 rather than A6's anchor-only optimum of 2.4e5: at 2.4e5 the
# seed-0 combined cell is 0.4969, already 0.028 below relu, so that cell is not
# worth three runs. 7.6e4 is where abs+LwF is competitive.
#
# Chained behind the running cautious-rule jobs on purpose: three concurrent
# 10-task runs already distort the runtime column, and seven would be worse.
set -u

REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/absom_lwf
mkdir -p "$LOGS"

while pgrep -f "expt_name caut" > /dev/null; do
  sleep 60
done

launch () {
  local seed=$1
  local name="absom-lwf-lam76000-s${seed}"
  echo "[launch] $name"
  PYTHONUNBUFFERED=1 $PY $REPO/main.py \
    --config configs/base.yaml \
    --config configs/models/til/woe_si_lwf.yaml \
    --n_epochs 1 --inner_steps 2 --lr 0.003 \
    --woe_omega_transform abs --woe_anchor_mode proximal \
    --woe_lambda 76000.0 --woe_lwf_lambda 1.0 \
    --single-seed --seed "$seed" --no-save_checkpoints \
    --expt_name "$name" > "$LOGS/$name.log" 2>&1
}

launch 39 &
launch 55 &
wait
echo "[absom-lwf] done"
