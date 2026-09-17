#!/bin/bash
# Is the DS content effective or still inert, now that dropout is gone?
#
# Two contrasts, both against the no-dropout i2 arm already measured at n=3
# (0.5823 +/- 0.0080, scripts/logs/dropout_none):
#
#   ce       -- B6's control. A plain loss-based importance with no DS content.
#               If it ties i2 again, the DS-specific content is still inert.
#   conflict -- the DS-specific half of I_2 (the other half, `logit`, is just
#               confidence). If DS content carries anything, this is where.
#
# Lambda is NOT guessed. The shadow dump (scripts/run_dropout_ds_shadow.sh)
# carries every scalar's Omega field on the same trajectory, so lambda is set to
# put the SAME NUMBER of parameters at b >= 0.1 as i2 at 2.4e5 -- count, not
# mass, is what transfers lambda here. That calibration returns 72 for ce, and
# B6's independently tuned ce peak at p=0.2 was 75, so the rule reproduces the
# empirical value and 75 is kept. conflict calibrates to 2.95e5.
#
# The seed-0 bracket at 40 / 140 guards the recurring failure mode in this
# project: an n=3 "confirmation" run at the wrong lambda.
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/ds_nodropout
mkdir -p "$LOGS"
# shellcheck source=/dev/null
. "$REPO/scripts/lib_joblimit.sh"

arm () {
  local scalar=$1 lam=$2 seed=$3
  local name="dsnd_${scalar}_lam${lam}_s${seed}"
  job_slot
  echo "[launch] $name"
  RESNET1D_DROPOUT=0 WOE_LC_DEBUG=1 PYTHONUNBUFFERED=1 $PY $REPO/main.py \
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

for seed in 0 39 55; do arm ce 75 "$seed"; done
for seed in 0 39 55; do arm conflict 295000 "$seed"; done
arm ce 40 0
arm ce 140 0
wait
echo "[ds content no-dropout] done"
