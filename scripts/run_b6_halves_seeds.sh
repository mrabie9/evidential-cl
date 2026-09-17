#!/bin/bash
# n=3 on the two B6-halves peaks. Both grids were unimodal and bracketed at
# seed 0, and both peaks landed at lambda*Omega ~ 3.0e7 against the i2 control's
# 2.94e7 -- so the halves are being anchored at the same effective strength as
# the whole, which is what makes the comparison readable.
#
# Seed 0: logit 0.5053 (lam 7e5), conflict 0.5041 (lam 3.2e5), against i2's
# 0.5008. Both are +0.003 to +0.005, i.e. 1-1.5x the i2 control's seed sd of
# 0.0031 -- exactly the size of effect this campaign has repeatedly seen reverse
# (B6's own seed-0 ordering did not replicate; see the `ce` arm). n=1 settles
# nothing here; the paired seeds are the result.
#
# Control for pairing (abs/proximal/sum, lambda 2.4e5):
#   i2  s0 0.5008  s39 0.5039  s55 0.4977   -> 0.5008 +/- 0.0031
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/b6_halves
mkdir -p "$LOGS"
# shellcheck source=/dev/null
. "$REPO/scripts/lib_joblimit.sh"

run_arm () {
  local scalar=$1 lam=$2 seed=$3
  local name="b6h_${scalar}_lam${lam}_s${seed}"
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

for seed in 39 55; do
  run_arm logit 700000 "$seed"
  run_arm conflict 320000 "$seed"
done
wait
echo "[b6 halves seeds] done"
