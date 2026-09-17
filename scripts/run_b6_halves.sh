#!/bin/bash
# B6, split down the middle: which half of I_2 carries the ranking?
#
# I_2 = ||z'||^2 + 2 sum_k w+_k w-_k  ("logit" + "conflict", _least_commitment_terms).
# The path integral is *linear* in the tracked scalar, so
# omega_i2 = omega_logit + omega_conflict exactly -- the only additive
# decomposition of omega in this family. Every other B6 arm is a substitution
# (`ce`) or a factor-removal (`displacement`); this one is a true split.
#
# The abs projection at consolidation does NOT preserve that additivity
# (|a+b| != |a|+|b|), which is why the halves must be run live rather than read
# off the i2 field.
#
# Prior (shadow_s0 dumps, i2 trajectory, abs transform): Omega totals are
# i2 1.2258e2, conflict 9.198e1, logit 4.223e1, and the i2 field agrees with
# conflict at rho 0.972 / cos 0.992 against logit's 0.850 / 0.985. So the field
# evidence says conflict is nearly all of Omega. Whether that reaches accuracy is
# the question; note the halves are themselves collinear (cos 0.962), so the
# fields cannot attribute and only live arms can.
#
# NOTE `logit` is NOT `z2`. Under centered_uniform sum_j w_jk = z_k - beta_k.mu,
# and the divisor is J^2 rather than the active-class count (z2's total Omega is
# 4.52e6 against logit's 42.2). B6's claim that z2 and i2 "differ only by the
# conflict term" identifies z2 with a quantity it is not.
#
# Lambda: match lambda*Omega to i2's tuned peak, 2.4e5 * 122.58 = 2.94e7.
#   logit    2.94e7 / 42.23 = 6.97e5  -> centre 7e5
#   conflict 2.94e7 / 91.98 = 3.20e5  -> centre 3.2e5
# A decade either side. Omega ratios place grids, never single cells: i1 was
# predicted at 6.6e3 and peaked near 1.8e3, A7's max was 1.76x predicted against
# 3x real. The shadow Omegas are also measured on the *i2* trajectory, so the
# live arms will not reproduce them exactly -- another reason for a wide grid.
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/b6_halves
mkdir -p "$LOGS"
# shellcheck source=/dev/null
. "$REPO/scripts/lib_joblimit.sh"

run_arm () {
  local scalar=$1 lam=$2 seed=${3:-0}
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

for lam in 70000 220000 700000 2200000 7000000; do
  run_arm logit "$lam"
done
for lam in 32000 100000 320000 1000000 3200000; do
  run_arm conflict "$lam"
done
wait
echo "[b6 halves] done"
