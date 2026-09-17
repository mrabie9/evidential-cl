#!/bin/bash
# Does un-flooring SI's path-length denominator help?
#
# xi = 1e-3 against a median Delta^2 of 4.7e-8 leaves the denominator constant
# for 99.99% of parameters, so Omega has been a raw path integral throughout the
# project rather than the curvature-like quantity SI intends. This is the first
# change to Omega here that is not DS-motivated.
#
# Choosing xi. Measured offline from the xiparts dump (numerator and Delta^2 are
# stored separately, so Omega rebuilds at any xi without retraining):
#
#   xi      floored frac   Omega mass ratio   Spearman vs 1e-3
#   1e-6      0.8945            529x               0.913
#   1e-7      0.6103           3271x               0.854
#   1e-8      0.2997          16048x               0.790
#
# 1e-6 would have been a near-no-op -- 89% still floored -- which is why the
# grid was sized from the dump before any run was launched. 1e-8 is the first
# value where the denominator is live for most parameters. Both are run: 1e-6 as
# the marginal case, 1e-8 as the real one.
#
# Choosing lambda. Matched anchor is 2.4e5 / mass_ratio -> ~450 at 1e-6 and ~15
# at 1e-8. This project has four recorded instances of an Omega ratio failing to
# predict the optimal lambda (off by up to 3.7x), so each grid spans a decade
# either side of the prediction rather than testing the predicted cell.
#
# Read against xi=1e-3, lambda=2.4e5: 0.5206 / 0.5008 / -0.0198 (n=3, +/-0.0031).
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/xi_fix
mkdir -p "$LOGS"
MAXJOBS=3
MINFREEGB=60
PIDS=()

wait_slot () {
  while true; do
    local alive=()
    for pid in ${PIDS[@]+"${PIDS[@]}"}; do
      kill -0 "$pid" 2>/dev/null && alive+=("$pid")
    done
    PIDS=(${alive[@]+"${alive[@]}"})
    local freegb
    freegb=$(free -g | awk '/^Mem:/ {print $7}')
    if [ "${#PIDS[@]}" -lt "$MAXJOBS" ] && [ "$freegb" -ge "$MINFREEGB" ]; then break; fi
    sleep 30
  done
  sleep 45
}

launch () {
  local xi=$1 lam=$2
  local name="xifix_xi${xi}_lam${lam}"
  wait_slot
  echo "[launch] $name"
  WOE_LC_DEBUG=1 PYTHONUNBUFFERED=1 $PY $REPO/main.py \
    --config configs/base.yaml \
    --config configs/models/til/woe_si_lc.yaml \
    --n_epochs 1 --inner_steps 2 --lr 0.003 \
    --woe_omega_transform abs --woe_anchor_mode proximal \
    --woe_omega_accum sum --woe_xi "$xi" --woe_lambda "$lam" \
    --single-seed --seed 0 --no-save_checkpoints \
    --expt_name "$name" > "$LOGS/$name.log" 2>&1 &
  PIDS+=($!)
}

for lam in 45 150 450 1350 4050; do launch 1e-6 "$lam"; done
for lam in 1.5 5 15 45 135; do launch 1e-8 "$lam"; done
wait
echo "[xi fix sweep] done"
