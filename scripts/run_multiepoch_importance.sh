#!/bin/bash
# Does a longer path integral rescue per-parameter importance on this benchmark?
#
# Every anchor number on record here was measured one-shot (--n_epochs 1
# --inner_steps 2), and the one-shot proximal frontiers are all within 0.031 of
# each other -- SI 0.5101, RWalk 0.4942, EWC 0.4792, WoE-SI 0.4788 -- with the
# same shape: diagonal traded monotonically for BWT until the network is nearly
# frozen. That is what an uninformative Omega looks like. The competing reading
# is that importance here is merely under-estimated: SI's path integral and EWC's
# Fisher both get ~3x more steps per task at --n_epochs 3.
#
# The two readings separate on the uniform control. If a longer run lets the
# MEASURED Omega pull further ahead of the uniform one, the estimate was noisy
# and more steps fix it. If the measured-vs-uniform gap is the same in both
# regimes, importance is uninformative in this domain however well it is
# estimated, and the anchor is L2-SP with decoration.
#
# Lambda placement. Uniform Omega counts tasks and does not move with epochs, so
# its grids are the ones already bracketed (SI ~2.6, EWC ~1.2, WoE-SI peaked at 3
# in the A9 sweep). Measured Omega GROWS with steps, so each measured peak should
# move DOWN; the grids are centred on the one-shot peak with a half-decade below.
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/multiepoch
mkdir -p "$LOGS"
# shellcheck source=/dev/null
. "$REPO/scripts/lib_joblimit.sh"

EPOCHS=3

launch () {
  local name=$1; shift
  job_slot
  echo "[launch] $name"
  PYTHONUNBUFFERED=1 $PY $REPO/main.py \
    --config configs/base.yaml \
    --n_epochs $EPOCHS --inner_steps 2 \
    --single-seed --seed 0 --no-save_checkpoints \
    --expt_name "$name" "$@" > "$LOGS/$name.log" 2>&1 &
  # REQUIRED by lib_joblimit.sh: job_slot counts this array, so without the
  # append the limiter sees no running jobs and admits every launch at once.
  PIDS+=($!)
}

for lam in 100 300 1000; do
  launch "me3_si_meas_lam${lam}" --config configs/models/til/si.yaml \
    --anchor_mode proximal --si_c "$lam"
done
for lam in 1 3 10; do
  launch "me3_si_unif_lam${lam}" --config configs/models/til/si.yaml \
    --anchor_mode proximal --anchor_omega_uniform --si_c "$lam"
done
for lam in 300000 1000000 3000000; do
  launch "me3_ewc_meas_lam${lam}" --config configs/models/til/ewc.yaml \
    --anchor_mode proximal --lamb "$lam"
done
for lam in 0.4 1.2 4; do
  launch "me3_ewc_unif_lam${lam}" --config configs/models/til/ewc.yaml \
    --anchor_mode proximal --anchor_omega_uniform --lamb "$lam"
done
for lam in 300000 1000000 3000000; do
  launch "me3_woe_meas_lam${lam}" --config configs/models/til/woe_si_lc.yaml \
    --lr 0.003 --woe_lambda "$lam"
done
for lam in 1 3 10; do
  launch "me3_woe_unif_lam${lam}" --config configs/models/til/woe_si_lc.yaml \
    --lr 0.003 --woe_omega_transform uniform --woe_lambda "$lam"
done
wait
echo "[done] multi-epoch importance campaign"
