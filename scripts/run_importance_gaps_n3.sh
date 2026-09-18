#!/bin/bash
# Put the whole of section F's table on n=3, at each cell's already-bracketed peak.
#
# The WoE-SI 3-epoch pair was run at n=3 first and REFUTED the n=1 reading (seed 0
# gave a x1.00 "flat gap"; seeds 39/55 gave x1.66 and x1.60, mean x1.42). The
# reason is that the UNIFORM control is the seed-noisy arm -- sd 0.0197 against
# the measured arm's 0.0062 -- so a gap, being a difference of two arms, carries
# sd ~0.02 at n=1. Every other cell in section F is still n=1 and sits under
# exactly that risk, so they get the same treatment rather than a hedge.
#
# Peaks are not re-swept; the brackets are closed and only the seed moves.
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/gaps_n3
mkdir -p "$LOGS"
# shellcheck source=/dev/null
. "$REPO/scripts/lib_joblimit.sh"

launch () {
  local name=$1; local cfg=$2; local ep=$3; local seed=$4; shift 4
  job_slot
  echo "[launch] $name"
  PYTHONUNBUFFERED=1 $PY $REPO/main.py \
    --config configs/base.yaml \
    --config "configs/models/til/${cfg}.yaml" \
    --n_epochs "$ep" --inner_steps 2 \
    --single-seed --seed "$seed" --no-save_checkpoints \
    --expt_name "$name" "$@" > "$LOGS/$name.log" 2>&1 &
  PIDS+=($!)
}

for seed in 39 55; do
  # --- 1 epoch (peaks: si 1e3/3, ewc 1e6/12, woe 2.4e5 abs / 3 uniform) ---
  launch "gn3_e1_si_meas_s${seed}"  si  1 "$seed" --anchor_mode proximal --si_c 1000
  launch "gn3_e1_si_unif_s${seed}"  si  1 "$seed" --anchor_mode proximal --anchor_omega_uniform --si_c 3
  launch "gn3_e1_ewc_meas_s${seed}" ewc 1 "$seed" --anchor_mode proximal --lamb 1000000
  launch "gn3_e1_ewc_unif_s${seed}" ewc 1 "$seed" --anchor_mode proximal --anchor_omega_uniform --lamb 12
  launch "gn3_e1_woe_meas_s${seed}" woe_si_lc 1 "$seed" --lr 0.003 --woe_omega_transform abs --woe_lambda 240000
  launch "gn3_e1_woe_unif_s${seed}" woe_si_lc 1 "$seed" --lr 0.003 --woe_omega_transform uniform --woe_lambda 3
  # --- 3 epoch (peaks: si 300/3, ewc 1e6/40; woe already n=3) ---
  launch "gn3_e3_si_meas_s${seed}"  si  3 "$seed" --anchor_mode proximal --si_c 300
  launch "gn3_e3_si_unif_s${seed}"  si  3 "$seed" --anchor_mode proximal --anchor_omega_uniform --si_c 3
  launch "gn3_e3_ewc_meas_s${seed}" ewc 3 "$seed" --anchor_mode proximal --lamb 1000000
  launch "gn3_e3_ewc_unif_s${seed}" ewc 3 "$seed" --anchor_mode proximal --anchor_omega_uniform --lamb 40
done
wait
echo "[chain] importance gaps n=3 complete"
