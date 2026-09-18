#!/bin/bash
# Is per-parameter importance carrying anything on this benchmark, or is every
# anchor here L2-SP with a decorative Omega?
#
# WoE-SI already has the answer for itself: woe_omega_transform='uniform' scores
# 0.4345 against abs's 0.5008, so its Omega is worth +0.066. SI and EWC had no
# such control, so their proximal frontiers (SI 0.5101 at lambda 1e3, EWC 0.4792
# at 1e6, one-shot) could not be read. --anchor_omega_uniform adds it.
#
# Lambda placement is arithmetic, not a guess: uniform Omega is ones, and the
# measured medians after 9 tasks are 2.65e-3 (SI) and 1.22e-6 (EWC), so the
# matched strengths are ~1e3/380 ~ 2.6 and ~1e6/8.2e5 ~ 1.2. Each grid spans a
# decade either side of that.
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/omega_uniform
mkdir -p "$LOGS"
# shellcheck source=/dev/null
. "$REPO/scripts/lib_joblimit.sh"

for lam in 0.4 1.2 4 12; do
  name="omguni_ewc_lam${lam}"
  job_slot
  echo "[launch] $name"
  PYTHONUNBUFFERED=1 $PY $REPO/main.py \
    --config configs/base.yaml \
    --config configs/models/til/ewc.yaml \
    --n_epochs 1 --inner_steps 2 \
    --anchor_mode proximal --anchor_omega_uniform \
    --lamb "$lam" \
    --single-seed --seed 0 --no-save_checkpoints \
    --expt_name "$name" > "$LOGS/$name.log" 2>&1 &
  PIDS+=($!)   # REQUIRED: job_slot counts this array (lib_joblimit.sh)
done
wait
echo "[done] omega uniform control"
