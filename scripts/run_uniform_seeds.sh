#!/bin/bash
# n=3 on the uniform-Omega peak. The margin against the path-integral anchor is
# 0.057 -- 18x the seed sd -- so this is unlikely to reverse, but four of the
# last five single-seed readings in this campaign did, and this cell now carries
# the claim that the path integral earns its place.
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/uniform_omega
mkdir -p "$LOGS"
# shellcheck source=/dev/null
. "$REPO/scripts/lib_joblimit.sh"

for seed in 39 55; do
  name="unif_lam3_s${seed}"
  job_slot
  echo "[launch] $name"
  WOE_LC_DEBUG=1 PYTHONUNBUFFERED=1 $PY $REPO/main.py \
    --config configs/base.yaml \
    --config configs/models/til/woe_si_lc.yaml \
    --n_epochs 1 --inner_steps 2 --lr 0.003 \
    --woe_omega_transform uniform --woe_anchor_mode proximal \
    --woe_omega_accum sum --woe_lambda 3 \
    --single-seed --seed "$seed" --no-save_checkpoints \
    --expt_name "$name" > "$LOGS/$name.log" 2>&1 &
  PIDS+=($!)
done
wait
echo "[uniform seeds] done"
