#!/bin/bash
# Re-run of the seed-55 arm of run_absom_lwf_seeds.sh: the first attempt was
# SIGKILLed at task 6 while sharing the box with the tail of the cautious batch
# (swap was 6/7 GB at the time). Run alone, same configuration otherwise.
set -u

REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/absom_lwf
mkdir -p "$LOGS"

PYTHONUNBUFFERED=1 $PY $REPO/main.py \
  --config configs/base.yaml \
  --config configs/models/til/woe_si_lwf.yaml \
  --n_epochs 1 --inner_steps 2 --lr 0.003 \
  --woe_omega_transform abs --woe_anchor_mode proximal \
  --woe_lambda 76000.0 --woe_lwf_lambda 1.0 \
  --single-seed --seed 55 --no-save_checkpoints \
  --expt_name absom-lwf-lam76000-s55 > "$LOGS/absom-lwf-lam76000-s55.log" 2>&1
echo "[absom-lwf s55] done"
