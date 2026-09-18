#!/bin/bash
# The xi=1e-8 arm was still rising at its top cell (0.4695 at lambda=135), so it
# is unbracketed above and cannot be compared against a bracketed baseline --
# the error this project has now made twice in the other direction.
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/xi_fix
mkdir -p "$LOGS"
for lam in 405 1215 3645; do
  while [ "$(free -g | awk '/^Mem:/ {print $7}')" -lt 60 ]; do sleep 30; done
  name="xifix_xi1e-8_lam${lam}"
  echo "[launch] $name"
  WOE_LC_DEBUG=1 PYTHONUNBUFFERED=1 $PY $REPO/main.py \
    --config configs/base.yaml \
    --config configs/models/til/woe_si_lc.yaml \
    --n_epochs 1 --inner_steps 2 --lr 0.003 \
    --woe_omega_transform abs --woe_anchor_mode proximal \
    --woe_omega_accum sum --woe_xi 1e-8 --woe_lambda "$lam" \
    --single-seed --seed 0 --no-save_checkpoints \
    --expt_name "$name" > "$LOGS/$name.log" 2>&1 &
  sleep 45
done
wait
echo "[xi bracket] done"
