#!/bin/bash
# One run whose dump carries the numerator and Delta^2 separately, so Omega can
# be recomputed offline at any xi. Needed because the campaign's framing now
# rests on B6, and a floored SI denominator is a competing explanation for it:
# if Omega = |sum h.Delta| / xi with xi constant, Omega is essentially a
# displacement measure and "the tracked scalar does no distinguishing work" is
# what you would observe whatever scalar you tracked.
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/ds_gates
DUMPS=$REPO/scripts/logs/omega_dumps
mkdir -p "$LOGS" "$DUMPS"
MINFREEGB=60
while [ "$(free -g | awk '/^Mem:/ {print $7}')" -lt "$MINFREEGB" ]; do sleep 30; done
WOE_LC_DEBUG=1 WOE_OMEGA_DUMP="$DUMPS/xiparts_s0" PYTHONUNBUFFERED=1 $PY $REPO/main.py \
  --config configs/base.yaml \
  --config configs/models/til/woe_si_lc.yaml \
  --n_epochs 1 --inner_steps 2 --lr 0.003 \
  --woe_omega_transform abs --woe_anchor_mode proximal \
  --woe_omega_accum sum --woe_lambda 240000.0 \
  --single-seed --seed 0 --no-save_checkpoints \
  --expt_name xiparts_s0 > "$LOGS/xiparts_s0.log" 2>&1
echo "[xi parts dump] done"
