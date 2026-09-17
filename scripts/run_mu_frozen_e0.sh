#!/bin/bash
# PR-3, E0: the bit-identity gate.
#
# `ema` must reproduce the recorded i2 control on the B6 anchor-only host
# exactly: diagonal 0.5206 / final 0.5008 / backward -0.0198 (seed 0, lambda
# 2.4e5). The pre-pass runs in this arm too -- that is the whole point of the
# gate, since the pre-pass is the only thing that could break the equality.
#
# Nothing else in the campaign launches until this passes.
#
# Checkpoints are ON deliberately: every recent probe script passed
# --no-save_checkpoints, which left the 2026-08 campaign unrecoverable for
# retrospective analysis. 741 MB/run against 954 GB free.
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/mu_frozen
mkdir -p "$LOGS"

WOE_LC_DEBUG=1 PYTHONUNBUFFERED=1 $PY $REPO/main.py \
  --config configs/base.yaml \
  --config configs/models/til/woe_si_lc.yaml \
  --n_epochs 1 --inner_steps 2 --lr 0.003 \
  --woe_omega_transform abs --woe_anchor_mode proximal \
  --woe_omega_accum sum --woe_importance_scalar i2 \
  --woe_lambda 240000.0 \
  --woe_mu_mode ema \
  --single-seed --seed 0 --save_checkpoints \
  --expt_name mufz_e0_ema_lam240000_s0 > "$LOGS/mufz_e0_ema_lam240000_s0.log" 2>&1

echo "[E0] exit=$?"
grep -E "Diagonal F1:|Final F1:|Backward:" $REPO/logs/woe_si_lc/mufz_e0_ema_lam240000_s0-*/0/results.txt 2>/dev/null | tail -3
