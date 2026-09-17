#!/bin/bash
# One instrumented run carrying every candidate importance scalar's Omega on a
# single trajectory (WOE_OMEGA_SHADOW), so the between-scalar agreement can be
# measured without the confound that B6's dumps each came from their own run at
# their own anchor strength.
#
# The live scalar stays `i2` at its tuned lambda (2.4e5 on the abs/proximal
# anchor-only host), because the trajectory should be the one the campaign's
# conclusions were drawn on. `logit` and `conflict` are the two halves of I_2;
# `ce`, `z2` and `phi2` are B6's rivals. Cost is one extra forward/backward per
# scalar per importance window and nothing else -- the shadows never touch the
# loss, the feature mean, or the BatchNorm statistics (tests/test_omega_shadow.py).
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/ds_gates
DUMPS=$REPO/scripts/logs/omega_dumps
mkdir -p "$LOGS" "$DUMPS"

SEED=${1:-0}

WOE_LC_DEBUG=1 \
WOE_OMEGA_SHADOW=logit,conflict,ce,z2,phi2 \
WOE_OMEGA_DUMP="$DUMPS/shadow_s${SEED}" \
PYTHONUNBUFFERED=1 $PY $REPO/main.py \
  --config configs/base.yaml \
  --config configs/models/til/woe_si_lc.yaml \
  --n_epochs 1 --inner_steps 2 --lr 0.003 \
  --woe_omega_transform abs --woe_anchor_mode proximal \
  --woe_omega_accum sum --woe_lambda 240000.0 \
  --woe_importance_scalar i2 \
  --single-seed --seed "$SEED" --no-save_checkpoints \
  --expt_name "shadow_s${SEED}" > "$LOGS/shadow_s${SEED}.log" 2>&1

echo "[done] $LOGS/shadow_s${SEED}.log"
$PY $REPO/scripts/omega_shadow_correlate.py "$DUMPS/shadow_s${SEED}"
