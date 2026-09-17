#!/bin/bash
# Is the DS-specific content still inert once trunk dropout is removed?
#
# B6's null -- the DS scalar `i2` ties the plain `ce` control -- has a mechanistic
# explanation: the two extract almost the same parameter ranking, so whichever is
# tracked, Omega comes out the same. That explanation was measured at p=0.2, where
# the anchor was barely engaged (93% of parameters at b<0.1). At p=0 it engages on
# 3.6x more parameters and importance is ~3x less concentrated, so the null is
# worth re-testing rather than assumed.
#
# WOE_OMEGA_SHADOW differentiates `ce`, `logit` and `conflict` on the *same*
# trajectory as the tracked `i2`, so their Omega fields are compared with no
# lambda confound and no separate run per scalar. `conflict` is the DS-specific
# half of I_2; `logit` is the half that is just confidence.
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/dropout_ds
DUMPS=$REPO/scripts/logs/omega_dumps
mkdir -p "$LOGS"
# shellcheck source=/dev/null
. "$REPO/scripts/lib_joblimit.sh"

arm () {
  local tag=$1 sched=$2
  local name="dsshadow_${tag}_s0"
  job_slot
  echo "[launch] $name (RESNET1D_DROPOUT=$sched)"
  RESNET1D_DROPOUT="$sched" WOE_OMEGA_SHADOW="ce,logit,conflict" \
  WOE_OMEGA_DUMP="$DUMPS/dsshadow_${tag}_s0" \
  WOE_LC_DEBUG=1 PYTHONUNBUFFERED=1 $PY $REPO/main.py \
    --config configs/base.yaml \
    --config configs/models/til/woe_si_lc.yaml \
    --n_epochs 1 --inner_steps 2 --lr 0.003 \
    --woe_omega_transform abs --woe_anchor_mode proximal \
    --woe_omega_accum sum --woe_importance_scalar i2 \
    --woe_lambda 240000.0 \
    --single-seed --seed 0 --no-save_checkpoints \
    --expt_name "$name" > "$LOGS/$name.log" 2>&1 &
  PIDS+=($!)
}

arm none "0"
arm flat02 "0.2"
wait
echo "[ds shadow] done"
