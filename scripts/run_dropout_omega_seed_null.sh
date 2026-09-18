#!/bin/bash
# Seed null for the SUPPORT half of scripts/dropout_omega_compare.py.
#
# The seed-0 dumps say the arms rank different parameters (Spearman 0.61 between
# no-dropout and flat p=0.2, top-1% overlap 0.45). That number is uninterpretable
# on its own: two runs of the *same* configuration at different seeds are also
# two different trajectories and would disagree by some unknown amount. This adds
# seed 39 for two arms so the within-arm, across-seed agreement can be measured
# and the between-arm figure read against it.
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/dropout_omega
DUMPS=$REPO/scripts/logs/omega_dumps
mkdir -p "$LOGS"
# shellcheck source=/dev/null
. "$REPO/scripts/lib_joblimit.sh"

arm () {
  local tag=$1 sched=$2 seed=$3
  local name="omdump_${tag}_s${seed}"
  job_slot
  echo "[launch] $name (RESNET1D_DROPOUT=$sched)"
  RESNET1D_DROPOUT="$sched" WOE_OMEGA_DUMP="$DUMPS/drop_${tag}_s${seed}" \
  WOE_LC_DEBUG=1 PYTHONUNBUFFERED=1 $PY $REPO/main.py \
    --config configs/base.yaml \
    --config configs/models/til/woe_si_lc.yaml \
    --n_epochs 1 --inner_steps 2 --lr 0.003 \
    --woe_omega_transform abs --woe_anchor_mode proximal \
    --woe_omega_accum sum --woe_importance_scalar i2 \
    --woe_lambda 240000.0 \
    --single-seed --seed "$seed" --no-save_checkpoints \
    --expt_name "$name" > "$LOGS/$name.log" 2>&1 &
  PIDS+=($!)
}

arm none "0" 39
arm flat02 "0.2" 39
wait
echo "[dropout omega seed null] done"
