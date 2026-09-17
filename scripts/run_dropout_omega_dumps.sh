#!/bin/bash
# Does trunk dropout change the *distribution* of importance, or only its scale?
#
# The accuracy arms answered "does it help" (no dropout wins by 0.0859) but the
# only distributional readout available from their logs was nonzero_frac, which
# was 0.9958 in all three arms -- i.e. importance is not concentrated by zeroing
# parameters, so that statistic cannot see the tail where Omega actually lives.
#
# WOE_OMEGA_DUMP writes each task's path integral and the cumulative Omega to
# disk, which gives Gini, top-1% mass share and the anchor's b-regime histogram
# per arm. Seed 0 only: this is a distributional question about one trajectory,
# not an effect size.
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/dropout_omega
DUMPS=$REPO/scripts/logs/omega_dumps
mkdir -p "$LOGS"
# shellcheck source=/dev/null
. "$REPO/scripts/lib_joblimit.sh"

arm () {
  local tag=$1 sched=$2
  local name="omdump_${tag}_s0"
  job_slot
  echo "[launch] $name (RESNET1D_DROPOUT=$sched)"
  RESNET1D_DROPOUT="$sched" WOE_OMEGA_DUMP="$DUMPS/drop_${tag}_s0" \
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
arm sched "0.5,0.4,0.3,0.2"
wait
echo "[dropout omega dumps] done"
