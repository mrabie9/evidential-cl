#!/bin/bash
# B6 re-run on a host where the anchor has room to matter.
#
# B6 concluded the DS content of the tracked scalar does no distinguishing work,
# on a proximal-anchor + LwF host. But there the anchor contributes only +0.030
# over LwF alone (0.5223 vs 0.4921) against seed sds of 0.0067-0.0169 -- under 2
# sigma of headroom. That is the same weak-vehicle objection that just retired
# C6 Claim 1: a null between scalars inside a mechanism contributing 0.03 cannot
# discriminate the scalars.
#
# The anchor-only host (`woe_si_lc`, no LwF) gives the anchor +0.207 over no
# anchor (0.5008 vs 0.2938) -- 7x the headroom. And the four scalars provably
# build *different* Omega there: pairwise Spearman 0.43-0.84 from the xiparts
# dumps. So the ingredients differ; whether that reaches accuracy is untested.
#
# Lambda per scalar, from each scalar's measured total Omega on this host:
#   i2   1.23e2   (x1)        peak 2.4e5   [already n=3: 0.5008 +/- 0.0031]
#   ce   3.93e5   (x3206)     matched ~75
#   phi2 1.78e5   (x1455)     matched ~165
#   z2   3.23e6   (x26341)    matched ~9.1
# Each grid spans a decade either side, because this project has five recorded
# instances of an Omega ratio mispredicting the optimal lambda.
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/b6_anchor_only
mkdir -p "$LOGS"
# shellcheck source=/dev/null
. "$REPO/scripts/lib_joblimit.sh"

run_cell () {
  local scalar=$1 lam=$2
  local name="b6ao_${scalar}_lam${lam}"
  job_slot
  echo "[launch] $name"
  WOE_LC_DEBUG=1 PYTHONUNBUFFERED=1 $PY $REPO/main.py \
    --config configs/base.yaml \
    --config configs/models/til/woe_si_lc.yaml \
    --n_epochs 1 --inner_steps 2 --lr 0.003 \
    --woe_omega_transform abs --woe_anchor_mode proximal \
    --woe_omega_accum sum --woe_importance_scalar "$scalar" \
    --woe_lambda "$lam" \
    --single-seed --seed 0 --no-save_checkpoints \
    --expt_name "$name" > "$LOGS/$name.log" 2>&1 &
  PIDS+=($!)
}

for lam in 7.5 25 75 225 750; do run_cell ce "$lam"; done
for lam in 16.5 55 165 495 1650; do run_cell phi2 "$lam"; done
for lam in 0.9 3 9 27 90; do run_cell z2 "$lam"; done
wait
echo "[b6 anchor-only] done"
