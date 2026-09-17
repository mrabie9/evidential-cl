#!/bin/bash
# Replicate the si proximal peak on seeds 39 and 55.
#
# The seed-0 bracket is unimodal and the peak is now INTERIOR, which is what the
# first grid failed to establish (it topped out at its own edge):
#
#   si_c     1      10     100    1000    3000   10000   30000
#   final  0.3193 0.3429 0.5020 0.5876  0.5520  0.5240  0.5107
#   BWT    -0.362 -0.340 -0.152 -0.018  -0.006  -0.002  +0.002
#
# Past 1000 the anchor keeps buying retention (BWT reaches zero and turns
# positive) but the diagonal falls faster than the forgetting it prevents, so
# 1000 is the trade-off point rather than a boundary.
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/si_proximal
mkdir -p "$LOGS"
MAXJOBS=3
PIDS=()

slot () {
  while true; do
    local alive=()
    for pid in ${PIDS[@]+"${PIDS[@]}"}; do
      kill -0 "$pid" 2>/dev/null && alive+=("$pid")
    done
    PIDS=(${alive[@]+"${alive[@]}"})
    [ "${#PIDS[@]}" -lt "$MAXJOBS" ] && break
    sleep 20
  done
  sleep 20
}

run_si () {
  local sic=$1 seed=$2 sched=$3 tag=$4
  local name="siprox_${tag}c${sic}_s${seed}"
  slot
  echo "[$(date +%H:%M:%S)] launch $name"
  RESNET1D_DROPOUT="$sched" PYTHONUNBUFFERED=1 $PY $REPO/main.py \
    --config configs/base.yaml \
    --config configs/models/til/si.yaml \
    --n_epochs 1 --inner_steps 2 \
    --anchor_mode proximal --si_c "$sic" \
    --single-seed --seed "$seed" --no-save_checkpoints \
    --expt_name "$name" > "$LOGS/$name.log" 2>&1 &
  PIDS+=($!)
}

run_si 1000 39 "0" ""
run_si 1000 55 "0" ""
# p=0.2 companion at the same si_c, so the dropout delta for si is measured with
# a working anchor rather than the broken loss-form one.
run_si 1000 0 "0.2" "flat"
run_si 1000 39 "0.2" "flat"
run_si 1000 55 "0.2" "flat"
wait
echo "[$(date +%H:%M:%S)] si proximal seeds done"
