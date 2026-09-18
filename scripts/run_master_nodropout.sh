#!/bin/bash
# Master driver: DS-content arms, then the eralg4 and si baselines at p=0.
#
# Launched with setsid so the runs survive whatever has been reaping the
# background task -- two earlier attempts were killed mid-flight (once after
# 4 min, once after 2 min) and produced nothing. This script is also its own
# limiter: lib_joblimit.sh's MINFREEGB gate stalled the queue for 49 minutes on
# the second attempt, so the memory check is dropped and only the hard count of
# 3 concurrent runs is kept, which is what every batch that finished today used.
#
# Phase 1  DS content at p=0 (8 runs) -- see run_ds_content_nodropout.sh for why.
# Phase 2  eralg4 at p=0 and p=0.2, 3 seeds each.
# Phase 3  si at p=0 and p=0.2, 3 seeds each.
#
# Phases 2 and 3 run BOTH dropout settings rather than pairing against a recorded
# number: no baseline for these two exists on today's tree at this harness
# (n_epochs 1, inner_steps 2), and the recorded 0.6082 for eralg4 was measured
# under settings this script cannot verify. Each method keeps its own config lr
# (eralg4 0.01, si 0.003 with si_c 0.4), so the dropout delta is within-method.
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs
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

woe () {
  local scalar=$1 lam=$2 seed=$3
  local name="dsnd_${scalar}_lam${lam}_s${seed}"
  mkdir -p "$LOGS/ds_nodropout"
  slot
  echo "[$(date +%H:%M:%S)] launch $name"
  RESNET1D_DROPOUT=0 WOE_LC_DEBUG=1 PYTHONUNBUFFERED=1 $PY $REPO/main.py \
    --config configs/base.yaml \
    --config configs/models/til/woe_si_lc.yaml \
    --n_epochs 1 --inner_steps 2 --lr 0.003 \
    --woe_omega_transform abs --woe_anchor_mode proximal \
    --woe_omega_accum sum --woe_importance_scalar "$scalar" \
    --woe_lambda "$lam" \
    --single-seed --seed "$seed" --no-save_checkpoints \
    --expt_name "$name" > "$LOGS/ds_nodropout/$name.log" 2>&1 &
  PIDS+=($!)
}

baseline () {
  local model=$1 tag=$2 sched=$3 seed=$4
  local name="base_${model}_${tag}_s${seed}"
  mkdir -p "$LOGS/baselines_dropout"
  slot
  echo "[$(date +%H:%M:%S)] launch $name (RESNET1D_DROPOUT=$sched)"
  RESNET1D_DROPOUT="$sched" PYTHONUNBUFFERED=1 $PY $REPO/main.py \
    --config configs/base.yaml \
    --config "configs/models/til/${model}.yaml" \
    --n_epochs 1 --inner_steps 2 \
    --single-seed --seed "$seed" --no-save_checkpoints \
    --expt_name "$name" > "$LOGS/baselines_dropout/$name.log" 2>&1 &
  PIDS+=($!)
}

echo "=== phase 1: DS content at p=0 ==="
for seed in 0 39 55; do woe ce 75 "$seed"; done
for seed in 0 39 55; do woe conflict 295000 "$seed"; done
woe ce 40 0
woe ce 140 0

echo "=== phase 2: eralg4, both dropout settings ==="
for seed in 0 39 55; do baseline eralg4 dropnone "0" "$seed"; done
for seed in 0 39 55; do baseline eralg4 dropflat "0.2" "$seed"; done

echo "=== phase 3: si, both dropout settings ==="
for seed in 0 39 55; do baseline si dropnone "0" "$seed"; done
for seed in 0 39 55; do baseline si dropflat "0.2" "$seed"; done

wait
echo "[$(date +%H:%M:%S)] master done"
