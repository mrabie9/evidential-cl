#!/bin/bash
# si with the proximal anchor at p=0, swept then replicated.
#
# The loss-form arm (base_si_drop*, si_c=0.4) scored ~0.33 with BWT ~ -0.35
# against a 0.65-0.69 diagonal: it learned each task and lost it. si_c=0.4 is a
# loss-mode value and carries no information about proximal, which needs its own
# strength -- scripts/run_anchor_mode_benchmark.py sizes si's proximal grid at
# 1e1-1e2, two to three decades above the as-configured 0.4.
#
# Phase 1 brackets si_c on seed 0 across three decades. Phase 2 replicates the
# peak on seeds 39 and 55 and is launched by run_si_proximal_seeds.sh once the
# bracket is read -- picking the peak automatically from inside one script is how
# a mis-bracketed n=3 gets published.
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
  local sic=$1 seed=$2
  local name="siprox_c${sic}_s${seed}"
  slot
  echo "[$(date +%H:%M:%S)] launch $name"
  RESNET1D_DROPOUT=0 PYTHONUNBUFFERED=1 $PY $REPO/main.py \
    --config configs/base.yaml \
    --config configs/models/til/si.yaml \
    --n_epochs 1 --inner_steps 2 \
    --anchor_mode proximal --si_c "$sic" \
    --single-seed --seed "$seed" --no-save_checkpoints \
    --expt_name "$name" > "$LOGS/$name.log" 2>&1 &
  PIDS+=($!)
}

for sic in 1 10 100 1000; do run_si "$sic" 0; done
wait
echo "[$(date +%H:%M:%S)] si proximal bracket done"
