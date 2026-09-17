#!/bin/bash
# Extend the si proximal si_c bracket: the first grid peaked at its EDGE.
#
# Seed 0, p=0, proximal:  1 -> 0.3193, 10 -> 0.3429, 100 -> 0.5020, 1000 -> 0.5876
# with BWT climbing -0.362 -> -0.018 and the diagonal falling 0.682 -> 0.605. The
# anchor is still buying more retention than the plasticity it costs at the top of
# the grid, so 1000 is a boundary, not a peak, and replicating it would be
# replicating a clipped bracket.
#
# The grid ran two to three decades above run_anchor_mode_benchmark.py's 1e1-1e2
# sizing for si proximal, so that guidance is wrong for this harness (n_epochs 1,
# inner_steps 2, p=0) and is not a reason to stop here.
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

for sic in 3000 10000 30000; do run_si "$sic" 0; done
wait
echo "[$(date +%H:%M:%S)] si proximal extension done"
