#!/bin/bash
# Bracket the *normalised* arms above their current top cell. `sum_norm` was
# still rising at 4.8e5 (0.4411 -> 0.4775 -> 0.4776), so comparing a bracketed
# `max_norm` against an unbracketed `sum_norm` would repeat exactly the error the
# first A7 pass made in the other direction.
#
# NOTE on concurrency: the previous driver's `wait_slot` counted jobs via
# $(jobs -rp | wc -l), and command substitution runs in a subshell where `jobs`
# reports nothing -- so the cap never engaged and only the memory gate throttled.
# Tracked here with an explicit PID array instead.
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/ds_gates
mkdir -p "$LOGS"
MAXJOBS=3
MINFREEGB=60
PIDS=()

wait_slot () {
  while true; do
    local alive=()
    for pid in ${PIDS[@]+"${PIDS[@]}"}; do
      kill -0 "$pid" 2>/dev/null && alive+=("$pid")
    done
    PIDS=(${alive[@]+"${alive[@]}"})
    local freegb
    freegb=$(free -g | awk '/^Mem:/ {print $7}')
    if [ "${#PIDS[@]}" -lt "$MAXJOBS" ] && [ "$freegb" -ge "$MINFREEGB" ]; then
      break
    fi
    sleep 30
  done
  sleep 45
}

launch () {
  local name=$1 accum=$2 lam=$3
  wait_slot
  echo "[launch] $name accum=$accum lambda=$lam"
  WOE_LC_DEBUG=1 PYTHONUNBUFFERED=1 $PY $REPO/main.py \
    --config configs/base.yaml \
    --config configs/models/til/woe_si_lc.yaml \
    --n_epochs 1 --inner_steps 2 --lr 0.003 \
    --woe_omega_transform abs --woe_anchor_mode proximal \
    --woe_omega_accum "$accum" --woe_lambda "$lam" \
    --single-seed --seed 0 --no-save_checkpoints \
    --expt_name "$name" > "$LOGS/$name.log" 2>&1 &
  PIDS+=($!)
}

launch cn_sumnorm_lam960000  sum_norm   960000.0
launch cn_sumnorm_lam1920000 sum_norm  1920000.0
launch cn_maxnorm_lam1920000 max_norm  1920000.0
wait
echo "[norm bracket] done"
