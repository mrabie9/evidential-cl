#!/bin/bash
# Does SI's own path-length correction survive the anchor, or does xi floor it?
#
# WoE-SI keeps canonical SI's denominator exactly:
#   Omega_i^t = |omega_i^t| / ((Delta_i^t)^2 + xi)      (model/woe_si.py:2000)
# with Delta_i^t the task's total displacement. So the 10-30x suppression of
# later tasks' path integrals under the anchor is already *post*-correction --
# which is either a real statement about importance estimation under sequential
# penalties, or an artefact of xi.
#
# The third possibility neither reading covers: the correction only divides by
# path length while Delta^2 >> xi. Once the anchor pushes |Delta| below
# sqrt(xi) = 0.0316 (xi = 1e-3 here) the denominator floors at xi and the
# correction switches off -- precisely when it is most needed. Then omega is
# proportional to Delta rather than inverse in it, and the suppression is
# expected rather than surprising.
#
# `delta_rms` and `delta_floored_frac` in the [LC] trace decide between them.
# Two runs: anchor off (the uncorrected baseline) and anchor at the peak.
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/ds_gates
mkdir -p "$LOGS"
MAXJOBS=2
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
    if [ "${#PIDS[@]}" -lt "$MAXJOBS" ] && [ "$freegb" -ge "$MINFREEGB" ]; then break; fi
    sleep 30
  done
  sleep 45
}

launch () {
  local name=$1 lam=$2
  wait_slot
  echo "[launch] $name lambda=$lam"
  WOE_LC_DEBUG=1 PYTHONUNBUFFERED=1 $PY $REPO/main.py \
    --config configs/base.yaml \
    --config configs/models/til/woe_si_lc.yaml \
    --n_epochs 1 --inner_steps 2 --lr 0.003 \
    --woe_omega_transform abs --woe_anchor_mode proximal \
    --woe_omega_accum sum --woe_lambda "$lam" \
    --single-seed --seed 0 --no-save_checkpoints \
    --expt_name "$name" > "$LOGS/$name.log" 2>&1 &
  PIDS+=($!)
}

launch dsat_lam0      0.0
launch dsat_lam240000 240000.0
wait
echo "[delta saturation] done"
