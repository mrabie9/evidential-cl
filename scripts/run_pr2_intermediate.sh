#!/bin/bash
# PR-2a: the no-retrain test of whether xi's floor erases between-scalar
# differences in Omega. One dump per candidate tracked scalar, each carrying the
# numerator and Delta^2 separately, so Omega can be rebuilt offline at any xi and
# the scalars' profiles compared at 1e-3 against 1e-6. No retraining, so no
# lambda-retuning confound -- which is what disqualifies the naive fail branch.
#
# Plus two seeds of the CE rehearsal bar on the *injection* host: PR-1's band
# structure inherits a seed sd measured on `woe_si_lc` (anchored, lr 0.003), and
# the gate it governs runs on `woe_si_injection` (anchor off, lr 0.001). The
# threshold cannot be trusted until that host's own spread is measured.
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/ds_gates
DUMPS=$REPO/scripts/logs/omega_dumps
mkdir -p "$LOGS" "$DUMPS"
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

scalar_dump () {
  local scalar=$1
  wait_slot
  echo "[launch] xiparts_${scalar}"
  WOE_LC_DEBUG=1 WOE_OMEGA_DUMP="$DUMPS/xiparts_${scalar}" PYTHONUNBUFFERED=1 $PY $REPO/main.py \
    --config configs/base.yaml \
    --config configs/models/til/woe_si_lc.yaml \
    --n_epochs 1 --inner_steps 2 --lr 0.003 \
    --woe_omega_transform abs --woe_anchor_mode proximal \
    --woe_omega_accum sum --woe_lambda 240000.0 \
    --woe_importance_scalar "$scalar" \
    --single-seed --seed 0 --no-save_checkpoints \
    --expt_name "xiparts_${scalar}" > "$LOGS/xiparts_${scalar}.log" 2>&1 &
  PIDS+=($!)
}

ce_seed () {
  local seed=$1
  wait_slot
  echo "[launch] veh_ce_s${seed}"
  PYTHONUNBUFFERED=1 $PY $REPO/main.py \
    --config configs/base.yaml \
    --config configs/models/til/woe_si_injection.yaml \
    --n_epochs 1 --inner_steps 2 \
    --woe_replay_memories 256 --woe_replay_mode ce \
    --single-seed --seed "$seed" --no-save_checkpoints \
    --expt_name "veh_ce_s${seed}" > "$LOGS/veh_ce_s${seed}.log" 2>&1 &
  PIDS+=($!)
}

ce_seed 39
ce_seed 55
scalar_dump ce
scalar_dump phi2
scalar_dump z2
wait
echo "[pr2 intermediate] done"
