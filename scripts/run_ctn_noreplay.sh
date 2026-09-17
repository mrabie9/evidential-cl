#!/bin/bash
# CTN no-replay ablation (T3): n_memories=0 removes all retained data.
# Same regime as the other no-AMP CTN ablations (n_epochs=1, inner_steps=2,
# --no-amp). The three seeds run IN PARALLEL (one single-seed sweep each) to
# cut wall time to ~1 seed-run; aggregate stats are computed from the per-seed
# metrics npz afterwards.
#
# Usage:  bash scripts/run_ctn_noreplay.sh
set -uo pipefail

REPO="/home/lunet/wsmr11/repos/evidential-cl"
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
BASE_CONFIG="$REPO/configs/base.yaml"
LOGDIR="$REPO/scripts/logs/ctn_ablations"
mkdir -p "$LOGDIR"

run_seed() {
  local seed="$1"
  local out_log="$LOGDIR/ctn_noreplay_se_noamp_s${seed}.log"
  echo "[run_ctn_noreplay] START seed $seed -> $out_log"
  "$PY" "$REPO/main.py" \
    --config "$BASE_CONFIG" \
    --config "$REPO/configs/models/til/ctn_noreplay.yaml" \
    --n_epochs 1 --inner_steps 2 --no-amp \
    --expt_name "noreplay-ablation_se_noamp_til_s${seed}" \
    --seeds "$seed" >"$out_log" 2>&1
  echo "[run_ctn_noreplay] DONE  seed $seed (exit $?)"
}

run_seed 0  & PID0=$!
run_seed 39 & PID39=$!
run_seed 55 & PID55=$!

wait "$PID0";  S0=$?
wait "$PID39"; S39=$?
wait "$PID55"; S55=$?

echo "[run_ctn_noreplay] all done. exits: s0=$S0 s39=$S39 s55=$S55"
echo "  logs/ctn_noreplay/noreplay-ablation_se_noamp_til_s*-*/results.txt"
