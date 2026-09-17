#!/bin/bash
# CIL E0 (Res-ER full method) re-run with the sister-repo two-forward loop.
#
# The default eralg4 ER path concatenates replay+current into a single batch and forwards
# once; that mixes BatchNorm statistics across old/new data and corrupts the retention loss
# (see memory: eralg4 BN mixing breaks replay). La-MAML/model/eralg4.py instead runs the
# current live batch and the replay batch through *separate* forwards and sums the losses.
# This repo mirrors that path behind the --eralg4_joint_er PROBE flag (ER_joint()).
#
# This script re-runs the CIL E0 baseline row with that probe enabled, at the same operating
# point as scripts/run_cil_ablations.sh (cil/eralg4.yaml, lr 0.01, inner_steps 2, 1 epoch),
# over the full n=9 seed set so it pairs directly against the existing E0 baseline
# (logs/eralg4/eralg4_resER_se_cil-*).
set -uo pipefail

REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1   # main.py writes logs/ relative to cwd; anchor it here
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
BASE="$REPO/configs/base.yaml"
CFG="$REPO/configs/models/cil"

SEEDS="${SEEDS:-0,39,55,7,13,21,1,2,3}"
LOGDIR="${LOGDIR:-$REPO/scripts/logs/cil_e0_joint}"
mkdir -p "$LOGDIR"

echo "[e0joint] START eralg4_resER_joint_se_cil  seeds=$SEEDS  $(date)"
"$PY" "$REPO/main.py" \
  --config "$BASE" \
  --config "$CFG/eralg4.yaml" \
  --n_epochs 1 --inner_steps 2 --lr 0.01 --no-save_checkpoints \
  --eralg4_joint_er \
  --expt_name eralg4_resER_joint_se_cil \
  --seeds "$SEEDS" > "$LOGDIR/eralg4_resER_joint_se_cil.log" 2>&1
echo "[e0joint] END   exit=$?  $(date)"
echo "[e0joint] results: logs/eralg4/eralg4_resER_joint_se_cil-<ts>/{$SEEDS}/results.txt"
