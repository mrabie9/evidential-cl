#!/bin/bash
# GOAL PROBE (answer candidate): is C-MAML's TIL edge over Res-ER actually
# Res-ER's sensitivity to bf16 AMP?
#
# Single-task (task 0 only) bisection, identical harness, AMP toggled:
#
#            fp32     bf16 AMP    AMP cost
#   Res-ER   84.06      82.32       -1.74
#   C-MAML   82.59      82.42       -0.17
#
# In fp32 Res-ER is the BETTER single-task learner; bf16 AMP removes 1.74 points
# from it and only 0.17 from C-MAML. Every run in the grid is trained under
# autocast(bfloat16) (main.py wraps model.observe), so the reported "C-MAML
# advantage" may be an AMP artefact rather than an algorithmic property. This is
# the same failure class as the CTN bf16 regression that --no-amp recovered.
#
# Runs the full 10-task TIL sequence with --no-amp for both families at their
# grid operating points. If the +1.42 gap shrinks or reverses without AMP, the
# paper claim changes: it is a precision artefact, not an algorithmic advantage.
#
# Seeds 0,39,55 to pair against the existing n=6 pools.
set -uo pipefail
REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
BASE="$REPO/configs/base.yaml"
CFG="$REPO/configs/models/til"
SEEDS="${SEEDS:-0,39,55}"
LOGDIR="${LOGDIR:-$REPO/scripts/logs/amp_probe_til}"
mkdir -p "$LOGDIR"

echo "[amp] ===== START seeds=$SEEDS $(date) ====="

( echo "[amp] START e0_noamp $(date '+%H:%M:%S')"
  "$PY" "$REPO/main.py" --config "$BASE" --config "$CFG/eralg4.yaml" \
    --n_epochs 1 --inner_steps 2 --no-save_checkpoints \
    --eralg4_joint_er --lr 0.01 --no-amp \
    --expt_name e0_noamp_joint_se_til --seeds "$SEEDS" \
    > "$LOGDIR/e0_noamp.log" 2>&1
  echo "[amp] END   e0_noamp exit=$? $(date '+%H:%M:%S')" ) &

( echo "[amp] START m0_noamp $(date '+%H:%M:%S')"
  "$PY" "$REPO/main.py" --config "$BASE" --config "$CFG/cmaml.yaml" \
    --n_epochs 1 --inner_steps 2 --no-save_checkpoints \
    --second_order --cmaml_joint_er --no-amp \
    --expt_name m0_noamp_joint_se_til --seeds "$SEEDS" \
    > "$LOGDIR/m0_noamp.log" 2>&1
  echo "[amp] END   m0_noamp exit=$? $(date '+%H:%M:%S')" ) &

wait
echo "[amp] ===== ALL DONE $(date) ====="
