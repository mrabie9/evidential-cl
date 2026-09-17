#!/bin/bash
# Res-ER (eralg4) + LwF distillation, CIL -- does a function-space term help where the
# buffer-side one was never tested?
#
# LwF distills the CURRENT task's batch toward a teacher frozen at the previous task
# boundary, over the columns of completed classes (model/lwf_regulariser.py, the same
# code path si / rwalk / woe_si use). It is NOT --er_distill, which puts KL on the
# REPLAY rows inside each row's own task class slice.
#
# Single knob off the pinned CIL E0 baseline (logs/ablations/cil/res-er/E0,
# expt eralg4_resER_joint_se_cil): same lr 0.01, is2, single epoch, --eralg4_joint_er,
# AMP on. Seeds 0,39,55 are the first three of the canonical n=9 set, so the comparison
# is paired seed-for-seed against E0 (13.71 / 14.63 / 11.69 final F1).
#
# Usage:  bash scripts/run_cil_eralg4_lwf.sh
#         SEEDS=7,13,21 bash scripts/run_cil_eralg4_lwf.sh   # top-up
set -uo pipefail

REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1   # main.py writes logs/ relative to cwd; anchor it here
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"

SEEDS="${SEEDS:-0,39,55}"
LOGDIR="${LOGDIR:-$REPO/scripts/logs/cil_eralg4_lwf}"
mkdir -p "$LOGDIR"

echo "[lwf] ===== eralg4 + LwF (CIL)  seeds=$SEEDS  START $(date) ====="
"$PY" "$REPO/main.py" \
  --config "$REPO/configs/base.yaml" \
  --config "$REPO/configs/models/cil/eralg4_lwf.yaml" \
  --n_epochs 1 --inner_steps 2 --no-save_checkpoints \
  --expt_name "eralg4_resER_joint_lwf_se_cil" \
  --seeds "$SEEDS" > "$LOGDIR/eralg4_resER_joint_lwf_se_cil.log" 2>&1
echo "[lwf] ===== END exit=$?  $(date) ====="
echo "[lwf] results: logs/eralg4_lwf/eralg4_resER_joint_lwf_se_cil-*/{${SEEDS//,/,}}/results.txt"
