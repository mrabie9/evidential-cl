#!/bin/bash
# CIL counterpart of scripts/run_e0_gradavg.sh: does the gradient-averaging explanation
# for C-MAML's edge over Res-ER carry over from TIL to CIL?
#
# In TIL the whole M0-vs-E0 gap is gradient-noise averaging. resnet1d keeps four
# Dropout(p=0.2) layers and ResNet1D.forward forces train mode, so every forward is
# stochastic; C-MAML's meta_batches=3 loop incidentally averages three of them per
# optimizer step where eralg4 uses one. Giving Res-ER the same via --eralg4_grad_avg 3
# bought it +1.02 F1 (p 0.012, 8/9) and closed the M0 gap from +1.21 (p 0.004, 9/9) to
# +0.20 (p 0.570). Averaging saturates at K=3, so only K=3 is run here.
#
# CIL has never had this control: --eralg4_grad_avg has only ever been passed in TIL.
#
# MATCHED TO THE no-AMP E0 ARM. Config is the CIL E0 operating point exactly
# (configs/models/cil/eralg4.yaml -> lr 0.01, is2, one epoch, --eralg4_joint_er) with
# --no-amp, mirroring logs/eralg4/eralg4_resER_joint_noamp_se_cil-*, plus K=3. The
# comparison against M0 is therefore mixed-precision: M0's CIL arms all ran AMP on, on
# the grounds that C-MAML is AMP-insensitive -- established in TIL, never measured in
# CIL. That assumption is the one soft joint in the contrast.
#
# Canonical nine seeds so the pairing lands on the same seeds as both existing arms.
#
# Usage:  bash scripts/run_cil_e0_gradavg.sh
set -uo pipefail

REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1   # main.py resolves data_path and logs/ against cwd
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
EXPT="e0_gradavg3_joint_noamp_se_cil"
LOGDIR="${LOGDIR:-$REPO/scripts/logs/cil_e0_gradavg}"
mkdir -p "$LOGDIR"

SHARDS=("0,1,2,3,7" "13,21,39,55")

run_shard() {
  local idx="$1" seeds="$2"
  echo "[k3cil] START shard$idx seeds=$seeds $(date '+%F %H:%M:%S')"
  "$PY" "$REPO/main.py" \
    --config "$REPO/configs/base.yaml" \
    --config "$REPO/configs/models/cil/eralg4.yaml" \
    --n_epochs 1 --inner_steps 2 --lr 0.01 --no-save_checkpoints \
    --eralg4_joint_er --no-amp --eralg4_grad_avg 3 \
    --expt_name "$EXPT" \
    --seeds "$seeds" > "$LOGDIR/shard${idx}.log" 2>&1
  echo "[k3cil] END   shard$idx exit=$? $(date '+%F %H:%M:%S')"
}

echo "[k3cil] ===== START $EXPT $(date) ====="
for i in "${!SHARDS[@]}"; do
  run_shard "$i" "${SHARDS[$i]}" &
done
wait
echo "[k3cil] ===== ALL DONE $(date) ====="
echo "[k3cil] results: logs/eralg4/${EXPT}-<ts>/{seed}/results.txt"
