#!/bin/bash
# GOAL PROBE (causal test): is C-MAML's TIL edge over Res-ER just gradient
# variance reduction under dropout + clipping?
#
# resnet1d carries four Dropout(p=0.2) layers and ResNet1D.forward forces
# train mode on every call, so every forward is stochastic. Measured on a fixed
# batch and weights (K draws averaged, vs a 60-draw reference gradient):
#
#   K=1  cosine 0.694  ||g|| 5.57   <- eralg4: clipped every step (thr 5.0)
#   K=3  cosine 0.854  ||g|| 4.52   <- C-MAML meta_batches: NOT clipped
#   K=10 cosine 0.944  ||g|| 4.10
#
# Both methods clip on every step, so the update is a fixed-size step whose only
# free variable is direction -- and C-MAML averages 3 stochastic forwards of the
# same batch (meta_batches=3) where eralg4 uses one. That predicts every
# observation: the edge is plasticity (+2.58 diagonal), M3 (K_meta=1) is the one
# non-inert C-MAML ablation, M1/M4 are inert, and the gap is widest on the tasks
# where learning is hardest.
#
# --eralg4_grad_avg K gives eralg4 the same averaging. K=3 is COMPUTE-MATCHED to
# C-MAML (3 forwards per optimizer step). If E0 at K=3 reaches M0's 63.8, the
# "advantage" is variance reduction, not meta-learning, and the paper claim
# changes accordingly.
#
# Everything else at the E0 operating point: eralg4.yaml, --eralg4_joint_er,
# --lr 0.01, single-epoch TIL, --inner_steps 2, seeds 0,39,55.
set -uo pipefail
REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
BASE="$REPO/configs/base.yaml"
CFG="$REPO/configs/models/til"
SEEDS="${SEEDS:-0,39,55}"
LOGDIR="${LOGDIR:-$REPO/scripts/logs/e0_gradavg}"
mkdir -p "$LOGDIR"

run_job() {
  local k="$1"
  echo "[gavg] START K=$k $(date '+%H:%M:%S')"
  "$PY" "$REPO/main.py" \
    --config "$BASE" --config "$CFG/eralg4.yaml" \
    --n_epochs 1 --inner_steps 2 --no-save_checkpoints \
    --eralg4_joint_er --lr 0.01 --eralg4_grad_avg "$k" \
    --expt_name "e0_gradavg${k}_joint_se_til" --seeds "$SEEDS" \
    > "$LOGDIR/e0_gradavg${k}.log" 2>&1
  echo "[gavg] END   K=$k exit=$? $(date '+%H:%M:%S')"
}

echo "[gavg] ===== START seeds=$SEEDS $(date) ====="
run_job 3 &
run_job 5 &
wait
echo "[gavg] ===== ALL DONE $(date) ====="
