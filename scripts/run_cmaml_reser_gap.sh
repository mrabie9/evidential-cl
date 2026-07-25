#!/bin/bash
# Why is C-MAML (M0) 1.6 F1 below Res-ER (E0) in single-epoch TIL?
#
#   M0 59.1+-0.7   Diag 64.9   BWT -5.8   (n=9)
#   E0 60.7+-0.8   Diag 65.5   BWT -4.8   (n=9)
#   paired M0-E0: F1 -1.63, Diag -0.64, BWT -0.98  -> mostly retention.
#
# Both methods draw the SAME replay batch (global reservoir, 5120 slots, 128
# replay rows appended before 256 current rows), apply the SAME per-sample TIL
# mask, take the SAME number of SGD(0.01, m0.9) steps (inner_steps 2), and clip
# at the same norm. The meta machinery is inert (M1/M4 ~0). What differs is how
# the replay and current blocks are reduced into one scalar:
#
#   Res-ER (eralg4._weighted_multitask_loss):
#       loss = CE(current) + memory_loss_lambda * CE(replay)      # two means
#              -> replay share pinned at 1/2, total loss scale ~2 units
#   C-MAML (lamaml_cifar.meta_loss, historical):
#       loss = CE(concat(replay, current))                        # one mean
#              -> replay share set by the pooled inverse-frequency class weights
#
# That last point is the whole story: old-task classes are rare in the pooled
# batch, so replay rows collect much larger weights and their share of the loss
# GROWS with each task -- measured 0.34 (task 0) -> 0.74 (task 4) -> 0.85 (task
# 9), i.e. by the end the current task receives 15% of the gradient. Res-ER sits
# at 0.50 throughout. --cmaml_replay_loss_mode separates share from scale:
#
#   pooled      escalating share, scale 1  = M0 baseline (n=9, not re-run here)
#   split_norm  share 1:1,        scale 1  = replay share alone
#   split       share 1:1,        scale 2  = eralg4's reduction exactly
#   pooled + --opt_wt 0.02                 = step size alone (share untouched)
#
# RESULT (n=6, paired vs M0): split +3.03, split_norm +2.67 (both p=0.031, the
# permutation floor), opt_wt 0.02 +0.17 (p=0.78). All of the gain is diagonal
# F1; BWT is unchanged. See docs/cmaml_vs_reser_til.md.
#
# All three rows below use the M0 operating point otherwise: til/cmaml.yaml,
# --second_order, --inner_steps 2, --n_epochs 1, opt_wt 0.01 from the yaml.
# Seeds match the M0/E0 pools so every comparison is paired.
set -uo pipefail
REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1   # main.py resolves data_path and logs/ against cwd
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
BASE="$REPO/configs/base.yaml"
CFG="$REPO/configs/models/til"
SEEDS="${SEEDS:-0,39,55,7,13,21}"
LOGDIR="${LOGDIR:-$REPO/scripts/logs/cmaml_reser_gap}"
mkdir -p "$LOGDIR"

run_job() {
  local expt="$1" extra="$2"
  echo "[gap] START $expt extra='$extra' $(date '+%H:%M:%S')"
  "$PY" "$REPO/main.py" \
    --config "$BASE" --config "$CFG/cmaml.yaml" \
    --n_epochs 1 --inner_steps 2 --no-save_checkpoints \
    --second_order \
    --expt_name "$expt" --seeds "$SEEDS" $extra > "$LOGDIR/${expt}.log" 2>&1
  echo "[gap] END   $expt exit=$? $(date '+%H:%M:%S')"
}

echo "[gap] ===== START seeds=$SEEDS $(date) ====="
run_job "cmaml_splitloss_se_til"     "--cmaml_replay_loss_mode split" &
run_job "cmaml_splitnorm_se_til"     "--cmaml_replay_loss_mode split_norm" &
run_job "cmaml_pooled_lr02_se_til"   "--opt_wt 0.02" &
wait
echo "[gap] ===== ALL DONE $(date) ====="
echo "[gap] results: logs/cmaml/{cmaml_splitloss,cmaml_splitnorm,cmaml_pooled_lr02}_se_til-<ts>/<seed>/results.txt"
