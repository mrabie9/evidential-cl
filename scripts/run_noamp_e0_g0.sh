#!/bin/bash
# Re-run the TIL baselines E0 (Res-ER) and G0 (GEM) with --no-amp.
#
# WHY: bf16 AMP is not method-neutral here. Measured on the full 10-task TIL
# sequence (n=3), --no-amp is worth +0.64 F1 / +1.73 diagonal to Res-ER and
# nothing to C-MAML (-0.10 / -0.22), which accounts for about half the reported
# C-MAML-over-Res-ER gap. See docs/cmaml_advantage_is_numerical.md. Any
# cross-method claim measured under AMP is therefore suspect, so the remaining
# baselines need an AMP-free reading.
#
# E0: eralg4.yaml --eralg4_joint_er --lr 0.01, matching the existing e0_noamp
#     runs exactly. Seeds 7,13,21 only -- 0,39,55 already exist, so this tops
#     that pool from n=3 to the full n=6.
# G0: gem.yaml --lr 0.01 (the operating point of logs/gem/gem_full_lr01abl-* (the bf16 pool this replaced),
#     whose runs were amp=True), inner_steps 2. All six seeds.
#
# GEM is the slow one (~24 min/seed vs ~14 for Res-ER), so G0 is split across
# two lanes by seed triplet to keep wall-clock down.
set -uo pipefail
REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
BASE="$REPO/configs/base.yaml"
CFG="$REPO/configs/models/til"
LOGDIR="${LOGDIR:-$REPO/scripts/logs/noamp_e0_g0}"
mkdir -p "$LOGDIR"

run_job() {
  local tag="$1" cfgfile="$2" expt="$3" seeds="$4" extra="$5"
  echo "[noamp] START $tag seeds=$seeds $(date '+%H:%M:%S')"
  "$PY" "$REPO/main.py" \
    --config "$BASE" --config "$CFG/${cfgfile}.yaml" \
    --n_epochs 1 --inner_steps 2 --no-save_checkpoints --no-amp \
    --expt_name "$expt" --seeds "$seeds" $extra \
    > "$LOGDIR/${tag}_${seeds//,/-}.log" 2>&1
  echo "[noamp] END   $tag seeds=$seeds exit=$? $(date '+%H:%M:%S')"
}

echo "[noamp] ===== START $(date) ====="
run_job "e0" "eralg4" "e0_noamp_joint_se_til" "7,13,21" "--eralg4_joint_er --lr 0.01" &
run_job "g0a" "gem" "g0_noamp_lr01_se_til" "0,39,55" "--lr 0.01" &
run_job "g0b" "gem" "g0_noamp_lr01_se_til" "7,13,21" "--lr 0.01" &
wait
echo "[noamp] ===== ALL DONE $(date) ====="
echo "[noamp] analyse: $PY scripts/analyse_noamp_baselines.py"
