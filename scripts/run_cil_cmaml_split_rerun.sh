#!/bin/bash
# Rerun the CIL C-MAML rows under the CURRENT meta-loss reduction (--cmaml_replay_loss_mode
# split), which the curated rows predate.
#
# WHY. Every curated CIL C-MAML run (logs/ablations/cil/cmaml/M0..M4) was launched on
# 2026-07-24/25 BEFORE commit 0b977d1c (2026-07-25 15:12, "score replay and current blocks
# separately in meta_loss"). That commit both added --cmaml_replay_loss_mode and changed the
# behaviour: before it, meta_loss took a single pooled CE over the concatenated replay+current
# rows; after it, the default is 'split' (score the blocks separately, replay share pinned
# 1:1). Those runs therefore carry no such key in training_parameters.json -- absence of the
# key IS the marker that a run predates the flag -- and they ran the legacy pooled CE, which
# this repo measures at ~2.7 F1 of lost plasticity. The TIL block was already redone under
# the fix (the *_joint_splitfix_se_til runs from 2026-07-25 15:48), so TIL and CIL currently
# sit on different objectives.
#
# ORDER. M0 first: it is the family baseline, so every M1-M4 delta is measured against it. If
# M0 moves, M3 and M4 must be redone before their verdicts mean anything. M1 is a second-order
# variant and would follow; M2 (no replay) is expected to be INERT to both flags, since each
# governs how the replay block is forwarded and combined and M2 has no replay block -- worth
# confirming in meta_loss rather than assuming, but it is why M2 was skipped in the TIL wave.
#
# --cmaml_joint_er is passed explicitly: it lives in the rerun script's CLI rather than in the
# cil/cmaml*.yaml configs, and all five curated rows carry it, so omitting it silently drops a
# mechanism worth +2.4 F1 (note b of tab:cil_ablation).
#
# 9 canonical seeds in 3 concurrent shards, AMP on, inner_steps 2 -- matching every other row.
#
# Usage:  bash scripts/run_cil_cmaml_split_rerun.sh          # M0 (default)
#         ROW=M3 bash scripts/run_cil_cmaml_split_rerun.sh
#         ROW=M4 bash scripts/run_cil_cmaml_split_rerun.sh
set -uo pipefail

REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1   # main.py writes logs/ relative to cwd; anchor it here
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"

ROW="${ROW:-M0}"
case "$ROW" in
  M0) CFG_STEM="cmaml";              EXPT="m0_split_se_cil";           EXTRA="" ;;
  M1) CFG_STEM="cmaml";              EXPT="m1_secondorder_split_se_cil"; EXTRA="--second_order" ;;
  M3) CFG_STEM="cmaml_single_inner"; EXPT="m3_singleinner_split_se_cil"; EXTRA="" ;;
  M4) CFG_STEM="cmaml_alpha0";       EXPT="m4_alpha0_split_se_cil";      EXTRA="" ;;
  *)  echo "unknown ROW=$ROW (expected M0|M1|M3|M4)"; exit 2 ;;
esac

LOGDIR="${LOGDIR:-$REPO/scripts/logs/cil_cmaml_split/$ROW}"
mkdir -p "$LOGDIR"

SHARDS=("0,1,2" "3,39,55" "7,13,21")

run_shard() {
  local idx="$1" seeds="$2"
  echo "[$ROW] START shard$idx seeds=$seeds $(date '+%H:%M:%S')"
  "$PY" "$REPO/main.py" \
    --config "$REPO/configs/base.yaml" \
    --config "$REPO/configs/models/cil/${CFG_STEM}.yaml" \
    --n_epochs 1 --inner_steps 2 --no-save_checkpoints \
    --cmaml_joint_er --cmaml_replay_loss_mode split \
    --expt_name "$EXPT" \
    --seeds "$seeds" $EXTRA > "$LOGDIR/shard${idx}.log" 2>&1
  echo "[$ROW] END   shard$idx exit=$? $(date '+%H:%M:%S')"
}

echo "[$ROW] ===== $EXPT (cfg=$CFG_STEM extra='$EXTRA')  3 shards x 3 seeds  START $(date) ====="
i=0
for s in "${SHARDS[@]}"; do
  run_shard "$i" "$s" &
  i=$((i + 1))
done
wait
echo "[$ROW] ===== ALL DONE $(date) ====="
echo "[$ROW] results: logs/$CFG_STEM/$EXPT-<ts>/<seed>/results.txt"
