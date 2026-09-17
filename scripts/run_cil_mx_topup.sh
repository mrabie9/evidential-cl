#!/bin/bash
# Seed top-up for the MX contrast (M0 vs E0), taking each arm from n=9 to n=29.
#
# WHY. MX currently reads +0.1 [-1.6,+1.9]: the point estimate is essentially zero but the
# interval is wider than the delta=1 band, so the verdict is "inconclusive" -- cannot rule
# out an effect larger than the smallest one of interest. Reaching an equivalence (inert)
# verdict needs the CI inside +/-1, i.e. half-width < 0.88 given |mean| = 0.12. With the
# observed paired sd of 2.29 that crosses at n ~= 29, hence 20 new seeds per arm.
#
# BOTH ARMS ARE REQUIRED. The design is paired: a delta exists only for seeds present in
# both arms, so 20 extra M0 seeds against E0 frozen at 9 would contribute nothing. Paired
# also happens to be far cheaper here -- an unpaired design with E0 fixed would need ~86 M0
# seeds (~40 GPU-h) against 40 runs (~16 GPU-h) for the paired top-up.
#
# CAVEAT. n=29 is a planning estimate assuming the point estimate and sd hold. The mean has
# se 0.77 and will drift as seeds land; if it settles at 0.3 the requirement rises to ~44,
# at 0.5 to ~84. Watch the running sd/CI rather than assuming 29 suffices -- the companion
# check is scripts/../scratchpad monitoring, or just rerun the MX stats as seeds arrive.
#
# Seeds 101-120 are new (the canonical set is 0,1,2,3,7,13,21,39,55), so nothing collides
# and the topup pools with the base run by expt_name.
#
# ROW=M0 uses the split-era config (--cmaml_joint_er --cmaml_replay_loss_mode split) and the
# SAME expt_name as the base rerun, so the dirs pool. ROW=E0 mirrors the curated E0 exactly
# (lr 0.01, inner_steps 2, --eralg4_joint_er, AMP on); note its base run lives in the curated
# tree logs/ablations/cil/res-er/E0 while this topup lands in logs/eralg4/, so analysis must
# pool the two locations explicitly.
#
# Usage:  ROW=M0 bash scripts/run_cil_mx_topup.sh
#         ROW=E0 bash scripts/run_cil_mx_topup.sh
set -uo pipefail

REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1   # main.py writes logs/ relative to cwd; anchor it here
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"

ROW="${ROW:-M0}"
case "$ROW" in
  # C-MAML: joint probe + split meta-loss, AMP ON (C-MAML is AMP-insensitive; the TIL
  # m0_noamp probe established that). Same expt_name as the base rerun so the dirs pool.
  # 20 new seeds only -- the base 9 already ran this exact configuration.
  M0) CFG="cmaml";  EXPT="m0_split_se_cil"
      EXTRA="--cmaml_joint_er --cmaml_replay_loss_mode split"
      SHARDS=("101,102,103,104,105,106,107"
              "108,109,110,111,112,113,114"
              "115,116,117,118,119,120") ;;
  # Res-ER: joint probe + --no-amp. There is no --cmaml_replay_loss_mode analogue to pass:
  # eralg4 calls _weighted_multitask_loss unconditionally, which IS the split reduction (it
  # is what 'split' was written to match), so Res-ER is already split by construction.
  #
  # ALL 29 SEEDS, not 20. The curated E0 (n=9) ran with AMP ON, so its seeds cannot be
  # pooled with no-AMP ones -- that would make the arm a mixture of two precisions. Going
  # no-AMP means rebuilding the whole arm, hence the canonical 9 are rerun here too. New
  # expt_name accordingly; this is a new arm, not a topup of the old one.
  E0) CFG="eralg4"; EXPT="eralg4_resER_joint_noamp_se_cil"
      EXTRA="--eralg4_joint_er --no-amp"
      SHARDS=("0,1,2,3,7,13,21,101,102,103"
              "104,105,106,107,108,109,110,111,112"
              "39,55,113,114,115,116,117,118,119,120") ;;
  # Res-ER with AMP ON: 20 new seeds to bring the ORIGINAL (AMP) arm to n=29, matching the
  # no-AMP arm. Gives a like-for-like n=29 AMP-vs-no-AMP contrast on Res-ER in CIL -- the
  # n=9 estimate of that effect was +1.71 -- and lets MX be evaluated against both
  # precisions at equal n. Only the 20 new seeds are needed here: the canonical 9 already
  # ran this exact configuration and are the curated E0.
  #
  # NOTE the curated E0 base run lives in logs/ablations/cil/res-er/E0/ and is NO LONGER in
  # logs/eralg4/, so this topup cannot auto-pool by glob; analysis must read both locations
  # explicitly even though the expt_name matches.
  E0amp) CFG="eralg4"; EXPT="eralg4_resER_joint_se_cil"
      EXTRA="--eralg4_joint_er"
      SHARDS=("101,102,103,104,105,106,107"
              "108,109,110,111,112,113,114"
              "115,116,117,118,119,120") ;;
  *)  echo "unknown ROW=$ROW (expected M0|E0|E0amp)"; exit 2 ;;
esac

LOGDIR="${LOGDIR:-$REPO/scripts/logs/cil_mx_topup/$ROW}"
mkdir -p "$LOGDIR"

run_shard() {
  local idx="$1" seeds="$2"
  echo "[$ROW] START shard$idx seeds=$seeds $(date '+%H:%M:%S')"
  "$PY" "$REPO/main.py" \
    --config "$REPO/configs/base.yaml" \
    --config "$REPO/configs/models/cil/${CFG}.yaml" \
    --n_epochs 1 --inner_steps 2 --no-save_checkpoints \
    $EXTRA \
    --expt_name "$EXPT" \
    --seeds "$seeds" > "$LOGDIR/shard${idx}.log" 2>&1
  echo "[$ROW] END   shard$idx exit=$? $(date '+%H:%M:%S')"
}

echo "[$ROW] ===== $EXPT topup, 20 seeds over 3 shards  START $(date) ====="
i=0
for s in "${SHARDS[@]}"; do
  run_shard "$i" "$s" &
  i=$((i + 1))
done
wait
echo "[$ROW] ===== ALL DONE $(date) ====="
