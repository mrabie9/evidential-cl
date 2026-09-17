#!/bin/bash
# n=9 probe: C-MAML with BOTH meta mechanisms removed (M3 + M4 combined).
#
# alpha_init 0 (no inner adaptation) + meta_batches 1 (no gradient averaging) reduces
# C-MAML to plain replay SGD via opt_wt, one forward per update. This is the C-MAML
# analogue of D2 and the candidate MX row: the arm that removes the last asymmetry against
# Res-ER's E0 (which also uses a single forward and takes one optimiser step per inner step
# at the same effective lr, opt_wt 0.01 vs cfg.lr 0.01).
#
# PURPOSE OF THE PROBE: measure the point estimate and the paired-delta sd, which together
# size any top-up. The equivalence verdict at delta=1 is extremely sensitive to where the
# estimate lands -- ~21-34 seeds if it comes in near -0.4, but ~539 at -0.85 -- so n=9 first
# and decide afterwards. Additivity of M3 (+0.3) and M4 (+0.2) is only a conjecture here;
# both component effects are individually underpowered.
#
# The 9 seeds are split into 3 concurrent 3-seed jobs. All three share one expt_name, so
# they land in three sibling timestamped run dirs and analyse_cil_ablations.load_expt pools
# their seed subdirs by glob -- no PIN needed, and the pooled set is the canonical 9.
#
# AMP is left ON to match M0/M3/M4 and E0 (the CIL block's precision). inner_steps 2 and the
# canonical 9 seeds match every other row so pairing against E0 is complete.
#
# --cmaml_joint_er is REQUIRED: every C-MAML row in tab:cil_ablation carries it (note b,
# worth +2.4 F1), and it lives in the rerun script's CLI rather than in the cil/cmaml*.yaml
# configs, so building this arm from cil/cmaml_alpha0.yaml alone silently drops it. A first
# attempt did exactly that and read -3.77 vs E0, most of which was the missing flag; those
# runs are kept under the OLD expt name cmaml_alpha0_singleinner_se_cil as the pooled-BN
# variant and must not be pooled with these.
#
# CAVEAT -- meta-loss reduction. --cmaml_replay_loss_mode split scores the replay and
# current blocks separately (the current default, commit 0b977d1c, 2026-07-25). The curated
# M0/M3/M4 rows all PREDATE that commit and carry no such key, i.e. they ran the legacy
# pooled CE, which this repo measures as ~2.7 F1 of lost plasticity. So this arm is directly
# comparable to E0 (eralg4 is unaffected by the flag) but NOT to the table's C-MAML rows;
# placing it in the C-MAML block would need M0 rerun at split as well.
#
# Usage:  bash scripts/run_cil_cmaml_m3m4_probe.sh
set -uo pipefail

REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1   # main.py writes logs/ relative to cwd; anchor it here
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"

# MODE selects the setting; both blocks are split-era and carry the joint-BN probe, and
# both curated M0 arms run AMP on, so the arm matches its own block on every axis.
MODE="${MODE:-cil}"
case "$MODE" in
  cil) SUFFIX="se_cil" ;;
  til) SUFFIX="se_til" ;;
  *)   echo "unknown MODE=$MODE (expected cil|til)"; exit 2 ;;
esac

LOGDIR="${LOGDIR:-$REPO/scripts/logs/cmaml_m3m4/$MODE}"
mkdir -p "$LOGDIR"

# One shard per line; each runs its seeds sequentially, the 3 shards run concurrently.
# TOPUP=1 switches to the second block of seeds. The topup seeds must already exist in the
# COMPARATOR arms or they contribute nothing: these contrasts are paired, so a seed present
# only in this arm has no partner. In CIL, M0 (n=29) and E0 (n=29) both cover 101-120, so
# 101-109 pair immediately. In TIL, M0 and E0 hold only the canonical 9, so a TIL topup also
# requires the same new seeds on M0 and E0 before it is usable.
SHARDS=("0,1,2" "3,39,55" "7,13,21")
if [ "${TOPUP:-0}" = "1" ]; then
  case "$MODE" in
    cil) SHARDS=("101,102,103" "104,105,106" "107,108,109") ;;
    til) SHARDS=("101,102" "103,104" "105,106") ;;
  esac
fi

run_shard() {
  local idx="$1" seeds="$2"
  echo "[m3m4-$MODE] START shard$idx seeds=$seeds $(date '+%H:%M:%S')"
  "$PY" "$REPO/main.py" \
    --config "$REPO/configs/base.yaml" \
    --config "$REPO/configs/models/${MODE}/cmaml_alpha0_singleinner.yaml" \
    --n_epochs 1 --inner_steps 2 --no-save_checkpoints \
    --cmaml_joint_er --cmaml_replay_loss_mode split \
    --expt_name "cmaml_a0si_jointer_split_${SUFFIX}" \
    --seeds "$seeds" > "$LOGDIR/shard${idx}.log" 2>&1
  echo "[m3m4-$MODE] END   shard$idx exit=$? $(date '+%H:%M:%S')"
}

echo "[m3m4] ===== 3 shards x 3 seeds, concurrent  START $(date) ====="
i=0
for s in "${SHARDS[@]}"; do
  run_shard "$i" "$s" &
  i=$((i + 1))
done
wait
echo "[m3m4] ===== ALL DONE $(date) ====="
echo "[m3m4] per-shard logs: $LOGDIR/shard{0,1,2}.log"
