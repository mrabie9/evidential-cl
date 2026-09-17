#!/bin/bash
# Why does BCL-Dual (4.8) beat Ring-ER in CIL? -- the matched-control pair.
#
# Post-mask-fix (2026-07-24, see docs/se_cil_ablations.tex note c) BCL-Dual's honest CIL
# baseline is 4.8 +/- 0.3, while the leave-one-out grid finds every BCL mechanism inert
# except replay. That looks contradictory only until the Ring-ER comparison is matched on
# the two knobs that differ from BCL's config, learning rate and gradient budget:
#
#   er_ring static  is2  lr 0.01   ->  3.4 +/- 0.3    (the number BCL was compared against)
#   er_ring static  is2  lr 0.001  ->  4.5 +/- 0.7    lr-matched: BCL's edge drops to -0.4 (p=0.21)
#   er_ring static  is4  lr 0.01   ->  4.2 +/- 0.5    budget-matched only
#   er_ring static  is4  lr 0.001  ->  ???            <-- THIS RUN: both matched
#   bcl_dual        is2  lr 0.001  ->  4.8 +/- 0.3    (B0; bilevel = 4 SGD steps/batch)
#
# BCL at is2 takes 4 SGD steps per batch (bilevel), so is4 is the budget-matched Ring-ER
# cell. If lr and budget are roughly additive (+1.1 and +0.8 from the rows above), the
# missing cell lands near 5.3 -- i.e. ABOVE BCL, making BCL's remaining edge negative and
# fully explaining the grid's "everything inert" verdict.
#
# Job 2 tests the converse: BCL's config lr of 0.001 is a 10x outlier against every other
# CIL replay method (er_ring/eralg4/agem all use 0.01). The bcl_noreplay lr-0.01 probe
# scored 5.8 +/- 0.6 -- BCL *without* replay at lr 0.01 beats BCL *with* replay at lr 0.001
# -- which suggests the whole BCL family has been graded at a mis-set operating point.
#
# Both jobs are n=9 seeds so they pair directly against the existing B0/E-row seed sets.
# Usage:  bash scripts/run_bcl_vs_ring_matched.sh
set -uo pipefail

REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1   # main.py writes logs/ relative to cwd; anchor it here
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
BASE="$REPO/configs/base.yaml"
CFG="$REPO/configs/models/cil"

SEEDS="${SEEDS:-0,39,55,7,13,21,1,2,3}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
LOGDIR="${LOGDIR:-$REPO/scripts/logs/bcl_vs_ring_matched}"
mkdir -p "$LOGDIR"

# run_job <config_stem> <expt_name> <inner_steps> <extra_cli_args>
run_job() {
  local stem="$1" expt="$2" isteps="$3" extra="$4"
  echo "[match] START $expt  (cfg=$stem is=$isteps extra='$extra')  $(date '+%H:%M:%S')"
  "$PY" "$REPO/main.py" \
    --config "$BASE" \
    --config "$CFG/${stem}.yaml" \
    --n_epochs 1 --inner_steps "$isteps" --no-save_checkpoints \
    --expt_name "$expt" \
    --seeds "$SEEDS" $extra > "$LOGDIR/${expt}.log" 2>&1
  echo "[match] END   $expt  exit=$?  $(date '+%H:%M:%S')"
}

JOBS=(
  # R1: Ring-ER matched to BCL on BOTH lr and gradient budget -- the decisive control.
  "er_ring|er_ring_static_is4_lr001_se_cil|4|--lr 0.001 --memory_loss_lambda 1"
  # R2: BCL-Dual at the lr every other CIL replay method uses.
  "bcl_dual|bcl_dual_lr01_se_cil|2|--lr 0.01"
)

echo "[match] ===== BCL-vs-Ring matched controls  seeds=$SEEDS  MAX_PARALLEL=$MAX_PARALLEL  START $(date) ====="

running=0
for spec in "${JOBS[@]}"; do
  IFS='|' read -r j_stem j_expt j_is j_extra <<< "$spec"
  run_job "$j_stem" "$j_expt" "$j_is" "$j_extra" &
  running=$((running + 1))
  if (( running >= MAX_PARALLEL )); then
    wait -n
    running=$((running - 1))
  fi
done
wait

echo "[match] ===== ALL DONE  $(date) ====="
echo "[match] per-job logs: $LOGDIR/*.log"
