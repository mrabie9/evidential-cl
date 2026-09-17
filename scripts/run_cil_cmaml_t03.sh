#!/bin/bash
# C-MAML ablation block (M0-M4) under CIL on a TRUNCATED 4-task stream: slots 0-3 of the
# canonical order (t2-rcn, t0-rcn, t2-uclresm-25noise, t0-deeprad).
#
# WHY. The published CIL M block sits near the CIL floor (M0 ~ 12.6 F1), where every
# leave-one-out delta is small and the block reads as "only replay is load-bearing". A
# 4-task stream is the same problem with less interference: fewer classes in the shared
# head, a shorter forgetting horizon, so a higher operating point and more room above the
# floor. The question this answers is whether the M mechanisms (second order, replay, meta
# -batch averaging, inner adaptation) have effects that are genuinely nil or merely
# compressed against the floor -- if headroom magnifies them, the 10-task verdicts are a
# ceiling artefact of the regime, not a property of the mechanisms.
#
# Truncation is a PREFIX of the canonical order, so tasks 0-3 here are trained on exactly
# the stream the 10-task runs see before their task 4. What is NOT held fixed: the replay
# budget. n_memories stays at the canonical 5120 total, which the loader partitions over
# n_tasks, so each task gets 1280 slots here vs 512 in the 10-task runs. That is deliberate
# (same total budget, shorter stream) but it means M2 (no replay) is contrasted against a
# better-fed replay baseline than in the 10-task block -- read cross-regime deltas with
# that in mind; within-block M1-M4 vs M0 contrasts are unaffected.
#
# CONFIGURATION matches the split-era CIL M rows exactly (scripts/run_cil_cmaml_split_rerun.sh):
# --cmaml_joint_er --cmaml_replay_loss_mode split, inner_steps 2, n_epochs 1, AMP on, and
# the canonical 9 seeds. The flags are passed to every row including M2; with memories: 0
# both degrade to no-ops (meta_loss falls through to the single forward / pooled CE when
# replay_count is 0), so the block stays single-knob.
#
# M2 IS INCLUDED here even though the split rerun skipped it -- it is inert to the split
# flags, but "does replay matter" is the one M verdict most likely to move with headroom.
#
# 15 jobs (5 rows x 3 seed shards) through a bounded pool of MAX_PARALLEL. Rows are
# interleaved across the queue so that a partial run still yields comparable seed counts on
# every row rather than three finished rows and two empty ones.
#
# Usage:  bash scripts/run_cil_cmaml_t03.sh
#         MAX_PARALLEL=2 bash scripts/run_cil_cmaml_t03.sh
set -uo pipefail

REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1   # main.py writes logs/ relative to cwd; anchor it here
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"

TASKS="t2-rcn,t0-rcn,t2-uclresm-25noise,t0-deeprad"
SHARDS=("0,1,2" "3,39,55" "7,13,21")
MAX_PARALLEL="${MAX_PARALLEL:-3}"

LOGDIR="${LOGDIR:-$REPO/scripts/logs/cil_cmaml_t03}"
mkdir -p "$LOGDIR"

# "row|config_stem|expt_name|extra_cli_args" -- no inline trailing comments, they fuse into
# the string and silently corrupt the element.
ROWS=(
  "M0|cmaml|m0_split_t03_se_cil|"
  "M1|cmaml|m1_secondorder_split_t03_se_cil|--second_order"
  "M2|cmaml_no_replay|m2_noreplay_split_t03_se_cil|"
  "M3|cmaml_single_inner|m3_singleinner_split_t03_se_cil|"
  "M4|cmaml_alpha0|m4_alpha0_split_t03_se_cil|"
)

run_job() {
  local row="$1" stem="$2" expt="$3" extra="$4" idx="$5" seeds="$6"
  echo "[$row] START shard$idx seeds=$seeds $(date '+%H:%M:%S')"
  "$PY" "$REPO/main.py" \
    --config "$REPO/configs/base.yaml" \
    --config "$REPO/configs/models/cil/${stem}.yaml" \
    --task-order-files "$TASKS" \
    --n_epochs 1 --inner_steps 2 --no-save_checkpoints \
    --cmaml_joint_er --cmaml_replay_loss_mode split \
    --expt_name "$expt" \
    --seeds "$seeds" $extra > "$LOGDIR/${row}_shard${idx}.log" 2>&1
  echo "[$row] END   shard$idx exit=$? $(date '+%H:%M:%S')"
}

echo "[t03] ===== C-MAML M0-M4, CIL tasks 0-3, n=9  MAX_PARALLEL=$MAX_PARALLEL  START $(date) ====="
echo "[t03] stream: $TASKS"

running=0
for i in "${!SHARDS[@]}"; do
  for spec in "${ROWS[@]}"; do
    IFS='|' read -r r_row r_stem r_expt r_extra <<< "$spec"
    run_job "$r_row" "$r_stem" "$r_expt" "$r_extra" "$i" "${SHARDS[$i]}" &
    running=$((running + 1))
    if (( running >= MAX_PARALLEL )); then
      wait -n
      running=$((running - 1))
    fi
  done
done
wait

echo "[t03] ===== ALL DONE $(date) ====="
echo "[t03] per-job logs: $LOGDIR/M*_shard*.log"
echo "[t03] results:      logs/<config_stem>/<expt_name>-<ts>/<seed>/results.txt"
