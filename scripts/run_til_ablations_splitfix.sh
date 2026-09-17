#!/bin/bash
# Re-run the TIL ablation rows E0 and M0-M4 on post-fix code.
#
# WHY: commit 0b977d1c changed lamaml_cifar's meta-loss reduction from a single
# pooled class-weighted CE to eralg4's split
# ``current + memory_loss_lambda * replay``. Under the pooled CE the
# inverse-frequency weights were derived from the mixed batch, so replay's share
# of the loss escalated with task count (0.34 after task 0 -> 0.85 after task 9)
# and starved the current task. Every M row was therefore measured against a
# handicapped baseline; probe runs put the correction at about +3 F1, all of it
# plasticity (docs/cmaml_vs_reser_til.md).
#
# E0 runs with --eralg4_joint_er (the sister-repo two-forward loop: current and
# replay in SEPARATE forwards, so BatchNorm does not pool their statistics). That
# is a NEW operating point, not a reproduction of the logged E0=60.75, which
# predates the flag -- treat its delta vs the old pool as the joint-ER effect in
# TIL, previously only measured in CIL (+2.4 F1 there).
#
# NB this leaves the grid with NO drift control: nothing here is expected to
# reproduce an old number, so a shared-code regression (the failure mode behind
# CTN 0.50 -> 0.38) would pass unnoticed. Add a plain "eralg4|...|--lr 0.01" row
# back if that check is wanted. M2 (replay removed) was dropped on request; it was
# a no-op for the fix anyway, since memories:0 means the split branch never fires.
#
# EVERY row carries the BN split -- --eralg4_joint_er on E0 and its twin
# --cmaml_joint_er on M0-M4 -- so the two families are matched on that axis and
# the E-vs-M comparison isolates the algorithms rather than the forward layout.
# Both old pools predate their flag, so no row's delta vs the old pool is a pure
# reproduction: for the M rows it mixes the loss-reduction fix with the BN split,
# for E0 it is the BN split alone.
#
# Row -> config (these runs are now the curated pools in logs/ablations/til/{cmaml,res-er}/):
#   E0  eralg4.yaml            --lr 0.01 --eralg4_joint_er   full Res-ER
#   M0  cmaml.yaml             --second_order --cmaml_joint_er   full C-MAML
#   M1  cmaml.yaml             --cmaml_joint_er   second-order removed (first-order)
#   M3  cmaml_single_inner.yaml --cmaml_joint_er  meta-batches removed (K_meta 1)
#   M4  cmaml_alpha0.yaml      --cmaml_joint_er   inner adaptation removed (alpha 0)
#
# All rows: single-epoch TIL, --n_epochs 1 --inner_steps 2, n=6 seeds
# (0,39,55,7,13,21) -- the point at which the exact sign-flip permutation test can
# first reach p<0.05 (floor 2/2**6 = 0.031), and enough for the +3 F1 effect the
# probe measured. Each row pairs seed-for-seed against its old pool.
#
# TOP-UP POLICY: n=6 is final for rows that come back load-bearing or inert; only
# rows whose verdict is INCONCLUSIVE or BORDERLINE get seeds 1,2,3 for n=9. Run
# the analyser first, then top up just those rows:
#
#   SEED_GROUP_SPEC="1,2,3" ROWS_ONLY=m3,m4 bash scripts/run_til_ablations_splitfix.sh
#
# Jobs are (row x seed-triplet) and ordered GROUP-MAJOR, so all rows reach n=3
# together and then n=6 -- the full grid is readable early instead of after the
# last row finishes. MAX_PARALLEL defaults to 3 because the GPU is usually already
# carrying other work; raise it if the box is idle.
#
# Usage:  bash scripts/run_til_ablations_splitfix.sh
#         MAX_PARALLEL=4 bash scripts/run_til_ablations_splitfix.sh
#         ROWS_ONLY=e0,m0 bash scripts/run_til_ablations_splitfix.sh
set -uo pipefail

REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1   # main.py resolves data_path and logs/ against cwd
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
BASE="$REPO/configs/base.yaml"
CFG="$REPO/configs/models/til"
IS=2
ER_LR="0.01"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
LOGDIR="${LOGDIR:-$REPO/scripts/logs/til_ablations_splitfix}"
mkdir -p "$LOGDIR"

# run_job <config_stem> <expt_name> <seeds> <extra_cli_args>
run_job() {
  local stem="$1" expt="$2" seeds="$3" extra="$4"
  echo "[queue] START $expt seeds=$seeds (cfg=$stem extra='$extra') $(date '+%H:%M:%S')"
  "$PY" "$REPO/main.py" \
    --config "$BASE" --config "$CFG/${stem}.yaml" \
    --n_epochs 1 --inner_steps "$IS" --no-save_checkpoints \
    --expt_name "$expt" --seeds "$seeds" $extra \
    > "$LOGDIR/${expt}_${seeds//,/-}.log" 2>&1
  echo "[queue] END   $expt seeds=$seeds exit=$? $(date '+%H:%M:%S')"
}

# id|config_stem|expt_name|extra_cli_args   (id is what ROWS_ONLY filters on)
ROWS=(
  "e0|eralg4|e0_resER_joint_splitfix_se_til|--lr $ER_LR --eralg4_joint_er"
  "m0|cmaml|m0_cmaml_joint_splitfix_se_til|--second_order --cmaml_joint_er"
  "m1|cmaml|m1_firstorder_joint_splitfix_se_til|--cmaml_joint_er"
  "m3|cmaml_single_inner|m3_singleinner_joint_splitfix_se_til|--cmaml_joint_er"
  "m4|cmaml_alpha0|m4_alpha0_joint_splitfix_se_til|--cmaml_joint_er"
)
# Semicolon-separated seed triplets. Default is n=6; override for a top-up.
SEED_GROUP_SPEC="${SEED_GROUP_SPEC:-0,39,55;7,13,21}"
IFS=';' read -r -a SEED_GROUPS <<< "$SEED_GROUP_SPEC"
ROWS_ONLY="${ROWS_ONLY:-}"   # e.g. ROWS_ONLY=m3,m4 to top up only those rows

selected=()
for row in "${ROWS[@]}"; do
  rid="${row%%|*}"
  if [ -z "$ROWS_ONLY" ] || [[ ",$ROWS_ONLY," == *",$rid,"* ]]; then
    selected+=("$row")
  fi
done
if [ "${#selected[@]}" -eq 0 ]; then
  echo "[splitfix] ROWS_ONLY='$ROWS_ONLY' matched no rows; valid ids: e0 m0 m1 m3 m4" >&2
  exit 1
fi

echo "[splitfix] ===== START $(date) MAX_PARALLEL=$MAX_PARALLEL rows=${#selected[@]} groups=${#SEED_GROUPS[@]} jobs=$((${#selected[@]} * ${#SEED_GROUPS[@]})) ====="
echo "[splitfix] seed groups: ${SEED_GROUPS[*]}"
running=0
for group in "${SEED_GROUPS[@]}"; do
  for row in "${selected[@]}"; do
    IFS='|' read -r _rid stem expt extra <<< "$row"
    run_job "$stem" "$expt" "$group" "$extra" &
    running=$((running + 1))
    if (( running >= MAX_PARALLEL )); then
      wait -n          # free a slot as soon as ANY running job finishes
      running=$((running - 1))
    fi
  done
done
wait                   # drain the remaining in-flight jobs
echo "[splitfix] ===== ALL DONE $(date) ====="
echo "[splitfix] analyse: $PY scripts/analyse_til_ablations_splitfix.py"
