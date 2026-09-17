#!/bin/bash
# Multi-epoch rerun of the two published leave-one-out ablation grids: the TIL grid of
# docs/ablation_studies.tex (tab:ablation_optim) and its CIL complement in
# docs/se_cil_ablations.tex (tab:cil_ablation). Both were measured in the single-epoch
# regime (--n_epochs 1 --inner_steps 2); this driver re-measures every row at the
# multi-epoch operating point that backs the headline results tables:
#
#     --n_epochs 10 --inner_steps 1 --no-amp
#
# SETTINGS ARE MATCHED ACROSS THE TWO MODES, which the published pair is not:
#
#   * Precision. Every row here runs fp32 (--no-amp). The published TIL grid is a mix
#     (GEM/Res-ER/CTN fp32, C-MAML/BCL bf16) and the published CIL grid is bf16
#     throughout. bf16 is not free on this benchmark -- it costs GEM 1.43 F1 and flips
#     cross-method ordering -- so a mixed-precision grid cannot be read across families.
#   * BatchNorm forward layout. Every replay row splits the replay and current-task
#     forwards (--eralg4_joint_er on E0, --cmaml_joint_er on M0-M4), in both modes.
#   * Loss reduction. --cmaml_replay_loss_mode now defaults to "split" for every C-MAML
#     row in both modes; the published CIL M rows predate the flag and ran pooled.
#   * C-MAML row semantics. M0 is the SECOND-ORDER full method and M1 removes the
#     Hessian term, in BOTH modes. The published CIL grid had these inverted (its M0 was
#     first-order and M1 switched second-order ON), so its M1 delta carried the opposite
#     sign to the TIL M1 delta under the same row ID.
#   * GEM learning rate. G2 runs at lr 0.01 alongside G0/G1 rather than at 0.03, so the
#     GEM family is single-lr and tab:ablation_optim's note (a) no longer applies.
#
# Row set is otherwise exactly the published one: GEM (G0-G2), Res-ER (E0-E2), C-MAML
# (M0-M4), BCL-Dual (B0-B5b) and CTN (T0-T3) in TIL; Res-ER, C-MAML and BCL-Dual in CIL
# (GEM sits on the CIL floor and CTN has no CIL config, as in the published grid).
#
# G1 is not a job: it is the same er_ring configuration as E1, so the analyser reads the
# E1 pool for both rows, exactly as the single-epoch tree does.
#
# BUDGET NOTE. inner_steps is 1 for every row, so BCL-Dual's bilevel round takes 2 SGD
# steps/batch. B5a flattens it at half budget (--inner_steps 1, one step) and B5b at
# matched budget (--inner_steps 2, two steps), mirroring the is2/is4 pair single-epoch.
#
# Results land in logs/<config_stem>/<row>_me_<mode>-<timestamp>/<seed>/results.txt. The
# expt_name is unique per row, so scripts/analyse_me_ablations.py pools a row by globbing
# its (stem, expt_name) pair -- no curated tree and no config-signature matching needed.
# Re-running a seed group creates a new timestamped dir alongside the old one and the
# analyser lets the newest run win a duplicate seed.
#
# Jobs are (row x seed-triplet), ordered GROUP-MAJOR so the whole grid reaches n=3 before
# any row reaches n=6: the table is readable early rather than after the last row lands.
#
# Usage:
#   bash scripts/run_me_ablations.sh                       # n=6, both modes
#   MAX_PARALLEL=2 bash scripts/run_me_ablations.sh        # lighter GPU share
#   MODES=cil bash scripts/run_me_ablations.sh             # one mode only
#   ROWS_ONLY=m3,m4 SEED_GROUP_SPEC="1,2,3" bash scripts/run_me_ablations.sh   # n=9 top-up
#   SMOKE=1 bash scripts/run_me_ablations.sh               # 2 tasks, 512 samples, seed 0
set -uo pipefail

REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1   # main.py resolves data_path and logs/ against cwd
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
BASE="$REPO/configs/base.yaml"

N_EPOCHS="${N_EPOCHS:-10}"
IS=1
MAX_PARALLEL="${MAX_PARALLEL:-3}"
SMOKE="${SMOKE:-0}"
LOGDIR="${LOGDIR:-$REPO/scripts/logs/me_ablations}"
mkdir -p "$LOGDIR"

# id|mode|config_stem|extra_cli_args
# ROWS_ONLY filters on id, MODES on mode.
ROWS=(
  # ---- TIL: GEM ----
  "g0|til|gem|--lr 0.01"
  "g2|til|gem_noqp|--lr 0.01"
  # ---- TIL: Res-ER (E1 doubles as GEM's G1) ----
  "e0|til|eralg4|--lr 0.01 --memory_loss_lambda 1 --eralg4_joint_er"
  "e1|til|er_ring|--lr 0.01 --memory_loss_lambda 1"
  "e2|til|er_ring|--lr 0.01 --memory_loss_lambda 1 --er_dynamic_ring"
  # ---- TIL: C-MAML ----
  "m0|til|cmaml|--second_order --cmaml_joint_er"
  "m1|til|cmaml|--cmaml_joint_er"
  "m2|til|cmaml_no_replay|--cmaml_joint_er"
  "m3|til|cmaml_single_inner|--cmaml_joint_er"
  "m4|til|cmaml_alpha0|--cmaml_joint_er"
  # ---- TIL: BCL-Dual ----
  "b0|til|bcl_dual|"
  "b1|til|bcl_nodistill|"
  "b3|til|bcl_nodualmem|"
  "b4|til|bcl_noreplay|"
  "b5a|til|bcl_singlelevel|--inner_steps 1"
  "b5b|til|bcl_singlelevel|--inner_steps 2"
  # ---- TIL: CTN ----
  "t0|til|ctn|"
  "t1|til|ctn_nofilm|"
  "t2|til|ctn_nodistill|"
  "t3|til|ctn_noreplay|"
  # ---- CIL: Res-ER ----
  "e0|cil|eralg4|--lr 0.01 --memory_loss_lambda 1 --eralg4_joint_er"
  "e1|cil|er_ring|--lr 0.01 --memory_loss_lambda 1"
  "e2|cil|er_ring|--lr 0.01 --memory_loss_lambda 1 --er_dynamic_ring"
  # ---- CIL: C-MAML ----
  "m0|cil|cmaml|--second_order --cmaml_joint_er"
  "m1|cil|cmaml|--cmaml_joint_er"
  "m2|cil|cmaml_no_replay|--cmaml_joint_er"
  "m3|cil|cmaml_single_inner|--cmaml_joint_er"
  "m4|cil|cmaml_alpha0|--cmaml_joint_er"
  # ---- CIL: BCL-Dual ----
  "b0|cil|bcl_dual|"
  "b1|cil|bcl_nodistill|"
  "b3|cil|bcl_nodualmem|"
  "b4|cil|bcl_noreplay|"
  "b5a|cil|bcl_singlelevel|--inner_steps 1"
  "b5b|cil|bcl_singlelevel|--inner_steps 2"
)

SEED_GROUP_SPEC="${SEED_GROUP_SPEC:-0,39,55;7,13,21}"
ROWS_ONLY="${ROWS_ONLY:-}"
MODES="${MODES:-til,cil}"

if [ "$SMOKE" = "1" ]; then
  SEED_GROUP_SPEC="0"
  N_EPOCHS=1
  LOGDIR="$REPO/scripts/logs/me_ablations_smoke"
  mkdir -p "$LOGDIR"
fi

IFS=';' read -r -a SEED_GROUPS <<< "$SEED_GROUP_SPEC"

# run_job <mode> <config_stem> <expt_name> <seeds> <extra_cli_args>
run_job() {
  local mode="$1" stem="$2" expt="$3" seeds="$4" extra="$5"
  local out="$LOGDIR/${expt}_${seeds//,/-}.log"
  echo "[me] START $expt seeds=$seeds (cfg=$mode/$stem extra='$extra') $(date '+%H:%M:%S')"
  if [ "$SMOKE" = "1" ]; then
    "$PY" "$REPO/main.py" \
      --config "$BASE" --config "$REPO/configs/models/${mode}/${stem}.yaml" \
      --n_epochs 1 --inner_steps "$IS" --no-amp --no-save_checkpoints \
      --task-order-files "t0-deeprad,t1-deeprad" --samples_per_task 512 \
      --expt_name "$expt" --seeds "$seeds" $extra > "$out" 2>&1
  else
    "$PY" "$REPO/main.py" \
      --config "$BASE" --config "$REPO/configs/models/${mode}/${stem}.yaml" \
      --n_epochs "$N_EPOCHS" --inner_steps "$IS" --no-amp --no-save_checkpoints \
      --expt_name "$expt" --seeds "$seeds" $extra > "$out" 2>&1
  fi
  echo "[me] END   $expt seeds=$seeds exit=$? $(date '+%H:%M:%S')"
}

selected=()
for row in "${ROWS[@]}"; do
  IFS='|' read -r rid rmode _stem _extra <<< "$row"
  if [ -n "$ROWS_ONLY" ] && [[ ",$ROWS_ONLY," != *",$rid,"* ]]; then continue; fi
  if [[ ",$MODES," != *",$rmode,"* ]]; then continue; fi
  selected+=("$row")
done
if [ "${#selected[@]}" -eq 0 ]; then
  echo "[me] ROWS_ONLY='$ROWS_ONLY' MODES='$MODES' matched no rows" >&2
  exit 1
fi

echo "[me] ===== START $(date) regime=${N_EPOCHS}ep/is${IS}/no-amp smoke=$SMOKE"
echo "[me]       rows=${#selected[@]} groups=${#SEED_GROUPS[@]} jobs=$(( ${#selected[@]} * ${#SEED_GROUPS[@]} )) MAX_PARALLEL=$MAX_PARALLEL"
echo "[me]       seed groups: ${SEED_GROUPS[*]}"

running=0
for group in "${SEED_GROUPS[@]}"; do
  for row in "${selected[@]}"; do
    IFS='|' read -r rid rmode stem extra <<< "$row"
    expt="${rid}_me_${rmode}"
    # Smoke runs get their own expt_name so their throwaway run dirs can never be
    # pooled into a published row by the analyser's (stem, expt_name) glob.
    if [ "$SMOKE" = "1" ]; then expt="${expt}_smoke"; fi
    run_job "$rmode" "$stem" "$expt" "$group" "$extra" &
    running=$((running + 1))
    if (( running >= MAX_PARALLEL )); then
      wait -n
      running=$((running - 1))
    fi
  done
done
wait
echo "[me] ===== ALL DONE $(date) ====="
echo "[me] per-job logs: $LOGDIR/*.log"
echo "[me] analyse: $PY scripts/analyse_me_ablations.py"
