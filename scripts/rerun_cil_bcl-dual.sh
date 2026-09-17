#!/bin/bash
# CIL leave-one-out ablation grid -- the class-incremental complement of the single-pass
# TIL grid in docs/ablation_studies.tex (Table tab:ablation_optim). Core three families
# only (Res-ER / C-MAML / BCL-Dual), the methods with headroom above the CIL floor; GEM is
# at the floor in CIL and is excluded, and CTN is not in the CIL results table.
#
# Each row is the CIL twin of a TIL ablation: same single-knob change, loader swapped to
# class_incremental_loader (via the cil/*.yaml configs). Row-0 baselines sit at the repo's
# canonical CIL operating points (configs/models/cil/{eralg4,cmaml,bcl_dual}.yaml).
#
# Regime: single-epoch, inner_steps 2 (=> 4 SGD steps/batch for C-MAML/BCL, 2 for Res-ER),
# n=6 seeds (0,39,55,7,13,21) -- the canonical set used by the TIL grid.
#
# Bounded worker pool: MAX_PARALLEL concurrent processes (default 2, kept low because this
# shares the GPU with an already-running C-MAML top-up). Each job runs all 6 seeds
# internally so every config gets one run dir with a cross-seed summary:
#   logs/<config_stem>/<expt_name>-<ts>/{0,39,55,7,13,21}/results.txt
#
# Usage:  bash scripts/run_cil_ablations.sh
#         MAX_PARALLEL=3 bash scripts/run_cil_ablations.sh
set -uo pipefail

REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1   # main.py writes logs/ relative to cwd; anchor it here
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
BASE="$REPO/configs/base.yaml"
CFG="$REPO/configs/models/cil"

# SEEDS/LOGDIR overridable from env so the same JOBS list serves the n=6 base run
# (SEEDS=0,39,55,7,13,21) and the n=9 top-up (SEEDS=1,2,3, separate LOGDIR). Expt names are
# identical across runs, so the topup lands in a new timestamped run dir alongside the base
# one and analyse_cil_ablations.py pools their seed subdirs.
SEEDS="${SEEDS:-0,1,2,3,39,55,7,13,21}"
IS=2
ER_LR="0.01"                        # Res-ER (E0) / Ring-ER-dynamic (E1) learning rate
MAX_PARALLEL="${MAX_PARALLEL:-2}"

LOGDIR="${LOGDIR:-$REPO/scripts/logs/cil_ablations}"
mkdir -p "$LOGDIR"

# run_job <config_stem> <expt_name> <inner_steps> <extra_cli_args>
run_job() {
  local stem="$1" expt="$2" isteps="$3" extra="$4"
  echo "[queue] START $expt  (cfg=$stem is=$isteps extra='$extra')  $(date '+%H:%M:%S')"
  "$PY" "$REPO/main.py" \
    --config "$BASE" \
    --config "$CFG/${stem}.yaml" \
    --n_epochs 1 --inner_steps "$isteps" \
    --expt_name "$expt" \
    --seeds "$SEEDS" $extra > "$LOGDIR/${expt}.log" 2>&1
  echo "[queue] END   $expt  exit=$?  $(date '+%H:%M:%S')"
}

# Job list: "stem|expt|inner_steps|extra". Interleaved across families (Res-ER / C-MAML /
# BCL-Dual) so the pool mixes light and heavy work. Grid row IDs in comments.
JOBS=(
  # baselines
  # "eralg4|eralg4_resER_se_cil|$IS|--lr $ER_LR"                                              # E0
  # "cmaml|cmaml_full_se_cil|$IS|"                                                            # M0 (first-order baseline)
  # "bcl_dual|bcl_dual_se_cil|$IS|"                                                           # B0
  # # Res-ER
  # "er_ring|er_ring_dynring_se_cil|$IS|--lr $ER_LR --memory_loss_lambda 1 --er_dynamic_ring" # E1
  # BCL-Dual
  "bcl_dual|fixed-cil-mask_se_cil|$IS|"                                                     # B1
  "bcl_nodistill|bcl_nodistill_se_cil|$IS|"                                                 # B1
  "bcl_nodualmem|bcl_nodualmem_se_cil|$IS|"                                                 # B3
  "bcl_noreplay|bcl_noreplay_se_cil|$IS|"                                                   # B4
  "bcl_singlelevel|bcl_singlelevel_is2_se_cil|2|"                                           # B5a (half budget)
  "bcl_singlelevel|bcl_singlelevel_is4_se_cil|4|"                                           # B5b (matched budget)
)

echo "[queue] ===== CIL ablation grid  seeds=$SEEDS  MAX_PARALLEL=$MAX_PARALLEL  jobs=${#JOBS[@]}  START $(date) ====="

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

echo "[queue] ===== ALL DONE  $(date) ====="
echo "[queue] per-job logs: $LOGDIR/*.log"
echo "[queue] results:      logs/<config_stem>/<expt_name>-<ts>/{0,1,2,3,39,55,7,13,21}/results.txt"
