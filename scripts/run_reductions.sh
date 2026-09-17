#!/bin/bash
# Reduction-experiment runner (see docs/reduction_experiments.md).
#
# Triggers when the in-flight beta=1 BCL grid (rerun_cil_bcl-dual.sh) finishes, then works
# through the pending reduction runs with a 3-wide worker pool. CIL jobs are listed first so the
# pool always drains them before touching TIL. Each job runs all n=9 seeds internally into one
# timestamped run dir with a cross-seed summary.
#
# Excluded: anything needing the not-yet-built C-MAML separate-forward probe (X2) -- T-M2 and the
# cmaml-joint half of C-M1. Excluded: rows already covered by the running grid (B3/B4 at beta=1)
# or by existing data (M4 = cmaml_alpha0).
#
# Usage:  bash scripts/run_reductions.sh              # waits for the grid, then runs
#         WAIT=0 bash scripts/run_reductions.sh       # skip the wait, run immediately
set -uo pipefail

REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1   # main.py writes logs/ relative to cwd; anchor it here
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
BASE="$REPO/configs/base.yaml"
MODELS="$REPO/configs/models"

SEEDS="${SEEDS:-0,39,55,7,13,21,1,2,3}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
WAIT="${WAIT:-1}"
TRIGGER="rerun_cil_bcl-dual.sh"   # current run whose completion we gate on
LOGDIR="${LOGDIR:-$REPO/scripts/logs/cil_reductions}"
mkdir -p "$LOGDIR"

# --- Trigger: block until the current grid's driver process is gone ------------------------
if [ "$WAIT" = "1" ]; then
  echo "[reduce] waiting for '$TRIGGER' to finish before starting  $(date '+%H:%M:%S')"
  while pgrep -f "$TRIGGER" >/dev/null 2>&1; do
    sleep 60
  done
  echo "[reduce] trigger cleared; starting reduction runs  $(date '+%H:%M:%S')"
fi

# run_job <models-relative stem> <expt_name> <inner_steps> <extra_cli_args>
run_job() {
  local stem="$1" expt="$2" isteps="$3" extra="$4"
  echo "[reduce] START $expt  (cfg=$stem is=$isteps extra='$extra')  $(date '+%H:%M:%S')"
  "$PY" "$REPO/main.py" \
    --config "$BASE" \
    --config "$MODELS/${stem}.yaml" \
    --n_epochs 1 --inner_steps "$isteps" --no-save_checkpoints \
    --expt_name "$expt" \
    --seeds "$SEEDS" $extra > "$LOGDIR/${expt}.log" 2>&1
  echo "[reduce] END   $expt  exit=$?  $(date '+%H:%M:%S')"
}

# Job list: "stem|expt|inner_steps|extra". CIL first (priority), then TIL.
# Job list: "stem|expt|inner_steps|extra". Order = user priority (2026-07-23):
#   (1) CIL "why B3>>Ring-ER"  (2) Z2/Z3 + reservoir reduction  (3) C-M1 [blocked on X2]
#   (4) TIL C-MAML  (5) TIL BCL.  The MAX_PARALLEL pool drains the list top-down, so earlier
#   groups are fully in-flight before later ones start.
JOBS=(
  # ===== 1. CIL: "why does B3 (static ring) 13.7 >> Ring-ER 3.8?" -- isolate the loop on the
  #        STATIC ring (the one B3 actually uses). NB: the 3.8 baseline (E1) is the DYNAMIC ring;
  #        the static bare baseline below has never been run and is the linchpin (it is also the
  #        static-vs-dynamic test). er_distill defaults reg=1,temp=5 == BCL's distillation.
  "cil/er_ring|er_ring_static_base_se_cil|2|--lr 0.01 --memory_loss_lambda 1"                 # rung-0: static bare baseline
  "cil/er_ring|er_ring_static_lr001_se_cil|2|--lr 0.001 --memory_loss_lambda 1"               # L-lr
  "cil/er_ring|er_ring_static_is4_se_cil|4|--lr 0.01 --memory_loss_lambda 1"                  # L-budget
  "cil/er_ring|er_ring_static_distill_se_cil|2|--lr 0.01 --memory_loss_lambda 1 --er_distill" # L-distill
  # ===== 2. CIL: lock the combination (Z2), B4 diagnostic (Z3, matched budget), reservoir reduction.
  "cil/bcl_b3_gres|bcl_b3_gres_b1_se_cil|2|"                                                  # Z2: B3+gres @ beta=1 (vs grid B3 @ beta=1)
  "cil/bcl_noreplay|bcl_noreplay_nobilevel_is4_se_cil|4|--no_bilevel"                         # Z3: no-mem no-bilevel @ MATCHED budget (isolate structure)
  "cil/eralg4|eralg4_resER_reduce_se_cil|4|--lr 0.001 --eralg4_joint_er --er_distill"         # reduction: bare reservoir @ B3+gres's point -> 16.4?
  # ===== 3. CIL: C-M1 cmaml-joint vs eralg4-joint -- BLOCKED on X2 (lamaml_cifar separate-forward
  #        probe not built yet). eralg4-joint side = X1 (done). Uncomment once X2 exists.
  # "cil/cmaml|cmaml_joint_se_cil|2|<cmaml separate-forward flag from X2>"
  # ===== 4. TIL: C-MAML reduction. (T-M2 cmaml-joint TIL also blocked on X2.)
  "til/er_ring|er_ring_lr01_se_til|2|--lr 0.01 --memory_loss_lambda 1"                        # T-M1: er_ring@0.01 comparator for cmaml
  "til/eralg4|eralg4_unmasked_se_til|2|--lr 0.01 --eralg4_unmasked_loss"                      # T-M3: unmask eralg4 -> should drop to cmaml level
  # ===== 5. TIL: BCL reduction.
  "til/er_ring|er_ring_lr001_is4_se_til|4|--lr 0.001 --memory_loss_lambda 1"                    # T-B1: Ring-ER @ B5b's point
  "til/er_ring|er_ring_lr001_is4_distill_se_til|4|--lr 0.001 --memory_loss_lambda 1 --er_distill" # T-B2: + distill
)

echo "[reduce] ===== reduction grid  seeds=$SEEDS  MAX_PARALLEL=$MAX_PARALLEL  jobs=${#JOBS[@]}  START $(date) ====="

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

echo "[reduce] ===== ALL DONE  $(date) ====="
echo "[reduce] per-job logs: $LOGDIR/*.log"
echo "[reduce] results:      logs/<config_stem>/<expt_name>-<ts>/{seeds}/results.txt"
