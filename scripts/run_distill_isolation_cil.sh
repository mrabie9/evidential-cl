#!/bin/bash
# Is BCL's deficit vs Ring-ER the KL distillation term? -- the two complementary tests.
#
# Setup. At matched lr (0.001) and matched gradient budget (is2, single loop) BCL-Dual
# stripped of its bilevel loop is slightly WORSE than plain static Ring-ER:
#
#   er_ring static        is2 lr.001            4.5 +/- 0.7   FWT 12.6
#   B5a bcl_singlelevel   is2 lr.001            4.1 +/- 0.4   FWT 11.7
#   paired delta (B5a - ring)                  -0.39 F1      -0.93 FWT (p_sign .0039)
#
# B5a still carries distillation + dual memory; Ring-ER carries neither. Removing distill
# from BCL recovers almost exactly that deficit (B1 - B0 = +0.43 F1, +0.77 FWT), which
# points at the KL term. But the converse test at lr 0.01 disagrees: ADDING the identical
# distillation (--er_distill, reg=1 temp=5) to Ring-ER costs nothing (+0.58 F1 ns) and moves
# FWT by exactly zero (+0.00 [-0.33,+0.34]). Those two tests sit at different learning
# rates, so the disagreement is confounded and neither settles it.
#
# The two jobs below close the loop at ONE matched operating point (lr 0.001, is2), and are
# exact complements -- if distillation is the whole story then D1 lands on B5a and D2 lands
# on Ring-ER:
#
#   D1  Ring-ER + distill    expect ~4.1 / FWT ~11.7  (reproduces B5a)
#   D2  B5a - distill        expect ~4.5 / FWT ~12.6  (reproduces Ring-ER)
#
# If instead D1 stays at 4.5 and D2 stays at 4.1, the KL term is exonerated and the residual
# belongs to BCL's other non-bilevel machinery (the dual/validation memory path).
# D2 disables distillation the same way ablation B1 does, via memory_strength=0.
#
# Usage:  bash scripts/run_distill_isolation_cil.sh
set -uo pipefail

REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1   # main.py writes logs/ relative to cwd; anchor it here
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
BASE="$REPO/configs/base.yaml"
CFG="$REPO/configs/models/cil"

SEEDS="${SEEDS:-0,39,55,7,13,21,1,2,3}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
LOGDIR="${LOGDIR:-$REPO/scripts/logs/distill_isolation_cil}"
mkdir -p "$LOGDIR"

# run_job <config_stem> <expt_name> <inner_steps> <extra_cli_args>
run_job() {
  local stem="$1" expt="$2" isteps="$3" extra="$4"
  echo "[distill] START $expt  (cfg=$stem is=$isteps extra='$extra')  $(date '+%H:%M:%S')"
  "$PY" "$REPO/main.py" \
    --config "$BASE" \
    --config "$CFG/${stem}.yaml" \
    --n_epochs 1 --inner_steps "$isteps" --no-save_checkpoints \
    --expt_name "$expt" \
    --seeds "$SEEDS" $extra > "$LOGDIR/${expt}.log" 2>&1
  echo "[distill] END   $expt  exit=$?  $(date '+%H:%M:%S')"
}

JOBS=(
  # D1: add BCL-equivalent distillation to Ring-ER, at BCL's lr and budget.
  "er_ring|er_ring_static_lr001_distill_se_cil|2|--lr 0.001 --memory_loss_lambda 1 --er_distill"
  # D2: take distillation away from B5a (memory_strength 0, as ablation B1 does).
  "bcl_singlelevel|bcl_singlelevel_is2_nodistill_se_cil|2|--memory_strength 0"
)

echo "[distill] ===== distillation isolation (CIL)  seeds=$SEEDS  MAX_PARALLEL=$MAX_PARALLEL  START $(date) ====="

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

echo "[distill] ===== ALL DONE  $(date) ====="
echo "[distill] per-job logs: $LOGDIR/*.log"
