#!/bin/bash
# TIL: does BCL-Dual's B5a reduce to Ring-ER + distillation?
#
# B5a is BCL-Dual with the bilevel round collapsed to a single loop at B0's inner-loop
# budget (inner_steps 2). What survives that collapse is replay + KL distillation on frozen
# soft targets -- which is exactly what er_ring --er_distill computes: model/er_ring.py:473
# scales self.reg * KL against frozen buffer targets, and self.reg = memory_strength in both
# er_ring.py:61 and bcl_dual.py:89, so the coefficient path is identical.
#
# Config matched to B5a (logs/ablations/til/bcl-dual/B5a/.../training_parameters.json):
#   lr 0.001            (er_ring's TIL yaml already defaults to this -- BCL's tuned rate)
#   inner_steps 2       memory_strength 1     temperature 5.0
#   n_memories 5120     memory_loss_lambda 1  mem_sampling ring   bf16 AMP
# Seeds are B5a's exact set (n=6) so the pairing is complete.
#
# Two arms. The no-distill control is what makes the test informative: if B5a - distill_arm
# is null but B5a - control is not, the equivalence is real and distillation is the term
# doing the work. Without the control a null result could just mean both are weak.
#
# Residual known difference: B5a retains BCL's second (meta) memory buffer, which er_ring
# does not have. TIL row B3 measures that buffer at -0.4 fcl (inconclusive), so it is the
# expected size of any leftover gap.
set -uo pipefail

REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
SEEDS="${SEEDS:-0,39,55,7,13,21}"
LOGDIR="$REPO/scripts/logs/ering_distill_b5a"
mkdir -p "$LOGDIR"

run_job() {
  local expt="$1" seeds="$2" extra="$3"
  echo "[b5a] START $expt seeds=$seeds $(date '+%H:%M:%S')"
  "$PY" "$REPO/main.py" \
    --config "$REPO/configs/base.yaml" \
    --config "$REPO/configs/models/til/er_ring.yaml" \
    --n_epochs 1 --inner_steps 2 --no-save_checkpoints \
    --lr 0.001 --memory_loss_lambda 1 --memory_strength 1 --temperature 5.0 \
    --expt_name "$expt" --seeds "$seeds" $extra \
    > "$LOGDIR/${expt}_${seeds//,/-}.log" 2>&1
  echo "[b5a] END   $expt seeds=$seeds exit=$? $(date '+%H:%M:%S')"
}

echo "[b5a] ===== START $(date) seeds=$SEEDS ====="
run_job "ering_distill_b5amatch_se_til"   "0,39,55" "--er_distill" &
run_job "ering_distill_b5amatch_se_til"   "7,13,21" "--er_distill" &
run_job "ering_nodistill_b5amatch_se_til" "0,39,55" "" &
run_job "ering_nodistill_b5amatch_se_til" "7,13,21" "" &
wait
echo "[b5a] ===== ALL DONE $(date) ====="
