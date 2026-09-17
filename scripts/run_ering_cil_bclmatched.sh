#!/bin/bash
# Ring-ER in CIL matched to BCL-Dual, to test "BCL-Dual reduces to Ring-ER".
#
# WHY new runs: the existing CIL Ring-ER row (E1, er_ring_dynring_se_cil) cannot serve as
# the comparator. It differs from BCL-Dual's B0 on two axes at once:
#   lr        0.01 vs BCL's 0.001  (10x)
#   buffer    dynamic ring vs BCL's static ring
# Either alone would confound method with configuration.
#
# Two arms, because BCL's bilevel round performs 2 SGD steps per inner step, so B0 at
# inner_steps 2 spends a 4-step budget (this is why the B5b flattening row matches at is4):
#   is2 -- config-matched   (same inner_steps as B0, half the SGD steps)
#   is4 -- budget-matched   (same total SGD steps as B0)
# This repo has repeatedly found budget masquerading as mechanism, so both are needed
# before any "reduces to" claim.
#
# Everything else mirrors B0 (fixed-cil-mask_se_cil-2026-07-24_16-58-55-9781): lr 0.001,
# static ring, n_memories 5120, memory_loss_lambda 1, bf16 AMP (the CIL block's precision),
# n_epochs 1, class_incremental loader. Seeds are B0's exact set so pairing is complete
# at n=9.
set -uo pipefail

REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
SEEDS="${SEEDS:-0,1,2,3,7,13,21,39,55}"
LOGDIR="$REPO/scripts/logs/ering_cil_bclmatched"
mkdir -p "$LOGDIR"

run_job() {
  local isteps="$1" expt="$2" seeds="$3"
  echo "[eringcil] START $expt is=$isteps seeds=$seeds $(date '+%H:%M:%S')"
  "$PY" "$REPO/main.py" \
    --config "$REPO/configs/base.yaml" \
    --config "$REPO/configs/models/cil/er_ring.yaml" \
    --n_epochs 1 --inner_steps "$isteps" --no-save_checkpoints \
    --lr 0.001 --memory_loss_lambda 1 \
    --expt_name "$expt" --seeds "$seeds" \
    > "$LOGDIR/${expt}_${seeds//,/-}.log" 2>&1
  echo "[eringcil] END   $expt seeds=$seeds exit=$? $(date '+%H:%M:%S')"
}

echo "[eringcil] ===== START $(date) seeds=$SEEDS ====="
run_job 2 "ering_static_lr001_is2_bclmatch_se_cil" "0,1,2,3,7" &
run_job 2 "ering_static_lr001_is2_bclmatch_se_cil" "13,21,39,55" &
run_job 4 "ering_static_lr001_is4_bclmatch_se_cil" "0,1,2,3,7" &
run_job 4 "ering_static_lr001_is4_bclmatch_se_cil" "13,21,39,55" &
wait
echo "[eringcil] ===== ALL DONE $(date) ====="
