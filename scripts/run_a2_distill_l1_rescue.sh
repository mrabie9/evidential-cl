#!/bin/bash
# Rescue launch for gem_bob A2 (KL distillation, lambda=1), --no-amp, lr 0.01.
#
# WHY standalone: the arm was queued inside run_noamp_gembob_ering.sh but its job-table entry
# had a trailing '#' fused to the closing quote, so it launched as '--distill_lambda 1#' and
# argparse killed it in 3 seconds on 2026-07-27 10:38. The table is fixed now, but the driver
# had already expanded its array in memory, so the corrected entry will not be picked up by
# the running batch. A2 is a Phase-1 mechanism arm and carries a Holm verdict, so it cannot be
# dropped -- it runs here on its own lane instead of waiting for the batch to drain.
#
# Settings match the batch exactly: n=6 seeds, inner_steps 2, --no-amp, distill_lambda 1
# (BCL-matched coefficient; see configs/models/til/gem_bob_distill.yaml).
set -uo pipefail

REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
LOGDIR="$REPO/scripts/logs/noamp_gembob_ering"
mkdir -p "$LOGDIR"

echo "[a2rescue] START $(date)"
"$PY" "$REPO/main.py" \
  --config "$REPO/configs/base.yaml" \
  --config "$REPO/configs/models/til/gem_bob_distill.yaml" \
  --n_epochs 1 --inner_steps 2 --no-save_checkpoints --no-amp \
  --expt_name gembob_a2_distill_l1_noamp \
  --seeds 0,39,55,7,13,21 \
  --distill_lambda 1 \
  > "$LOGDIR/gembob_a2_distill_l1_noamp.log" 2>&1
echo "[a2rescue] END exit=$? $(date)"
