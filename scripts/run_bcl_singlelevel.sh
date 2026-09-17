#!/bin/bash
# BCL-Dual ablation B5: bilevel two-loop structure removed (single-level reduction
# to replay + distillation). Single-epoch TIL, 3 seeds. See docs/cmaml_bcl_ablations.md.
#
# Two runs:
#   * is4 = BUDGET-MATCHED to B0 (B0 = 2 rounds x 2 steps = 4 SGD steps;
#           this = 4 rounds x 1 fused step = 4 SGD steps). This is the decisive run:
#           same step budget, bilevel structure removed.
#   * is2 = un-matched (4 -> 2 SGD steps) to show the raw drop from losing the
#           second step as well, for the budget-vs-structure decomposition.
#
# Read against anchors already in the doc: er_ring + distill (55.6) below,
# bcl@beta=1 / no-meta-amp (60.0) above, B0 bcl_dual (62.6) top.
set -uo pipefail
REPO="/home/lunet/wsmr11/repos/evidential-cl"
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"

echo "[singlelevel] ===== START bcl_singlelevel is4 (budget-matched) ====="
"$PY" "$REPO/main.py" \
  --config "$REPO/configs/base.yaml" \
  --config "$REPO/configs/models/til/bcl_singlelevel.yaml" \
  --n_epochs 1 --inner_steps 4 \
  --expt_name "bcl_singlelevel_is4_se_til" \
  --seeds "0,39,55"
echo "[singlelevel] ===== END bcl_singlelevel is4 exit=$? ====="

echo "[singlelevel] ===== START bcl_singlelevel is2 (un-matched) ====="
"$PY" "$REPO/main.py" \
  --config "$REPO/configs/base.yaml" \
  --config "$REPO/configs/models/til/bcl_singlelevel.yaml" \
  --n_epochs 1 --inner_steps 2 \
  --expt_name "bcl_singlelevel_is2_se_til" \
  --seeds "0,39,55"
echo "[singlelevel] ===== END bcl_singlelevel is2 exit=$? ====="

echo "[singlelevel] ALL DONE"
echo "[singlelevel] results: logs/bcl_singlelevel/bcl_singlelevel_is{2,4}_se_til-*/{0,39,55}/results.txt"
