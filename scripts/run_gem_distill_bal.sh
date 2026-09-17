#!/bin/bash
# 3-seed single-epoch TIL run of gem_distill + A1 class-balanced replay (CBRS), at MATCHED
# hyperparameters vs gem_distill (lr/margin/distill_lambda inherited; only balanced_replay flipped).
# Clean A/B to isolate the effect of class-balanced buffer admission.
set -uo pipefail
REPO="/home/lunet/wsmr11/repos/evidential-cl"
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"

"$PY" "$REPO/main.py" \
  --config "$REPO/configs/base.yaml" \
  --config "$REPO/configs/models/til/gem_distill_bal.yaml" \
  --n_epochs 1 --inner_steps 2 \
  --expt_name gem_distill_bal_se_til \
  --seeds "0,39,55"
echo "[run_gem_distill_bal] exit=$?"
echo "[run_gem_distill_bal] results: logs/gem_distill_bal/gem_distill_bal_se_til-*/results.txt"
