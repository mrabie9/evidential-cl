#!/bin/bash
# 3-seed single-epoch TIL run of gem_distill + signal-only class-balanced replay, matched hyperparams.
set -uo pipefail
REPO="/home/lunet/wsmr11/repos/evidential-cl"
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
"$PY" "$REPO/main.py" \
  --config "$REPO/configs/base.yaml" \
  --config "$REPO/configs/models/til/gem_distill_balsig.yaml" \
  --n_epochs 1 --inner_steps 2 \
  --expt_name gem_distill_balsig_se_til \
  --seeds "0,39,55"
echo "[run_gem_distill_balsig] exit=$?"
