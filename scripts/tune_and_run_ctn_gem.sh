#!/bin/bash
# Tune ctn_gem (B2): grid lr x memory_strength x gem_margin (task_emb/ctx_lr fixed from config),
# write best params back into the model config, then launch the 3-seed single-epoch TIL run.
set -uo pipefail
REPO="/home/lunet/wsmr11/repos/evidential-cl"
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
CFG="$REPO/configs/models/til/ctn_gem.yaml"

echo "==================== TUNING ctn_gem ===================="
"$PY" "$REPO/tuning/tune_ctn_gem.py" \
  --config "$REPO/configs/tuning_defaults.yaml" \
  --config "$CFG" \
  --override n_epochs=1 --override inner_steps=2 \
  --num-samples 40
tune_status=$?
echo "[tune_and_run] tuning exit=$tune_status"
if [ "$tune_status" -ne 0 ]; then
  echo "[tune_and_run] tuning failed; NOT launching 3-seed run."; exit 1
fi

echo "==================== TUNED CONFIG ===================="
cat "$CFG"

echo "==================== 3-SEED TIL RUN ctn_gem ===================="
"$PY" "$REPO/main.py" \
  --config "$REPO/configs/base.yaml" \
  --config "$CFG" \
  --n_epochs 1 --inner_steps 2 \
  --expt_name ctn_gem_se_til \
  --seeds "0,39,55"
echo "[tune_and_run] 3-seed run exit=$?"
echo "[tune_and_run] results: logs/ctn_gem/ctn_gem_se_til-*/results.txt"
