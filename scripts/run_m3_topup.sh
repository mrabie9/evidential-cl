#!/bin/bash
# n=6 -> n=9 top-up for C-MAML row M3 (meta-batch averaging off, meta_batches=1).
# Adds seeds 1,2,3 so the paired M3-vs-M0 comparison extends to 9 matched pairs.
# Config/args match the existing M3 runs in logs/cmaml_single_inner/ (pre-split-CE pool):
#   cmaml_single_inner.yaml, --inner_steps 2, opt_wt 0.01 (set in the yaml), no extra flags.
# Results land in logs/cmaml_single_inner/cmaml_singleinner_topup-<ts>/{1,2,3}/ ; move after.
set -uo pipefail
REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
LOGDIR="$REPO/scripts/logs/m3_topup"
mkdir -p "$LOGDIR"

echo "[m3-topup] ===== START seeds=1,2,3 $(date) ====="
"$PY" "$REPO/main.py" \
  --config "$REPO/configs/base.yaml" \
  --config "$REPO/configs/models/til/cmaml_single_inner.yaml" \
  --n_epochs 1 --inner_steps 2 \
  --expt_name "cmaml_singleinner_topup" \
  --seeds "1,2,3" > "$LOGDIR/cmaml_singleinner_topup.log" 2>&1
echo "[m3-topup] ===== END exit=$? $(date) ====="
echo "[m3-topup] results: logs/cmaml_single_inner/cmaml_singleinner_topup-*/{1,2,3}/results.txt"
