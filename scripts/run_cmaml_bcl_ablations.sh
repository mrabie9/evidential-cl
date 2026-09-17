#!/bin/bash
# C-MAML and BCL-Dual ablation sweep: single-epoch TIL, 3 seeds, matched hyperparameters.
# Baselines first (anchors), then single-knob ablations. See docs/cmaml_bcl_ablations.md.
# Run sequentially (one config at a time) so this stays a single job alongside any other run.
set -uo pipefail
REPO="/home/lunet/wsmr11/repos/evidential-cl"
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"

CONFIGS=(
  bcl_dual
  cmaml
  bcl_nodistill
  bcl_nobilevel
  bcl_nodualmem
  cmaml_first_order
  cmaml_no_replay
  cmaml_single_inner
)

for cfg in "${CONFIGS[@]}"; do
  echo "[ablation-sweep] ===== START $cfg ====="
  "$PY" "$REPO/main.py" \
    --config "$REPO/configs/base.yaml" \
    --config "$REPO/configs/models/til/$cfg.yaml" \
    --n_epochs 1 --inner_steps 2 \
    --expt_name "${cfg}_se_til" \
    --seeds "0,39,55"
  echo "[ablation-sweep] ===== END $cfg exit=$? ====="
done
echo "[ablation-sweep] ALL DONE"
echo "[ablation-sweep] results: logs/<cfg>/<cfg>_se_til-*/{0,39,55}/results.txt"
