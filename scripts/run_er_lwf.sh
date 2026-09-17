#!/bin/bash
# LwF leverage test: does current-data distillation (LwF) add retention independent of
# replay / on-buffer distill? Single-epoch TIL, is4 (matched to the er_ring is4 anchors),
# 3 seeds. Factorial completes the 2x2 with existing rows:
#   er_ring is4 (replay CE)        58.6 / -10.1   [have]
#   er_ring + distill is4          59.0 /  -5.5   [have]
#   er_ring + lwf is4              <-- this run
#   er_ring + distill + lwf is4   <-- this run
# See docs/cmaml_bcl_ablations.md.
set -uo pipefail
REPO="/home/lunet/wsmr11/repos/evidential-cl"
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"

for cfg in er_ring_lwf er_ring_distill_lwf; do
  echo "[er-lwf] ===== START $cfg is4 ====="
  "$PY" "$REPO/main.py" \
    --config "$REPO/configs/base.yaml" \
    --config "$REPO/configs/models/til/$cfg.yaml" \
    --n_epochs 1 --inner_steps 4 \
    --expt_name "${cfg}_is4_se_til" \
    --seeds "0,39,55"
  echo "[er-lwf] ===== END $cfg is4 exit=$? ====="
done
echo "[er-lwf] ALL DONE"
echo "[er-lwf] results: logs/er_ring/{er_ring_lwf,er_ring_distill_lwf}_is4_se_til-*/{0,39,55}/results.txt"
