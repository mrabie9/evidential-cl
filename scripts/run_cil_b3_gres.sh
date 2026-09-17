#!/bin/bash
# CIL combination experiment: B3 (bcl_nodualmem) + global reservoir (E0's mechanism).
# Single-knob change from B3 (bcl_global_reservoir true); n=9 seeds matching the B3 baseline
# (bcl_nodualmem_se_cil) so B3+gres pairs directly against B3's F1 13.7. CIL, single-epoch,
# inner_steps 2 (4 SGD steps/round for BCL's bilevel), checkpoints off (disk).
set -uo pipefail
REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
SEEDS="${SEEDS:-0,39,55,7,13,21,1,2,3}"
LOGDIR="$REPO/scripts/logs/cil_b3_gres"
mkdir -p "$LOGDIR"
echo "[gres] START bcl_b3_gres_se_cil seeds=$SEEDS $(date)"
"$PY" "$REPO/main.py" \
  --config "$REPO/configs/base.yaml" \
  --config "$REPO/configs/models/cil/bcl_b3_gres.yaml" \
  --n_epochs 1 --inner_steps 2 --no-save_checkpoints \
  --expt_name bcl_b3_gres_se_cil \
  --seeds "$SEEDS" > "$LOGDIR/bcl_b3_gres_se_cil.log" 2>&1
echo "[gres] END exit=$? $(date)"
