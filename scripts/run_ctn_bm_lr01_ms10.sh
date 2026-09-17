#!/bin/bash
# CTN budget-match cell at lr=0.01 with NATIVE distill weight (memory_strength 10),
# filling the missing table entry: the existing lr=0.01 run was equalized (ms 10->1),
# which guts CTN's distillation. Same regime as run_budget_match_lr.sh: n_epochs 1,
# inner_steps 2 (2 rounds x 2 = 4 steps), --no-amp on CLI, seeds 0,39,55.
set -uo pipefail
REPO="/home/lunet/wsmr11/repos/evidential-cl"
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
cd "$REPO"

"$PY" "$REPO/main.py" \
  --config "$REPO/configs/base.yaml" \
  --config "$REPO/configs/models/til/ctn.yaml" \
  --lr 0.01 --n_epochs 1 --inner_steps 2 \
  --memory_strength 10 \
  --expt_name "ctn_bm_lr01_ms10_se_noamp_til" \
  --seeds "0,39,55" --no-amp
echo "===== END ctn_bm_lr01_ms10 exit=$? ====="
