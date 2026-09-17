#!/bin/bash
# n=6 -> n=9 top-up for ablation row B5b (BCL-Dual bilevel structure -> single
# loop, budget-matched at inner_steps=4). Adds seeds 1,2,3 so the paired B5b-vs-B0
# comparison extends to 9 matched pairs. Config/args match the existing B5b runs
# in logs/ablations/til/bcl-dual/B5b/ (bcl_singlelevel.yaml, --inner_steps 4, no --beta).
# Results land in logs/bcl_singlelevel/bcl_singlelevel_is4_topup-<ts>/{1,2,3}/ ;
# move into the ablations tree after.
set -uo pipefail
REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
LOGDIR="$REPO/scripts/logs/b5b_topup"
mkdir -p "$LOGDIR"

echo "[b5b] ===== START bcl_singlelevel is4 topup seeds=1,2,3 $(date) ====="
"$PY" "$REPO/main.py" \
  --config "$REPO/configs/base.yaml" \
  --config "$REPO/configs/models/til/bcl_singlelevel.yaml" \
  --n_epochs 1 --inner_steps 4 \
  --expt_name "bcl_singlelevel_is4_topup" \
  --seeds "1,2,3" > "$LOGDIR/bcl_singlelevel_is4_topup.log" 2>&1
echo "[b5b] ===== END exit=$? $(date) ====="
echo "[b5b] results: logs/bcl_singlelevel/bcl_singlelevel_is4_topup-*/{1,2,3}/results.txt"
