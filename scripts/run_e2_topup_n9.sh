#!/bin/bash
# n=6 -> n=9 top-up for TIL ablation row E2 (reservoir -> fully-utilised dynamic ring).
#
# E2's pool at logs/ablations/til/res-er/E2/ holds seeds 0,39,55,7,13,21 while its
# pairing baseline E0 already has 9 (adds 1,2,3). Adding the same three seeds here
# extends the paired E2-E0 difference in analyse_ering_block_noamp.py to 9 matched
# pairs -- the standing threshold for a verdict on this repo's paired effects.
#
# Config is copied verbatim from the E2 line in run_noamp_gembob_ering.sh and
# re-verified against the logged training_parameters.json of the existing run:
#   er_ring, --no-amp, lr 0.01, memory_loss_lambda 1, --er_dynamic_ring,
#   inner_steps 2, n_epochs 1, memories 5120, batch 256.
#
# expt_name matches the existing pool exactly so organise_ablations.py's
# "er_ring/ering_dynring_lr01_noamp_se_til-*" glob picks the new timestamped run up
# and moves it under logs/ablations/til/res-er/E2/ alongside the n=6 run.
#
# Usage:  bash scripts/run_e2_topup_n9.sh
set -uo pipefail
REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
BASE="$REPO/configs/base.yaml"
CFG="$REPO/configs/models/til"
SEEDS="${SEEDS:-1,2,3}"
LOGDIR="$REPO/scripts/logs/e2_topup_n9"
mkdir -p "$LOGDIR"

EXPT="ering_dynring_lr01_noamp_se_til"

echo "[e2topup] ===== START $(date) seeds=$SEEDS ====="
"$PY" "$REPO/main.py" \
  --config "$BASE" --config "$CFG/er_ring.yaml" \
  --n_epochs 1 --inner_steps 2 --no-save_checkpoints --no-amp \
  --lr 0.01 --memory_loss_lambda 1 --er_dynamic_ring \
  --expt_name "$EXPT" --seeds "$SEEDS" \
  > "$LOGDIR/${EXPT}_topup.log" 2>&1
echo "[e2topup] END exit=$? $(date)"
echo "[e2topup] organise: $PY scripts/organise_ablations.py"
echo "[e2topup] analyse:  $PY scripts/analyse_ering_block_noamp.py"
