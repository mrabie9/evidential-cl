#!/bin/bash
# "Best of the best" combinations: pair each family's strongest ingredient with the
# transferable retention channel (distillation / LwF / reservoir admission). Single-epoch
# TIL, n_epochs 1 / inner_steps 2, 3 seeds (0,39,55), AMP on -- matched to the existing
# gem_distill winner (66.2) so all rows drop straight into docs/cmaml_bcl_ablations_v2.md.
#
#   gem_distill        GEM (QP) + on-buffer KL distillation        [config exists, re-run]
#   gem_lwf            GEM (QP) + current-data LwF                  [config exists, re-run]
#   eralg4_distill     Res-ER (reservoir) + teacher-snapshot KL     [NEW]
#   bcl_dual_reservoir BCL-Dual, reservoir buffer admission         [NEW]
#
# NOTE: CTN is NOT in this table -- CTN must run --no-amp, so it lives only in the
# budget-match table (run_budget_match.sh) with an explicit AMP footnote.
set -uo pipefail
REPO="/home/lunet/wsmr11/repos/evidential-cl"
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"

CONFIGS=(
  gem_distill
  gem_lwf
  eralg4_distill
  bcl_dual_reservoir
)

for cfg in "${CONFIGS[@]}"; do
  echo "[best-of-best] ===== START $cfg ====="
  "$PY" "$REPO/main.py" \
    --config "$REPO/configs/base.yaml" \
    --config "$REPO/configs/models/til/$cfg.yaml" \
    --n_epochs 1 --inner_steps 2 \
    --expt_name "${cfg}_bob_se_til" \
    --seeds "0,39,55"
  echo "[best-of-best] ===== END $cfg exit=$? ====="
done
echo "[best-of-best] ALL DONE"
echo "[best-of-best] results: logs/<cfg>/<cfg>_bob_se_til-*/{0,39,55}/results.txt"
