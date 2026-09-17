#!/bin/bash
# Driver: run the "best of the best" combos then the budget-match ablation table,
# sequentially (one GPU job at a time). Each sub-script handles its own 3-seed sweeps.
set -uo pipefail
REPO="/home/lunet/wsmr11/repos/evidential-cl"

echo "[driver] ===== BEST-OF-BEST START $(date) ====="
bash "$REPO/scripts/run_best_of_best.sh"
echo "[driver] ===== BEST-OF-BEST END exit=$? $(date) ====="

echo "[driver] ===== BUDGET-MATCH START $(date) ====="
bash "$REPO/scripts/run_budget_match.sh"
echo "[driver] ===== BUDGET-MATCH END exit=$? $(date) ====="

echo "[driver] ALL DONE $(date)"
