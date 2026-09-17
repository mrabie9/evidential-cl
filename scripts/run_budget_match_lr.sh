#!/bin/bash
# Parametrized budget-match: same design as run_budget_match.sh (hold step budget = 4 SGD
# steps AND a common lr equal across families) but the common lr is passed in, so the whole
# table can be swept at a second operating point. Everything else (each method's other tuned
# params) is unchanged; only lr + step budget move.
#
#   $1 = common lr (e.g. 0.01)
#   $2 = lr tag for expt_name (e.g. 01)   -> suffix _bm_lr<tag>_se_til
#
# Step budget: replay methods (GEM/Ring-ER/Res-ER) --inner_steps 4; meta methods
# (CMAML/BCL-Dual/CTN) --inner_steps 2 (2 rounds x 2 = 4). CMAML's weight lr is opt_wt, not
# --lr (inert for lamaml_cifar), so it gets --opt_wt <lr>. CTN gets --no-amp on the CLI
# (ctn.yaml's no_amp key is inert) and is footnoted as not step-comparable. Seeds 0,39,55.
#
# Memory-strength confound also removed: every run is forced to
# --memory_strength 1 --memory_loss_lambda 1 so the replay/memory-regularization weight is
# equal across methods. This overrides native values (CTN 10->1 distill weight; er_ring /
# eralg4 0.1->1 replay weight; GEM margin / BCL already 1; CMAML uses neither -> no-op).
set -uo pipefail
REPO="/home/lunet/wsmr11/repos/evidential-cl"
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"

LR="${1:?usage: run_budget_match_lr.sh <lr> <lrtag>}"
TAG="${2:?usage: run_budget_match_lr.sh <lr> <lrtag>}"

run() {
  local cfg="$1"; local isteps="$2"; local name="$3"; shift 3
  echo "[budget-match-lr$TAG] ===== START $name (cfg=$cfg lr=$LR inner_steps=$isteps) ====="
  "$PY" "$REPO/main.py" \
    --config "$REPO/configs/base.yaml" \
    --config "$REPO/configs/models/til/$cfg.yaml" \
    --lr "$LR" --n_epochs 1 --inner_steps "$isteps" \
    --memory_strength 1 --memory_loss_lambda 1 \
    --expt_name "$name" \
    --seeds "0,39,55" "$@"
  echo "[budget-match-lr$TAG] ===== END $name exit=$? ====="
}

run gem      4 "gem_bm_lr${TAG}_se_til"
run er_ring  4 "er_ring_bm_lr${TAG}_se_til"
run eralg4   4 "eralg4_bm_lr${TAG}_se_til"
run cmaml    2 "cmaml_bm_lr${TAG}_se_til" --opt_wt "$LR"
run bcl_dual 2 "bcl_dual_bm_lr${TAG}_se_til"
run ctn      2 "ctn_bm_lr${TAG}_se_noamp_til" --no-amp

echo "[budget-match-lr$TAG] ALL DONE"
echo "[budget-match-lr$TAG] results: logs/<model>/<expt_name>-*/{0,39,55}/results.txt"
