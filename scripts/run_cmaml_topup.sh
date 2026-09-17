#!/bin/bash
# n=6 -> n=9 top-up for C-MAML rows M0 (baseline, second-order) and M4 (alpha0,
# inner-loop adaptation off). Adds seeds 1,2,3 so the paired M4-vs-M0 comparison
# extends to 9 matched pairs. Configs/args match the existing runs pooled in
# logs/{cmaml,cmaml_alpha0}/ (pre-split-CE pools):
#   M0 = cmaml.yaml --second_order   (params show second_order=True)
#   M4 = cmaml_alpha0.yaml            (alpha_init=0, no --second_order)
# both --inner_steps 2, opt_wt 0.01 (set in the yaml). Results land in
# logs/<stem>/<expt>-<ts>/{1,2,3}/ ; move into the ablations tree after.
set -uo pipefail
REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
BASE="$REPO/configs/base.yaml"
CFG="$REPO/configs/models/til"
LOGDIR="$REPO/scripts/logs/cmaml_topup"
mkdir -p "$LOGDIR"

run_job() {
  local stem="$1" expt="$2" extra="$3"
  echo "[cmaml-topup] START $expt (cfg=$stem extra='$extra') $(date '+%H:%M:%S')"
  "$PY" "$REPO/main.py" \
    --config "$BASE" --config "$CFG/${stem}.yaml" \
    --n_epochs 1 --inner_steps 2 \
    --expt_name "$expt" --seeds "1,2,3" $extra > "$LOGDIR/${expt}.log" 2>&1
  echo "[cmaml-topup] END   $expt exit=$? $(date '+%H:%M:%S')"
}

echo "[cmaml-topup] ===== START seeds=1,2,3 $(date) ====="
run_job "cmaml"        "cmaml_secondorder_topup" "--second_order" &
run_job "cmaml_alpha0" "cmaml_alpha0_topup"      "" &
wait
echo "[cmaml-topup] ===== ALL DONE $(date) ====="
echo "[cmaml-topup] results: logs/{cmaml,cmaml_alpha0}/*_topup-*/{1,2,3}/results.txt"
