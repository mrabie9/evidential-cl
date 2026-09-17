#!/bin/bash
# Rerun the CTN diagnostic ablations in the single-epoch TIL regime with AMP
# disabled (n_epochs=1, inner_steps=2, --no-amp), plus a matched CTN baseline
# so FiLM / distillation impact can be judged against the same regime.
#
#   1. ctn           (full model, baseline)        -> seeds 0, 39, 55
#   2. ctn_nofilm    (FiLM head disabled)          -> seeds 0, 39, 55
#   3. ctn_nodistill (KL-distillation disabled)    -> seeds 0, 39, 55
#
# All three run IN PARALLEL (one GPU process each); seeds within each run
# sequentially via main.py's built-in --seeds sweep. Cross-seed summary lands at
#   logs/<config_stem>/<expt_name>-<timestamp>/results.txt
#
# NOTE: FWT (the "Forward:" line in results.txt) is structurally 0 in this code
# path. These runs yield valid F1 / BWT only.
#
# Usage:  bash scripts/run_ctn_ablations_me.sh
set -uo pipefail

REPO="/home/lunet/wsmr11/repos/evidential-cl"
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
BASE_CONFIG="$REPO/configs/base.yaml"
LOGDIR="$REPO/scripts/logs/ctn_ablations"
mkdir -p "$LOGDIR"

run_ablation() {
  local config_stem="$1"   # e.g. ctn_nofilm
  local expt_name="$2"     # e.g. film-ablation_me_til
  local seeds="$3"         # e.g. "0,39,55"
  local out_log="$4"

  echo "[run_ctn_ablations_me] START $config_stem | seeds=$seeds | expt=$expt_name -> $out_log"
  "$PY" "$REPO/main.py" \
    --config "$BASE_CONFIG" \
    --config "$REPO/configs/models/til/${config_stem}.yaml" \
    --n_epochs 1 --inner_steps 2 --no-amp \
    --expt_name "$expt_name" \
    --seeds "$seeds" >"$out_log" 2>&1
  echo "[run_ctn_ablations_me] DONE  $config_stem (exit $?)"
}

run_ablation "ctn"           "baseline_se_noamp_til"        "0,39,55" "$LOGDIR/ctn_se_noamp.log"           &
PID_BASE=$!
run_ablation "ctn_nofilm"    "film-ablation_se_noamp_til"   "0,39,55" "$LOGDIR/ctn_nofilm_se_noamp.log"    &
PID_FILM=$!
run_ablation "ctn_nodistill" "distill-ablation_se_noamp_til" "0,39,55" "$LOGDIR/ctn_nodistill_se_noamp.log" &
PID_DISTILL=$!

wait "$PID_BASE";    STATUS_BASE=$?
wait "$PID_FILM";    STATUS_FILM=$?
wait "$PID_DISTILL"; STATUS_DISTILL=$?

echo "[run_ctn_ablations_me] all done. ctn exit=$STATUS_BASE ctn_nofilm exit=$STATUS_FILM ctn_nodistill exit=$STATUS_DISTILL"
echo "[run_ctn_ablations_me] summaries:"
echo "  logs/ctn/baseline_se_noamp_til-*/results.txt"
echo "  logs/ctn_nofilm/film-ablation_se_noamp_til-*/results.txt"
echo "  logs/ctn_nodistill/distill-ablation_se_noamp_til-*/results.txt"
