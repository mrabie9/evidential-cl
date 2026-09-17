#!/bin/bash
# Run the two CTN diagnostic ablations in the single-epoch (1 epoch, 2 inner passes) TIL regime.
# The two ablations run IN PARALLEL (one GPU process each); seeds within each run sequentially.
#
#   1. ctn_nofilm    (FiLM head disabled)        -> seeds 39, 55   (seed 0 already done)
#   2. ctn_nodistill (KL-distillation disabled)  -> seeds 0, 39, 55
#
# Each invocation drives main.py's built-in seed sweep (--seeds), which relaunches one
# child process per seed under a shared experiment dir and writes a cross-seed summary at
#   logs/<config_stem>/<expt_name>-<timestamp>/results.txt
# (top-level dir name = stem of the LAST --config file: ctn_nofilm / ctn_nodistill).
#
# NOTE: FWT (the "Forward:" line in results.txt) is structurally 0 in this code path -- the
# evaluator only ever sees tasks <= current, so the confusion-matrix forward cells and the
# at-init baseline row are never populated. These runs yield valid F1 / BWT only.
#
# Usage:  bash scripts/run_ctn_ablations.sh
set -uo pipefail

REPO="/home/lunet/wsmr11/repos/evidential-cl"
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
BASE_CONFIG="$REPO/configs/base.yaml"
LOGDIR="$REPO/scripts/logs/ctn_ablations"
mkdir -p "$LOGDIR"

run_ablation() {
  local config_stem="$1"   # e.g. ctn_nofilm
  local expt_name="$2"     # e.g. film-ablation_se_til
  local seeds="$3"         # e.g. "39,55"
  local out_log="$4"

  echo "[run_ctn_ablations] START $config_stem | seeds=$seeds | expt=$expt_name -> $out_log"
  "$PY" "$REPO/main.py" \
    --config "$BASE_CONFIG" \
    --config "$REPO/configs/models/til/${config_stem}.yaml" \
    --n_epochs 1 --inner_steps 2 \
    --expt_name "$expt_name" \
    --seeds "$seeds" >"$out_log" 2>&1
  echo "[run_ctn_ablations] DONE  $config_stem (exit $?)"
}

# Launch both ablations concurrently.
run_ablation "ctn_nofilm"    "film-ablation_se_til"    "39,55"   "$LOGDIR/ctn_nofilm.log"    &
PID_FILM=$!
run_ablation "ctn_nodistill" "distill-ablation_se_til" "0,39,55" "$LOGDIR/ctn_nodistill.log" &
PID_DISTILL=$!

wait "$PID_FILM";    STATUS_FILM=$?
wait "$PID_DISTILL"; STATUS_DISTILL=$?

echo "[run_ctn_ablations] all done. ctn_nofilm exit=$STATUS_FILM ctn_nodistill exit=$STATUS_DISTILL"
echo "[run_ctn_ablations] summaries:"
echo "  logs/ctn_nofilm/film-ablation_se_til-*/results.txt"
echo "  logs/ctn_nodistill/distill-ablation_se_til-*/results.txt"
