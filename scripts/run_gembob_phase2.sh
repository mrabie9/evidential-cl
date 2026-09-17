#!/bin/bash
# gem_bob Phase 2: combine the Phase-1 load-bearing mechanisms.
#
# Phase 1 (lr=0.01, n=9) found two load-bearing add-ons over plain GEM: the fully-utilised
# ring (A1, +1.2 F1) and KL distillation at lambda=1 (A2, +1.8 F1, reaching 66.2 == the
# gem_distill headline). Bilevel (A3) and meta-batch (A5) were inconclusive/inert on F1.
# Phase 2 builds on the A1+A2 pair and asks whether the two non-winners add anything on top:
#
#   C1  A1 + A2            ring + distill(1)                 -- do the two winners stack above 66.2?
#   C2  A1 + A2 + A3       + bilevel (inner_steps 1)         -- budget-matched (bilevel = 2 SGD steps)
#   C3  A1 + A2 + A5       + meta-batch K=3                  -- budget-neutral
#
# All lr=0.01, n=9, paired against the SAME A0 baseline as Phase 1 (gembob_a0), so C-vs-A0 and
# C-vs-A1/A2 deltas are directly comparable. 3 configs x 9 seeds = 27 runs.
#
# Analyse:  la-maml_env/bin/python scripts/compare_gembob.py --phase 2 --verbose
#
# Usage:  bash scripts/run_gembob_phase2.sh   [MAX_PARALLEL=4 ...]
set -uo pipefail

REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1   # main.py writes logs/ relative to cwd; anchor it here
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
BASE="$REPO/configs/base.yaml"
CFG="$REPO/configs/models/til"

SEEDS="${SEEDS:-0,39,55,7,13,21,1,2,3}"   # same n=9 set as Phase 1
MAX_PARALLEL="${MAX_PARALLEL:-3}"

LOGDIR="$REPO/scripts/logs/gembob_phase2"
mkdir -p "$LOGDIR"

# run_job <config_stem> <expt_name> <inner_steps>
run_job() {
  local stem="$1" expt="$2" isteps="$3"
  echo "[queue] START $expt  (cfg=$stem is=$isteps)  $(date '+%H:%M:%S')"
  "$PY" "$REPO/main.py" \
    --config "$BASE" \
    --config "$CFG/${stem}.yaml" \
    --n_epochs 1 --inner_steps "$isteps" \
    --expt_name "$expt" \
    --seeds "$SEEDS" > "$LOGDIR/${expt}.log" 2>&1
  echo "[queue] END   $expt  exit=$?  $(date '+%H:%M:%S')"
}

# "stem|expt|inner_steps". C2 runs inner_steps 1 (bilevel's inner+outer = 2 SGD steps,
# matching the baseline budget); C1/C3 run inner_steps 2.
JOBS=(
  "gem_bob_c1|gembob_c1|2"
  "gem_bob_c2|gembob_c2|1"
  "gem_bob_c3|gembob_c3|2"
)

echo "=== gem_bob Phase 2: ${#JOBS[@]} configs x $(echo "$SEEDS" | tr ',' '\n' | wc -l) seeds"
echo "=== seeds=$SEEDS  MAX_PARALLEL=$MAX_PARALLEL  logs -> $LOGDIR"
echo "=== started $(date)"

for job in "${JOBS[@]}"; do
  IFS='|' read -r stem expt isteps <<< "$job"
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
    sleep 20
  done
  run_job "$stem" "$expt" "$isteps" &
done
wait

echo "=== all jobs finished $(date)"
echo "=== analyse: $PY $REPO/scripts/compare_gembob.py --phase 2 --verbose"
