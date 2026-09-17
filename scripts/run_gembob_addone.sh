#!/bin/bash
# gem_bob ("GEM best-of-best") Phase 1: add-one study.
#
# Takes GEM and adds, ONE AT A TIME, each mechanism the leave-one-out grid in
# docs/ablation_studies.tex found load-bearing, to test whether they transfer to a parent
# that already has GEM's gradient projection (G1, -3.6 F1 -- the strongest single mechanism
# in the grid). Phase 2 combines whichever arms show a meaningful effect.
#
#   A0  baseline, all gates off        -- reproduces plain GEM exactly (correctness gate)
#   A1  + fully-utilised ring buffer   -- E2, +2.1 F1 on Res-ER
#   A2  + KL distillation              -- B1 -1.2, T2 -4.7
#   A3  + bilevel, budget-matched      -- B5b, -1.4 F1
#   A4  + bilevel, 2x budget           -- diagnostic: A3-vs-A4 separates structure from compute
#   A5  + meta-batch averaging K=3     -- M3, -1.2 F1
#
# BUDGET MATCHING. A0/A1/A2/A5 run --inner_steps 2 (2 SGD steps per batch). A3 runs
# --inner_steps 1 because its bilevel round takes an inner AND an outer step, so it lands on
# the same 2 steps. A4 is the deliberately UNMATCHED rerun at --inner_steps 2 (4 steps): the
# write-up's own finding is that raw step count is worth more than most mechanisms
# ("budget masquerades as mechanism", B5a vs B5b = 3.4 F1), so without A4 a bilevel win at
# matched budget cannot be told apart from a compute win.
#
# lr 0.01 for every arm: the config-matched operating point (GEM 63.4 +/- 0.1) rather than
# GEM's native 0.03 (63.5 +/- 2.1), because the paired tests need the lower seed variance.
#
# n=9 seeds, matching the topped-up rows of the grid so results are directly comparable.
# 6 configs x 9 seeds = 54 runs. At ~26 min/seed measured for GEM (distillation and the
# bilevel outer step cost somewhat more), expect ~35-50 GPU-h, ~15-20 h wall at MAX_PARALLEL=3.
#
# Results land in logs/gem_bob/<expt_name>-<timestamp>/<seed>/results.txt and are analysed by
#   la-maml_env/bin/python scripts/compare_gembob.py --phase 1 --verbose
#
# Usage:  bash scripts/run_gembob_addone.sh
#         MAX_PARALLEL=4 bash scripts/run_gembob_addone.sh
set -uo pipefail

REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1   # main.py writes its logs/ dir relative to cwd; anchor it here
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
BASE="$REPO/configs/base.yaml"
CFG="$REPO/configs/models/til"

SEEDS="${SEEDS:-0,39,55,7,13,21,1,2,3}"   # the n=9 set used by the topped-up grid rows
MAX_PARALLEL="${MAX_PARALLEL:-3}"

LOGDIR="$REPO/scripts/logs/gembob_addone"
mkdir -p "$LOGDIR"

# run_job <config_stem> <expt_name> <inner_steps> <extra_cli_args>
run_job() {
  local stem="$1" expt="$2" isteps="$3" extra="$4"
  echo "[queue] START $expt  (cfg=$stem is=$isteps extra='$extra')  $(date '+%H:%M:%S')"
  "$PY" "$REPO/main.py" \
    --config "$BASE" \
    --config "$CFG/${stem}.yaml" \
    --n_epochs 1 --inner_steps "$isteps" \
    --expt_name "$expt" \
    --seeds "$SEEDS" $extra > "$LOGDIR/${expt}.log" 2>&1
  echo "[queue] END   $expt  exit=$?  $(date '+%H:%M:%S')"
}

# "stem|expt|inner_steps|extra". Ordered so the baseline finishes first (nothing can be
# interpreted without it) and the two heaviest arms (A2 distillation, A3/A4 bilevel) are
# spread across the pool rather than bunched.
JOBS=(
  "gem_bob|gembob_a0|2|"                                    # A0 baseline
  "gem_bob_distill|gembob_a2_distill|2|"                    # A2
  "gem_bob_dynring|gembob_a1_dynring|2|"                    # A1
  "gem_bob_bilevel|gembob_a3_bilevel_matched|1|"            # A3 budget-matched
  "gem_bob_meta|gembob_a5_meta|2|"                          # A5
  "gem_bob_bilevel|gembob_a4_bilevel_unmatched|2|"          # A4 unmatched diagnostic
)

echo "=== gem_bob add-one study: ${#JOBS[@]} configs x $(echo "$SEEDS" | tr ',' '\n' | wc -l) seeds"
echo "=== seeds=$SEEDS  MAX_PARALLEL=$MAX_PARALLEL  logs -> $LOGDIR"
echo "=== started $(date)"

for job in "${JOBS[@]}"; do
  IFS='|' read -r stem expt isteps extra <<< "$job"
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
    sleep 20
  done
  run_job "$stem" "$expt" "$isteps" "$extra" &
done
wait

echo "=== all jobs finished $(date)"
echo "=== analyse with: $PY $REPO/scripts/compare_gembob.py --phase 1 --verbose"
