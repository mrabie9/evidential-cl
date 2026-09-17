#!/bin/bash
# gem_bob add-one study, lr=0.03 replica.
#
# Identical design to scripts/run_gembob_addone.sh (see that file's header for the mechanism
# list and budget-matching rationale) but at GEM's NATIVE lr=0.03 instead of 0.01. This is the
# operating point of the ablation grid's G0 (63.5 +- 2.1) and of GEM's best absolute F1, so it
# answers "do the add-one effects survive at the lr where GEM is actually run", not just at the
# low-variance 0.01 point chosen to sharpen the paired tests.
#
# CAVEAT -- POWER. GEM's seed variance at lr=0.03 is ~4x that at 0.01 (the G0 row: +-2.1 vs
# +-0.5). The mechanism effects here are ~1 F1 point, so even n=9 may be underpowered: expect
# wider CIs and more "inconclusive" verdicts than the 0.01 study. Read a non-significant row as
# underpowered, not as evidence of no effect (the write-up's stated protocol).
#
# --lr 0.03 is passed on the CLI, overriding the lr: 0.01 baked into each shared config
# (CLI wins over YAML: main.py applies YAML into the namespace, then argparse overrides it with
# any explicit CLI flag -- the same mechanism the budget-match scripts rely on). expt names get
# a _lr03 suffix so they pool separately from the 0.01 study; analyse with
#   la-maml_env/bin/python scripts/compare_gembob.py --phase 1 --tag _lr03 --verbose
#
# 6 configs x 9 seeds = 54 runs. Higher-variance rate does not change runtime materially.
#
# Usage:  bash scripts/run_gembob_addone_lr03.sh
#         MAX_PARALLEL=4 bash scripts/run_gembob_addone_lr03.sh
set -uo pipefail

REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1   # main.py writes its logs/ dir relative to cwd; anchor it here
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
BASE="$REPO/configs/base.yaml"
CFG="$REPO/configs/models/til"

LR="0.03"
SEEDS="${SEEDS:-0,39,55,7,13,21,1,2,3}"   # the n=9 set used by the topped-up grid rows
MAX_PARALLEL="${MAX_PARALLEL:-3}"

LOGDIR="$REPO/scripts/logs/gembob_addone_lr03"
mkdir -p "$LOGDIR"

# run_job <config_stem> <expt_name> <inner_steps> <extra_cli_args>
run_job() {
  local stem="$1" expt="$2" isteps="$3" extra="$4"
  echo "[queue] START $expt  (cfg=$stem lr=$LR is=$isteps extra='$extra')  $(date '+%H:%M:%S')"
  "$PY" "$REPO/main.py" \
    --config "$BASE" \
    --config "$CFG/${stem}.yaml" \
    --lr "$LR" --n_epochs 1 --inner_steps "$isteps" \
    --expt_name "$expt" \
    --seeds "$SEEDS" $extra > "$LOGDIR/${expt}.log" 2>&1
  echo "[queue] END   $expt  exit=$?  $(date '+%H:%M:%S')"
}

# "stem|expt|inner_steps|extra". Same ordering as the 0.01 study.
JOBS=(
  "gem_bob|gembob_a0_lr03|2|"                                    # A0 baseline
  "gem_bob_distill|gembob_a2_distill_lr03|2|"                    # A2
  "gem_bob_dynring|gembob_a1_dynring_lr03|2|"                    # A1
  "gem_bob_bilevel|gembob_a3_bilevel_matched_lr03|1|"           # A3 budget-matched
  "gem_bob_meta|gembob_a5_meta_lr03|2|"                          # A5
  "gem_bob_bilevel|gembob_a4_bilevel_unmatched_lr03|2|"         # A4 unmatched diagnostic
)

echo "=== gem_bob add-one study (lr=$LR): ${#JOBS[@]} configs x $(echo "$SEEDS" | tr ',' '\n' | wc -l) seeds"
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
echo "=== analyse: $PY $REPO/scripts/compare_gembob.py --phase 1 --tag _lr03 --verbose"
