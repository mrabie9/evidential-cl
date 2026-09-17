#!/bin/bash
# Re-run the GEM (G) and CTN (T) leave-one-out ablation rows at lr=0.01 instead of
# the tuned lr=0.03. Motivation: at lr=0.03 the G/T paired comparisons are noisy
# (e.g. G1 projection-vs-replay came back inconclusive, CI [-7.2,+1.1], driven by a
# single seed-13 reversal; CTN T0-T3 carry large BWT stds). A gentler step should
# tighten the per-seed differences. ONLY the learning rate changes -- every other
# knob stays at the tuned operating point (GEM memory_strength=1; CTN
# memory_strength=10 + --no-amp, since ctn.yaml's no_amp key is silently dropped).
#
# n=6 seeds per row (0,39,55,7,13,21) in ONE run each, so every row gets a clean
# cross-seed summary dir:  logs/<config_stem>/<expt_name>-<ts>/{seed}/results.txt
# Bounded worker pool, G/T interleaved so at most ~2 heavy CTN jobs run at once.
#
# Usage:  bash scripts/run_gt_ablations_lr01.sh   [MAX_PARALLEL=3]
set -uo pipefail

REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1   # main.py writes logs/ relative to cwd; anchor it here
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
BASE="$REPO/configs/base.yaml"
CFG="$REPO/configs/models/til"

SEEDS="0,39,55,7,13,21"
IS=2
LR="0.01"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
LOGDIR="$REPO/scripts/logs/gt_lr01"
mkdir -p "$LOGDIR"

# run_job <config_stem> <expt_name> <extra_cli_args>
run_job() {
  local stem="$1" expt="$2" extra="$3"
  echo "[gt-lr01] START $expt (cfg=$stem extra='$extra') $(date '+%H:%M:%S')"
  "$PY" "$REPO/main.py" \
    --config "$BASE" \
    --config "$CFG/${stem}.yaml" \
    --n_epochs 1 --inner_steps "$IS" --lr "$LR" \
    --expt_name "$expt" \
    --seeds "$SEEDS" $extra > "$LOGDIR/${expt}.log" 2>&1
  echo "[gt-lr01] END   $expt exit=$? $(date '+%H:%M:%S')"
}

# "stem|expt|extra" — G (light) and T (heavy, --no-amp) interleaved.
JOBS=(
  "gem|gem_full_lr01abl|"                          # G0
  "ctn|ctn_full_lr01abl|--no-amp"                  # T0
  "er_ring|ering_rehearsal_lr01abl|"               # G1 (plain Ring-ER rehearsal on the buffer)
  "ctn_nofilm|ctn_nofilm_lr01abl|--no-amp"         # T1
  "gem|gem_noreplay_lr01abl|--n_memories 0"        # G2 (empty buffer -> plain SGD)
  "ctn_nodistill|ctn_nodistill_lr01abl|--no-amp"   # T2
  "ctn_noreplay|ctn_noreplay_lr01abl|--no-amp"     # T3 (config also sets n_memories 0)
)

echo "[gt-lr01] ===== G/T ablations @ lr=$LR  seeds=$SEEDS  MAX_PARALLEL=$MAX_PARALLEL  jobs=${#JOBS[@]}  START $(date) ====="
running=0
for spec in "${JOBS[@]}"; do
  IFS='|' read -r j_stem j_expt j_extra <<< "$spec"
  run_job "$j_stem" "$j_expt" "$j_extra" &
  running=$((running + 1))
  if (( running >= MAX_PARALLEL )); then wait -n; running=$((running - 1)); fi
done
wait
echo "[gt-lr01] ===== ALL DONE $(date) ====="
echo "[gt-lr01] per-job logs: $LOGDIR/*.log"
echo "[gt-lr01] results:      logs/<config_stem>/<expt>_lr01abl-<ts>/{seed}/results.txt"
