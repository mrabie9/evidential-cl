#!/bin/bash
# Fill-in re-runs to close out clean paired n=6 on the ablation grid. Three fixes:
#   (1) BCL B1/B3/B4 at --beta 1 (their configs hardcode beta:3.0; the canonical
#       B-family and its baseline bcl_b1 are beta=1). Re-run at 7,13,21 -> pools
#       with the historical bcl_b1_* (beta1) 0,39,55 runs.
#   (2) CTN T0-T3 with --no-amp (the ctn*.yaml no_amp key is silently dropped, so
#       the earlier runs used AMP). Re-run at 7,13,21 -> pools with the amp=False
#       *_se_noamp originals at 0,39,55.
#   (3) Res-ER single-epoch E0: eralg4 at 0,39,55 (is2, lr0.01) to match the
#       single-epoch dynring E1, giving a clean n=6 dynring-vs-ResER comparison.
#
# Expt names REUSE the s7-13-21 stems so compare_ablation.py's globs pick these
# (newest) runs as the reference; config-signature pooling then excludes the old
# mis-configured dirs automatically. Bounded worker pool, MAX_PARALLEL default 2,
# CTN interleaved with light jobs so at most one heavy job runs at a time.
#
# Usage:  bash scripts/run_ablation_fillin.sh   [MAX_PARALLEL=2]
set -uo pipefail

REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1   # main.py writes its logs/ dir relative to cwd; anchor it here
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
BASE="$REPO/configs/base.yaml"
CFG="$REPO/configs/models/til"
IS=2
MAX_PARALLEL="${MAX_PARALLEL:-2}"
LOGDIR="$REPO/scripts/logs/ablation_fillin"
mkdir -p "$LOGDIR"

# run_job <stem> <expt> <inner_steps> <seeds> <extra>
run_job() {
  local stem="$1" expt="$2" isteps="$3" seeds="$4" extra="$5"
  echo "[fillin] START $expt (cfg=$stem is=$isteps seeds=$seeds extra='$extra') $(date '+%H:%M:%S')"
  "$PY" "$REPO/main.py" \
    --config "$BASE" \
    --config "$CFG/${stem}.yaml" \
    --data_path "$REPO/data/rff/radar/full" \
    --n_epochs 1 --inner_steps "$isteps" \
    --expt_name "$expt" \
    --seeds "$seeds" $extra > "$LOGDIR/${expt}.log" 2>&1
  echo "[fillin] END   $expt exit=$? $(date '+%H:%M:%S')"
}

# "stem|expt|is|seeds|extra" — interleave CTN (heavy) with light jobs.
JOBS=(
  "ctn|ctn_full_s7-13-21|$IS|7,13,21|--no-amp"                 # T0
  "bcl_nodistill|bcl_nodistill_s7-13-21|$IS|7,13,21|--beta 1"  # B1 @ beta1
  "ctn_nofilm|ctn_nofilm_s7-13-21|$IS|7,13,21|--no-amp"        # T1
  "bcl_nodualmem|bcl_nodualmem_s7-13-21|$IS|7,13,21|--beta 1"  # B3 @ beta1
  "ctn_nodistill|ctn_nodistill_s7-13-21|$IS|7,13,21|--no-amp"  # T2
  "bcl_noreplay|bcl_noreplay_s7-13-21|$IS|7,13,21|--beta 1"    # B4 @ beta1
  "ctn_noreplay|ctn_noreplay_s7-13-21|$IS|7,13,21|--no-amp"    # T3
  "eralg4|eralg4_resER_s0-39-55|$IS|0,39,55|--lr 0.01 --memory_loss_lambda 1"  # E0 single-epoch, other 3 seeds
)

echo "[fillin] ===== START seeds-fillin  MAX_PARALLEL=$MAX_PARALLEL  jobs=${#JOBS[@]}  $(date) ====="
running=0
for spec in "${JOBS[@]}"; do
  IFS='|' read -r j_stem j_expt j_is j_seeds j_extra <<< "$spec"
  run_job "$j_stem" "$j_expt" "$j_is" "$j_seeds" "$j_extra" &
  running=$((running + 1))
  if (( running >= MAX_PARALLEL )); then wait -n; running=$((running - 1)); fi
done
wait
echo "[fillin] ===== ALL DONE $(date) ====="
echo "[fillin] logs: $LOGDIR/*.log"
