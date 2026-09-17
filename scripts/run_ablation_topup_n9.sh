#!/bin/bash
# n=6 -> n=9 top-up for the lr=0.03 inconclusive ablation families. Adds seeds
# 1,2,3 to each row so the PAIRED comparison (ablation vs its baseline) extends to
# 9 matched pairs -- hence the baselines (G0,T0,B0) are topped up too, not just the
# inconclusive ablations (G1; T1,T2; B3). Same 3 seeds across a family so pairing holds.
# Configs exactly match the existing runs in logs/ablations/til/<algo>/<ID>/:
#   G0 gem lr0.03 | G1 er_ring --lr 0.03 | T0/T1/T2 ctn* --no-amp lr0.03 (ms10)
#   B0 bcl_dual (beta1) | B3 bcl_nodualmem --beta 1
# Results land in logs/<stem>/<expt>-<ts>/{1,2,3}/ ; move into the ablations tree after.
#
# Usage:  bash scripts/run_ablation_topup_n9.sh   [MAX_PARALLEL=4]
set -uo pipefail
REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
BASE="$REPO/configs/base.yaml"
CFG="$REPO/configs/models/til"
SEEDS="1,2,3"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
LOGDIR="$REPO/scripts/logs/topup_n9"
mkdir -p "$LOGDIR"

run_job() {
  local stem="$1" expt="$2" extra="$3"
  echo "[topup] START $expt (cfg=$stem extra='$extra') $(date '+%H:%M:%S')"
  "$PY" "$REPO/main.py" \
    --config "$BASE" --config "$CFG/${stem}.yaml" \
    --n_epochs 1 --inner_steps 2 \
    --expt_name "$expt" --seeds "$SEEDS" $extra > "$LOGDIR/${expt}.log" 2>&1
  echo "[topup] END   $expt exit=$? $(date '+%H:%M:%S')"
}

# "stem|expt|extra" — heavy CTN interleaved with light jobs.
JOBS=(
  "ctn|ctn_full_topup_lr03|--no-amp"                # T0
  "gem|gem_full_topup_lr03|"                        # G0 (lr0.03 native)
  "ctn_nofilm|ctn_nofilm_topup_lr03|--no-amp"       # T1
  "er_ring|ering_rehearsal_topup_lr03|--lr 0.03"    # G1
  "ctn_nodistill|ctn_nodistill_topup_lr03|--no-amp" # T2
  "bcl_dual|bcl_b1_topup|--beta 1"                  # B0
  "bcl_nodualmem|bcl_nodualmem_topup|--beta 1"      # B3
)

echo "[topup] ===== n=9 top-up  seeds=$SEEDS  MAX_PARALLEL=$MAX_PARALLEL  jobs=${#JOBS[@]}  START $(date) ====="
running=0
for spec in "${JOBS[@]}"; do
  IFS='|' read -r j_stem j_expt j_extra <<< "$spec"
  run_job "$j_stem" "$j_expt" "$j_extra" &
  running=$((running + 1))
  if (( running >= MAX_PARALLEL )); then wait -n; running=$((running - 1)); fi
done
wait
echo "[topup] ===== ALL DONE $(date) ====="
echo "[topup] logs: $LOGDIR/*.log ; results in logs/<stem>/*_topup*-<ts>/{1,2,3}/results.txt"
