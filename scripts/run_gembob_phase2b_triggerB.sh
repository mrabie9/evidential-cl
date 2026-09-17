#!/bin/bash
# gem_bob Phase 2b, Trigger B: launches the remaining pairs when C1 AND C3 finish (~7h),
# freeing 2 GPU slots. A2+A5 (Trigger A) is still running at this point, so this pool caps
# concurrency at 2 to hold the total at ~3 processes.
#
# Meta-first ordering (C3 made meta look promising): A1+A5 before the bilevel pairs A2+A3, A1+A3.
# Bilevel pairs run inner_steps=1 (2 SGD steps, budget-matched); the meta pair runs inner_steps=2.
#
# Gates on the phase-2 driver log's "END gembob_c1" AND "END gembob_c3" markers.
set -uo pipefail
REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
DRIVER="$REPO/scripts/logs/gembob_phase2_driver.log"
LOGDIR="$REPO/scripts/logs/gembob_phase2b"
mkdir -p "$LOGDIR"
SEEDS="${SEEDS:-0,39,55,7,13,21,1,2,3}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"   # 2 here + A2+A5 from Trigger A = ~3 total on the GPU

run_job() {
  local stem="$1" expt="$2" isteps="$3"
  echo "[triggerB] START $expt (cfg=$stem is=$isteps) $(date '+%H:%M:%S')"
  "$PY" "$REPO/main.py" \
    --config "$REPO/configs/base.yaml" \
    --config "$REPO/configs/models/til/${stem}.yaml" \
    --n_epochs 1 --inner_steps "$isteps" \
    --expt_name "$expt" \
    --seeds "$SEEDS" > "$LOGDIR/${expt}.log" 2>&1
  echo "[triggerB] END   $expt exit=$? $(date '+%H:%M:%S')"
}

echo "[triggerB] waiting for C1 AND C3 to finish ($(date))"
until grep -q "END   gembob_c1" "$DRIVER" 2>/dev/null && grep -q "END   gembob_c3" "$DRIVER" 2>/dev/null; do
  sleep 60
done
echo "[triggerB] C1 & C3 done; launching remaining pairs $(date)"

# "stem|expt|inner_steps", meta-first.
JOBS=(
  "gem_bob_a1a5|gembob_a1a5|2"   # A1+A5 ring + meta
  "gem_bob_a2a3|gembob_a2a3|1"   # A2+A3 distill + bilevel
  "gem_bob_a1a3|gembob_a1a3|1"   # A1+A3 ring + bilevel
)
for job in "${JOBS[@]}"; do
  IFS='|' read -r stem expt isteps <<< "$job"
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do sleep 20; done
  run_job "$stem" "$expt" "$isteps" &
done
wait
echo "[triggerB] all remaining pairs finished $(date)"
echo "[triggerB] analyse: $PY $REPO/scripts/compare_gembob.py --phase 2 --verbose"
