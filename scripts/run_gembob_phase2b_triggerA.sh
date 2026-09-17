#!/bin/bash
# gem_bob Phase 2b, Trigger A: fills the slot freed when C2 finishes (~3.5h before C1/C3).
#
# C2 runs inner_steps=1 (bilevel budget-match) so it finishes ~3.5h ahead of C1/C3. Rather
# than leave that GPU slot idle, launch the highest-priority pair here. C3 (A1+A2+meta) looked
# promising, so the meta-batch pairs get priority; this slot runs A2+A5 (meta on the strongest
# single, distillation), leaving A1+A5 and the bilevel pairs for Trigger B.
#
# Gates on the phase-2 driver log's "END gembob_c2" marker, then launches one n=9 job.
set -uo pipefail
REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
DRIVER="$REPO/scripts/logs/gembob_phase2_driver.log"
LOGDIR="$REPO/scripts/logs/gembob_phase2b"
mkdir -p "$LOGDIR"
SEEDS="${SEEDS:-0,39,55,7,13,21,1,2,3}"

echo "[triggerA] waiting for C2 to finish ($(date))"
until grep -q "END   gembob_c2" "$DRIVER" 2>/dev/null; do sleep 60; done
echo "[triggerA] C2 done; launching A2+A5 (meta on distill) $(date)"

"$PY" "$REPO/main.py" \
  --config "$REPO/configs/base.yaml" \
  --config "$REPO/configs/models/til/gem_bob_a2a5.yaml" \
  --n_epochs 1 --inner_steps 2 \
  --expt_name gembob_a2a5 \
  --seeds "$SEEDS" > "$LOGDIR/gembob_a2a5.log" 2>&1
echo "[triggerA] A2+A5 finished exit=$? $(date)"
