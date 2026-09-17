#!/bin/bash
# RWalk counterpart to scripts/run_si_epsilon_sweep.sh -- but deliberately NOT
# the same experiment.
#
# scripts/probe_rwalk_riemannian.py (full data) puts RWalk's median
# 0.5*F*delta^2 at 1e-14 of eps, versus SI's delta^2 at 1e-5 of it. SI's
# normalisation was inert but reachable: eps ~3e-7 made it live and the sweep
# found it actively harmful. RWalk's is *unreachable* -- engaging a typical
# parameter needs eps ~1e-16, at which s accumulates terms of order 1e16 and
# the penalty overflows any usable scale. The reason is structural: SI's
# numerator is whole-task displacement, RWalk's is a single optimiser step
# squared times a squared-gradient EMA, a product of two tiny quantities.
#
# So a full eps sweep would be hours spent confirming a flat line the
# arithmetic already predicts. Instead:
#
#   1. Two compensated eps arms (lamb/eps held at 1e5, matching the SI sweep's
#      compensation) as a cheap falsification check. These MUST come out flat.
#      If either moves, the arithmetic above is wrong and the full sweep is
#      back on.
#   2. The Fisher ablation, which is the only live question for RWalk. F
#      supplies 0.1-0.9% of the penalty and its s<0 clamp never fires, so it
#      looks vestigial -- but magnitude share is not effect. `alpha` is the
#      Fisher EMA rate and fisher_running starts at zeros, so --alpha 0.0 pins
#      F == 0 for the whole run (rwalk.py:277,317,339) with no code change.
#
# Reference is the existing rwalk_anchoronly_noamp_se_til pool (eps=0.01,
# lamb=1000, alpha=0.9), same config and seeds, so it is not rerun.
set -uo pipefail
REPO="/home/lunet/wsmr11/repos/evidential-cl"
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
cd "$REPO" || exit 1

JOBS=(
  "rwalk_eps1em5:--eps 0.00001 --lamb 1.0"
  "rwalk_eps1em9:--eps 0.000000001 --lamb 0.0001"
  "rwalk_alpha0:--alpha 0.0"
)

echo "[rwalk-fisher] ${#JOBS[@]} jobs queued"
for job in "${JOBS[@]}"; do
  name="${job%%:*}"
  extra="${job#*:}"
  echo "[rwalk-fisher] ===== START $name ($extra) ====="
  # shellcheck disable=SC2086
  "$PY" "$REPO/main.py" \
    --config "$REPO/configs/base.yaml" \
    --config "$REPO/configs/models/til/rwalk_lwf.yaml" \
    --rwalk_lwf_lambda 0.0 \
    --n_epochs 1 --inner_steps 2 \
    --no-amp \
    --no-save_checkpoints \
    --expt_name "${name}_noamp_se_til" \
    --seeds "0,39,55" \
    $extra
  echo "[rwalk-fisher] ===== END $name exit=$? ====="
done
echo "[rwalk-fisher] ALL DONE"
