#!/bin/bash
# Does SI's per-parameter normalisation do anything once it is numerically live?
#
# scripts/probe_si_epsilon.py showed that at the shipped si_epsilon=0.01 the
# denominator of `omega += W / (delta^2 + epsilon)` is epsilon for all but
# ~1 in 20,000 parameters, so omega is an *unnormalised* path integral. This
# sweep lowers epsilon until delta^2 competes: ~2e-5 engages the top 1% of
# parameters, ~3e-7 engages the median.
#
# The catch is that lowering epsilon also multiplies omega by 1/epsilon, so a
# naive sweep varies anchor strength over five orders of magnitude and measures
# that instead. While epsilon dominates, si_c and 1/epsilon are perfectly
# degenerate, so si_c is compensated to hold si_c/epsilon fixed at the swept
# operating point (1000 / 0.01 = 1e5). That keeps the penalty on the typical
# (still-dominated) parameter constant and leaves only the tail discounting to
# vary -- which is the question.
#
# Compensation cannot be exact for parameters that *do* engage (their omega
# becomes W/delta^2, not W/epsilon), but those are the minority the experiment
# is about. The final arm is deliberately uncompensated as a control: if it
# tracks its compensated twin, the effect is shape; if it tracks nothing, the
# effect was scale all along.
#
# Anchor-only throughout (si_lwf_lambda 0) so LwF cannot mask the difference.
# Matched to scripts/run_si_rwalk_lwf.sh: single-epoch TIL, inner_steps 2,
# seeds 0/39/55, no AMP -- so the eps=1e-2 reference cell is the existing
# si_anchoronly_noamp_se_til pool and does not need rerunning.
set -uo pipefail
REPO="/home/lunet/wsmr11/repos/evidential-cl"
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
cd "$REPO" || exit 1

JOBS=(
  "si_eps1em3:--si_epsilon 0.001 --si_c 100.0"
  "si_eps1em4:--si_epsilon 0.0001 --si_c 10.0"
  "si_eps1em5:--si_epsilon 0.00001 --si_c 1.0"
  "si_eps1em6:--si_epsilon 0.000001 --si_c 0.1"
  "si_eps1em7:--si_epsilon 0.0000001 --si_c 0.01"
  "si_eps1em6_uncomp:--si_epsilon 0.000001 --si_c 1000.0"
)

echo "[si-eps-sweep] ${#JOBS[@]} jobs queued"
for job in "${JOBS[@]}"; do
  name="${job%%:*}"
  extra="${job#*:}"
  echo "[si-eps-sweep] ===== START $name ($extra) ====="
  # shellcheck disable=SC2086
  "$PY" "$REPO/main.py" \
    --config "$REPO/configs/base.yaml" \
    --config "$REPO/configs/models/til/si_lwf.yaml" \
    --si_lwf_lambda 0.0 \
    --n_epochs 1 --inner_steps 2 \
    --no-amp \
    --no-save_checkpoints \
    --expt_name "${name}_noamp_se_til" \
    --seeds "0,39,55" \
    $extra
  echo "[si-eps-sweep] ===== END $name exit=$? ====="
done
echo "[si-eps-sweep] ALL DONE"
