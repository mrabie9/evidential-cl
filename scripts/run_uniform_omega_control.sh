#!/bin/bash
# Does the path integral do any work at all?
#
# The xi sweep showed SI's *denominator* is not merely inert here but actively
# harmful (best final: xi=1e-3 0.5008, 1e-6 0.4833, 1e-8 0.4695). In the floored
# regime Omega = |sum h.Delta| / xi, so the anchor is driven entirely by the
# *numerator*. Whether that numerator earns its place is a separate question and
# has never had a control.
#
# `--woe_omega_transform uniform` sets Omega constant across parameters: the
# anchor becomes a plain proximal pull toward theta* (L2-SP with a
# stability-preserving update), with no per-parameter importance at all. If it
# matches 0.5008, the path integral contributes nothing beyond uniform shrinkage.
#
# Sizing lambda. Uniform Omega = t after t tasks, so b = 2*lr*lambda*t = 0.054*
# lambda at t=9. The abs arm's median b at its peak is 1.67e-2, i.e. lambda ~0.31
# matched on the median; the grid spans that up to b ~= 54 (effectively frozen).
#
# Concurrency is capped at 3 by scripts/lib_joblimit.sh -- see the note there on
# why a free-memory gate alone is not sufficient.
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/uniform_omega
mkdir -p "$LOGS"
# shellcheck source=/dev/null
. "$REPO/scripts/lib_joblimit.sh"

for lam in "$@"; do
  name="unif_lam${lam}"
  job_slot
  echo "[launch] $name"
  WOE_LC_DEBUG=1 PYTHONUNBUFFERED=1 $PY $REPO/main.py \
    --config configs/base.yaml \
    --config configs/models/til/woe_si_lc.yaml \
    --n_epochs 1 --inner_steps 2 --lr 0.003 \
    --woe_omega_transform uniform --woe_anchor_mode proximal \
    --woe_omega_accum sum --woe_lambda "$lam" \
    --single-seed --seed 0 --no-save_checkpoints \
    --expt_name "$name" > "$LOGS/$name.log" 2>&1 &
  PIDS+=($!)
done
wait
echo "[uniform omega] done"
