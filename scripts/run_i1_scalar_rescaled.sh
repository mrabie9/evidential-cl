#!/bin/bash
# Re-measure the `i1` tracked importance scalar at an anchor strength sized for
# its own Omega scale.
#
# The I_p sweep ran `woe_importance_scalar=i1` at abs's peak of 2.4e5, which was
# tuned for I_2, and the result (final 0.4438, diagonal 0.4457, BWT -0.0019) has
# the unmistakable signature of a *frozen* network: near-zero forgetting bought
# with a collapsed diagonal. That is over-anchoring, not a verdict on the scalar.
#
# The scale factor is the same one the objective grids needed: both scalars are
# per-feature averages (I_p / J^p), so their ratio is ~ J / w ~ 132 with J = 512
# and the measured w ~ 3.9. The path integral, and hence Omega, is therefore
# ~132x larger under I_1, and the matched anchor is 2.4e5 / 132 ~ 1.8e3. Both
# 1.8e3 and 1.8e4 are run because 132 is an estimate, and one decade of bracket
# is cheap next to drawing a conclusion from a mis-scaled cell.
#
# Read against the i2 control on the identical host: final 0.5008.
set -u

REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/i1_scalar
mkdir -p "$LOGS"

for lam in 1800.0 18000.0; do
  name="i1scalar_lam${lam%.0}_s0"
  echo "[launch] $name"
  WOE_LC_DEBUG=1 PYTHONUNBUFFERED=1 $PY $REPO/main.py \
    --config configs/base.yaml \
    --config configs/models/til/woe_si_lc.yaml \
    --n_epochs 1 --inner_steps 2 \
    --lr 0.003 \
    --woe_omega_transform abs \
    --woe_anchor_mode proximal \
    --woe_importance_scalar i1 \
    --woe_lambda "$lam" \
    --single-seed --seed 0 --no-save_checkpoints \
    --expt_name "$name" > "$LOGS/$name.log" 2>&1 &
done
wait

echo
printf "%-24s %9s %9s %9s\n" run diag final bwt
for lam in 1800 18000; do
  d=$(ls -d $REPO/logs/*/i1scalar_lam${lam}_s0-*/0/results.txt 2>/dev/null | tail -1)
  if [ -n "$d" ]; then
    printf "%-24s %9s %9s %9s\n" "lam=$lam" \
      "$(grep -oP '^Diagonal \S+: \K[-0-9.]+' "$d")" \
      "$(grep -oP '^Final \S+: \K[-0-9.]+' "$d")" \
      "$(grep -oP '^Backward: \K[-0-9.]+' "$d")"
  fi
done
