#!/bin/bash
# Follow-up to A7. Two questions the first sweep left open.
#
# 1. Is the max curve really interior-peaked? The recorded peak (3.6e5, final
#    0.4857) beats the next cell down (2.4e5, 0.4848) by 0.0009, which is a
#    fifth of this host's seed spread -- i.e. the curve is *flat*, not peaked,
#    and the turnover was asserted on noise. Structurally it must turn over
#    somewhere, because at lambda -> 0 both rules collapse to the same
#    unanchored network; the open question is whether the turn happens above or
#    below the sum control's 0.5008. Four cells below the flat region settle it.
#
# 2. What does Omega actually look like? Only totals and nonzero_frac were ever
#    measured. WOE_OMEGA_DUMP writes the per-task path integral before it is
#    combined, so a single `sum` run yields the counterfactual max-Omega on an
#    identical trajectory -- the two rules can then be compared per parameter
#    without the confound that the arms diverge in training after task 0.
set -u

REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/cautious_lowlam
DUMPS=$REPO/scripts/logs/omega_dumps
mkdir -p "$LOGS" "$DUMPS"

launch () {
  local name=$1 accum=$2 lam=$3 dump=$4
  echo "[launch] $name accum=$accum lambda=$lam dump=${dump:-none}"
  WOE_LC_DEBUG=1 WOE_OMEGA_DUMP="$dump" PYTHONUNBUFFERED=1 $PY $REPO/main.py \
    --config configs/base.yaml \
    --config configs/models/til/woe_si_lc.yaml \
    --n_epochs 1 --inner_steps 2 \
    --lr 0.003 \
    --woe_omega_transform abs \
    --woe_anchor_mode proximal \
    --woe_omega_accum "$accum" \
    --woe_lambda "$lam" \
    --single-seed --seed 0 --no-save_checkpoints \
    --expt_name "$name" > "$LOGS/$name.log" 2>&1
}

launch caut_max_lam120000_s0 max 120000.0 "" &
launch caut_max_lam60000_s0  max  60000.0 "" &
launch caut_dump_sum_s0      sum 240000.0 "$DUMPS/sum_s0" &
wait

launch caut_max_lam30000_s0 max 30000.0 "" &
launch caut_max_lam15000_s0 max 15000.0 "" &
launch caut_dump_max_s0     max 240000.0 "$DUMPS/max_s0" &
wait

echo
printf "%-26s %9s %9s %9s\n" run diag final bwt
for name in caut_max_lam120000_s0 caut_max_lam60000_s0 caut_max_lam30000_s0 caut_max_lam15000_s0 caut_dump_sum_s0 caut_dump_max_s0; do
  d=$(ls -d $REPO/logs/*/${name}-*/0/results.txt 2>/dev/null | tail -1)
  if [ -n "$d" ]; then
    printf "%-26s %9s %9s %9s\n" "$name" \
      "$(grep -oP '^Diagonal \S+: \K[-0-9.]+' "$d")" \
      "$(grep -oP '^Final \S+: \K[-0-9.]+' "$d")" \
      "$(grep -oP '^Backward: \K[-0-9.]+' "$d")"
  fi
done
