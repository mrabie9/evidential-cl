#!/bin/bash
# A7's commensurability control -- the gate the refutation has to pass.
#
# Denoeux's cautious rule combines weights of evidence *on a common scale*. The
# raw per-task path integrals are not on one: task 0 runs from random
# initialisation and takes the whole drop in the tracked scalar, and every later
# task both starts from a trained representation and travels less under a
# progressively stronger anchor -- the accumulation rule feeds back into the
# quantities it will later combine. `max` keeps only the largest term and is
# therefore maximally exposed to that; `sum` partially launders it. So A7 as it
# stands cannot answer "your max lost because your weights weren't commensurable",
# which is a fair objection and not a claim about cautious combination at all.
#
# `*_norm` scales each task's omega to unit total mass before combining, then
# rescales the result to the raw-sum arm's total so lambda transfers. The 2x2
# {sum,max} x {raw,norm} is needed because normalised-max against raw-sum would
# confound the rule with the normalisation.
#
# Outcomes. If normalised-max still loses at matched BWT, the refutation holds
# with its own precondition granted. If it ties or wins, A7 was measuring the
# instrumentation. Either is worth more than the current result.
#
# The `sum` arm is re-run as a regression check: consolidation was refactored
# into two passes to compute the global normaliser, and it must still return
# 0.5206 / 0.5008 / -0.0198 bit-identically.
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/cautious_norm
DUMPS=$REPO/scripts/logs/omega_dumps
mkdir -p "$LOGS" "$DUMPS"

launch () {
  local name=$1 accum=$2 lam=$3 dump=${4:-}
  echo "[launch] $name accum=$accum lambda=$lam"
  WOE_LC_DEBUG=1 WOE_OMEGA_DUMP="$dump" PYTHONUNBUFFERED=1 $PY $REPO/main.py \
    --config configs/base.yaml \
    --config configs/models/til/woe_si_lc.yaml \
    --n_epochs 1 --inner_steps 2 --lr 0.003 \
    --woe_omega_transform abs --woe_anchor_mode proximal \
    --woe_omega_accum "$accum" --woe_lambda "$lam" \
    --single-seed --seed 0 --no-save_checkpoints \
    --expt_name "$name" > "$LOGS/$name.log" 2>&1
}

launch cn_sum_regression   sum       240000.0 &
launch cn_maxnorm_lam60000 max_norm   60000.0 &
launch cn_maxnorm_lam120000 max_norm 120000.0 &
launch cn_maxnorm_lam240000 max_norm 240000.0 &
launch cn_maxnorm_lam480000 max_norm 480000.0 &
launch cn_sumnorm_lam240000 sum_norm 240000.0 &
wait

launch cn_maxnorm_lam960000 max_norm 960000.0 &
launch cn_sumnorm_lam120000 sum_norm 120000.0 &
launch cn_sumnorm_lam480000 sum_norm 480000.0 &
# Anchor OFF, with a dump: does task 0 still dominate the path integral when no
# penalty is shrinking later tasks' travel? This separates the two candidate
# causes of incommensurability -- initialisation (reason 1) from anchor feedback
# (reason 2). If task 0 still takes ~37% of the mass at lambda=0, reason 1 is
# doing the work and normalisation is the only available fix.
launch cn_lam0_dump sum 0.0 "$DUMPS/lam0_s0" &
wait
echo "[normalised control] done"
