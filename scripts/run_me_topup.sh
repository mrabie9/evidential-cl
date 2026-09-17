#!/bin/bash
# n=9 top-up (seeds 1,2,3) for the multi-epoch ablation grids, run AFTER the n=6 campaign
# of scripts/run_me_ablations.sh has drained. The two modes run SEQUENTIALLY so the GPU
# never carries more than MAX_PARALLEL jobs in total.
#
# Row selection has two parts.
#
# (1) BY PROTOCOL -- sec:stat_protocol extends a row to n=9 only when its verdict is
#     "underpowered" (interval spans zero AND is wider than the delta=1 SESOI). These come
#     straight from `analyse_me_ablations.py --topup`:
#         til: E1, M1, M3        cil: M4, B3, B5b
#
# (2) BY JUDGEMENT -- four rows verdicted "ms" (marginal) are added deliberately. Each has
#     an interval that EXCLUDES zero but a permutation p pinned at 0.062, one step above
#     the n=6 floor of 0.031, so a single extra seed can decide them; and each REVERSES the
#     sign of its published single-epoch row, which makes it exactly the kind of claim this
#     rerun exists to settle. Leaving them at "marginal" would be the one place the study
#     fails to answer its own question.
#         til: E2 (+1.8, published -0.3), B3 (+0.6, published -0.4)
#         cil: M3 (+1.9, published -0.5), B5a (-0.8, p_Holm 0.055 -- misses ss by 0.005)
#
# Family baselines (E0/M0/B0) are pulled in wherever any of their ablations is topped up,
# since every delta is paired against them and the pairing uses common seeds only.
#
# Rows NOT topped up are final at n=6: everything already ss or ns, per the protocol's
# "ablations that remain underpowered at n=9 are left as-is" and its converse.
#
# Usage:  bash scripts/run_me_topup.sh          [MAX_PARALLEL=3]
set -uo pipefail

REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1
MAX_PARALLEL="${MAX_PARALLEL:-3}"

TIL_ROWS="e0,e1,e2,m0,m1,m3,b0,b3"
CIL_ROWS="m0,m3,m4,b0,b3,b5a,b5b"

echo "[topup] ===== START $(date) MAX_PARALLEL=$MAX_PARALLEL ====="
echo "[topup] til rows: $TIL_ROWS"
echo "[topup] cil rows: $CIL_ROWS"

SEED_GROUP_SPEC="1,2,3" MODES=til ROWS_ONLY="$TIL_ROWS" \
  MAX_PARALLEL="$MAX_PARALLEL" LOGDIR="$REPO/scripts/logs/me_topup" \
  bash "$REPO/scripts/run_me_ablations.sh"

echo "[topup] ----- til done, starting cil $(date) -----"

SEED_GROUP_SPEC="1,2,3" MODES=cil ROWS_ONLY="$CIL_ROWS" \
  MAX_PARALLEL="$MAX_PARALLEL" LOGDIR="$REPO/scripts/logs/me_topup" \
  bash "$REPO/scripts/run_me_ablations.sh"

echo "[topup] ===== ALL DONE $(date) ====="
echo "[topup] analyse: la-maml_env/bin/python scripts/analyse_me_ablations.py"
