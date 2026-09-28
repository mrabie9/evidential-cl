#!/bin/bash
# Re-run the BCL-Dual family (B0, B1, B2, B5a, B5b) of the CIL or TIL ablation campaign after
# two fixes that change every BCL-Dual number:
#   * bcl_dual.observe now clears the inner-step gradient before the outer step (it used
#     to be re-applied, so the "validation" step was mostly a second current-batch step);
#   * AdcIqAdapter mixes the three ADCs with softmax(w) instead of w / sum(w), which was
#     singular at sum(w) = 0 and collapsed b0 seed 9 / b2 seed 8 at 1 epoch.
#
# Same protocol as run_cil_ablation_campaign.sh, restricted to the BCL-Dual family:
#   phase 1: seeds 7,13,21;1,2,3    every B row (n=6)
#   phase 2: seeds 4,5,6;8,9,10     B0 + the B rows still INC (n=12), once
# CIL: the Res-ER and C-MAML families are not re-run; their stem directories are
# symlinked from the pre-fix tree ($SRC_ROOT) so the analyser and the LaTeX table see a
# complete grid. They ran with the old adapter normalisation; none of their 216
# checkpoints had a degenerate mix. TIL: only the BCL-Dual family is reported (the other
# TIL families' pools are not on this machine), and no LaTeX table is written.
#
# Usage:
#   bash scripts/run_bcl_gradfix_rerun.sh                 # CIL
#   MODE=til bash scripts/run_bcl_gradfix_rerun.sh
#   REGIMES=1 MAX_PARALLEL=4 bash scripts/run_bcl_gradfix_rerun.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$REPO/la-maml_env/bin/python}"
DRIVER="$REPO/scripts/run_ablations_noise_removed.sh"
ANALYSER="$REPO/scripts/analyse_ablations_noise_removed.py"
TEXER="$REPO/scripts/tex_ablation_table.py"
REGIMES="${REGIMES:-1 5}"
MODE="${MODE:-cil}"
SRC_ROOT="${SRC_ROOT:-logs/ablations_cil_rerun}"
LOG_ROOT="${LOG_ROOT:-logs/ablations_${MODE}_bclfix}"
STATE_ROOT="$REPO/$LOG_ROOT"
B_ROWS="b0,b1,b2,b5a,b5b"

mkdir -p "$STATE_ROOT/$MODE"
if [ "$MODE" = "cil" ]; then
  for stem in eralg4 er_ring eralg4_distill eralg4_distill_ms05 cmaml cmaml_alpha0 cmaml_single_inner; do
    if [ ! -e "$STATE_ROOT/cil/$stem" ]; then
      ln -s "$REPO/$SRC_ROOT/cil/$stem" "$STATE_ROOT/cil/$stem"
    fi
  done
fi

export MODES="$MODE" PY MAX_PARALLEL="${MAX_PARALLEL:-3}"
export RUN_LOGROOT="$LOG_ROOT" ABLATION_LOG_ROOT="$LOG_ROOT"
export DRIVER_LOGDIR="$STATE_ROOT/driver"

# bcl_topup <tag>: B0 plus the BCL-Dual rows whose verdict is INC, comma-separated.
bcl_topup() {
  local line rows
  line=$("$PY" "$ANALYSER" --tag "$1" --modes "$MODE" --topup)
  rows="${line##*ROWS_ONLY=}"
  "$PY" -c 'import sys; print(",".join(r for r in sys.argv[1].split(",") if r.startswith("b")))' "$rows"
}

campaign() {
  local ne="$1" tag="$2"
  local state="$STATE_ROOT/$tag"
  mkdir -p "$state"
  export N_EPOCHS="$ne" TAG="$tag"
  if [ ! -f "$state/phase1.done" ]; then
    ROWS_ONLY="$B_ROWS" SEED_GROUP_SPEC="7,13,21;1,2,3" bash "$DRIVER" || return 1
    touch "$state/phase1.done"
  fi
  if [ ! -f "$state/phase2.done" ]; then
    local rows
    rows=$(bcl_topup "$tag")
    echo "[$tag] BCL-Dual n=6 top-up list: '${rows}'"
    echo "$rows" > "$state/topup_rows.txt"
    if [ -n "$rows" ]; then
      ROWS_ONLY="$rows" SEED_GROUP_SPEC="4,5,6;8,9,10" bash "$DRIVER" || return 1
    fi
    touch "$state/phase2.done"
  fi
  "$PY" "$ANALYSER" --tag "$tag" --modes "$MODE" --stats | tee "$state/final_table.txt"
}

rc=0
for ne in $REGIMES; do
  if [ "$ne" = "1" ]; then
    campaign 1 nrm1e || rc=1
  else
    campaign 5 nrm || rc=1
  fi
done

if [ "$MODE" = "cil" ]; then
  "$PY" "$TEXER" --mode cil --tag-1e nrm1e --tag-5e nrm \
    --out "$STATE_ROOT/cil_ablations_1e_5e.tex"
fi
exit $rc
