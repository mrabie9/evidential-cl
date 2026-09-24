#!/bin/bash
# CIL ablation re-run, single-pass (1 epoch) AND offline (5 epochs), paper protocol
# (sec:stat_protocol), ending in one LaTeX table with both regimes side by side.
#
# Why a re-run: the CIL C-MAML snapshot configs pinned use_old_task_memory: false while
# configs/models/cil/cmaml.yaml runs true (the CIL twin of the TIL fix in 80364996), so the
# M family was not anchored on the C-MAML the main experiments run.
#
# Everything this campaign writes lives in its own tree, $LOG_ROOT (default
# logs/ablations_cil_rerun/), so it never mixes with the pre-fix pools in
# logs/ablations_noise_removed/:
#   $LOG_ROOT/cil/<stem>/<ts>_<row>_<tag>_cil/<seed>/   run logs (what the analyser reads)
#   $LOG_ROOT/driver/<n>e_cil/<row>/<seeds>.log          one stdout capture per job
#   $LOG_ROOT/<tag>/                                     phase markers, top-up list, table
#   $LOG_ROOT/cil_ablations_1e_5e.tex                    the final table
# Tags are the grid's usual ones: nrm1e (1 epoch) and nrm (5 epochs).
#
# Per regime (1e first, it is ~5x cheaper):
#   phase 1: seeds 7,13,21;1,2,3    every CIL row of run_ablations_noise_removed.sh (n=6)
#   phase 2: seeds 4,5,6;8,9,10     INC rows + their family baselines (n=12), ONCE.
#                                   Rows still INC at n=12 stay INC.
# A phase writes its done-marker only once it finishes with every row at n>=6, so a rerun
# after an interruption skips completed phases and repeats an unfinished one in full.
#
# Usage:
#   bash scripts/run_cil_ablation_campaign.sh
#   MAX_PARALLEL=4 bash scripts/run_cil_ablation_campaign.sh
#   REGIMES=5 bash scripts/run_cil_ablation_campaign.sh     # one regime only
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$REPO/la-maml_env/bin/python}"
DRIVER="$REPO/scripts/run_ablations_noise_removed.sh"
ANALYSER="$REPO/scripts/analyse_ablations_noise_removed.py"
TEXER="$REPO/scripts/tex_ablation_table.py"
TAG1="${TAG1:-nrm1e}"
TAG5="${TAG5:-nrm}"
REGIMES="${REGIMES:-1 5}"
LOG_ROOT="${LOG_ROOT:-logs/ablations_cil_rerun}"
STATE_ROOT="$REPO/$LOG_ROOT"
mkdir -p "$STATE_ROOT"

export MODES=cil PY MAX_PARALLEL="${MAX_PARALLEL:-3}"
export RUN_LOGROOT="$LOG_ROOT" ABLATION_LOG_ROOT="$LOG_ROOT"
export DRIVER_LOGDIR="$STATE_ROOT/driver"

# short_rows <tag>: CIL table rows with fewer than 6 seeds (pending rows included).
short_rows() {
  "$PY" - "$ANALYSER" "$1" <<'PYEOF'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("a", sys.argv[1])
a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
for rows in a.GRIDS["cil"].values():
    for rid, _l, stem, pid in rows:
        n = len(a.pool("cil", stem, pid, sys.argv[2]))
        if n < 6:
            print("{} n={}".format(rid, n))
PYEOF
}

# campaign <n_epochs> <tag>
campaign() {
  local ne="$1" tag="$2"
  local state="$STATE_ROOT/$tag"
  mkdir -p "$state"
  export N_EPOCHS="$ne" TAG="$tag"

  if [ ! -f "$state/phase1.done" ]; then
    SEED_GROUP_SPEC="7,13,21;1,2,3" bash "$DRIVER"
    local short
    short=$(short_rows "$tag")
    if [ -n "$short" ]; then
      echo "[$tag] phase 1 incomplete, rows below n=6:" >&2
      echo "$short" >&2
      return 1
    fi
    touch "$state/phase1.done"
  fi

  if [ ! -f "$state/phase2.done" ]; then
    local line rows
    line=$("$PY" "$ANALYSER" --tag "$tag" --modes cil --topup)
    rows="${line##*ROWS_ONLY=}"
    echo "[$tag] n=6 top-up list: '${rows}'"
    echo "$rows" > "$state/topup_rows.txt"
    if [ -n "$rows" ]; then
      ROWS_ONLY="$rows" SEED_GROUP_SPEC="4,5,6;8,9,10" bash "$DRIVER"
    fi
    touch "$state/phase2.done"
  fi

  "$PY" "$ANALYSER" --tag "$tag" --modes cil --stats | tee "$state/final_table.txt"
}

rc=0
for ne in $REGIMES; do
  if [ "$ne" = "1" ]; then
    campaign 1 "$TAG1" || rc=1
  else
    campaign 5 "$TAG5" || rc=1
  fi
done

"$PY" "$TEXER" --mode cil --tag-1e "$TAG1" --tag-5e "$TAG5" \
  --out "$STATE_ROOT/cil_ablations_1e_5e.tex"
exit $rc
