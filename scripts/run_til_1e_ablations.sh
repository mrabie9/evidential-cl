#!/bin/bash
# Single-pass (1-epoch) TIL ablation campaign on the frozen nrm grid, paper protocol
# (sec:stat_protocol): n=6 for every row, then ONE top-up of 6 more seeds (n=12) for the
# rows whose verdict is INC, plus their family baselines so the paired deltas stay paired.
# Rows still INC at n=12 stay INC -- there is no second top-up.
#
#   phase 1: seeds 7,13,21;1,2,3    every TIL row of run_ablations_noise_removed.sh
#   phase 2: seeds 4,5,6;8,9,10     INC rows + baselines, from the analyser's --topup list
#
# A phase writes its done-marker only once it finishes with every row at n>=6, so a rerun
# after an interruption skips completed phases and repeats an unfinished one in full.
#
# Usage:
#   bash scripts/run_til_1e_ablations.sh
#   MAX_PARALLEL=4 bash scripts/run_til_1e_ablations.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$REPO/la-maml_env/bin/python}"
DRIVER="$REPO/scripts/run_ablations_noise_removed.sh"
ANALYSER="$REPO/scripts/analyse_ablations_noise_removed.py"
TAG="nrm1e"
STATE="$REPO/scripts/logs/ablations_noise_removed/1e_til"
mkdir -p "$STATE"

export N_EPOCHS=1 TAG MODES=til PY MAX_PARALLEL="${MAX_PARALLEL:-3}"

# short_rows: TIL table rows with fewer than 6 seeds (pending rows included), one per line.
short_rows() {
  "$PY" - "$ANALYSER" "$TAG" <<'PYEOF'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("a", sys.argv[1])
a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
for rows in a.GRIDS["til"].values():
    for rid, _l, stem, pid in rows:
        n = len(a.pool("til", stem, pid, sys.argv[2]))
        if n < 6:
            print("{} n={}".format(rid, n))
PYEOF
}

if [ ! -f "$STATE/phase1.done" ]; then
  SEED_GROUP_SPEC="7,13,21;1,2,3" bash "$DRIVER"
  short=$(short_rows)
  if [ -n "$short" ]; then
    echo "[1e-til] phase 1 incomplete, rows below n=6:" >&2
    echo "$short" >&2
    exit 1
  fi
  touch "$STATE/phase1.done"
fi

if [ ! -f "$STATE/phase2.done" ]; then
  line=$("$PY" "$ANALYSER" --tag "$TAG" --modes til --topup)
  rows="${line##*ROWS_ONLY=}"
  echo "[1e-til] n=6 top-up list: '${rows}'"
  echo "$rows" > "$STATE/topup_rows.txt"
  if [ -n "$rows" ]; then
    ROWS_ONLY="$rows" SEED_GROUP_SPEC="4,5,6;8,9,10" bash "$DRIVER"
  fi
  touch "$STATE/phase2.done"
fi

"$PY" "$ANALYSER" --tag "$TAG" --modes til --stats | tee "$STATE/final_table.txt"
