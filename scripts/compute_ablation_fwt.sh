#!/bin/bash
# Compute ZS forward-transfer (FWT) for the CTN ablations, per seed and aggregated.
#
# FWT_t = trained-model zero-shot F1 on task t  -  untrained-model zero-shot F1 on task t
# averaged over tasks 1-9 (task 0 excluded, no prior tasks to transfer from).
#
# Requires the untrained baseline JSON produced by:
#   scripts/collect_zero_shot_baselines_all_models.py --mode til \
#     --models ctn,ctn_nofilm,ctn_nodistill --output logs/fwt/zs_baseline_til.json
#
# Usage:  bash scripts/compute_ablation_fwt.sh
set -uo pipefail

REPO="/home/lunet/wsmr11/repos/evidential-cl"
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
BASELINE="$REPO/logs/fwt/zs_baseline_til.json"
OUT="$REPO/logs/fwt"
mkdir -p "$OUT"

[ -f "$BASELINE" ] || { echo "ERROR: baseline JSON missing: $BASELINE"; exit 1; }

for algo in ctn_nofilm ctn_nodistill; do
  for seed in 0 39 55; do
    # latest metrics dir for this algo/seed (seed 0 of ctn_nofilm lives in an older run dir)
    mdir=$(find "$REPO/logs/$algo" -maxdepth 3 -type d -path "*/$seed/metrics" 2>/dev/null | sort | tail -1)
    if [ -z "$mdir" ]; then
      echo "[fwt] WARN: no metrics dir for $algo seed $seed"; continue
    fi
    "$PY" "$REPO/scripts/collect_zero_shot_validation_metrics.py" \
      --algo "$algo" \
      --metrics-dir "$mdir" \
      --baseline-json "$BASELINE" \
      --csv "$OUT/fwt_${algo}_seed${seed}.csv" >/dev/null 2>&1 \
      && echo "[fwt] wrote $OUT/fwt_${algo}_seed${seed}.csv  (from $mdir)" \
      || echo "[fwt] FAIL $algo seed $seed (from $mdir)"
  done
done

echo "=== Aggregating FWT (mean over tasks 1-9 per seed, then mean +/- sample std) ==="
"$PY" - "$OUT" <<'PY'
import csv, glob, math, os, sys
out = sys.argv[1]
def avg_tasks_1_9(path):
    vals=[]
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                t=int(float(row["task_index"])); v=float(row["forward_transfer_total_f1_zs"])
            except (KeyError,ValueError):
                continue
            if 1<=t<=9 and not math.isnan(v):
                vals.append(v)
    return sum(vals)/len(vals) if vals else float("nan")
for algo in ("ctn_nofilm","ctn_nodistill"):
    per_seed=[]
    for seed in (0,39,55):
        p=os.path.join(out,f"fwt_{algo}_seed{seed}.csv")
        if os.path.exists(p):
            a=avg_tasks_1_9(p); per_seed.append((seed,a))
    seeds=[v for _,v in per_seed if not math.isnan(v)]
    print(f"\n{algo}:")
    for s,a in per_seed:
        print(f"  seed {s}: FWT(1-9) = {a*100:+.2f}")
    if len(seeds)>=2:
        m=sum(seeds)/len(seeds)
        sd=(sum((x-m)**2 for x in seeds)/(len(seeds)-1))**0.5
        print(f"  MEAN +/- std (x100): {m*100:+.2f} +/- {sd*100:.2f}")
    elif seeds:
        print(f"  MEAN (x100): {seeds[0]*100:+.2f}")
PY
