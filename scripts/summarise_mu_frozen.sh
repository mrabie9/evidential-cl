#!/bin/bash
# Tabulate every PR-3 cell that has finished: diagonal / final / BWT, plus the
# per-task mu divergence the runs print under WOE_LC_DEBUG=1.
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
printf "%-42s %-5s %8s %8s %9s\n" RUN SEED DIAG FINAL BWT
for d in $REPO/logs/woe_si_lc/mufz_*/; do
  n=$(basename "$d" | sed 's/-2026.*//')
  for sd in "$d"*/; do
    r="$sd/results.txt"
    [ -f "$r" ] || continue
    dg=$(grep -m1 "Diagonal F1:" "$r" | awk '{print $3}')
    fn=$(grep -m1 "Final F1:" "$r" | awk '{print $3}')
    bw=$(grep -m1 "Backward:" "$r" | awk '{print $2}')
    printf "%-42s %-5s %8s %8s %9s\n" "$n" "$(basename $sd)" "$dg" "$fn" "$bw"
  done
done
echo
echo "--- I_2 halves (per task; logit is the mu-free half, conflict the mu-dependent one) ---"
for f in $REPO/scripts/logs/mu_frozen/mufz_*.log; do
  [ -f "$f" ] || continue
  hits=$(grep -c "\[LC\] split" "$f" 2>/dev/null || echo 0)
  [ "$hits" -gt 0 ] || continue
  echo "== $(basename $f .log)"
  grep "\[LC\] split" "$f"
done
echo
echo "--- mu divergence (per task, end of task) ---"
for f in $REPO/scripts/logs/mu_frozen/mufz_*.log; do
  [ -f "$f" ] || continue
  hits=$(grep -c "\[WOE_MU\]" "$f" 2>/dev/null || echo 0)
  [ "$hits" -gt 0 ] || continue
  echo "== $(basename $f .log)"
  grep "\[WOE_MU\]" "$f"
done

echo
echo "=============================================================="
/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python "$REPO/scripts/mu_frozen_report.py"
