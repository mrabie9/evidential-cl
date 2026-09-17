#!/bin/bash
# Kill the A4 and A2' arms of the running noamp gem_bob batch as soon as they launch.
#
# WHY a watchdog and not an edit: run_noamp_gembob_ering.sh was already running when these
# two arms were dropped, and bash expands "${JOBS[@]}" once when the for-loop starts, so the
# queue is fixed in the driver's memory. Editing the file only affects a future relaunch.
# Restarting the driver would have discarded ~3.5h of in-flight seeds; this costs <1 min.
#
# Killing the arm's main.py makes run_job return, so the driver's `wait -n` immediately
# starts the next queued arm. The pattern matches both the multi-seed launcher and its
# --single-seed child. "gembob_a2_distill_noamp" does NOT match "gembob_a2_distill_l1_noamp"
# (the retained A2 row) because the strings diverge right after "distill_".
#
# Self-terminates once both arms have been seen and killed, or after MAXWAIT.
set -uo pipefail

PATTERNS=("expt_name gembob_a4_bilevel_unmatched_noamp" "expt_name gembob_a2_distill_noamp")
LOG="/home/lunet/wsmr11/repos/evidential-cl/scripts/logs/drop_arms_watchdog.log"
MAXWAIT=$((60 * 60 * 48))
INTERVAL=30

killed_a4=0
killed_a2p=0
elapsed=0

echo "[watchdog] start $(date)" >> "$LOG"
while [ "$elapsed" -lt "$MAXWAIT" ]; do
  if pgrep -f "${PATTERNS[0]}" > /dev/null 2>&1; then
    pkill -f "${PATTERNS[0]}"
    killed_a4=1
    echo "[watchdog] killed A4 $(date)" >> "$LOG"
  fi
  if pgrep -f "${PATTERNS[1]}" > /dev/null 2>&1; then
    pkill -f "${PATTERNS[1]}"
    killed_a2p=1
    echo "[watchdog] killed A2prime $(date)" >> "$LOG"
  fi
  if [ "$killed_a4" -eq 1 ] && [ "$killed_a2p" -eq 1 ]; then
    echo "[watchdog] both arms handled, exiting $(date)" >> "$LOG"
    exit 0
  fi
  if ! pgrep -f "run_noamp_gembob_ering.sh" > /dev/null 2>&1; then
    echo "[watchdog] driver gone, exiting $(date)" >> "$LOG"
    exit 0
  fi
  sleep "$INTERVAL"
  elapsed=$((elapsed + INTERVAL))
done
echo "[watchdog] MAXWAIT reached, exiting $(date)" >> "$LOG"
