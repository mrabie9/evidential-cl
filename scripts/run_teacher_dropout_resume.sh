#!/bin/bash
# Resume the --woe_teacher_dropout 2x2x3 after the first pass lost 8 of 12 runs.
#
# Cause: this box is also running scripts/run_importance_gaps_n3.sh (3 concurrent
# 10-task jobs). Two of mine on top of three of those exhausts the 7 GB of swap,
# and the cgroup OOM killer takes one job per batch at around task 6 -- the same
# failure that already cost run_absom_lwf_seeds.sh its seed-55 arm. So: one job
# at a time, and a retry, since a kill leaves no `# val:` line to collect.
#
# Idempotent. A run whose log already carries a `# val:` line is skipped, so the
# survivors of earlier passes are not repeated and this can be re-run freely.
#
# Serial was not enough on its own: by 14:41 the box was also running two of the
# user's campaigns (run_importance_gaps_n3.sh plus a b6h_logit_* sweep), six
# jobs at 22-23 GB RSS each, with all 7 GB of swap gone. The real hazard there is
# not that my job dies -- it is that the OOM killer picks the *largest* victim,
# so a job of mine can get one of theirs killed instead. Hence `wait_for_room`:
# never start a run unless the box has headroom, and check often enough that a
# short quiet window is used rather than missed.
set -u

# Available memory (GB) and other-job count required before a run may start. Two
# of my own runs peaked near 24 GB, so this asks for roughly two runs' worth.
#
# Gate on *available*, not on free swap: swap stays allocated long after the jobs
# that filled it exit (the box sat idle at 11/251 GB used with swap still 6/7
# full), so a free-swap gate never reopens and the queue stalls indefinitely.
MIN_AVAIL_GB=60
MAX_OTHER_JOBS=3

other_jobs () {
  pgrep -f "main.py" 2>/dev/null | while read -r pid; do
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "expt_name td-" || echo "$pid"
  done | wc -l
}

avail_gb () {
  free -g | awk '/^Mem:/ {print $7}'
}

wait_for_room () {
  local name=$1 waited=0
  while :; do
    local avail jobs
    avail=$(avail_gb)
    jobs=$(other_jobs)
    if [ "$avail" -ge "$MIN_AVAIL_GB" ] && [ "$jobs" -le "$MAX_OTHER_JOBS" ]; then
      [ "$waited" -gt 0 ] && echo "[room $(date +%H:%M:%S)] $name after ${waited}m (${avail}GB available, $jobs other jobs)"
      return 0
    fi
    if [ "$waited" -eq 0 ]; then
      echo "[wait $(date +%H:%M:%S)] $name held: ${avail}GB available, $jobs other jobs running"
    fi
    sleep 120
    waited=$((waited + 2))
  done
}

REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/teacher_dropout
mkdir -p "$LOGS"

run_one () {
  local cell=$1 lam=$2 drop=$3 seed=$4
  local name="td-${cell}-${drop}-s${seed}"
  local log="$LOGS/$name.log"

  if [ -f "$log" ] && grep -q "# val:" "$log"; then
    echo "[skip $(date +%H:%M:%S)] $name (already complete)"
    return 0
  fi

  local attempt
  for attempt in 1 2 3; do
    wait_for_room "$name"
    echo "[run $(date +%H:%M:%S)] $name attempt $attempt"
    PYTHONUNBUFFERED=1 $PY $REPO/main.py \
      --config configs/base.yaml \
      --config configs/models/til/woe_si_lwf.yaml \
      --n_epochs 1 --inner_steps 2 --lr 0.003 \
      --woe_omega_transform relu --woe_anchor_mode proximal \
      --woe_lambda "$lam" --woe_lwf_lambda 1.0 \
      --woe_teacher_dropout "$drop" \
      --single-seed --seed "$seed" --no-save_checkpoints \
      --expt_name "$name" > "$log" 2>&1
    if grep -q "# val:" "$log"; then
      echo "[ok $(date +%H:%M:%S)] $name"
      return 0
    fi
    echo "[retry $(date +%H:%M:%S)] $name produced no result (killed?)"
  done
  echo "[FAIL $(date +%H:%M:%S)] $name after 3 attempts"
  return 1
}

for seed in 0 39 55; do
  run_one lwfonly 0.0 keep    "$seed"
  run_one lwfonly 0.0 disable "$seed"
done
for seed in 0 39 55; do
  run_one both 1000000.0 keep    "$seed"
  run_one both 1000000.0 disable "$seed"
done

echo "[teacher-dropout resume] all done"
