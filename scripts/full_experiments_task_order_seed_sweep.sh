#!/bin/bash
# Run scripts/full_experiments.sh once per entry in TASK_ORDER_SEEDS, passing
# --task-order-seed for each run. Each iteration gets -d task_order_seed_<n> so
# logs land under a distinct logs/full_experiments/run_*_task_order_seed_<n>/
# folder.
#
# Usage:
#   ./scripts/full_experiments_task_order_seed_sweep.sh
#   ./scripts/full_experiments_task_order_seed_sweep.sh --mode cil
#   ./scripts/full_experiments_task_order_seed_sweep.sh --one-shot
#
# full_experiments.sh activates la-maml_env when present (see AGENTS.md).
# On hosts whose short hostname is not lnx-elkk-1 or lnx-elkk-2, you may need
#   export HOST_KEY=lnx-elkk-1   # or lnx-elkk-2
# so the cached schedule JSON matches the intended machine.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONE_SHOT=0
FORWARDED_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --one-shot)
      ONE_SHOT=1
      shift
      ;;
    *)
      FORWARDED_ARGS+=("$1")
      shift
      ;;
  esac
done

# Edit this list to choose which task-order seeds to run (order is preserved).
TASK_ORDER_SEEDS=(57 1040 329 83)

overall_exit=0

for task_order_seed in "${TASK_ORDER_SEEDS[@]}"; do
  echo "[full_experiments_task_order_seed_sweep] === task_order_seed=${task_order_seed} ==="
  cmd=(
    "$SCRIPT_DIR/full_experiments.sh"
    -d "task_order_seed_${task_order_seed}"
    --task-order-seed "$task_order_seed"
    --single-seed
  )
  if [ "$ONE_SHOT" -eq 1 ]; then
    cmd+=(--one-shot)
  fi
  cmd+=("${FORWARDED_ARGS[@]}")
  if ! "${cmd[@]}"; then
    overall_exit=1
  fi
done

exit "$overall_exit"
