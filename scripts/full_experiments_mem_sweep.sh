#!/bin/bash
# Run scripts/full_experiments.sh once per entry in MEM_VALS, passing the same
# replay capacity to both CLI flags used by main.py (see parser.py: --memories,
# --n_memories). Each iteration gets -d mem_<n> so logs land under a distinct
# logs/full_experiments/run_*_mem_<n>/ folder.
#
# Usage:
#   ./scripts/full_experiments_mem_sweep.sh
#   ./scripts/full_experiments_mem_sweep.sh --mode cil
#   ./scripts/full_experiments_mem_sweep.sh --one-shot
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

# Edit this list to choose which buffer sizes to run (order is preserved).
MEM_VALS=(1024 2048 8192 16384)

overall_exit=0

for n_mem in "${MEM_VALS[@]}"; do
  echo "[full_experiments_mem_sweep] === memories=${n_mem} (also n_memories=${n_mem}) ==="
  cmd=(
    "$SCRIPT_DIR/full_experiments.sh"
    -d "mem_${n_mem}"
    --memories "$n_mem"
    --n_memories "$n_mem"
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
