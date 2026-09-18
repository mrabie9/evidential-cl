#!/bin/bash
# Shared concurrency limiter for experiment drivers. Source this and call
# `job_slot` before every launch, then `PIDS+=($!)` after.
#
# Why a hard count and not just a memory gate: a run is ~1-2 GB for its first
# minutes and only grows to ~17-23 GB once the task loaders are warm. A gate that
# checks free memory at *launch* time therefore passes repeatedly and admits far
# more jobs than will fit -- which is exactly how six concurrent runs got started
# on a box that holds three comfortably. The count is the binding constraint; the
# memory check is a secondary backstop for anything else sharing the machine.
#
# MAXJOBS is 3 unless the caller sets it first. Do not raise it.
: "${MAXJOBS:=3}"
: "${MINFREEGB:=40}"
: "${STAGGER:=45}"
PIDS=()

job_slot () {
  while true; do
    local alive=()
    for pid in ${PIDS[@]+"${PIDS[@]}"}; do
      kill -0 "$pid" 2>/dev/null && alive+=("$pid")
    done
    PIDS=(${alive[@]+"${alive[@]}"})
    local freegb
    freegb=$(free -g | awk '/^Mem:/ {print $7}')
    if [ "${#PIDS[@]}" -lt "$MAXJOBS" ] && [ "$freegb" -ge "$MINFREEGB" ]; then
      break
    fi
    sleep 30
  done
  sleep "$STAGGER"
}
