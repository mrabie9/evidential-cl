#!/bin/bash
# Ad-hoc fill-in for ft and iid2: their earlier seed-100/390/550 extension
# runs (done.txt labels seedext_{1e,5e}_{til,cil}_{ft,iid2}) had their output
# directories deleted from disk (logs/ft, logs/iid2/*seedext* no longer
# exist), and ft never had a seed-0/39/55 baseline in the first place (absent
# from logs/00_sync/{1e,5e}_{TIL,CIL}/saved_models). This reruns both models
# from scratch with fresh, distinct labels (so the stale seedext_* done.txt
# entries are not mistaken for coverage and are not touched):
#   ft   : full 6-seed sweep (0,39,55,100,390,550), all 4 epoch/mode combos.
#   iid2 : 3-seed sweep (0,39,55), all 4 epoch/mode combos.
#
# Usage:
#   scripts/pipeline_60h/exp1b_ft_iid2_baseline.sh          # run (MAX_JOBS=4)
#   scripts/pipeline_60h/exp1b_ft_iid2_baseline.sh --list   # print job specs

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_queue.sh"

IID2_SEEDS="0,39,55"

emit_ft_iid2_baseline_jobs() {
    local epochs epoch_flags mode
    for epochs in 1e 5e; do
        epoch_flags="$EPOCHS_1E"
        [ "$epochs" = "5e" ] && epoch_flags="$EPOCHS_5E"
        for mode in til cil; do
            echo "ftbaseline_${epochs}_${mode}|${mode}|ft|--seeds ${ALL_SEEDS} ${epoch_flags}"
            echo "iid2baseline_${epochs}_${mode}|${mode}|iid2|--seeds ${IID2_SEEDS} ${epoch_flags}"
        done
    done
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    experiment_main emit_ft_iid2_baseline_jobs "$@"
fi
