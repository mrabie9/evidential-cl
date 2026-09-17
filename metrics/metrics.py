### We directly copied the metrics.py model file from the GEM project https://github.com/facebookresearch/GradientEpisodicMemory

# Copyright 2019-present, IBM Research
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function

import os
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

import torch


def task_changes(result_t):
    n_tasks = int(result_t.max() + 1)
    changes = []
    current = result_t[0]
    for i, t in enumerate(result_t):
        if t != current:
            changes.append(i)
            current = t

    return n_tasks, changes


def transfer_stats(result_t, result_a):
    """Reduce a per-round score matrix to one row per task and derive transfer stats.

    The eval log has one row per evaluation round, so a task can own several
    rows. Only the last row of each task is kept, giving a T x T matrix whose
    row ``t`` holds the score on every task right after training task ``t``.

    Args:
        result_t: 1D tensor of task ids, one entry per evaluation round.
        result_a: 2D tensor (rounds x tasks) of per-task scores.

    Returns:
        Tuple ``(baseline, reduced, diag, final, bwt, fwt)``: the first-round
        row, the T x T matrix, and per-task tensors for the diagonal, the last
        row, backward transfer (last row minus diagonal) and forward transfer.

    Usage:
        baseline, reduced, diag, final, bwt, fwt = transfer_stats(val_t, val_a)
    """
    nt, changes = task_changes(result_t)

    baseline = result_a[0]
    changes = torch.LongTensor(changes + [result_a.size(0)]) - 1
    result = result_a[changes]

    # acc[t] equals result[t,t]
    acc = result.diag()
    fin = result[nt - 1]
    # bwt[t] equals result[T,t] - acc[t]
    bwt = result[nt - 1] - acc

    # fwt[t] equals result[t-1,t] - baseline[t]
    fwt = torch.zeros(nt)
    for t in range(1, nt):
        fwt[t] = result[t - 1, t] - baseline[t]

    return baseline, result, acc, fin, bwt, fwt


def append_metric_block(path, title, result_t, result_a):
    """Append one metric's task matrix and transfer stats to ``results.txt``.

    Args:
        path: File to append to.
        title: Metric name used in the header and stat labels, e.g. ``"F1"``.
        result_t: 1D tensor of task ids, one entry per evaluation round.
        result_a: 2D tensor (rounds x tasks) of per-task scores for this metric.

    Returns:
        Dict with float ``diag``, ``final``, ``bwt`` and ``fwt`` means, or
        ``None`` when the matrix is empty or does not line up with ``result_t``.

    Usage:
        stats = append_metric_block(results_path, "F1", val_t, val_f1)
    """
    if result_a.numel() == 0 or result_a.size(0) != result_t.numel():
        return None
    baseline, result, acc, fin, bwt, fwt = transfer_stats(result_t, result_a)
    stats = {
        "diag": float(acc.mean()),
        "final": float(fin.mean()),
        "bwt": float(bwt.mean()),
        "fwt": float(fwt.mean()),
    }
    try:
        with open(path, "a", encoding="utf-8") as f:
            print("", file=f)
            print(
                "{} (per-task macro, rows = after training task t):".format(title),
                file=f,
            )
            print(" ".join(["%.4f" % r for r in baseline]), file=f)
            print("|", file=f)
            for row in range(result.size(0)):
                print(" ".join(["%.4f" % r for r in result[row]]), file=f)
            print("Diagonal %s: %.4f" % (title, stats["diag"]), file=f)
            print("Final %s: %.4f" % (title, stats["final"]), file=f)
            print("Backward %s: %.4f" % (title, stats["bwt"]), file=f)
            print("Forward %s: %.4f" % (title, stats["fwt"]), file=f)
    except OSError:
        pass
    return stats


def confusion_matrix(result_t, result_a, log_dir, fname=None, metric_name="Accuracy"):
    baseline, result, acc, fin, bwt, fwt = transfer_stats(result_t, result_a)

    if fname is not None:
        f = open(os.path.join(log_dir, fname), "w")

        print(" ".join(["%.4f" % r for r in baseline]), file=f)
        print("|", file=f)
        for row in range(result.size(0)):
            print(" ".join(["%.4f" % r for r in result[row]]), file=f)
        print("", file=f)
        print("Diagonal %s: %.4f" % (metric_name, acc.mean()), file=f)
        print("Final %s: %.4f" % (metric_name, fin.mean()), file=f)
        print("Backward: %.4f" % bwt.mean(), file=f)
        print("Forward:  %.4f" % fwt.mean(), file=f)
        f.close()

    colors = cm.nipy_spectral(np.linspace(0, 1, len(result)))
    figure = plt.figure(figsize=(8, 8))
    ax = plt.gca()
    # Convert to a plain NumPy array without relying on the deprecated
    # ``copy=`` semantics in NumPy 2.x.
    data = np.asarray(result_a)
    for i in range(len(data[0])):
        plt.plot(
            range(data.shape[0]), data[:, i], label=str(i), color=colors[i], linewidth=2
        )

    plt.savefig(log_dir + "/" + "task_wise_f1.png")

    stats = []
    stats.append(acc.mean())
    stats.append(fin.mean())
    stats.append(bwt.mean())
    stats.append(fwt.mean())

    return stats
