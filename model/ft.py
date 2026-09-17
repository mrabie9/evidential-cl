"""Naive fine-tuning lower bound: train on each new task's data only."""

from model import iid2


class Net(iid2.Net):
    """Sequential fine-tuning: the iid2 learner without any replay.

    Task ``t`` trains only on task ``t``'s data, with no memory, regulariser or
    other continual-learning mechanism, so it forgets freely and bounds the
    baselines from below. Everything else (backbone, SGD optimiser, loss and
    logit masking) is inherited from :class:`model.iid2.Net`, so the lower and
    upper bounds differ only in which training data each task sees. Batches are
    single-task, so iid2's per-row TIL mask reduces to the task-``t`` mask.

    Usage:
        model = Net(n_inputs, n_outputs, n_tasks, args)
        loss, rec, logits = model.observe(x, y, t)
    """

    # Read by ``main.life_experience``: train each task on its own data only.
    cumulative_replay = False


__all__ = ["Net"]
