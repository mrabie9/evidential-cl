"""Tests for the frozen per-task ``mu`` reference (PR-3, ``--woe_mu_mode``).

``I_2 = 1/2 ||z||^2 + 1/2 sum_k (sum_j |w_jk|)^2`` and ``sum_j w_jk = z_k``
identically, so ``mu`` cancels from the first half and the whole DS-specific
content of the tracked scalar is measured against it. ``frozen_pretask``
replaces the within-task EMA with the unweighted mean of ``phi`` over the task's
full training set, computed before the task's first gradient step.

Three properties are load-bearing and tested here:

1. the frozen ``mu`` does **not** move within a task;
2. the pre-pass leaves BatchNorm running statistics, the RNG state and the
   train/eval flag untouched;
3. **both** arms run the pre-pass, so its cost is common-mode and the arms
   differ only in which ``mu`` the evidence reads.

A caution about (2), worth reading before trusting it. These tests drive a stub
loader on CPU with no AMP and never touch ``IQDataGenerator``; they establish
that the pass does not perturb *through the paths they exercise*. At full scale
on CUDA the pass **does** shift the trajectory -- reproducibly, by ~0.8 sigma --
and saving/restoring RNG did not change that, so the mechanism is not a stolen
draw. That is why (3) exists: the perturbation is neutralised by being paid in
both arms rather than by being argued away. See PR-3 Amendment 1.

The general lesson, which cost a gate failure to learn: these tests verify an
argument about a code path, not a measurement of the run. Only the full-scale
regression can do the latter.
"""

# ruff: noqa: E402

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.woe_si import Net
from tests.test_woe_si import _make_args

CLASSES_PER_TASK = [3, 3]
N_OUTPUTS = 6
N_TASKS = 2
SEQ_LEN = 128


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _task_batches(task: int, n_batches: int = 3, batch_size: int = 4):
    """Deterministic (x, y) batches for ``task``, disjoint labels per task."""
    generator = torch.Generator().manual_seed(1000 + task)
    batches = []
    low = task * CLASSES_PER_TASK[task]
    high = low + CLASSES_PER_TASK[task]
    for _ in range(n_batches):
        x = torch.randn(batch_size, 2, SEQ_LEN, generator=generator)
        y = torch.randint(low, high, (batch_size,), generator=generator)
        batches.append((x, y))
    return batches


class _StubLoaderFn:
    """Stands in for the bound ``IncrementalLoader.get_tasks`` main.py attaches.

    Returns a fixed list of per-task iterables. Like the real one it is
    sequential and rebuildable, so iterating it draws no RNG -- which is the
    property the pre-pass depends on.
    """

    def __init__(self, per_task_batches):
        self._per_task = per_task_batches
        self.calls = 0

    def __call__(self, dataset_type="train"):
        assert dataset_type == "train"
        self.calls += 1
        return [list(batches) for batches in self._per_task]


def _build(mu_mode: str, loader_fn=None, seed: int = 0) -> Net:
    torch.manual_seed(seed)
    args = _make_args(
        "task_incremental_loader",
        classes_per_task=CLASSES_PER_TASK,
        woe_anchor_mode="proximal",
        woe_omega_transform="abs",
    )
    args.woe_mu_mode = mu_mode
    args.get_task_train_loader = loader_fn
    return Net(SEQ_LEN, N_OUTPUTS, N_TASKS, args)


def _bn_running_stats(model: Net):
    """Snapshot of every BatchNorm running statistic in the backbone."""
    out = {}
    for name, buf in model.net.model.named_buffers():
        if name.endswith(("running_mean", "running_var", "num_batches_tracked")):
            out[name] = buf.detach().clone()
    return out


# ----------------------------------------------------------------------
# 1. The frozen mu does not move within a task
# ----------------------------------------------------------------------
def test_frozen_mu_constant_within_task() -> None:
    batches = [_task_batches(0), _task_batches(1)]
    model = _build("frozen_pretask", _StubLoaderFn(batches))

    x0, y0 = batches[0][0]
    model.observe(x0, y0, 0)
    assert bool(model.woe_mu_frozen_set.item()), "pre-pass did not run"
    mu_first = model.woe_mu_frozen.detach().clone()

    for x, y in batches[0][1:]:
        model.observe(x, y, 0)
    mu_last = model.woe_mu_frozen.detach().clone()

    assert torch.equal(mu_first, mu_last), "frozen mu moved within the task"
    # And it is genuinely what the evidence reads, not just a stored spare.
    assert torch.equal(model._mu_for_evidence(), mu_last)
    # The EMA is still maintained alongside (the divergence diagnostic needs it)
    # and *has* moved, which is what makes the two modes distinguishable at all.
    assert bool(model.woe_feature_mean_initialised.item())
    assert not torch.equal(model.woe_feature_mean, mu_last)


def test_frozen_mu_recomputed_at_task_boundary() -> None:
    batches = [_task_batches(0), _task_batches(1)]
    loader_fn = _StubLoaderFn(batches)
    model = _build("frozen_pretask", loader_fn)

    for x, y in batches[0]:
        model.observe(x, y, 0)
    mu_task0 = model.woe_mu_frozen.detach().clone()

    x1, y1 = batches[1][0]
    model.observe(x1, y1, 1)
    mu_task1 = model.woe_mu_frozen.detach().clone()

    assert not torch.equal(mu_task0, mu_task1), "mu not recomputed at boundary"
    assert loader_fn.calls == 2, "pre-pass should run exactly once per task"


def test_frozen_mu_is_unweighted_mean_over_samples() -> None:
    """Not an EMA and not a batch-mean average: an exact per-sample mean.

    Batches are unequal in size so a mean-of-batch-means would differ from a
    mean-over-samples, which is the failure this pins down.
    """
    torch.manual_seed(7)
    uneven = [
        (torch.randn(6, 2, SEQ_LEN), torch.randint(0, 3, (6,))),
        (torch.randn(1, 2, SEQ_LEN), torch.randint(0, 3, (1,))),
    ]
    loader_fn = _StubLoaderFn([uneven, _task_batches(1)])
    model = _build("frozen_pretask", loader_fn)
    model._compute_pretask_mu(0)

    with torch.no_grad():
        feats = torch.cat(
            [model.net.forward_features(x, bn_training=False) for x, _ in uneven]
        )
    expected = feats.mean(dim=0)
    assert torch.allclose(model.woe_mu_frozen, expected, atol=1e-6)

    batch_mean_of_means = torch.stack(
        [
            model.net.forward_features(x, bn_training=False).mean(dim=0)
            for x, _ in uneven
        ]
    ).mean(dim=0)
    assert not torch.allclose(model.woe_mu_frozen, batch_mean_of_means, atol=1e-6)


# ----------------------------------------------------------------------
# 2. The pre-pass is inert: no BatchNorm update, no RNG consumption
# ----------------------------------------------------------------------
def test_prepass_leaves_batchnorm_and_rng_unchanged() -> None:
    batches = [_task_batches(0), _task_batches(1)]
    model = _build("frozen_pretask", _StubLoaderFn(batches))

    # Put BatchNorm into a non-trivial state first, so "unchanged" is a real
    # claim rather than "both are still at initialisation".
    model.net.train()
    with torch.no_grad():
        model.net.forward_features(batches[0][0][0], bn_training=True)

    bn_before = _bn_running_stats(model)
    torch.manual_seed(1234)
    rng_before = torch.get_rng_state()
    training_before = model.net.training

    model._compute_pretask_mu(0)

    bn_after = _bn_running_stats(model)
    assert set(bn_before) == set(bn_after) and bn_before, "no BN buffers found"
    for name, before in bn_before.items():
        assert torch.equal(before, bn_after[name]), f"pre-pass moved BN {name}"

    assert torch.equal(rng_before, torch.get_rng_state()), "pre-pass consumed RNG"
    assert model.net.training == training_before, "pre-pass left train/eval flipped"


def test_prepass_does_not_require_grad_or_leave_graph() -> None:
    batches = [_task_batches(0), _task_batches(1)]
    model = _build("frozen_pretask", _StubLoaderFn(batches))
    for param in model.net.parameters():
        param.grad = None
    model._compute_pretask_mu(0)
    assert not model.woe_mu_frozen.requires_grad
    assert all(p.grad is None for p in model.net.parameters())


# ----------------------------------------------------------------------
# 3. `ema` (and flag absent) is unchanged
# ----------------------------------------------------------------------
def _run_two_tasks(model: Net, batches) -> list:
    losses = []
    for task, task_batches in enumerate(batches):
        for x, y in task_batches:
            loss, _, _ = model.observe(x, y, task)
            losses.append(loss)
    return losses


def test_ema_mode_matches_flag_absent() -> None:
    """The default must be bit-identical to a build that never saw the flag."""
    batches = [_task_batches(0), _task_batches(1)]

    model_explicit = _build("ema", _StubLoaderFn(batches), seed=3)
    torch.manual_seed(99)
    losses_explicit = _run_two_tasks(model_explicit, batches)

    # Flag genuinely absent from args, and no loader hook attached either.
    torch.manual_seed(3)
    args = _make_args(
        "task_incremental_loader",
        classes_per_task=CLASSES_PER_TASK,
        woe_anchor_mode="proximal",
        woe_omega_transform="abs",
    )
    assert not hasattr(args, "woe_mu_mode")
    model_absent = Net(SEQ_LEN, N_OUTPUTS, N_TASKS, args)
    torch.manual_seed(99)
    losses_absent = _run_two_tasks(model_absent, batches)

    assert losses_explicit == losses_absent
    assert torch.equal(model_explicit.woe_feature_mean, model_absent.woe_feature_mean)


def test_ema_arm_runs_the_prepass_too() -> None:
    """Both arms run the pre-pass, so its cost is common-mode.

    An earlier version gated the pass to `frozen_pretask` so that `ema` would
    stay bit-identical to the recorded B6 control. That was withdrawn once the
    full-scale gate measured the pass to be *not* inert (0.5224 / 0.4984 /
    -0.0240 with it, against the recorded 0.5206 / 0.5008 / -0.0198 reproduced
    exactly without it, each trajectory reproducible across launches).

    `frozen_pretask` cannot run without the pass, so the only way the two arms
    can share a trajectory is for both to pay it. Bit-identity to a three-day-old
    run was only ever a proxy for "the arms differ by one knob"; running the pass
    in both arms serves that goal directly. See PR-3 Amendment 1.
    """
    batches = [_task_batches(0), _task_batches(1)]

    model_ema = _build("ema", _StubLoaderFn(batches))
    model_ema.observe(*batches[0][0], 0)
    assert bool(
        model_ema.woe_mu_frozen_set.item()
    ), "the ema arm must also run the pre-pass, so the cost is common-mode"

    model_frozen = _build("frozen_pretask", _StubLoaderFn(batches))
    model_frozen.observe(*batches[0][0], 0)
    assert bool(model_frozen.woe_mu_frozen_set.item())

    # Same mu computed in both arms; only which one the evidence *reads* differs.
    assert torch.equal(model_ema.woe_mu_frozen, model_frozen.woe_mu_frozen)
    assert torch.equal(model_ema._mu_for_evidence(), model_ema.woe_feature_mean)
    assert torch.equal(model_frozen._mu_for_evidence(), model_frozen.woe_mu_frozen)


def test_prepass_skipped_when_no_loader_hook_is_bound() -> None:
    """Without the hook (unit tests, other harnesses) `ema` still works."""
    batches = [_task_batches(0), _task_batches(1)]
    model = _build("ema", None)
    model.observe(*batches[0][0], 0)
    assert not bool(model.woe_mu_frozen_set.item())
    assert torch.equal(model._mu_for_evidence(), model.woe_feature_mean)


# ----------------------------------------------------------------------
# Guards
# ----------------------------------------------------------------------
def test_frozen_mode_without_loader_hook_is_a_hard_error() -> None:
    try:
        _build("frozen_pretask", None)
    except ValueError as exc:
        assert "get_task_train_loader" in str(exc)
    else:
        raise AssertionError("frozen_pretask without the loader hook must raise")


def test_unknown_mu_mode_rejected() -> None:
    try:
        _build("frozen", None)
    except ValueError as exc:
        assert "woe_mu_mode" in str(exc)
    else:
        raise AssertionError("an unknown woe_mu_mode must raise")


def test_frozen_mu_is_checkpointed() -> None:
    batches = [_task_batches(0), _task_batches(1)]
    model = _build("frozen_pretask", _StubLoaderFn(batches))
    model._compute_pretask_mu(0)
    state = model.state_dict()
    assert "woe_mu_frozen" in state
    assert "woe_mu_frozen_set" in state
    assert state["woe_mu_frozen"].shape == (model.feature_dim,)
