"""Task-0 replay gating for the reservoir-buffer learners.

``er_ring``, ``agem`` and ``gem`` gate replay on ``t > 0``, so task 0 trains on
its own data alone. ``eralg4`` (also ``la-er``) and the La-MAML family
(``lamaml``, ``cmaml``, ``smaml``) draw from a flat reservoir instead, and
without ``use_old_task_memory`` they start replaying the current task's own rows
from its second batch onwards. That inflates the CIL diagonal and depresses the
TIL one, which moves the reference point BWT is measured against and makes the
two groups incomparable.

These tests pin the gated behaviour under the flag, and pin the ungated
behaviour without it so the Algorithm 4 / reference La-MAML path stays intact.
"""

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.eralg4 import Net as ErAlg4Net
from model.lamaml_cifar import Net as LamamlNet

CLASSES_PER_TASK = 6
BATCH_SIZE = 8


def _make_args(use_old_task_memory: bool):
    """Minimal args-like object shared by both learners under test."""
    args = type("Args", (), {})()
    args.classes_per_task = [CLASSES_PER_TASK, CLASSES_PER_TASK]
    args.nc_per_task_list = ""
    args.nc_per_task = None
    args.get_samples_per_task = None
    args.samples_per_task = -1
    args.batch_size = 128
    args.clipgrad = 10.0
    args.grad_clip_norm = 10.0
    args.dataset = "iq"
    args.arch = "resnet1d"
    args.input_channels = 2
    args.cls_lambda = 1.0
    args.inner_steps = 1
    args.memories = 64
    args.replay_batch_size = 8
    args.cuda = False
    args.learn_lr = False
    args.second_order = False
    args.sync_update = False
    args.meta_batches = 2
    args.opt_wt = 0.1
    args.opt_lr = 0.1
    args.lr = 0.1
    args.alpha_init = 1e-3
    args.use_old_task_memory = use_old_task_memory
    return args


def _labels(task_id: int) -> torch.Tensor:
    """Global class labels for ``task_id``, one batch worth."""
    start = task_id * CLASSES_PER_TASK
    return torch.arange(BATCH_SIZE) % CLASSES_PER_TASK + start


def _run_task(model, task_id: int, n_batches: int) -> None:
    """Feed ``n_batches`` random batches of ``task_id`` through ``observe``."""
    for _ in range(n_batches):
        model.observe(torch.randn(BATCH_SIZE, 2, 512), _labels(task_id), task_id)


def _eralg4_pool_task_ids(model) -> list[int]:
    return [
        int(torch.as_tensor(entry[2]).flatten()[0]) for entry in model.replay_pool()
    ]


def _lamaml_pool_task_ids(model) -> list[int]:
    return [int(torch.as_tensor(entry[2]).flatten()[0]) for entry in model.M]


def test_eralg4_gated_has_no_replay_during_task_zero():
    """With the flag, eralg4's replay pool stays empty for the whole of task 0."""
    model = ErAlg4Net(n_inputs=1024, n_outputs=12, n_tasks=2, args=_make_args(True))
    model.train()

    for _ in range(5):
        assert model.replay_pool() == [], "task 0 must train on its own data alone"
        _run_task(model, task_id=0, n_batches=1)

    assert model.replay_pool() == []
    assert len(model.M) > 0, "the reservoir itself must still be filling"


def test_eralg4_gated_replays_only_earlier_tasks():
    """After the boundary the pool holds task-0 rows and never task-1 rows."""
    model = ErAlg4Net(n_inputs=1024, n_outputs=12, n_tasks=2, args=_make_args(True))
    model.train()

    _run_task(model, task_id=0, n_batches=4)
    _run_task(model, task_id=1, n_batches=4)

    pool_task_ids = _eralg4_pool_task_ids(model)
    assert pool_task_ids, "task 1 must have task-0 rows to replay"
    assert set(pool_task_ids) == {0}


def test_eralg4_ungated_replays_within_task_zero():
    """Without the flag, Algorithm 4's task-free replay is unchanged."""
    model = ErAlg4Net(n_inputs=1024, n_outputs=12, n_tasks=2, args=_make_args(False))
    model.train()

    _run_task(model, task_id=0, n_batches=3)

    assert set(_eralg4_pool_task_ids(model)) == {0}


def test_lamaml_gated_has_no_replay_during_task_zero():
    """With the flag, La-MAML draws from the boundary snapshot, empty at task 0."""
    model = LamamlNet(n_inputs=1024, n_outputs=12, n_tasks=2, args=_make_args(True))
    model.train()

    for _ in range(5):
        assert model.M == [], "task 0 must train on its own data alone"
        _run_task(model, task_id=0, n_batches=1)

    assert model.M == []
    assert len(model.M_new) > 0, "the live reservoir itself must still be filling"


def test_lamaml_gated_getbatch_returns_current_rows_only():
    """``getBatch`` adds no replay rows while the snapshot is empty."""
    model = LamamlNet(n_inputs=1024, n_outputs=12, n_tasks=2, args=_make_args(True))
    model.train()

    _run_task(model, task_id=0, n_batches=3)

    x = torch.randn(BATCH_SIZE, 2, 512).numpy()
    batch_x, _, batch_t = model.getBatch(x, _labels(0).numpy(), 0)
    assert batch_x.shape[0] == BATCH_SIZE
    assert set(batch_t.tolist()) == {0}


def test_lamaml_gated_replays_only_earlier_tasks():
    """After the boundary the snapshot holds task-0 rows and never task-1 rows."""
    model = LamamlNet(n_inputs=1024, n_outputs=12, n_tasks=2, args=_make_args(True))
    model.train()

    _run_task(model, task_id=0, n_batches=4)
    _run_task(model, task_id=1, n_batches=4)

    pool_task_ids = _lamaml_pool_task_ids(model)
    assert pool_task_ids, "task 1 must have task-0 rows to replay"
    assert set(pool_task_ids) == {0}


def test_lamaml_ungated_replays_within_task_zero():
    """Without the flag, the live-buffer replay path is unchanged."""
    model = LamamlNet(n_inputs=1024, n_outputs=12, n_tasks=2, args=_make_args(False))
    model.train()

    _run_task(model, task_id=0, n_batches=3)

    x = torch.randn(BATCH_SIZE, 2, 512).numpy()
    batch_x, _, _ = model.getBatch(x, _labels(0).numpy(), 0)
    assert batch_x.shape[0] > BATCH_SIZE, "live buffer must still be replayed"
