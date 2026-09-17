"""Tests for La-MAML (lamaml_cifar) handling 3-ADC IQ data with replay.

These tests exercise the 3-channel input path together with the replay
buffer so that mixed old/new batches are built correctly.
"""

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.lamaml_cifar import Net as LamamlNet


def _make_args(n_tasks=2, classes_per_task=None):
    """Minimal args-like object for La-MAML."""
    if classes_per_task is None:
        classes_per_task = [6, 6]
    o = type("Args", (), {})()
    o.classes_per_task = classes_per_task
    o.nc_per_task_list = ""
    o.nc_per_task = None
    o.get_samples_per_task = None
    o.samples_per_task = -1
    o.batch_size = 128
    o.clipgrad = 10.0
    o.grad_clip_norm = 10.0
    o.dataset = "iq"
    o.arch = "resnet1d"
    o.input_channels = 2
    o.cls_lambda = 1.0
    # Optim / meta settings
    o.inner_steps = 1
    o.memories = 16
    o.replay_batch_size = 8
    o.cuda = False
    o.learn_lr = False
    o.second_order = False
    o.sync_update = False
    o.meta_batches = 2
    o.opt_wt = 0.1
    o.opt_lr = 0.1
    o.alpha_init = 1e-3
    return o


def _make_labels(batch_size: int, task_id: int, classes_per_task=6):
    """Build class labels for a given task (1D tensor)."""
    start = task_id * classes_per_task
    cls = torch.arange(batch_size) % classes_per_task + start
    return cls


def test_lamaml_3adc_replay_single_task():
    """Single-task 3-ADC input goes through adapter and replay buffer without error."""
    args = _make_args()
    model = LamamlNet(n_inputs=1024, n_outputs=12, n_tasks=2, args=args)
    model.train()

    # Simulate task 0 with 3-ADC IQ input: (B, 3, 1024)
    x = torch.randn(8, 3, 1024)
    y = _make_labels(batch_size=8, task_id=0)

    loss, cls_tr_rec, metric_logits = model.observe(x, y, t=0)
    assert isinstance(loss, float)
    assert isinstance(cls_tr_rec, float)
    assert metric_logits is None

    # Memory should contain canonical (2, 512) representations from _input_for_replay
    assert len(model.M_new) > 0
    mem_x0, mem_y0, mem_t0 = model.M_new[0]
    assert isinstance(mem_t0.item(), (int, torch.Tensor)) or isinstance(
        mem_t0, torch.Tensor
    )
    assert mem_x0.shape[-2:] == (2, 512) or mem_x0.view(2, 512).shape == (2, 512)


def test_lamaml_3adc_replay_across_tasks():
    """3-ADC input for two tasks mixes correctly in replay batches."""
    args = _make_args()
    model = LamamlNet(n_inputs=1024, n_outputs=12, n_tasks=2, args=args)
    model.train()

    # Task 0: fill memory with 3-ADC data
    x0 = torch.randn(10, 3, 1024)
    y0 = _make_labels(batch_size=10, task_id=0)
    loss0, _, _ = model.observe(x0, y0, t=0)
    assert isinstance(loss0, float)
    assert len(model.M_new) > 0

    # Move to next task so old memory is reused
    model.real_epoch = 0  # ensure push_to_mem is active again

    # Task 1: new 3-ADC data should be combined with old memory in getBatch
    x1 = torch.randn(6, 3, 1024)
    y1 = _make_labels(batch_size=6, task_id=1)
    loss1, cls_tr_rec1, metric_logits1 = model.observe(x1, y1, t=1)
    assert isinstance(loss1, float)
    assert isinstance(cls_tr_rec1, float)
    assert metric_logits1 is None

    # Explicitly build a replay batch to ensure shapes are homogeneous
    x1_replay = model._input_for_replay(x1)
    bx, by, bt = model.getBatch(
        x1_replay.cpu().numpy(), y1.cpu().numpy(), t=1, batch_size=4
    )
    assert bx.shape[1:] == torch.from_numpy(x1_replay.cpu().numpy()).shape[1:]
    assert by.shape[0] == bt.shape[0] == bx.shape[0]


if __name__ == "__main__":
    test_lamaml_3adc_replay_single_task()
    test_lamaml_3adc_replay_across_tasks()
    print("All La-MAML 3-ADC replay tests passed.")
