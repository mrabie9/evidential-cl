"""Integration-style checks for adapter-training paths on 3-ADC inputs."""

# ruff: noqa: E402

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.bcl_dual import Net as BclDualNet
from model.ctn import Net as CtnNet
from model.eralg4 import Net as ErAlg4Net
from model.icarl import Net as IcarlNet
from model.lamaml_cifar import Net as LamamlCifarNet
from model.ucl_bresnet import Net as UclBresnetNet


def _make_args(classes_per_task=None):
    if classes_per_task is None:
        classes_per_task = [6, 6]
    o = type("Args", (), {})()
    o.classes_per_task = classes_per_task
    o.nc_per_task_list = ""
    o.nc_per_task = None
    o.batch_size = 8
    o.replay_batch_size = 8
    o.memories = 32
    o.n_memories = 32
    o.validation = 0.25
    o.inner_steps = 1
    o.n_meta = 1
    o.cuda = False
    o.learn_lr = False
    o.second_order = False
    o.sync_update = False
    o.meta_batches = 1
    o.opt_wt = 0.01
    o.opt_lr = 0.01
    o.alpha_init = 1e-3
    o.lr = 1e-3
    o.ctx_lr = 1e-3
    o.memory_strength = 0.5
    o.temperature = 5.0
    o.task_emb = 16
    o.samples_per_task = 16
    o.cls_lambda = 1.0
    o.arch = "resnet1d"
    o.dataset = "iq"
    o.class_weighted_ce = False
    o.grad_clip_norm = 5.0
    o.clipgrad = 5.0
    return o


def _labels(batch_size: int, task_id: int, classes_per_task: int = 6) -> torch.Tensor:
    start = task_id * classes_per_task
    return (torch.arange(batch_size) % classes_per_task) + start


def _assert_adapter_weight_changes(
    observe_callable, weight_tensor: torch.Tensor
) -> None:
    before = weight_tensor.detach().clone()
    observe_callable()
    after = weight_tensor.detach().clone()
    assert not torch.allclose(before, after), "Expected adapter weight to change."


def test_lamaml_cifar_updates_adapter_weight():
    args = _make_args()
    model = LamamlCifarNet(n_inputs=1024, n_outputs=12, n_tasks=2, args=args)
    model.train()
    x = torch.randn(8, 3, 1024)
    y = _labels(8, task_id=0)

    _assert_adapter_weight_changes(
        lambda: model.observe(x, y, t=0),
        model.net.model.input_adapter.weight,
    )


def test_eralg4_updates_adapter_weight():
    args = _make_args()
    model = ErAlg4Net(n_inputs=1024, n_outputs=12, n_tasks=2, args=args)
    model.train()
    x = torch.randn(8, 3, 1024)
    y = _labels(8, task_id=0)

    _assert_adapter_weight_changes(
        lambda: model.observe(x, y, t=0),
        model.net.model.input_adapter.weight,
    )


def test_icarl_updates_adapter_weight():
    args = _make_args()
    model = IcarlNet(n_inputs=1024, n_outputs=12, n_tasks=2, args=args)
    model.train()
    x = torch.randn(8, 3, 1024)
    y = _labels(8, task_id=0)

    _assert_adapter_weight_changes(
        lambda: model.observe(x, y, t=0),
        model.net.model.input_adapter.weight,
    )


def test_bcl_dual_updates_adapter_weight():
    args = _make_args()
    model = BclDualNet(n_inputs=1024, n_outputs=12, n_tasks=2, args=args)
    model.train()
    x = torch.randn(8, 3, 1024)
    y = _labels(8, task_id=0)

    _assert_adapter_weight_changes(
        lambda: model.observe(x, y, t=0),
        model.net.model.input_adapter.weight,
    )


def test_ctn_updates_adapter_weight():
    args = _make_args()
    model = CtnNet(n_inputs=1024, n_outputs=12, n_tasks=2, args=args)
    model.train()
    x = torch.randn(8, 3, 1024)
    y = _labels(8, task_id=0)

    _assert_adapter_weight_changes(
        lambda: model.observe(x, y, t=0),
        model.net.model.input_adapter.adapter.weight,
    )


def test_ucl_bresnet_optimizer_includes_adapter_params():
    args = _make_args()
    model = UclBresnetNet(n_inputs=1024, n_outputs=12, n_tasks=2, args=args)
    adapter_param_ids = {id(p) for p in model.model.input_adapter.parameters()}
    optimizer_param_ids = {
        id(parameter)
        for group in model.optimizer.param_groups
        for parameter in group["params"]
    }
    assert adapter_param_ids.issubset(optimizer_param_ids)
