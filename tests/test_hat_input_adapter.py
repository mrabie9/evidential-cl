"""Tests for HAT model input adapter: 2-channel and 3-channel inputs.

The HAT backbone uses HatInputAdapter (wrapping AdcIqAdapter from resnet1d).
These tests verify that 2-channel input passes through and 3-channel is
converted to 2-channel before the gated ResNet.
"""

import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model import task_bn  # noqa: E402
from model.hat import Net as HatNet  # noqa: E402


def _make_args(n_tasks=2, classes_per_task=None):
    """Minimal args-like object for HAT Net."""
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
    return o


def test_hat_2channel_3d():
    """2-channel 3D input (B, 2, L) bypasses adapter and reaches backbone."""
    args = _make_args()
    # n_inputs=1024 -> IQ seq_len=512, 2 channels
    model = HatNet(n_inputs=1024, n_outputs=13, n_tasks=2, args=args)
    model.eval()
    with torch.no_grad():
        x = torch.randn(4, 2, 512)
        logits = model(x, t=0)
    assert logits.shape == (4, 13)


def test_hat_2channel_2d():
    """2D flat input (B, 1024) is reshaped to (B, 2, 512) then forwarded."""
    args = _make_args()
    model = HatNet(n_inputs=1024, n_outputs=13, n_tasks=2, args=args)
    model.eval()
    with torch.no_grad():
        x = torch.randn(4, 1024)
        logits = model(x, t=0)
    assert logits.shape == (4, 13)


def test_hat_3channel_3d():
    """3-channel 3D input (B, 3, L) is adapted to 2-channel then forwarded."""
    args = _make_args()
    model = HatNet(n_inputs=1024, n_outputs=13, n_tasks=2, args=args)
    model.eval()
    with torch.no_grad():
        x = torch.randn(4, 3, 1024)
        logits = model(x, t=0)
    assert logits.shape == (4, 13)


def test_hat_2channel_and_3channel_forward():
    """Forward runs without error for both 2- and 3-channel inputs."""
    args = _make_args()
    model = HatNet(n_inputs=1024, n_outputs=13, n_tasks=2, args=args)
    model.eval()
    with torch.no_grad():
        x2 = torch.randn(2, 2, 512)
        out2 = model(x2, t=0)
        x3 = torch.randn(2, 3, 1024)
        out3 = model(x3, t=0)
    assert out2.shape == (2, 13)
    assert out3.shape == (2, 13)


def test_hat_grad_flow_3channel():
    """Gradients flow through the adapter for 3-channel input.

    ``HatInputAdapter`` reshapes ``(B, 3, L)`` to ``(B, 3, 2, L/2)``, so
    ``AdcIqAdapter`` uses the 4D einsum path (``weight``), not ``proj_3ch``.
    """
    args = _make_args()
    model = HatNet(n_inputs=1024, n_outputs=13, n_tasks=2, args=args)
    model.train()
    x = torch.randn(2, 3, 1024, requires_grad=True)
    logits = model(x, t=0)
    loss = logits.sum()
    loss.backward()
    adapter = model.bridge.input_adapter._adapter
    assert adapter.weight.grad is not None
    assert x.grad is not None


def test_hat_grad_flow_2channel():
    """Gradients flow for 2-channel input (adapter is identity)."""
    args = _make_args()
    model = HatNet(n_inputs=1024, n_outputs=13, n_tasks=2, args=args)
    model.train()
    x = torch.randn(2, 2, 512, requires_grad=True)
    logits = model(x, t=0)
    loss = logits.sum()
    loss.backward()
    assert x.grad is not None


def test_hat_forward_leaves_bn_state_untouched():
    """HAT's forward no longer swaps BatchNorm state; task_bn owns that now."""
    args = _make_args()
    model = HatNet(n_inputs=1024, n_outputs=13, n_tasks=2, args=args)
    model.eval()
    model.current_task = 0

    bn_modules = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
    assert bn_modules
    for index, batch_norm_module in enumerate(bn_modules):
        batch_norm_module.running_mean.fill_(float(index) + 5.0)
        batch_norm_module.running_var.fill_(2.0)
        batch_norm_module.num_batches_tracked.fill_(10)
    before = [m.running_mean.detach().clone() for m in bn_modules]

    x = torch.randn(2, 2, 512)
    model(x, t=0)
    model(x, t=1)

    for batch_norm_module, running_mean in zip(bn_modules, before):
        assert torch.equal(batch_norm_module.running_mean, running_mean)


def test_hat_supports_task_specific_bn_install():
    """task_bn converts HAT's hard-coded BatchNorm1d layers and keeps them working."""
    args = _make_args()
    args.bn_mode = "task_specific"
    args.loader = "task_incremental_loader"
    args.model = "hat"
    args.norm_type = "batchnorm"

    model = HatNet(n_inputs=1024, n_outputs=13, n_tasks=2, args=args)
    layers = task_bn.install(model, args, num_tasks=2)
    assert layers, "HAT backbone should expose convertible BatchNorm1d layers"

    x0 = torch.randn(4, 2, 512)
    x1 = torch.randn(4, 2, 512) * 4.0 + 9.0

    model.train()
    task_bn.set_active_task(model, 0)
    for _ in range(3):
        model(x0, t=0)
    task0_mean = layers[0].task_running_mean[0].detach().clone()

    task_bn.set_active_task(model, 1)
    for _ in range(3):
        model(x1, t=1)

    # Task 0's statistics survive training on task 1, and the two rows differ.
    assert torch.equal(layers[0].task_running_mean[0], task0_mean)
    assert not torch.allclose(
        layers[0].task_running_mean[0], layers[0].task_running_mean[1]
    )


if __name__ == "__main__":
    test_hat_2channel_3d()
    test_hat_2channel_2d()
    test_hat_3channel_3d()
    test_hat_2channel_and_3channel_forward()
    test_hat_grad_flow_3channel()
    test_hat_grad_flow_2channel()
    test_hat_forward_leaves_bn_state_untouched()
    test_hat_supports_task_specific_bn_install()
    print("All HAT input adapter tests passed.")
