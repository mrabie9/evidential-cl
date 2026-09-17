"""Smoke checks for the node/channel-tied UCL Bayesian backbone.

These tests confirm the per-node (linear) and per-channel (conv) sigma tying
introduced to match Ahn et al. 2019, and that both task-incremental and
class-incremental configurations still complete one ``observe`` step and one
evaluation ``forward`` with finite losses and correctly shaped logits.
"""

# ruff: noqa: E402

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.ucl_bresnet import BayesianConv1d, BayesianLinear
from model.ucl_bresnet import Net as UclBresnetNet


def _make_args(loader: str, classes_per_task=None):
    """Build a minimal args-like object for a UCL ``Net``.

    Args:
        loader: Incremental loader name selecting TIL vs CIL behaviour.
        classes_per_task: Optional per-task class counts.

    Returns:
        An attribute namespace consumable by ``Net.__init__``.
    """
    if classes_per_task is None:
        classes_per_task = [6, 6]
    args = type("Args", (), {})()
    args.classes_per_task = classes_per_task
    args.nc_per_task_list = ""
    args.nc_per_task = None
    args.loader = loader
    args.cuda = False
    args.lr = 1e-3
    args.clipgrad = 5.0
    args.class_weighted_ce = False
    args.inner_steps = 1
    return args


def _labels(batch_size: int, task_id: int, classes_per_task: int = 6) -> torch.Tensor:
    start = task_id * classes_per_task
    return (torch.arange(batch_size) % classes_per_task) + start


def test_tied_sigma_shapes():
    """Linear ties sigma per output node; conv ties per output channel."""
    linear = BayesianLinear(in_features=32, out_features=8)
    assert linear.weight_mu.shape == (8, 32)
    assert linear.weight_rho.shape == (8, 1)
    assert linear.weight_sigma.shape == (8, 1)

    conv = BayesianConv1d(in_channels=4, out_channels=6, kernel_size=3, padding=1)
    assert conv.weight_mu.shape == (6, 4, 3)
    assert conv.weight_rho.shape == (6, 1, 1)
    assert conv.weight_sigma.shape == (6, 1, 1)


def test_sampling_broadcasts_tied_sigma():
    """``Gaussian.sample`` keeps i.i.d. noise per weight with node-shared sigma."""
    linear = BayesianLinear(in_features=16, out_features=4)
    sample = linear.weight.sample()
    assert sample.shape == linear.weight_mu.shape
    assert torch.isfinite(sample).all()


def test_til_observe_and_eval_smoke():
    """TIL config completes one observe step and one eval forward."""
    args = _make_args("task_incremental_loader")
    model = UclBresnetNet(n_inputs=1024, n_outputs=12, n_tasks=2, args=args)

    x = torch.randn(8, 2, 1024)
    y = _labels(8, task_id=0)
    loss, recall, logits = model.observe(x, y, t=0)
    assert torch.isfinite(torch.tensor(loss))
    assert 0.0 <= recall <= 1.0
    assert logits is not None and logits.shape == (8, 6)

    model.eval()
    with torch.no_grad():
        eval_logits = model.forward(x, t=0)
    assert eval_logits.shape == (8, 6)
    assert torch.isfinite(eval_logits).all()


def test_cil_observe_and_eval_smoke():
    """CIL config completes one observe step and one eval forward."""
    args = _make_args("class_incremental_loader")
    model = UclBresnetNet(n_inputs=1024, n_outputs=12, n_tasks=2, args=args)

    x = torch.randn(8, 2, 1024)
    y = _labels(8, task_id=0)
    loss, recall, logits = model.observe(x, y, t=0)
    assert torch.isfinite(torch.tensor(loss))
    assert 0.0 <= recall <= 1.0
    assert logits is not None and logits.shape[0] == 8

    model.eval()
    with torch.no_grad():
        eval_logits = model.forward(x, t=0)
    assert eval_logits.shape[0] == 8
    assert torch.isfinite(eval_logits).all()


def test_regularisation_finite_across_task_boundary():
    """Crossing a task boundary activates UCL reg with a finite total loss."""
    args = _make_args("task_incremental_loader")
    model = UclBresnetNet(n_inputs=1024, n_outputs=12, n_tasks=2, args=args)

    model.observe(torch.randn(8, 2, 1024), _labels(8, task_id=0), t=0)
    loss, _, _ = model.observe(torch.randn(8, 2, 1024), _labels(8, task_id=1), t=1)
    assert model.saved
    assert torch.isfinite(torch.tensor(loss))


if __name__ == "__main__":
    test_tied_sigma_shapes()
    test_sampling_broadcasts_tied_sigma()
    test_til_observe_and_eval_smoke()
    test_cil_observe_and_eval_smoke()
    test_regularisation_finite_across_task_boundary()
    print("All UCL node-tied sigma smoke checks passed.")
