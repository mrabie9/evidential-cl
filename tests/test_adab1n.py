import os
import sys
from types import SimpleNamespace

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.adab1n import AdaB1N
from model.resnet1d import ADAB1N_MAX_TASKS, ResNet1D


def test_adab1n_train_eval_shapes():
    norm = AdaB1N(num_features=8, num_tasks=2)
    x = torch.randn(6, 8, 20)

    norm.train()
    out_train = norm(x)
    assert out_train.shape == x.shape

    norm.eval()
    out_eval = norm(x)
    assert out_eval.shape == x.shape


def test_adab1n_matches_plain_batchnorm_without_counts():
    torch.manual_seed(0)
    channels = 4
    x = torch.randn(10, channels, 16)

    ada = AdaB1N(num_features=channels, num_tasks=1, kappa=1.0)
    plain = torch.nn.BatchNorm1d(channels)
    with torch.no_grad():
        plain.weight.copy_(ada.weight)
        plain.bias.copy_(ada.bias)

    ada.train()
    plain.train()
    out_ada = ada(x)
    out_plain = plain(x)
    # Both use biased (population) variance to normalize the current batch,
    # so the forward output and the running_mean EMA line up exactly. The
    # running_var buffer does not: torch.nn.BatchNorm1d applies Bessel's
    # correction to its EMA update while AdaB1N (matching upstream AdaB2N)
    # does not, so the two running_var buffers are expected to diverge
    # slightly and are not compared here.
    assert torch.allclose(out_ada, out_plain, atol=1e-5)
    assert torch.allclose(ada.running_mean, plain.running_mean, atol=1e-5)


def test_adab1n_task_reweighted_forward():
    channels = 4
    norm = AdaB1N(num_features=channels, num_tasks=2)
    norm.train()

    x0 = torch.randn(5, channels, 10)
    norm(x0)
    norm.end_task()

    x1 = torch.randn(5, channels, 10)
    sample_task_indices = torch.tensor([0, 0, 1, 1, 1])
    sample_task_counts = torch.tensor([2, 2, 3, 3, 3])
    task_counts_extended = torch.tensor([2, 3])
    norm.set_counts(sample_task_indices, sample_task_counts, task_counts_extended)

    out = norm(x1)
    assert out.shape == x1.shape
    assert torch.isfinite(out).all()


def test_adab1n_end_task_bounds_checked():
    norm = AdaB1N(num_features=4, num_tasks=1)
    try:
        norm.end_task()
    except ValueError as exc:
        assert "num_tasks" in str(exc)
    else:
        raise AssertionError("Expected ValueError when exceeding num_tasks.")


def test_resnet1d_builds_with_adab1n_norm_type():
    args = SimpleNamespace(norm_type="adab1n", n_tasks=3, kappa=1.0)
    model = ResNet1D(num_classes=4, args=args)
    norm_modules = [m for m in model.modules() if isinstance(m, AdaB1N)]
    assert len(norm_modules) > 0
    # num_tasks is a ceiling, not an exact count: unused task_weight entries are
    # sliced out of the forward, so the backbone allocates ADAB1N_MAX_TASKS and
    # only grows it when an experiment declares more tasks than that.
    assert all(m.num_tasks == ADAB1N_MAX_TASKS for m in norm_modules)
    assert all(m.num_tasks >= args.n_tasks for m in norm_modules)

    x = torch.randn(2, 2, 32)
    out = model(x)
    assert out.shape == (2, 4)


if __name__ == "__main__":
    test_adab1n_train_eval_shapes()
    test_adab1n_matches_plain_batchnorm_without_counts()
    test_adab1n_task_reweighted_forward()
    test_adab1n_end_task_bounds_checked()
    test_resnet1d_builds_with_adab1n_norm_type()
    print("All AdaB1N tests passed.")
