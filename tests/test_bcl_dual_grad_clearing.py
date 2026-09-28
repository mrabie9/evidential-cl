"""BCL-Dual's outer (validation) step must not re-apply the inner gradient.

``observe`` takes an inner step on the current batch and then an outer step on the
validation buffer. Gradients used to be left in ``.grad`` between the two, so the outer
step applied ``grad(inner loss) + grad(outer loss)``: the inner gradient twice, and
a "validation" step that was mostly a second current-batch step.
"""

# ruff: noqa: E402

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.bcl_dual import Net as BclDualNet
from tests.test_adapter_training_paths import _labels, _make_args


def _grads_are_clear(model: torch.nn.Module) -> bool:
    return all(p.grad is None or not torch.any(p.grad) for p in model.parameters())


def _spy_outer_step(model: BclDualNet) -> list:
    """Record, at each outer-step sampling call, whether grads were already clear."""
    seen = []
    original = model.memory_sampling

    def spy(t, valid=False):
        if valid:
            seen.append(_grads_are_clear(model))
        return original(t, valid=valid)

    model.memory_sampling = spy
    return seen


def _model(**overrides) -> BclDualNet:
    args = _make_args()
    for key, value in overrides.items():
        setattr(args, key, value)
    model = BclDualNet(n_inputs=1024, n_outputs=12, n_tasks=2, args=args)
    model.train()
    return model


def test_outer_step_starts_from_clear_gradients():
    torch.manual_seed(0)
    model = _model()
    seen = _spy_outer_step(model)

    for t in (0, 0, 1, 1):
        model.observe(torch.randn(8, 3, 1024), _labels(8, task_id=t), t)

    assert seen, "the outer step never ran"
    assert all(seen), "stale inner-step gradients reached the outer step"
