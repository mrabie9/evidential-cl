"""Regression check for adapter-graph reuse in CTN and LA-ER.

This script runs small synthetic 3-ADC batches through `observe()` for:
- `model.ctn.Net`
- `model.eralg4.Net` (la-er)

and fails if either path raises the classic PyTorch error:
"Trying to backward through the graph a second time".
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

# Ensure repo root is importable when executed as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.ctn import Net as CtnNet
from model.eralg4 import Net as LaErNet


def build_common_args() -> SimpleNamespace:
    """Build common synthetic runtime args.

    Returns:
        SimpleNamespace: Minimal argument set used by both models.
    """
    return SimpleNamespace(
        arch="resnet1d",
        cuda=False,
        class_weighted_ce=True,
        data_scaling="normalize",
        use_iq_aug_features=False,
        iq_aug_feature_type="power",
        nc_per_task=4,
        classes_per_task=None,
    )


def build_ctn_args() -> SimpleNamespace:
    """Build CTN args for regression exercise.

    Returns:
        SimpleNamespace: Args compatible with `model.ctn.Net`.
    """
    args = build_common_args()
    args.memory_strength = 0.5
    args.temperature = 5.0
    args.task_emb = 16
    args.lr = 0.01
    args.ctx_lr = 0.01
    args.n_memories = 64
    args.validation = 0.2
    args.replay_batch_size = 16
    # Former 2×2 grid folds to four alternating rounds (see CtnConfig.from_args).
    args.inner_steps = 4
    args.batch_size = 8
    args.cls_lambda = 1.0
    return args


def build_laer_args() -> SimpleNamespace:
    """Build LA-ER args for regression exercise.

    Returns:
        SimpleNamespace: Args compatible with `model.eralg4.Net`.
    """
    args = build_common_args()
    args.alpha_init = 1e-3
    args.lr = 1e-2
    args.opt_lr = 1e-1
    args.learn_lr = True
    args.inner_steps = 1
    args.memories = 64
    args.replay_batch_size = 16
    args.grad_clip_norm = 2.0
    args.second_order = True
    args.meta_batches = 2
    args.dataset = "iq"
    args.cls_lambda = 1.0
    return args


def run_observe_regression() -> None:
    """Run the regression and raise on failure.

    Raises:
        RuntimeError: If either model hits a graph-reuse backward error.
    """
    torch.manual_seed(0)
    x = torch.randn(8, 3, 1024)
    y = torch.randint(0, 4, (8,), dtype=torch.long)

    ctn = CtnNet(n_inputs=1024, n_outputs=8, n_tasks=2, args=build_ctn_args())
    laer = LaErNet(n_inputs=1024, n_outputs=8, n_tasks=2, args=build_laer_args())

    for model_name, model in (("ctn", ctn), ("la-er", laer)):
        try:
            model.observe(x, y, 0)
        except RuntimeError as exc:
            message = str(exc)
            if "Trying to backward through the graph a second time" in message:
                raise RuntimeError(
                    f"{model_name} observe() still reuses a freed autograd graph."
                ) from exc
            raise


def test_observe_no_double_backward() -> None:
    """Pytest entrypoint: same checks as ``python tests/test_graph_reuse_regression.py``."""
    run_observe_regression()


if __name__ == "__main__":
    run_observe_regression()
    print("PASS: no graph-reuse backward error in CTN/LA-ER synthetic observe.")
