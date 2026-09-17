"""Regression: logit masking respects ``args.loader`` (TIL vs CIL)."""

# ruff: noqa: E402

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils import misc_utils


def test_til_loader_ignores_cil_arg() -> None:
    """Task-incremental loader must use per-task mask even if cil index is passed."""
    logits = torch.zeros(2, 20)
    logits[:, 3] = 1.0
    logits[:, 15] = 2.0
    classes_per_task = [5, 5, 5, 5]
    masked = misc_utils.apply_task_incremental_logit_mask(
        logits,
        task_index=0,
        nc_per_task=classes_per_task,
        n_outputs=20,
        cil_all_seen_upto_task=3,
        loader="task_incremental_loader",
    )
    assert torch.all(masked[:, 0:5] == logits[:, 0:5])
    assert torch.all(masked[:, 5:] <= -1e8)


def test_cil_loader_uses_cumulative_mask() -> None:
    logits = torch.arange(24, dtype=torch.float32).view(1, 24)
    classes_per_task = [6, 6, 6, 6]
    masked = misc_utils.apply_task_incremental_logit_mask(
        logits,
        task_index=0,
        nc_per_task=classes_per_task,
        n_outputs=24,
        cil_all_seen_upto_task=1,
        loader="class_incremental_loader",
    )
    assert torch.allclose(masked[0, 0:12], logits[0, 0:12])
    assert masked[0, 12].item() <= -1e8
