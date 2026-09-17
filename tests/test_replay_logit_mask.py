"""Training-time replay masking: CIL rows see every seen class, TIL rows their own block."""

import torch

from utils import misc_utils

CLASSES = [2, 3, 2]
N_OUT = sum(CLASSES)
CIL = "class_incremental_loader"
TIL = "task_incremental_loader"


def _active(masked):
    return (masked > -1e8).int()


def test_cil_masks_every_row_to_all_seen_classes():
    logits = torch.zeros(3, N_OUT)
    tasks = torch.tensor([0, 1, 1])
    masked = misc_utils.mask_replay_logits(
        logits, tasks, 1, CLASSES, N_OUT, loader=CIL
    )
    expected = torch.tensor([1, 1, 1, 1, 1, 0, 0]).expand(3, -1)
    assert torch.equal(_active(masked), expected)


def test_til_masks_each_row_to_its_own_task():
    logits = torch.zeros(3, N_OUT)
    tasks = torch.tensor([0, 1, 2])
    masked = misc_utils.mask_replay_logits(
        logits, tasks, 2, CLASSES, N_OUT, loader=TIL
    )
    expected = torch.tensor(
        [
            [1, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1],
        ]
    )
    assert torch.equal(_active(masked), expected)


def test_cil_replay_loss_penalises_newer_classes():
    """A replayed task-0 row scoring high on a task-1 class must be penalised."""
    logits = torch.tensor([[0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0]], requires_grad=True)
    tasks = torch.tensor([0])
    masked = misc_utils.mask_replay_logits(
        logits, tasks, 1, CLASSES, N_OUT, loader=CIL
    )
    loss = torch.nn.functional.cross_entropy(masked, torch.tensor([0]))
    loss.backward()
    assert logits.grad[0, 2] > 0
    assert logits.grad[0, 5] == 0


def test_input_is_not_modified():
    logits = torch.randn(2, N_OUT)
    before = logits.clone()
    misc_utils.mask_replay_logits(
        logits, torch.tensor([0, 2]), 2, CLASSES, N_OUT, loader=TIL
    )
    assert torch.equal(logits, before)
