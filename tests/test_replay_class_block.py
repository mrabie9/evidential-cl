"""Vectorised replay class-block helpers match the per-row loops they replaced."""

import torch

from model.replay_utils import (
    build_task_offsets_table,
    mask_padded_class_logits,
    task_class_gather_index,
)
from utils import misc_utils


def _reference_loop(
    full_logits: torch.Tensor,
    task_indices: torch.Tensor,
    classes_per_task: list[int],
    n_columns: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Original CTN / BCL-dual per-row gather and padding, kept as the oracle."""
    offsets = torch.tensor(
        [
            misc_utils.compute_offsets(int(task), classes_per_task)
            for task in task_indices.tolist()
        ]
    )
    mask = torch.zeros(task_indices.numel(), n_columns)
    for row in range(mask.size(0)):
        class_size = offsets[row][1] - offsets[row][0]
        mask[row, :class_size] = torch.arange(offsets[row][0], offsets[row][1])
    mask = mask.long()
    sizes = offsets[:, 1] - offsets[:, 0]
    block_logits = torch.gather(full_logits, 1, mask)
    for row, size in enumerate(sizes):
        if size < block_logits.size(1):
            block_logits[row, size:] = -1e9
    return mask, block_logits


def test_gather_index_and_padding_match_reference_loop() -> None:
    """Uneven class counts exercise both the index and the padding columns."""
    torch.manual_seed(0)
    classes_per_task = [4, 3, 5, 2, 6]
    n_columns = max(classes_per_task)
    offsets_table = build_task_offsets_table(
        [
            misc_utils.compute_offsets(task, classes_per_task)
            for task in range(len(classes_per_task))
        ]
    )
    task_indices = torch.randint(0, len(classes_per_task), (64,))
    full_logits = torch.randn(64, sum(classes_per_task))

    expected_mask, expected_logits = _reference_loop(
        full_logits, task_indices, classes_per_task, n_columns
    )
    gather_index, class_counts = task_class_gather_index(
        task_indices, offsets_table, n_columns
    )
    block_logits = mask_padded_class_logits(
        torch.gather(full_logits, 1, gather_index), class_counts
    )

    assert torch.equal(gather_index, expected_mask)
    assert torch.equal(block_logits, expected_logits)
