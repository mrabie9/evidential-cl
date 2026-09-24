from __future__ import annotations

from typing import Dict, Sequence, Tuple, Union

import torch
import torch.nn as nn

from utils.misc_utils import deinterleave_iq_last_axis


def classification_loss_zero_stub(cls_logits: torch.Tensor) -> torch.Tensor:
    """Scalar zero loss tied to logits (keeps autograd on an empty CE minibatch).

    Args:
        cls_logits: Classifier logits the returned loss should stay attached to.

    Returns:
        Zero-valued scalar tensor that still participates in the graph.

    Usage:
        loss = classification_loss_zero_stub(logits)
    """
    return cls_logits.sum() * 0.0


def unpack_y_to_class_labels(
    y: Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]],
) -> torch.Tensor:
    """Extract 1D class labels from the label payloads dataloaders emit.

    Args:
        y: Label batch as a tensor, a ``{"y_cls": ...}`` mapping, or a tuple
            whose first element holds the class ids.

    Returns:
        1D tensor of class ids.

    Usage:
        y_cls = unpack_y_to_class_labels(batch_y)
    """
    if isinstance(y, (tuple, list)) and len(y) == 2:
        y_cls = y[0]
    elif isinstance(y, dict):
        y_cls = y.get("y_cls", y.get("y"))
    else:
        y_cls = y
    if not torch.is_tensor(y_cls):
        y_cls = torch.as_tensor(y_cls)
    return y_cls


def build_task_offsets_table(
    task_class_offsets: Sequence[Tuple[int, int]],
    device: Union[torch.device, str, None] = None,
) -> torch.Tensor:
    """Stack per-task ``(start, end)`` class offsets into one lookup tensor.

    Args:
        task_class_offsets: ``(start, end)`` global class range for each task,
            in task order.
        device: Device to place the table on.

    Returns:
        Long tensor of shape ``(n_tasks, 2)``.

    Usage:
        offsets = build_task_offsets_table([(0, 4), (4, 8)], device="cuda")
    """
    return torch.tensor(list(task_class_offsets), dtype=torch.long, device=device)


def task_class_gather_index(
    task_indices: torch.Tensor,
    task_offsets_table: torch.Tensor,
    n_columns: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the per-row index that gathers each row's own task class block.

    Row ``r`` gets ``start_r, start_r + 1, ..., end_r - 1`` followed by zeros
    up to ``n_columns``; the zero columns are placeholders to be masked with
    :func:`mask_padded_class_logits`.

    Args:
        task_indices: Long tensor ``(B,)`` of task ids, one per row.
        task_offsets_table: Table from :func:`build_task_offsets_table`.
        n_columns: Width of the gathered block (the largest task class count).

    Returns:
        Tuple of the long gather index ``(B, n_columns)`` and the per-row class
        counts ``(B,)``.

    Usage:
        gather_index, class_counts = task_class_gather_index(t_idx, offsets, 4)
        block_logits = torch.gather(full_logits, 1, gather_index)
    """
    row_offsets = task_offsets_table[task_indices]
    class_counts = row_offsets[:, 1] - row_offsets[:, 0]
    column_positions = torch.arange(n_columns, device=task_offsets_table.device)
    gather_index = row_offsets[:, :1] + column_positions.unsqueeze(0)
    valid_columns = column_positions.unsqueeze(0) < class_counts.unsqueeze(1)
    gather_index = torch.where(
        valid_columns, gather_index, torch.zeros_like(gather_index)
    )
    return gather_index, class_counts


def mask_padded_class_logits(
    block_logits: torch.Tensor,
    class_counts: torch.Tensor,
    fill_value: float = -1e9,
) -> torch.Tensor:
    """Fill the padding columns beyond each row's class count.

    Args:
        block_logits: Gathered logits ``(B, n_columns)``.
        class_counts: Number of real classes per row ``(B,)``.
        fill_value: Value written into padding columns.

    Returns:
        New tensor with columns ``>= class_counts[r]`` set to ``fill_value``.

    Usage:
        block_logits = mask_padded_class_logits(block_logits, class_counts)
    """
    column_positions = torch.arange(block_logits.size(1), device=block_logits.device)
    padding = column_positions.unsqueeze(0) >= class_counts.unsqueeze(1)
    return block_logits.masked_fill(padding, fill_value)


class ReplayInputMixin:
    """Canonical input reshaping shared by replay-buffer based learners."""

    def _canonicalize_input(
        self,
        x: torch.Tensor,
        *,
        detach: bool,
    ) -> torch.Tensor:
        """Convert inputs to canonical replay/training shape.

        Canonical shape is typically `(B, 2, 512)` for IQ inputs after optional
        3-ADC adaptation.

        Args:
            x: Input batch in one of the supported formats.
            detach: Whether to detach from graph and disable gradient flow.

        Returns:
            Canonicalized tensor suitable for replay buffers or training.

        Usage:
            x = self._canonicalize_input(batch_x, detach=False)
        """
        if detach:
            x = x.detach()

        if x.dim() == 2:
            batch, features = x.shape
            if features % 2 == 0 and features % 3 != 0:
                x = deinterleave_iq_last_axis(x)
            return x

        if x.dim() == 3 and x.size(1) == 3:
            # (B, 3, 1024) -> (B, 3, 2, 512) then adapter
            batch, _, sequence_length = x.shape
            if sequence_length % 2 != 0:
                return x
            x = deinterleave_iq_last_axis(x)
            # fall through to 4D adapter path

        if x.dim() == 4 and x.size(1) == 3 and x.size(2) == 2:
            adapter = getattr(self, "net", None) and (
                getattr(self.net, "input_adapter", None)
                or getattr(getattr(self.net, "model", None), "input_adapter", None)
            )
            if adapter is not None and not isinstance(adapter, nn.Identity):
                sequence_length = x.size(3)
                if sequence_length > 512:
                    x = x[:, :, :, :512].contiguous()
                return adapter(x)
        return x

    def _input_for_replay(self, x: torch.Tensor) -> torch.Tensor:
        """Return detached canonical input for replay-buffer storage.

        Args:
            x: Input batch to store.

        Returns:
            Detached, canonicalized tensor.

        Usage:
            self.M.append([self._input_for_replay(x)[i], y[i], t])
        """
        with torch.no_grad():
            return self._canonicalize_input(x, detach=True)
