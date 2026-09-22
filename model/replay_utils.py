from __future__ import annotations

from typing import Dict, Tuple, Union

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
