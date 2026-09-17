"""Learning-without-Forgetting logit distillation as a reusable mixin.

``model.woe_si`` grew this term first, to make a 2x2 (parameter anchor x
function-space regulariser) run through a single code path with no
cross-implementation confound.  The same question applies to the other
quadratic-anchor learners in this repository -- ``model.si`` and
``model.rwalk`` -- so the mechanism lives here rather than being copied a third
and fourth time.

The numerics are those of ``model.lwf.Net._distillation_loss``: temperature
scaled KL between the student's and a frozen end-of-task teacher's softmax over
the columns of completed tasks, multiplied by ``T^2``.  Setting a host's own
anchor strength to 0 and the distillation weight to ``model.lwf``'s defaults
(1.0 / 5.0) therefore gives an LwF control that shares the host's optimiser,
task masking and BatchNorm handling.
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn

from utils import misc_utils


class LwfDistillationMixin:
    """Adds an optional LwF logit-distillation term to an anchor-based learner.

    The host must expose ``net`` (a ``ResNet1D``), ``classes_per_task``,
    ``n_outputs`` and ``_device()``.  Three hooks wire it up:
    ``_init_lwf_distillation`` in ``__init__``, ``_lwf_distillation_loss`` in
    the training step, and ``_snapshot_lwf_teacher`` at each task boundary.
    """

    # ------------------------------------------------------------------
    def _init_lwf_distillation(self, args: object | None, prefix: str) -> None:
        """Read ``<prefix>_lwf_lambda`` / ``<prefix>_lwf_temperature`` off ``args``.

        Args:
            args: Parsed arguments, or ``None`` for the defaults.
            prefix: Flag namespace of the host learner (``"si"``, ``"rwalk"``).

        Raises:
            ValueError: If the temperature is not positive.
        """
        self.lwf_prefix = prefix
        lwf_lambda = getattr(args, f"{prefix}_lwf_lambda", None)
        self.lwf_lambda = 0.0 if lwf_lambda is None else float(lwf_lambda)
        temperature = getattr(args, f"{prefix}_lwf_temperature", None)
        self.lwf_temperature = 5.0 if temperature is None else float(temperature)
        if self.lwf_temperature <= 0.0:
            raise ValueError(
                f"{prefix}_lwf_temperature must be positive, "
                f"got {self.lwf_temperature}"
            )
        self.lwf_kl = nn.KLDivLoss(reduction="batchmean")
        self.teacher: Optional[nn.Module] = None

    # ------------------------------------------------------------------
    def _previous_class_indices(self, t: int, device: torch.device) -> torch.Tensor:
        """Output columns of classes from *completed* tasks ``< t``.

        The cumulative prior-class span is ``[0, offset1)`` in both TIL and CIL,
        where ``offset1`` is the first column of the current task.  Returns an
        empty tensor on the first task.
        """
        offset1, _ = misc_utils.compute_offsets(t, self.classes_per_task)
        offset1 = min(self.n_outputs, offset1)
        indices = list(range(0, offset1))
        return torch.tensor(indices, dtype=torch.long, device=device)

    # ------------------------------------------------------------------
    def _lwf_distillation_loss(
        self, student_logits: torch.Tensor, x: torch.Tensor, t: int
    ) -> torch.Tensor:
        """Temperature-scaled KL against the frozen teacher on previous classes.

        Orthogonal to the host's quadratic anchor by construction: the anchor
        constrains *parameters*, this constrains the *function* at the readout,
        so both can be active at once.

        Args:
            student_logits: Unmasked class logits of the live network.
            x: The current batch, re-run through the frozen teacher.
            t: Current task index.  Returns 0 on the first task.

        Returns:
            Scalar distillation loss; exactly 0 before a teacher exists.
        """
        if self.teacher is None:
            return torch.zeros(1, device=student_logits.device)
        previous = self._previous_class_indices(t, student_logits.device)
        if previous.numel() == 0:
            return torch.zeros(1, device=student_logits.device)

        student_previous = student_logits.index_select(1, previous)
        with torch.no_grad():
            # bn_training=True scores the teacher on the *current batch's*
            # statistics rather than the running statistics it froze with. That
            # matches `model.lwf` (which calls `self.teacher(x)`, and ResNet1D's
            # forward runs `self.model.train(bn_training)`), and it is the right
            # choice here rather than an accident: consecutive tasks are
            # different radar datasets, so a teacher normalised with the
            # previous task's statistics is evaluated under distribution shift
            # and its targets are correspondingly degraded.
            #
            # `_snapshot_lwf_teacher` zeroes the teacher's BatchNorm momentum,
            # so this forward uses batch statistics *without* mutating the
            # frozen running buffers -- unlike `model.lwf`, where each
            # distillation pass updates them. Those buffers are never read in
            # this mode, so the numerics match `model.lwf` exactly; the teacher
            # simply stays genuinely frozen.
            teacher_logits = self.teacher(x, bn_training=True)
            teacher_probs = torch.softmax(
                teacher_logits.index_select(1, previous) / self.lwf_temperature,
                dim=1,
            )
        student_log_probs = torch.log_softmax(
            student_previous / self.lwf_temperature, dim=1
        )
        return self.lwf_kl(student_log_probs, teacher_probs) * (self.lwf_temperature**2)

    # ------------------------------------------------------------------
    def _snapshot_lwf_teacher(self) -> None:
        """Freeze the current net as the distillation teacher.

        Mirrors ``model.lwf.Net._update_teacher``: a ``deepcopy`` in eval mode
        with gradients disabled.  A no-op when distillation is switched off, so
        callers can invoke it unconditionally at a task boundary.
        """
        if self.lwf_lambda == 0.0:
            return
        self.teacher = copy.deepcopy(self.net)
        self.teacher.to(self._device())
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        # Zero the BatchNorm momentum so a `bn_training=True` teacher forward
        # normalises with batch statistics but leaves the running buffers
        # exactly where they were frozen: the update is
        # `(1 - momentum) * running + momentum * batch`. Without this, every
        # distillation pass would drift the "frozen" reference toward the
        # current task. `.eval()` alone cannot achieve it -- ResNet1D.forward
        # calls `self.model.train(bn_training)` and overrides module mode.
        for module in self.teacher.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.momentum = 0.0


__all__ = ["LwfDistillationMixin"]
