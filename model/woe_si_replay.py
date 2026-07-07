"""Weight-of-Evidence Synaptic Intelligence with reservoir experience replay.

``woe_si_replay`` is the WoE-SI continual learner (:mod:`model.woe_si`) augmented
with a rehearsal buffer. WoE-SI on its own is a *regularisation* method: it anchors
parameters via the Dempster-Shafer information-content path integral. This variant
adds an orthogonal *rehearsal* signal - an Experience Replay (Chaudhry et al. 2019,
"On Tiny Episodic Memories in Continual Learning") cross-entropy term computed on a
reservoir-sampled minibatch of previously-seen examples.

The two mechanisms are complementary and stack cleanly:

* the DS importance regularisation (inherited unchanged from :class:`model.woe_si.Net`)
  discourages drift of parameters that built committed evidence, and
* the replay CE term directly rehearses old class boundaries on stored raw inputs.

Only two hooks of the base learner are overridden -
:meth:`_classification_replay_loss` (adds the replay CE to the training loss) and
:meth:`_store_classification_replay` (writes each observed batch into the buffer).
Everything else - the ``I_2(m)`` importance path integral, consolidation, the
``parameter``/``channel``/``output`` regularisation levels - is reused verbatim.

The reservoir buffer is a single stream-wide buffer (not per-task), maintained with
Vitter's Algorithm R via :func:`utils.misc_utils.reservoir_slots`, so every stream
item is retained with equal probability ``capacity / seen``. Replay logits are
task-masked per stored sample so each rehearsed example is scored only over the
classes of the task it came from (matching ``model.eralg4``).

Config: ``configs/models/til/woe_si_replay.yaml``. Extra knobs:
``woe_replay_memories`` (buffer capacity), ``woe_replay_batch_size`` (replay draw
per step) and ``woe_replay_lambda`` (replay CE weight).
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import torch

from model.detection_replay import unpack_y_to_class_labels
from model.woe_si import Net as WoeSiNet
from utils import misc_utils
from utils.class_weighted_loss import classification_cross_entropy


class ReservoirReplayBuffer:
    """Fixed-capacity reservoir buffer of ``(input, label, task)`` stream items.

    Uses :func:`utils.misc_utils.reservoir_slots` (Vitter's Algorithm R) so that,
    after observing ``seen`` items, each is retained with probability
    ``capacity / seen``. Occupied slots stay dense in ``[0, len(self))`` so a draw
    can index the storage lists directly.

    Args:
        capacity: Maximum number of stored items (``<= 0`` disables the buffer).

    Usage:
        >>> buffer = ReservoirReplayBuffer(1000)
        >>> buffer.add(inputs, labels, task_id=0)
        >>> draw = buffer.sample(32)
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.inputs: List[torch.Tensor] = []
        self.labels: List[int] = []
        self.tasks: List[int] = []
        self.seen = 0

    def __len__(self) -> int:
        return len(self.inputs)

    def add(self, inputs: torch.Tensor, labels: torch.Tensor, task_id: int) -> None:
        """Insert a batch, respecting the reservoir replacement policy.

        Args:
            inputs: Batch of stored-ready inputs, shape ``(batch, ...)`` (CPU).
            labels: Integer class labels, shape ``(batch,)``.
            task_id: Task index the batch belongs to.
        """
        if self.capacity <= 0 or inputs.size(0) == 0:
            return
        slots, _filled, seen = misc_utils.reservoir_slots(
            inputs.size(0), len(self.inputs), self.seen, self.capacity
        )
        self.seen = seen
        inputs_cpu = inputs.detach().cpu()
        labels_cpu = labels.detach().cpu().long()
        for index, slot in enumerate(slots):
            if slot < 0:
                continue
            input_item = inputs_cpu[index].clone()
            label_item = int(labels_cpu[index].item())
            if slot < len(self.inputs):
                self.inputs[slot] = input_item
                self.labels[slot] = label_item
                self.tasks[slot] = int(task_id)
            else:
                # Fill phase: reservoir_slots hands back the next dense index.
                self.inputs.append(input_item)
                self.labels.append(label_item)
                self.tasks.append(int(task_id))

    def sample(
        self, batch_size: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Draw up to ``batch_size`` items uniformly without replacement.

        Args:
            batch_size: Requested number of replay items.

        Returns:
            Tuple ``(inputs, labels, tasks)`` as stacked tensors, or ``None`` when
            the buffer is empty or ``batch_size <= 0``.
        """
        if not self.inputs or batch_size <= 0:
            return None
        draw = min(int(batch_size), len(self.inputs))
        indices = random.sample(range(len(self.inputs)), draw)
        inputs = torch.stack([self.inputs[i] for i in indices])
        labels = torch.tensor([self.labels[i] for i in indices], dtype=torch.long)
        tasks = torch.tensor([self.tasks[i] for i in indices], dtype=torch.long)
        return inputs, labels, tasks


class Net(WoeSiNet):
    """WoE-SI learner with an added reservoir experience-replay CE term.

    Inherits the entire WoE-SI training loop and DS-importance machinery from
    :class:`model.woe_si.Net`; only the two rehearsal hooks are overridden.
    """

    def __init__(
        self, n_inputs: int, n_outputs: int, n_tasks: int, args: object
    ) -> None:
        super().__init__(n_inputs, n_outputs, n_tasks, args)
        self.replay_memories = int(getattr(args, "woe_replay_memories", 5120))
        self.replay_batch_size = int(getattr(args, "woe_replay_batch_size", 20))
        self.replay_lambda = float(getattr(args, "woe_replay_lambda", 1.0))
        self.replay_buffer = ReservoirReplayBuffer(self.replay_memories)

    # ------------------------------------------------------------------
    def _classification_replay_loss(self, t: int) -> torch.Tensor:
        """Reservoir-replay cross-entropy over previously-seen batches.

        Draws a replay minibatch, forwards it through the classification head,
        masks each sample's logits to its own task, and returns the
        ``woe_replay_lambda``-scaled CE. Returns ``0`` while the buffer is empty
        (e.g. the very first step), so task 0 sees no replay contribution.
        """
        del t
        device = self._device()
        if self.replay_lambda == 0.0 or len(self.replay_buffer) == 0:
            return torch.zeros(1, device=device)
        sample = self.replay_buffer.sample(self.replay_batch_size)
        if sample is None:
            return torch.zeros(1, device=device)

        replay_x, replay_y, replay_t = sample
        replay_x = replay_x.to(device)
        replay_y = replay_y.to(device).long()
        cls_logits = self.net.forward_heads(replay_x)[1]
        masked_logits = self._mask_replay_logits(cls_logits, replay_t)
        replay_ce = classification_cross_entropy(
            masked_logits,
            replay_y,
            class_weighted_ce=self.class_weighted_ce,
        )
        return self.replay_lambda * replay_ce

    # ------------------------------------------------------------------
    def _store_classification_replay(
        self, x: torch.Tensor, y: torch.Tensor, t: int
    ) -> None:
        """Add the observed batch to the reservoir buffer (canonicalised input)."""
        if self.replay_buffer.capacity <= 0:
            return
        y_cls = unpack_y_to_class_labels(y).long()
        stored_x = self._input_for_replay(x)
        self.replay_buffer.add(stored_x, y_cls, int(t))

    # ------------------------------------------------------------------
    def _mask_replay_logits(
        self, logits: torch.Tensor, tasks: torch.Tensor
    ) -> torch.Tensor:
        """Task-mask replay logits per stored sample (mirrors ``model.eralg4``).

        Each replay sample is scored only over the classes of the task it was
        drawn from, so cross-task samples in one draw are masked independently.

        Args:
            logits: Raw classification logits ``(replay_batch, n_outputs)``.
            tasks: Per-sample task ids ``(replay_batch,)``.

        Returns:
            Logits with each row masked to its sample's task classes.
        """
        if not self.is_task_incremental or tasks.numel() == 0:
            return logits
        masked = logits.clone()
        for task_id in torch.unique(tasks).tolist():
            row_selector = tasks == int(task_id)
            masked[row_selector] = misc_utils.apply_task_incremental_logit_mask(
                logits[row_selector],
                int(task_id),
                self.classes_per_task,
                self.n_outputs,
                cil_all_seen_upto_task=int(task_id),
                global_noise_label=self.noise_label,
                loader=self.incremental_loader_name,
            )
        return masked


__all__ = ["Net", "ReservoirReplayBuffer"]
