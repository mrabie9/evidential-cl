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

``woe_replay_mode`` selects what the reservoir contributes. Beyond the default
cross-entropy rehearsal, the buffer can additionally snapshot each stored item's DS
total evidence ``(w_plus, w_minus)`` at insertion time and later penalise that
evidence having *decayed*::

    L = mean_b sum_k [ relu(w+_stored - w+_now)^2 + relu(w-_now - w-_stored)^2 ]

Only deterioration is charged; the model is free to become *more* certain about a
rehearsed item, so the term never competes for capacity it does not need. This is a
functional constraint like ``woe_reg_level='output'``, but anchored to per-sample
evidence recorded when the sample was seen rather than to a frozen end-of-task
teacher network -- and, unlike output mode, the snapshot and the recomputation
centre with the *same* per-task feature mean, so an unchanged network scores exactly
zero penalty.

The evidence snapshot is stored *alongside* the input, not instead of it: penalising
decay requires re-evaluating the current model on the stored item, which needs the
input. The extra cost is ``2 * n_outputs`` floats per item against an input of a few
thousand, so it is a few percent of buffer memory.

Config: ``configs/models/til/woe_si_replay.yaml``. Extra knobs:
``woe_replay_memories`` (buffer capacity), ``woe_replay_batch_size`` (replay draw
per step), ``woe_replay_lambda`` (replay CE weight), ``woe_replay_mode``
(``ce``/``evidence``/``both``) and ``woe_evidence_lambda`` (evidence-decay weight).
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import torch

from model.detection_replay import unpack_y_to_class_labels
from model.woe_si import Net as WoeSiNet, per_class_total_evidence
from utils import misc_utils
from utils.class_weighted_loss import classification_cross_entropy

# What the reservoir contributes to the training loss:
#   "ce"       -- cross-entropy on rehearsed samples (the original behaviour).
#   "evidence" -- one-sided penalty on the DS total evidence of rehearsed samples
#                 falling below what it was when they were stored.
#   "both"     -- the sum of the two.
_REPLAY_MODES = ("ce", "evidence", "both")


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

    def __init__(self, capacity: int, store_evidence: bool = False) -> None:
        self.capacity = int(capacity)
        self.store_evidence = bool(store_evidence)
        self.inputs: List[torch.Tensor] = []
        self.labels: List[int] = []
        self.tasks: List[int] = []
        # Per-item snapshot of the DS total evidence at insertion time, stored at
        # full ``n_outputs`` width so class indices line up across tasks. Empty
        # unless ``store_evidence``.
        self.evidence_plus: List[torch.Tensor] = []
        self.evidence_minus: List[torch.Tensor] = []
        self.seen = 0

    def __len__(self) -> int:
        return len(self.inputs)

    def add(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        task_id: int,
        evidence_plus: Optional[torch.Tensor] = None,
        evidence_minus: Optional[torch.Tensor] = None,
    ) -> None:
        """Insert a batch, respecting the reservoir replacement policy.

        Args:
            inputs: Batch of stored-ready inputs, shape ``(batch, ...)`` (CPU).
            labels: Integer class labels, shape ``(batch,)``.
            task_id: Task index the batch belongs to.
            evidence_plus: Optional ``(batch, n_outputs)`` snapshot of ``w_plus``
                at insertion time. Required when the buffer stores evidence.
            evidence_minus: Optional ``(batch, n_outputs)`` snapshot of ``w_minus``.
        """
        if self.capacity <= 0 or inputs.size(0) == 0:
            return
        if self.store_evidence and (evidence_plus is None or evidence_minus is None):
            raise ValueError(
                "buffer was built with store_evidence=True but add() received no "
                "evidence snapshot"
            )
        slots, _filled, seen = misc_utils.reservoir_slots(
            inputs.size(0), len(self.inputs), self.seen, self.capacity
        )
        self.seen = seen
        inputs_cpu = inputs.detach().cpu()
        labels_cpu = labels.detach().cpu().long()
        plus_cpu = evidence_plus.detach().cpu() if self.store_evidence else None
        minus_cpu = evidence_minus.detach().cpu() if self.store_evidence else None
        for index, slot in enumerate(slots):
            if slot < 0:
                continue
            input_item = inputs_cpu[index].clone()
            label_item = int(labels_cpu[index].item())
            if slot < len(self.inputs):
                self.inputs[slot] = input_item
                self.labels[slot] = label_item
                self.tasks[slot] = int(task_id)
                if self.store_evidence:
                    self.evidence_plus[slot] = plus_cpu[index].clone()
                    self.evidence_minus[slot] = minus_cpu[index].clone()
            else:
                # Fill phase: reservoir_slots hands back the next dense index.
                self.inputs.append(input_item)
                self.labels.append(label_item)
                self.tasks.append(int(task_id))
                if self.store_evidence:
                    self.evidence_plus.append(plus_cpu[index].clone())
                    self.evidence_minus.append(minus_cpu[index].clone())

    def sample(self, batch_size: int, with_evidence: bool = False):
        """Draw up to ``batch_size`` items uniformly without replacement.

        Args:
            batch_size: Requested number of replay items.
            with_evidence: When ``True``, also return the stored ``(w_plus,
                w_minus)`` snapshots. Requires ``store_evidence``.

        Returns:
            Tuple ``(inputs, labels, tasks)``, or ``(inputs, labels, tasks,
            evidence_plus, evidence_minus)`` when ``with_evidence``. ``None`` when
            the buffer is empty or ``batch_size <= 0``.
        """
        if not self.inputs or batch_size <= 0:
            return None
        if with_evidence and not self.store_evidence:
            raise ValueError(
                "sample(with_evidence=True) needs a buffer built with "
                "store_evidence=True"
            )
        draw = min(int(batch_size), len(self.inputs))
        indices = random.sample(range(len(self.inputs)), draw)
        inputs = torch.stack([self.inputs[i] for i in indices])
        labels = torch.tensor([self.labels[i] for i in indices], dtype=torch.long)
        tasks = torch.tensor([self.tasks[i] for i in indices], dtype=torch.long)
        if not with_evidence:
            return inputs, labels, tasks
        plus = torch.stack([self.evidence_plus[i] for i in indices])
        minus = torch.stack([self.evidence_minus[i] for i in indices])
        return inputs, labels, tasks, plus, minus


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

        self.replay_mode = str(getattr(args, "woe_replay_mode", "ce"))
        if self.replay_mode not in _REPLAY_MODES:
            raise ValueError(
                f"woe_replay_mode must be one of {_REPLAY_MODES}, "
                f"got {self.replay_mode!r}"
            )
        self.evidence_lambda = float(getattr(args, "woe_evidence_lambda", 1.0))
        self.uses_evidence_replay = self.replay_mode in ("evidence", "both")
        self.uses_ce_replay = self.replay_mode in ("ce", "both")
        self.replay_buffer = ReservoirReplayBuffer(
            self.replay_memories, store_evidence=self.uses_evidence_replay
        )

        # Feature mean used to centre the evidence of each task's stored items.
        # Frozen at the first store for that task and reused verbatim when the
        # evidence is recomputed later, so the snapshot and the recomputation
        # centre identically. Without this the comparison inherits the artefact
        # that affects `woe_si`'s output mode, where the student centres with the
        # live EMA and the teacher with a stale snapshot -- there, distilling a
        # network against an exact copy of itself yields a non-zero penalty.
        self.register_buffer(
            "replay_task_feature_mean", torch.zeros(n_tasks, self.feature_dim)
        )
        self.register_buffer(
            "replay_task_feature_mean_set", torch.zeros(n_tasks, dtype=torch.bool)
        )

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
        if len(self.replay_buffer) == 0:
            return torch.zeros(1, device=device)
        sample = self.replay_buffer.sample(
            self.replay_batch_size, with_evidence=self.uses_evidence_replay
        )
        if sample is None:
            return torch.zeros(1, device=device)

        if self.uses_evidence_replay:
            replay_x, replay_y, replay_t, stored_plus, stored_minus = sample
        else:
            replay_x, replay_y, replay_t = sample
            stored_plus = stored_minus = None
        replay_x = replay_x.to(device)
        replay_y = replay_y.to(device).long()

        loss = torch.zeros(1, device=device)
        if self.uses_ce_replay and self.replay_lambda != 0.0:
            cls_logits = self.net.forward_heads(replay_x)[1]
            masked_logits = self._mask_replay_logits(cls_logits, replay_t)
            loss = loss + self.replay_lambda * classification_cross_entropy(
                masked_logits,
                replay_y,
                class_weighted_ce=self.class_weighted_ce,
            )
        if self.uses_evidence_replay and self.evidence_lambda != 0.0:
            loss = loss + self.evidence_lambda * self._evidence_decay_loss(
                replay_x, replay_t, stored_plus.to(device), stored_minus.to(device)
            )
        return loss

    # ------------------------------------------------------------------
    def _evidence_decay_loss(
        self,
        replay_x: torch.Tensor,
        replay_t: torch.Tensor,
        stored_plus: torch.Tensor,
        stored_minus: torch.Tensor,
    ) -> torch.Tensor:
        """One-sided penalty on stored evidence having decayed since insertion.

        For each rehearsed item the DS total evidence is recomputed with the
        current network over the classes of the task it came from, and compared
        with the snapshot taken when it was stored. Only *deterioration* is
        penalised -- support for the item's own task falling, or evidence against
        it rising::

            L = mean_b sum_k [ relu(w+_stored - w+_now)^2
                             + relu(w-_now - w-_stored)^2 ]

        Improvement is free, so the term never fights the current task for
        capacity it does not need. ``J^2``-normalised to sit on the same scale as
        the other DS losses in this family. The squared hinge is C^1 -- its
        derivative ``2*relu(.)`` is continuous at the kink -- so nothing here is
        discontinuous for the optimiser.

        Args:
            replay_x: Rehearsed inputs ``(batch, ...)`` already on device.
            replay_t: Per-item task ids ``(batch,)``.
            stored_plus: Snapshot ``w_plus`` ``(batch, n_outputs)``.
            stored_minus: Snapshot ``w_minus`` ``(batch, n_outputs)``.

        Returns:
            Scalar penalty; zero when no item can be scored.
        """
        features = self.net.forward_features(replay_x, bn_training=False)
        total = torch.zeros((), device=features.device)
        scored = 0
        for task_id in torch.unique(replay_t).tolist():
            rows = (replay_t == int(task_id)).to(features.device)
            if not bool(rows.any()):
                continue
            active = self._active_class_indices(int(task_id), features.device)
            if active.numel() == 0:
                continue
            weights = self._weights_of_evidence(
                features[rows],
                self.net.model.fc,
                active,
                self._task_feature_mean(int(task_id)),
            )
            now_plus, now_minus = per_class_total_evidence(weights)
            decay = torch.relu(stored_plus[rows][:, active] - now_plus).pow(
                2
            ) + torch.relu(now_minus - stored_minus[rows][:, active]).pow(2)
            total = total + decay.sum(dim=1).sum()
            scored += int(rows.sum().item())
        if scored == 0:
            return torch.zeros(1, device=features.device)
        feature_count = features.shape[1]
        return total / (scored * float(feature_count * feature_count))

    # ------------------------------------------------------------------
    def _task_feature_mean(self, task_id: int) -> torch.Tensor:
        """Feature mean frozen for ``task_id``, falling back to the live EMA."""
        if 0 <= task_id < self.replay_task_feature_mean.shape[0] and bool(
            self.replay_task_feature_mean_set[task_id].item()
        ):
            return self.replay_task_feature_mean[task_id]
        return self.woe_feature_mean

    # ------------------------------------------------------------------
    def _store_classification_replay(
        self, x: torch.Tensor, y: torch.Tensor, t: int
    ) -> None:
        """Add the observed batch to the reservoir buffer (canonicalised input).

        In evidence modes the current model's DS total evidence for the batch is
        snapshotted alongside it, so a later step can penalise that evidence having
        decayed. The snapshot is computed on the *canonicalised* tensor -- the same
        one that will be replayed -- so the two evaluations are directly comparable.
        """
        if self.replay_buffer.capacity <= 0:
            return
        y_cls = unpack_y_to_class_labels(y).long()
        stored_x = self._input_for_replay(x)
        if not self.uses_evidence_replay:
            self.replay_buffer.add(stored_x, y_cls, int(t))
            return
        plus, minus = self._snapshot_evidence(stored_x, int(t))
        self.replay_buffer.add(stored_x, y_cls, int(t), plus, minus)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _snapshot_evidence(
        self, stored_x: torch.Tensor, t: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Current ``(w_plus, w_minus)`` for a batch, at full ``n_outputs`` width.

        Scattered to full width so a replayed item's class indices line up
        regardless of which task it came from. Freezes this task's centring mean on
        first call so the later recomputation uses the identical ``mu``.
        """
        features = self.net.forward_features(
            stored_x.to(self._device()), bn_training=False
        )
        if 0 <= t < self.replay_task_feature_mean.shape[0] and not bool(
            self.replay_task_feature_mean_set[t].item()
        ):
            self.replay_task_feature_mean[t].copy_(features.mean(dim=0))
            self.replay_task_feature_mean_set[t].fill_(True)

        active = self._active_class_indices(t, features.device)
        plus = torch.zeros(features.shape[0], self.n_outputs, device=features.device)
        minus = torch.zeros_like(plus)
        if active.numel() > 0:
            weights = self._weights_of_evidence(
                features, self.net.model.fc, active, self._task_feature_mean(t)
            )
            w_plus, w_minus = per_class_total_evidence(weights)
            plus[:, active] = w_plus
            minus[:, active] = w_minus
        return plus, minus

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
