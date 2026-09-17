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
rehearsed item, so the term never competes for capacity it does not need.

``woe_evidence_scale`` chooses what the hinge is measured on. The default
``'weight'`` uses the raw ``(w_plus, w_minus)`` above. ``'belief'`` first maps each
channel through ``1 - exp(-w / tau)``, the mass that channel commits to its focal
set. Weights of evidence are unbounded above, so a *one-sided* penalty on them is
satisfiable by inflating the readout -- ``w`` is linear in it, so one global rescale
satisfies every "must not decrease" constraint at once, and current-task CE actively
pushes that way because it also sharpens the softmax. The symmetric ``output`` mode
pinned the scale; dropping the upper arm opened the hole. Beliefs saturate, so the
loophole stops paying. See :func:`model.woe_si.evidence_to_belief`.

Either way this is a functional constraint like ``woe_reg_level='output'``, but
anchored to per-sample evidence recorded when the sample was seen rather than to a
frozen end-of-task teacher network -- and, unlike output mode, the snapshot and the
recomputation centre with the *same* per-task feature mean, so an unchanged network
scores exactly zero penalty.

``woe_evidence_readout_only`` detaches the backbone features before the penalty is
formed, so rehearsed items constrain only the linear readout and never propagate a
gradient into the backbone. Paired with ``woe_replay_mode='evidence'`` this makes
the buffer purely a distillation signal: the backbone is driven entirely by the
current task while the readout is held to the evidence it previously assigned. It
also confines the mechanism to the one place the Dempster-Shafer construction is
exact -- for the backbone, ``d I / d theta`` is ``(d I / d phi)(d phi / d theta)``,
where only the first factor carries DS content.

The evidence snapshot is stored *alongside* the stored item, not instead of it:
penalising drift requires re-evaluating the current model, which needs something to
evaluate. The extra cost is ``2 * n_outputs`` floats per item against a stored item
of 512-1024, so it is a few percent of buffer memory.

``woe_replay_store`` decides what that stored item is. The default ``'input'`` keeps
the canonicalised network input; ``'feature'`` keeps the penultimate features ``phi``
and replays them straight into the readout. The Dempster-Shafer motive is that the
evidence is a function of ``phi`` at the readout, making ``phi`` a sufficient
statistic for it and the input a more expensive route to the same place -- worth 2x
the exemplars per byte here (512 floats against 1024). It costs staleness (stored
features are never re-encoded as the backbone drifts) and confines replay to the
readout, so the backbone gets no rehearsal gradient at all.

Config: ``configs/models/til/woe_si_replay.yaml``, or
``configs/models/til/woe_si_injection.yaml`` for the anchor-off, small-buffer host
the storage question is measured on. Extra knobs: ``woe_replay_memories`` (buffer
capacity), ``woe_replay_batch_size`` (replay draw per step), ``woe_replay_lambda``
(replay CE weight), ``woe_replay_mode``
(``ce``/``evidence``/``both``/``evidence_sym``/``logit``), ``woe_replay_store``
(``input``/``feature``), ``woe_evidence_lambda`` (weight on any non-CE term),
``woe_evidence_scale`` (``weight``/``belief``) and ``woe_evidence_belief_tau``.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import torch

from model.replay_utils import unpack_y_to_class_labels
from model.woe_si import Net as WoeSiNet, per_class_total_evidence
from utils import misc_utils
from utils.class_weighted_loss import classification_cross_entropy

# What the reservoir contributes to the training loss:
#   "ce"           -- cross-entropy on rehearsed samples (the original behaviour).
#   "evidence"     -- one-sided penalty on the DS total evidence of rehearsed
#                     samples falling below what it was when they were stored.
#   "both"         -- the sum of "ce" and "evidence".
#   "evidence_sym" -- symmetric squared drift of (w_plus, w_minus) from the
#                     snapshot: any movement is charged, not only decay.
#   "logit"        -- symmetric squared drift of the *logits* from the snapshot,
#                     i.e. Dark Experience Replay (Buzzega et al. 2020).
#
# The last two exist as a **matched pair**, and that is the whole point of them.
# Under a Dempster-Shafer reading, replay works by re-injecting evidence for old
# classes to counteract the conflict new evidence introduces. If that is really
# the mechanism, then what a buffer needs to carry is the weight-of-evidence
# vector rather than the input -- and distilling `w` should beat distilling the
# logits, because a logit is the *difference* w_plus - w_minus and therefore
# discards the ignorance degree of freedom: (8, 1) and (108, 101) are the same
# logit built from wildly different amounts of evidence, and only the second is
# a claim the model should be held to.
#
# "logit" and "evidence_sym" are deliberately the same functional form (a
# symmetric squared drift against a per-item snapshot taken at insertion) over
# the same items, drawn from the same buffer, at the same width. The only thing
# that differs is *what is distilled*, so the comparison isolates the DS content
# rather than confounding it with the shape of the penalty -- which is what the
# pre-existing one-sided "evidence" mode could not do against a DER baseline.
_REPLAY_MODES = (
    "ce",
    "evidence",
    "both",
    "evidence_sym",
    "logit",
    # DER++ analogues: distillation *on top of* CE rehearsal rather than instead
    # of it. C6 read a null between the `logit` and `evidence_sym` targets as
    # evidence about what a buffer should store, but both recovered only ~19% of
    # the gap CE rehearsal closes -- a null between two targets inside a vehicle
    # that barely delivers cannot discriminate the targets. These arms establish
    # whether distillation works at all on this host before that null is read.
    "ce_logit",
    "ce_evidence_sym",
)

# What the reservoir physically stores per item:
#   "input"   -- the canonicalised network input (the original behaviour).
#   "feature" -- the penultimate features phi, replayed straight into the readout.
#
# The DS motivation is the second half of the same hypothesis: the evidence
# w_jk = beta_kj * (phi_j - mu_j) + beta_0k / J is a function of phi at the
# readout, so phi is a *sufficient statistic* for everything the evidence
# reading cares about, and the input is a more expensive way of arriving at it.
# On this data that is a 2x footprint saving -- 512 floats against a 1024-float
# input -- so at a matched byte budget a feature buffer holds twice the
# exemplars. Measure at a matched *budget*, never at a matched item count, or the
# comparison hands one arm twice the memory and proves nothing.
#
# Two costs to state plainly rather than discover. First, stored features go
# stale as the backbone drifts, where a stored input is re-encoded by the current
# backbone every time it is drawn; this is the standard latent-replay trade and
# the reason the saving is not free. Second, replay then reaches only the
# readout -- the backbone receives no rehearsal gradient at all -- which makes
# this the buffer-side analogue of `woe_evidence_readout_only` and confines the
# mechanism to the one place the DS construction is exact.
_REPLAY_STORES = ("input", "feature")


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

    def __init__(
        self,
        capacity: int,
        store_evidence: bool = False,
        store_logits: bool = False,
    ) -> None:
        self.capacity = int(capacity)
        self.store_evidence = bool(store_evidence)
        self.store_logits = bool(store_logits)
        self.inputs: List[torch.Tensor] = []
        self.labels: List[int] = []
        self.tasks: List[int] = []
        # Per-item snapshot of the DS total evidence at insertion time, stored at
        # full ``n_outputs`` width so class indices line up across tasks. Empty
        # unless ``store_evidence``.
        self.evidence_plus: List[torch.Tensor] = []
        self.evidence_minus: List[torch.Tensor] = []
        # The same, for raw logits -- the Dark Experience Replay target that the
        # evidence snapshot is compared against. One of the two is populated, per
        # ``woe_replay_mode``; they are never both needed at once.
        self.logits: List[torch.Tensor] = []
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
        logits: Optional[torch.Tensor] = None,
    ) -> None:
        """Insert a batch, respecting the reservoir replacement policy.

        Args:
            inputs: Batch of stored-ready items, shape ``(batch, ...)`` (CPU).
                Network inputs, or penultimate features under
                ``woe_replay_store='feature'``.
            labels: Integer class labels, shape ``(batch,)``.
            task_id: Task index the batch belongs to.
            evidence_plus: Optional ``(batch, n_outputs)`` snapshot of ``w_plus``
                at insertion time. Required when the buffer stores evidence.
            evidence_minus: Optional ``(batch, n_outputs)`` snapshot of ``w_minus``.
            logits: Optional ``(batch, n_outputs)`` snapshot of the raw logits.
                Required when the buffer stores logits.
        """
        if self.capacity <= 0 or inputs.size(0) == 0:
            return
        if self.store_evidence and (evidence_plus is None or evidence_minus is None):
            raise ValueError(
                "buffer was built with store_evidence=True but add() received no "
                "evidence snapshot"
            )
        if self.store_logits and logits is None:
            raise ValueError(
                "buffer was built with store_logits=True but add() received no "
                "logit snapshot"
            )
        slots, _filled, seen = misc_utils.reservoir_slots(
            inputs.size(0), len(self.inputs), self.seen, self.capacity
        )
        self.seen = seen
        inputs_cpu = inputs.detach().cpu()
        labels_cpu = labels.detach().cpu().long()
        plus_cpu = evidence_plus.detach().cpu() if self.store_evidence else None
        minus_cpu = evidence_minus.detach().cpu() if self.store_evidence else None
        logits_cpu = logits.detach().cpu() if self.store_logits else None
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
                if self.store_logits:
                    self.logits[slot] = logits_cpu[index].clone()
            else:
                # Fill phase: reservoir_slots hands back the next dense index.
                self.inputs.append(input_item)
                self.labels.append(label_item)
                self.tasks.append(int(task_id))
                if self.store_evidence:
                    self.evidence_plus.append(plus_cpu[index].clone())
                    self.evidence_minus.append(minus_cpu[index].clone())
                if self.store_logits:
                    self.logits.append(logits_cpu[index].clone())

    def sample(
        self,
        batch_size: int,
        with_evidence: bool = False,
        with_logits: bool = False,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Draw up to ``batch_size`` items uniformly without replacement.

        Args:
            batch_size: Requested number of replay items.
            with_evidence: When ``True``, also return the stored ``(w_plus,
                w_minus)`` snapshots. Requires ``store_evidence``.
            with_logits: When ``True``, also return the stored logit snapshot.
                Requires ``store_logits``.

        Returns:
            A dict with keys ``inputs``, ``labels`` and ``tasks``, plus
            ``evidence_plus``/``evidence_minus`` when ``with_evidence`` and
            ``logits`` when ``with_logits``. ``None`` when the buffer is empty or
            ``batch_size <= 0``.

        Raises:
            ValueError: If a snapshot is requested that the buffer does not hold.
        """
        if not self.inputs or batch_size <= 0:
            return None
        if with_evidence and not self.store_evidence:
            raise ValueError(
                "sample(with_evidence=True) needs a buffer built with "
                "store_evidence=True"
            )
        if with_logits and not self.store_logits:
            raise ValueError(
                "sample(with_logits=True) needs a buffer built with "
                "store_logits=True"
            )
        draw = min(int(batch_size), len(self.inputs))
        indices = random.sample(range(len(self.inputs)), draw)
        batch = {
            "inputs": torch.stack([self.inputs[i] for i in indices]),
            "labels": torch.tensor([self.labels[i] for i in indices], dtype=torch.long),
            "tasks": torch.tensor([self.tasks[i] for i in indices], dtype=torch.long),
        }
        if with_evidence:
            batch["evidence_plus"] = torch.stack(
                [self.evidence_plus[i] for i in indices]
            )
            batch["evidence_minus"] = torch.stack(
                [self.evidence_minus[i] for i in indices]
            )
        if with_logits:
            batch["logits"] = torch.stack([self.logits[i] for i in indices])
        return batch


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
        self.evidence_readout_only = bool(
            getattr(args, "woe_evidence_readout_only", False)
        )
        self.replay_store = str(getattr(args, "woe_replay_store", "input"))
        if self.replay_store not in _REPLAY_STORES:
            raise ValueError(
                f"woe_replay_store must be one of {_REPLAY_STORES}, "
                f"got {self.replay_store!r}"
            )
        self.stores_features = self.replay_store == "feature"
        # woe_evidence_scale / woe_evidence_belief_tau are read and validated by
        # model.woe_si.Net.__init__, which the super() call above already ran.
        self.uses_evidence_replay = self.replay_mode in (
            "evidence",
            "both",
            "evidence_sym",
            "ce_evidence_sym",
        )
        # "evidence" and "both" charge decay only; the "*_sym" modes charge any
        # drift, which is the form that matches the DER logit baseline.
        self.evidence_drift_symmetric = self.replay_mode in (
            "evidence_sym",
            "ce_evidence_sym",
        )
        self.uses_ce_replay = self.replay_mode in (
            "ce",
            "both",
            "ce_logit",
            "ce_evidence_sym",
        )
        self.uses_logit_replay = self.replay_mode in ("logit", "ce_logit")
        self.replay_buffer = ReservoirReplayBuffer(
            self.replay_memories,
            store_evidence=self.uses_evidence_replay,
            store_logits=self.uses_logit_replay,
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
            self.replay_batch_size,
            with_evidence=self.uses_evidence_replay,
            with_logits=self.uses_logit_replay,
        )
        if sample is None:
            return torch.zeros(1, device=device)

        replay_x = sample["inputs"].to(device)
        replay_y = sample["labels"].to(device).long()
        replay_t = sample["tasks"]

        loss = torch.zeros(1, device=device)
        if self.uses_ce_replay and self.replay_lambda != 0.0:
            cls_logits = self._replay_logits(replay_x)
            masked_logits = self._mask_replay_logits(cls_logits, replay_t)
            loss = loss + self.replay_lambda * classification_cross_entropy(
                masked_logits,
                replay_y,
                class_weighted_ce=self.class_weighted_ce,
            )
        if self.uses_evidence_replay and self.evidence_lambda != 0.0:
            loss = loss + self.evidence_lambda * self._evidence_decay_loss(
                replay_x,
                replay_t,
                sample["evidence_plus"].to(device),
                sample["evidence_minus"].to(device),
            )
        if self.uses_logit_replay and self.evidence_lambda != 0.0:
            loss = loss + self.evidence_lambda * self._logit_distillation_loss(
                replay_x, replay_t, sample["logits"].to(device)
            )
        return loss

    # ------------------------------------------------------------------
    def _replay_features(self, replay_x: torch.Tensor) -> torch.Tensor:
        """Penultimate features for a replay draw, however the buffer stores it.

        Under ``woe_replay_store='feature'`` the stored tensor already *is* the
        feature vector, so there is no backbone pass to make -- which is both the
        2x footprint saving and its cost: the features were encoded by whatever
        backbone existed when the item was stored, and are never refreshed.

        ``bn_training=False`` on the input path so a replay draw never perturbs
        the BatchNorm running statistics, which the current-task pass owns.
        """
        if self.stores_features:
            return replay_x
        return self.net.forward_features(replay_x, bn_training=False)

    # ------------------------------------------------------------------
    def _replay_logits(self, replay_x: torch.Tensor) -> torch.Tensor:
        """Classification logits for a replay draw, from inputs or features."""
        if self.stores_features:
            return self.net.forward_classifier(replay_x, bn_training=False)
        return self.net.forward_classifier(self.net.forward_features(replay_x))

    # ------------------------------------------------------------------
    def _logit_distillation_loss(
        self,
        replay_x: torch.Tensor,
        replay_t: torch.Tensor,
        stored_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Dark Experience Replay: squared drift of the logits from the snapshot.

        The control arm for :meth:`_evidence_decay_loss` under
        ``woe_replay_mode='evidence_sym'``. Deliberately identical in every
        respect except the quantity distilled -- same buffer, same draw, same
        per-item snapshot taken at insertion, same symmetric squared form, same
        restriction to the columns of the item's own task -- so any difference
        between the two arms is attributable to ``w`` versus ``z`` and not to the
        shape of the penalty.

        The DS prediction is that this one loses, because ``z_k = w+_k - w-_k``
        retains only the difference of the two channels and throws away their
        common magnitude, which is precisely the ignorance the evidential reading
        says a rehearsal target should carry.

        Normalised per scored item and by the same ``_evidence_normaliser`` the
        evidence arm uses, so ``woe_evidence_lambda`` means a comparable thing in
        both -- comparable, not identical: a logit drift is one squared term per
        class where the evidence drift is two, so the grids overlap but the peaks
        need not coincide.

        Args:
            replay_x: Rehearsed inputs or features ``(batch, ...)``, on device.
            replay_t: Per-item task ids ``(batch,)``.
            stored_logits: Snapshot logits ``(batch, n_outputs)``.

        Returns:
            Scalar penalty; zero when no item can be scored.
        """
        logits = self._replay_logits(replay_x)
        total = torch.zeros((), device=logits.device)
        scored = 0
        for task_id in torch.unique(replay_t).tolist():
            rows = (replay_t == int(task_id)).to(logits.device)
            if not bool(rows.any()):
                continue
            active = self._active_class_indices(int(task_id), logits.device)
            if active.numel() == 0:
                continue
            drift = (logits[rows][:, active] - stored_logits[rows][:, active]).pow(2)
            total = total + drift.sum(dim=1).sum()
            scored += int(rows.sum().item())
        if scored == 0:
            return torch.zeros(1, device=logits.device)
        return total / (scored * self._evidence_normaliser(self.feature_dim))

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
        capacity it does not need. The squared hinge is C^1 -- its derivative
        ``2*relu(.)`` is continuous at the kink -- so nothing here is
        discontinuous for the optimiser.

        Under ``woe_evidence_scale='belief'`` both sides are first mapped through
        ``1 - exp(-w / tau)`` (see :meth:`model.woe_si.Net._to_penalty_scale`); the normalisation
        changes with the scale (see :meth:`model.woe_si.Net._evidence_normaliser`), so
        ``woe_evidence_lambda`` does not transfer between the two.

        Args:
            replay_x: Rehearsed inputs ``(batch, ...)`` already on device.
            replay_t: Per-item task ids ``(batch,)``.
            stored_plus: Snapshot ``w_plus`` ``(batch, n_outputs)``.
            stored_minus: Snapshot ``w_minus`` ``(batch, n_outputs)``.

        Returns:
            Scalar penalty; zero when no item can be scored.
        """
        features = self._replay_features(replay_x)
        if self.evidence_readout_only:
            # Detach so the penalty is a function of the readout alone: rehearsed
            # items constrain `fc` and never reach the backbone, which is then
            # driven purely by the current task. `bn_training=False` already keeps
            # the replay forward from touching BatchNorm running statistics, so
            # with this detach the backbone is untouched by replay entirely.
            features = features.detach()
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
            now_plus, now_minus = self._to_penalty_scale(
                *per_class_total_evidence(weights)
            )
            was_plus, was_minus = self._to_penalty_scale(
                stored_plus[rows][:, active], stored_minus[rows][:, active]
            )
            if self.evidence_drift_symmetric:
                # Charge any movement, matching the DER logit control's form.
                decay = (now_plus - was_plus).pow(2) + (now_minus - was_minus).pow(2)
            else:
                decay = torch.relu(was_plus - now_plus).pow(2) + torch.relu(
                    now_minus - was_minus
                ).pow(2)
            total = total + decay.sum(dim=1).sum()
            scored += int(rows.sum().item())
        if scored == 0:
            return torch.zeros(1, device=features.device)
        return total / (scored * self._evidence_normaliser(features.shape[1]))

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
        if self.uses_evidence_replay:
            plus, minus = self._snapshot_evidence(stored_x, int(t))
        else:
            plus = minus = None
        logits = self._snapshot_logits(stored_x) if self.uses_logit_replay else None
        # Encode last: the evidence and logit snapshots are taken on the input so
        # they are the values the *current* network assigns, and only then is the
        # item reduced to the representation the buffer keeps.
        if self.stores_features:
            stored_x = self._snapshot_features(stored_x)
        self.replay_buffer.add(
            stored_x,
            y_cls,
            int(t),
            evidence_plus=plus,
            evidence_minus=minus,
            logits=logits,
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _snapshot_features(self, stored_x: torch.Tensor) -> torch.Tensor:
        """Penultimate features of a batch, for a feature-storing buffer.

        ``bn_training=False`` so encoding an item for storage never perturbs the
        BatchNorm running statistics the current-task pass owns.
        """
        return self.net.forward_features(
            stored_x.to(self._device()), bn_training=False
        ).detach()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _snapshot_logits(self, stored_x: torch.Tensor) -> torch.Tensor:
        """Current classification logits for a batch, at full ``n_outputs`` width.

        The Dark Experience Replay target, snapshotted on the canonicalised input
        exactly as :meth:`_snapshot_evidence` snapshots the evidence, so the two
        rehearsal targets are recorded at the same moment from the same network.
        """
        device_x = stored_x.to(self._device())
        features = self.net.forward_features(device_x, bn_training=False)
        return self.net.forward_classifier(features, bn_training=False).detach()

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
                loader=self.incremental_loader_name,
            )
        return masked


__all__ = ["Net", "ReservoirReplayBuffer"]
