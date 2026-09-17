# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""iCaRL (Rebuffi et al., CVPR 2017) on the 1D ResNet backbone.

Follows the authors' reference implementation
(``srebuffi/iCaRL``, ``iCaRL-TheanoLasagne/main_cifar_100_theano.py``):

* **Training data.** Each batch of new-task data is joined by exemplars of every
  earlier class, in the proportion they would occupy had the exemplar set been
  concatenated to the task's training set.
* **Loss.** Sigmoid binary cross-entropy over every output unit. Targets are the
  one-hot labels, except that the units of earlier tasks' classes take the
  sigmoid outputs of a frozen copy of the network from the end of the previous
  task, on every row (new data and exemplars alike).
* **Exemplars.** At the end of a task each seen class keeps
  ``n_memories // seen classes`` exemplars: new classes are chosen by herding on
  L2-normalised features, earlier classes keep the head of their herding ranking.
* **Classifier.** Nearest mean of exemplars over L2-normalised features.

Adaptations to this repository's protocol:

* TIL scores only task ``t``'s classes; CIL scores the classes of tasks
  ``0..cil_all_seen_upto_task``, the same candidate sets every other learner uses.
* Feature forwards that stand in for the reference's ``deterministic=True`` run in
  eval mode under the evaluation BatchNorm policy (:mod:`model.task_bn`), so the
  frozen network, herding and class means normalise the way test batches do.
* The BCE is summed over output units and averaged over rows (the reference
  averages over both), a constant factor the learning rate absorbs. With
  ``class_weighted_ce`` the new-data rows are weighted by inverse class frequency
  in the batch; exemplar rows are class-balanced by construction and keep weight 1.
* Until a task's classes have exemplars (i.e. while it is still being trained)
  there are no class means for it, and ``forward`` returns the network's masked
  logits instead, as the reference's per-epoch validation printout does.
"""

import copy
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

import torch
import torch.nn.functional as F

from model import task_bn
from model.replay_utils import ReplayInputMixin, unpack_y_to_class_labels
from model.resnet1d import ResNet1D
from utils import misc_utils
from utils.class_weighted_loss import compute_inverse_frequency_class_weights
from utils.training_metrics import macro_recall

# Herding may re-pick a sample (which adds nothing), so bound the iterations. The
# reference stops after 1000 (Theano) or 1.1 * m (TensorFlow) iterations.
_HERDING_MAX_ITER_FACTOR = 10
_MASK_FILL = -1e9


@dataclass
class IcarlConfig:
    lr: float = 1e-3
    n_memories: int = 5120
    inner_steps: int = 1

    grad_clip_norm: Optional[float] = 0.0
    arch: str = "resnet1d"
    cuda: bool = True
    n_epochs: int = 1
    samples_per_task: int = -1
    icarl_feature_chunk_size: int = 512

    @staticmethod
    def from_args(args: object) -> "IcarlConfig":
        cfg = IcarlConfig()
        for field in cfg.__dataclass_fields__:
            if hasattr(args, field):
                setattr(cfg, field, getattr(args, field))
        return cfg


class Net(ReplayInputMixin, torch.nn.Module):
    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        super(Net, self).__init__()
        self.cfg = IcarlConfig.from_args(args)
        self.args = args
        self.nt = n_tasks
        self.n_memories = int(self.cfg.n_memories)
        self.n_classes = n_outputs
        self.n_outputs = n_outputs
        self.samples_per_task_resolver = getattr(args, "get_samples_per_task", None)
        self.samples_per_task = self.cfg.samples_per_task
        if self.samples_per_task_resolver is None:
            assert self.samples_per_task > 0, "Samples per task is <= 0"
        self.examples_seen = 0
        self.inner_steps = self.cfg.inner_steps

        if self.cfg.arch != "resnet1d":
            raise ValueError(
                f"Unsupported arch {self.cfg.arch}; only resnet1d is available now."
            )
        self.net = ResNet1D(n_outputs, args)
        self.n_feat = self.net.feature_dim
        self.opt = torch.optim.SGD(self.net.parameters(), lr=self.cfg.lr, momentum=0.9)
        self.class_weighted_ce = bool(getattr(args, "class_weighted_ce", True))

        # Network frozen at the end of the previous task (distillation targets).
        self.old_net: Optional[ResNet1D] = None
        # First-epoch training data of the current task, staged for herding.
        self.memx: Optional[torch.Tensor] = None
        self.memy: Optional[torch.Tensor] = None
        # Exemplar set on CPU: rows grouped by class, each class in herding order.
        self.exemplar_x: Optional[torch.Tensor] = None
        self.exemplar_y: Optional[torch.Tensor] = None

        # Class means depend on the weights and the exemplars; both bump this.
        self._weights_version = 0
        self._class_means_version = -1
        self._class_means_cache: dict[tuple[int, ...], torch.Tensor] = {}
        # Private generator so feature-chunk shuffling leaves training RNG alone.
        self._feature_generator = torch.Generator().manual_seed(0)

        self.gpu = self.cfg.cuda
        self.classes_per_task = misc_utils.build_task_class_list(
            n_tasks,
            n_outputs,
            nc_per_task=getattr(args, "nc_per_task_list", "")
            or getattr(args, "nc_per_task", None),
            classes_per_task=getattr(args, "classes_per_task", None),
        )
        self.nc_per_task = misc_utils.max_task_class_count(self.classes_per_task)
        self.incremental_loader_name = getattr(args, "loader", None)

    # ------------------------------------------------------------------
    def compute_offsets(self, task):
        offset1, offset2 = misc_utils.compute_offsets(task, self.classes_per_task)
        return int(offset1), int(offset2)

    def _get_samples_per_task(self, task):
        if self.samples_per_task_resolver is None:
            return self.samples_per_task
        return int(self.samples_per_task_resolver(task))

    def _device(self) -> torch.device:
        return next(self.net.parameters()).device

    def netforward(self, x):
        return self.net(self._canonicalize_input(x, detach=False))

    def _eval_normalization(self, net: torch.nn.Module):
        """BatchNorm context matching evaluation forwards (see :mod:`model.task_bn`)."""
        if task_bn.eval_uses_batch_statistics(self.args):
            return task_bn.batch_statistics(net)
        return nullcontext()

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        t: int,
        *,
        cil_all_seen_upto_task: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Score ``x`` with the nearest-mean-of-exemplars classifier.

        Args:
            x: Input batch.
            t: Task whose classes are scored in task-incremental runs.
            cil_all_seen_upto_task: In class-incremental runs, score the classes
                of tasks ``0..cil_all_seen_upto_task``.
            **kwargs: Swallows extra keys from the training loop.

        Returns:
            ``(batch, n_classes)`` scores: negative squared distance to each
            candidate class mean, ``-1e9`` for every other class. While a
            candidate task has no exemplars yet, the network's masked logits.
        """
        del kwargs
        cil_upto = cil_all_seen_upto_task
        if self.incremental_loader_name not in (None, "class_incremental_loader"):
            cil_upto = None
        tasks = [t] if cil_upto is None else list(range(cil_upto + 1))

        means, class_ids = self._class_means(tasks)
        if means is None:
            return misc_utils.apply_task_incremental_logit_mask(
                self.netforward(x),
                t,
                self.classes_per_task,
                self.n_classes,
                cil_all_seen_upto_task=cil_upto,
                loader=self.incremental_loader_name,
            )

        with torch.no_grad():
            feats = self.net.forward_features(self._canonicalize_input(x, detach=True))
            feats = F.normalize(feats.float(), p=2, dim=1)
            scores = feats.new_full((x.size(0), self.n_classes), _MASK_FILL)
            index = torch.as_tensor(class_ids, dtype=torch.long, device=feats.device)
            scores[:, index] = -torch.cdist(feats, means).pow(2)
        return scores

    def _class_means(
        self, tasks: list[int]
    ) -> tuple[Optional[torch.Tensor], list[int]]:
        """L2-normalised exemplar means of the classes of ``tasks``.

        Returns ``(None, [])`` when any of ``tasks`` has no exemplars yet.
        """
        if self.exemplar_y is None:
            return None, []
        present = set(torch.unique(self.exemplar_y).tolist())
        class_ids: list[int] = []
        for task in tasks:
            offset1, offset2 = self.compute_offsets(task)
            task_classes = [c for c in range(offset1, offset2) if c in present]
            if not task_classes:
                return None, []
            class_ids.extend(task_classes)

        if self._class_means_version != self._weights_version:
            self._class_means_cache = {}
            self._class_means_version = self._weights_version
        key = tuple(class_ids)
        if key not in self._class_means_cache:
            rows = torch.isin(self.exemplar_y, torch.as_tensor(class_ids))
            feats = self._eval_features(self.exemplar_x[rows])
            labels = self.exemplar_y[rows].to(feats.device)
            means = torch.stack([feats[labels == c].mean(dim=0) for c in class_ids])
            self._class_means_cache[key] = F.normalize(means, p=2, dim=1)
        return self._class_means_cache[key], class_ids

    def _eval_features(self, x: torch.Tensor) -> torch.Tensor:
        """L2-normalised penultimate features, computed the way evaluation is.

        The network runs in eval mode, under the evaluation BatchNorm policy, in
        chunks visited in a random order: under batch statistics each chunk then
        mixes classes like a test batch does, rather than normalising a class
        against itself.

        Args:
            x: Canonical inputs, on any device.

        Returns:
            ``(len(x), n_feat)`` float32 features on the network's device.
        """
        device = self._device()
        n = int(x.size(0))
        features = torch.empty((n, self.n_feat), device=device)
        if n == 0:
            return features
        chunk_size = max(int(self.cfg.icarl_feature_chunk_size), 1)
        order = torch.randperm(n, generator=self._feature_generator)
        was_training = self.net.training
        self.net.eval()
        try:
            with torch.no_grad(), self._eval_normalization(self.net):
                for start in range(0, n, chunk_size):
                    rows = order[start : start + chunk_size]
                    batch = x[rows.to(x.device)].to(device, non_blocking=True)
                    batch = self._canonicalize_input(batch, detach=True)
                    features[rows.to(device)] = self.net.forward_features(batch).float()
        finally:
            self.net.train(was_training)
        return F.normalize(features, p=2, dim=1)

    # ------------------------------------------------------------------
    def observe(self, x, y, t):
        batch_count = x.size(0)
        y_cls = unpack_y_to_class_labels(y).long()
        self.net.train()
        if self.gpu:
            self.net.cuda()
        device = self._device()
        samples_per_task = self._get_samples_per_task(t)
        assert samples_per_task > 0, "Samples per task is <= 0"

        # Stage the first pass over the task's data as the herding candidates.
        if self.examples_seen < samples_per_task:
            staged_x = self._input_for_replay(x).cpu().clone()
            staged_y = y_cls.detach().cpu().clone()
            if self.memx is None:
                self.memx, self.memy = staged_x, staged_y
            else:
                self.memx = torch.cat((self.memx, staged_x))
                self.memy = torch.cat((self.memy, staged_y))
        self.examples_seen += batch_count

        replay_x, replay_y = self._sample_exemplars(batch_count, samples_per_task)
        labels = y_cls.to(device)
        if replay_y is not None:
            labels = torch.cat((labels, replay_y))

        cls_tr_rec = []
        metric_logits = None
        for _ in range(self.inner_steps):
            inputs = self._canonicalize_input(x, detach=False)
            if replay_x is not None:
                inputs = torch.cat((inputs, replay_x))
            logits = self.net(inputs).float()
            targets = self._targets(x, replay_x, labels, t)
            loss = self._loss(logits, targets, labels, batch_count)

            self.opt.zero_grad()
            loss.backward()
            if self.cfg.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), self.cfg.grad_clip_norm
                )
            self.opt.step()
            self._weights_version += 1

            metric_logits = misc_utils.apply_task_incremental_logit_mask(
                logits[:batch_count].detach(),
                t,
                self.classes_per_task,
                self.n_classes,
                cil_all_seen_upto_task=t,
                loader=self.incremental_loader_name,
            )
            preds = torch.argmax(metric_logits, dim=1)
            cls_tr_rec.append(macro_recall(preds, labels[:batch_count]))

        # Last minibatch of the task across all epochs.
        n_epochs = int(getattr(self.args, "n_epochs", self.cfg.n_epochs))
        if self.examples_seen >= n_epochs * samples_per_task:
            self.examples_seen = 0
            self._end_task(t)

        avg_cls_tr_rec = sum(cls_tr_rec) / len(cls_tr_rec) if cls_tr_rec else 0.0
        return float(loss.item()), avg_cls_tr_rec, metric_logits

    def _sample_exemplars(
        self, batch_count: int, samples_per_task: int
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Draw exemplars in their share of the task's augmented training set.

        The reference concatenates the exemplars to the task's training data, so
        a batch of ``batch_count`` new rows carries on average
        ``batch_count * n_exemplars / samples_per_task`` exemplars.
        """
        if self.exemplar_x is None or self.exemplar_x.size(0) == 0:
            return None, None
        n_exemplars = int(self.exemplar_x.size(0))
        count = round(batch_count * n_exemplars / samples_per_task)
        count = min(n_exemplars, max(1, count))
        rows = torch.randperm(n_exemplars)[:count]
        device = self._device()
        return (
            self.exemplar_x[rows].to(device, non_blocking=True),
            self.exemplar_y[rows].to(device, non_blocking=True),
        )

    def _targets(
        self,
        x: torch.Tensor,
        replay_x: Optional[torch.Tensor],
        labels: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """One-hot labels with earlier tasks' units replaced by the frozen network's."""
        targets = F.one_hot(labels, self.n_classes).float()
        n_old_classes, _ = self.compute_offsets(t)
        if self.old_net is None or n_old_classes == 0:
            return targets
        # ``model.train()`` in the training loop also reaches this submodule.
        self.old_net.eval()
        with torch.no_grad(), self._eval_normalization(self.old_net):
            # New data goes through the frozen network's own input adapter;
            # exemplars are stored already canonical.
            old_inputs = ReplayInputMixin._canonicalize_input(
                SimpleNamespace(net=self.old_net), x, detach=True
            )
            if replay_x is not None:
                old_inputs = torch.cat((old_inputs, replay_x))
            old_logits = self.old_net(old_inputs).float()
        targets[:, :n_old_classes] = torch.sigmoid(old_logits[:, :n_old_classes])
        return targets

    def _loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        labels: torch.Tensor,
        batch_count: int,
    ) -> torch.Tensor:
        """Row-weighted mean of the per-row BCE summed over output units."""
        per_row = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        ).sum(dim=1)
        weights = torch.ones_like(per_row)
        if self.class_weighted_ce:
            new_labels = labels[:batch_count]
            class_weights = compute_inverse_frequency_class_weights(
                new_labels, self.n_classes, per_row.device
            )
            weights[:batch_count] = class_weights[new_labels]
        return (weights * per_row).sum() / weights.sum()

    # ------------------------------------------------------------------
    def _end_task(self, t: int) -> None:
        if self.memx is None or self.memy is None or self.memy.numel() == 0:
            return
        self.old_net = self._frozen_copy()
        self._update_exemplars(t)
        self._weights_version += 1
        if self.gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _frozen_copy(self) -> ResNet1D:
        # Share ``args`` rather than copying it: it holds the loader's bound
        # ``get_samples_per_task`` and, through it, the whole dataset.
        backbone_args = getattr(self.net, "args", None)
        frozen = copy.deepcopy(self.net, memo={id(backbone_args): backbone_args})
        frozen.eval()
        for param in frozen.parameters():
            param.requires_grad_(False)
        return frozen

    def _update_exemplars(self, t: int) -> None:
        """Reduce earlier classes' exemplar sets and herd the new classes'."""
        staged_x, staged_y = self.memx, self.memy
        self.memx = None
        self.memy = None

        offset1, offset2 = self.compute_offsets(t)
        new_classes = [
            c for c in torch.unique(staged_y).tolist() if offset1 <= c < offset2
        ]
        if len(new_classes) != self.classes_per_task[t]:
            print(
                "[WARNING][iCaRL] Task {} expected {} classes, found {}.".format(
                    t, self.classes_per_task[t], len(new_classes)
                )
            )
        old_classes = (
            [] if self.exemplar_y is None else torch.unique(self.exemplar_y).tolist()
        )
        n_seen = len(old_classes) + len(new_classes)
        if n_seen == 0:
            return
        per_class = self.n_memories // n_seen

        kept_x: list[torch.Tensor] = []
        kept_y: list[torch.Tensor] = []
        for c in old_classes:
            rows = (self.exemplar_y == c).nonzero(as_tuple=True)[0][:per_class]
            kept_x.append(self.exemplar_x[rows])
            kept_y.append(self.exemplar_y[rows])

        features = self._eval_features(staged_x)
        for c in new_classes:
            rows = (staged_y == c).nonzero(as_tuple=True)[0]
            ranking = self._herding_order(features[rows.to(features.device)], per_class)
            chosen = rows[ranking]
            kept_x.append(staged_x[chosen].clone())
            kept_y.append(staged_y[chosen].clone())

        self.exemplar_x = torch.cat(kept_x)
        self.exemplar_y = torch.cat(kept_y)

    @staticmethod
    def _herding_order(features: torch.Tensor, count: int) -> torch.Tensor:
        """Rank up to ``count`` rows of ``features`` by herding.

        Iterates ``w <- w + mu - phi(x*)`` with ``x* = argmax <w, phi(x)>`` over
        L2-normalised features, as the reference does; this is Algorithm 4's
        argmin of the distance between ``mu`` and the running exemplar mean.

        Args:
            features: ``(n, d)`` L2-normalised features of one class.
            count: Exemplars wanted.

        Returns:
            CPU indices into ``features``, best first.
        """
        count = min(int(count), int(features.size(0)))
        mean = features.mean(dim=0)
        direction = mean.clone()
        chosen = torch.zeros(features.size(0), dtype=torch.bool)
        order: list[int] = []
        iterations = 0
        while len(order) < count and iterations < _HERDING_MAX_ITER_FACTOR * count:
            index = int(torch.argmax(features @ direction))
            if not chosen[index]:
                chosen[index] = True
                order.append(index)
            direction = direction + mean - features[index]
            iterations += 1
        return torch.as_tensor(order, dtype=torch.long)
