"""Elastic Weight Consolidation (EWC) learner compatible with ``main.py``.

This version mirrors the behaviour of the original script—keeping the Fisher
information based regulariser and the per-task parameter snapshots—while
adapting it to the common ``Net`` interface used throughout the repository.  A
``ResNet1D`` backbone supplies task-agnostic features, and all interaction with
training happens through ``observe`` so it plugs directly into
``life_experience``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from model.resnet1d import ResNet1D
from model.replay_utils import (
    ReplayInputMixin,
    unpack_y_to_class_labels,
)
from utils.training_metrics import macro_recall
from utils import misc_utils
from utils.class_weighted_loss import classification_cross_entropy
from utils.proximal_anchor import (
    anchor_curvature,
    apply_proximal_anchor,
    log_importance_summary,
    optimizer_learning_rate,
    proximal_anchor_coefficient,
    resolve_anchor_mode,
)


@dataclass
class EwcConfig:
    """Hyper-parameters pulled from ``args`` with sensible fallbacks."""

    inner_steps: int = 1
    lr: float = 0.03
    optimizer: str = "sgd"
    lamb: float = 1.0
    clipgrad: float = 0.0
    cls_lambda: float = 1.0

    @staticmethod
    def from_args(args: object) -> "EwcConfig":
        cfg = EwcConfig()
        # Override defaults with any args attributes that match. `None` means
        # "not set on args" (the parser registers `lamb` with a None default),
        # so the dataclass default stands.
        for field in cfg.__dataclass_fields__:
            value = getattr(args, field, None)
            if value is not None:
                setattr(cfg, field, value)
        return cfg


class Net(ReplayInputMixin, nn.Module):
    """EWC continual learner built on top of ``ResNet1D``."""

    def __init__(
        self, n_inputs: int, n_outputs: int, n_tasks: int, args: object
    ) -> None:
        super().__init__()

        assert n_tasks > 0, "EWC requires a positive number of tasks"

        self.cfg = EwcConfig.from_args(args)
        self.n_tasks = n_tasks
        self.n_outputs = n_outputs
        self.classes_per_task = misc_utils.build_task_class_list(
            n_tasks,
            n_outputs,
            nc_per_task=getattr(args, "nc_per_task_list", "")
            or getattr(args, "nc_per_task", None),
            classes_per_task=getattr(args, "classes_per_task", None),
        )
        self.nc_per_task = misc_utils.max_task_class_count(self.classes_per_task)
        self.is_task_incremental = True

        self.net = ResNet1D(n_outputs, args)

        self.opt = self._build_optimizer()
        self.class_weighted_ce = bool(getattr(args, "class_weighted_ce", True))
        self.incremental_loader_name = getattr(args, "loader", None)

        self.lamb = float(self.cfg.lamb)
        self.omega_uniform = bool(getattr(args, "anchor_omega_uniform", False))
        self.anchor_mode = resolve_anchor_mode(args)
        self.use_proximal_anchor = self.anchor_mode == "proximal"
        self.clipgrad = float(self.cfg.clipgrad) if self.cfg.clipgrad > 0 else None
        self.cls_lambda = float(self.cfg.cls_lambda)

        self.current_task: Optional[int] = None
        self._tasks_consolidated = 0

        self.fisher: Dict[str, torch.Tensor] = {}
        self.param_star: Dict[str, torch.Tensor] = {}

        self._fisher_accum: Optional[Dict[str, torch.Tensor]] = None
        self._fisher_count: int = 0

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        t: int,
        *,
        cil_all_seen_upto_task: int | None = None,
    ) -> torch.Tensor:
        return misc_utils.apply_task_incremental_logit_mask(
            self.net(x),
            t,
            self.classes_per_task,
            self.n_outputs,
            cil_all_seen_upto_task=cil_all_seen_upto_task,
            loader=self.incremental_loader_name,
        )

    # ------------------------------------------------------------------
    def observe(
        self, x: torch.Tensor, y: torch.Tensor, t: int
    ) -> Tuple[float, float, torch.Tensor | None]:
        if self.current_task is None:
            self.current_task = t
        elif t != self.current_task:
            self._consolidate_current_task()
            self.current_task = t

        self.net.train()

        metric_logits = None
        for _ in range(self.cfg.inner_steps):
            y_cls = unpack_y_to_class_labels(y)
            cls_logits = self.net(x)
            logits_for_loss = cls_logits
            if self.is_task_incremental:
                logits_for_loss = misc_utils.apply_task_incremental_logit_mask(
                    cls_logits,
                    t,
                    self.classes_per_task,
                    self.n_outputs,
                    cil_all_seen_upto_task=t,
                    loader=self.incremental_loader_name,
                )
            targets_for_loss = y_cls.long()
            loss_ce = classification_cross_entropy(
                logits_for_loss,
                targets_for_loss,
                class_weighted_ce=self.class_weighted_ce,
            )
            preds = torch.argmax(logits_for_loss, dim=1)
            cls_tr_rec = macro_recall(preds, y_cls.long())

            # The empirical Fisher is the squared gradient of the *task* loss
            # alone, so it is taken on its own backward pass and the gradients
            # are then cleared: leaving them in place would add a second copy of
            # the cross-entropy gradient to the update below, silently doubling
            # the effective learning rate on the task loss.
            self.opt.zero_grad()
            loss_ce.backward(retain_graph=True)
            self._accumulate_fisher(int(y_cls.size(0)))
            self.opt.zero_grad()

            if self.use_proximal_anchor:
                # The anchor is applied in closed form after the optimiser step
                # instead, so it contributes nothing to this backward pass -- and
                # therefore nothing to the global gradient-norm clip budget.
                penalty = torch.zeros(1, device=self._device())
            else:
                penalty = self._ewc_penalty()
            loss = self.cls_lambda * loss_ce + 0.5 * self.lamb * penalty
            loss.backward()

            if self.clipgrad is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipgrad)

            self.opt.step()
            if self.use_proximal_anchor:
                self._apply_proximal_anchor()
            metric_logits = logits_for_loss.detach()

        return float(loss.item()), cls_tr_rec, metric_logits

    # ------------------------------------------------------------------
    def _build_optimizer(self) -> torch.optim.Optimizer:
        params = list(self.net.parameters())
        optim = self.cfg.optimizer.lower()
        lr = float(self.cfg.lr)

        if optim == "adam":
            return torch.optim.Adam(params, lr=lr)

        return torch.optim.SGD(params, lr=lr, momentum=0.9)

    # ------------------------------------------------------------------
    def _compute_offsets(self, task: int) -> Tuple[int, int]:
        offset1, offset2 = misc_utils.compute_offsets(task, self.classes_per_task)
        return offset1, min(self.n_outputs, offset2)

    # ------------------------------------------------------------------
    def _device(self) -> torch.device:
        return next(self.net.parameters()).device

    # ------------------------------------------------------------------
    def _accumulate_fisher(self, batch_size: int) -> None:
        if self._fisher_accum is None:
            self._fisher_accum = {
                name: torch.zeros_like(param, device=param.device)
                for name, param in self.net.named_parameters()
                if param.requires_grad
            }
            self._fisher_count = 0

        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue
            grad = param.grad
            if grad is None:
                continue
            self._fisher_accum[name] += grad.detach().clone().pow(2) * batch_size
        self._fisher_count += batch_size

    # ------------------------------------------------------------------
    def _consolidate_current_task(self) -> None:
        if self._fisher_accum is None or self._fisher_count == 0:
            self._reset_fisher_accum()
            return

        scale = 1.0 / float(self._fisher_count)
        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue
            fisher_est = self._fisher_accum.get(name)
            if fisher_est is None:
                continue
            fisher_est = fisher_est * scale
            if self.omega_uniform:
                # See --anchor_omega_uniform: keep the merge rule, drop the
                # ranking. EWC merges by running average, so a constant stays
                # constant rather than counting tasks as SI's sum does.
                fisher_est = torch.ones_like(fisher_est)
            if name in self.fisher:
                prev = self.fisher[name]
                merged = (prev * self._tasks_consolidated + fisher_est) / (
                    self._tasks_consolidated + 1
                )
                self.fisher[name] = merged
            else:
                self.fisher[name] = fisher_est
            self.param_star[name] = param.detach().clone()

        log_importance_summary("ewc", self.current_task, self.fisher.values())
        self._tasks_consolidated += 1
        self._reset_fisher_accum()

    # ------------------------------------------------------------------
    def _ewc_penalty(self) -> torch.Tensor:
        if not self.fisher:
            return torch.zeros(1, device=self._device())
        penalty = torch.zeros(1, device=self._device())
        for name, param in self.net.named_parameters():
            if name not in self.fisher:
                continue
            penalty += (
                self.fisher[name] * (param - self.param_star[name]).pow(2)
            ).sum()
        return penalty

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _apply_proximal_anchor(self) -> None:
        """Apply the EWC quadratic anchor as a closed-form post-step update.

        EWC writes its penalty as ``0.5 * lamb * sum_i F_i (theta_i -
        theta_i^*)^2``, so the anchor curvature is ``k = lamb`` -- half SI's
        factor, because the ``0.5`` is already explicit here. See
        ``utils.proximal_anchor`` for the derivation.

        A no-op before the first consolidation, when ``self.fisher`` is empty.
        """
        if not self.fisher:
            return
        coefficient = proximal_anchor_coefficient(
            optimizer_learning_rate(self.opt), anchor_curvature("ewc", self.lamb)
        )
        for name, param in self.net.named_parameters():
            fisher = self.fisher.get(name)
            star = self.param_star.get(name)
            if fisher is None or star is None:
                continue
            apply_proximal_anchor(param, fisher, star, coefficient)

    # ------------------------------------------------------------------
    def _reset_fisher_accum(self) -> None:
        self._fisher_accum = None
        self._fisher_count = 0

    # ------------------------------------------------------------------
    def on_task_end(self) -> None:
        """Optional hook for the training harness (called if available)."""
        self._consolidate_current_task()
