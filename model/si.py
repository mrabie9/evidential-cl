"""Synaptic Intelligence learner compatible with the repo training loop.

This rewrite mirrors the original SI implementation while exposing the same
``Net`` interface used elsewhere.  A ResNet1D backbone provides the shared
feature extractor and the SI path-integral bookkeeping lives inside the class so
``main.py`` can drive it through ``forward``/``observe``.
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
from model.lwf_regulariser import LwfDistillationMixin
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
class SiConfig:
    """Hyper-parameters with sensible fallbacks pulled from ``args``."""

    inner_steps: int = 1
    lr: float = 0.001
    si_c: float = 0.1
    si_epsilon: float = 0.01

    optimizer: str = "sgd"
    clipgrad: Optional[float] = 0.0
    cls_lambda: float = 1.0

    @staticmethod
    def from_args(args: object) -> "SiConfig":
        cfg = SiConfig()
        for field in cfg.__dataclass_fields__:
            # `None` means "not set on args" (the parser registers si_c and
            # si_epsilon with a None default), so the dataclass default stands.
            value = getattr(args, field, None)
            if value is not None:
                setattr(cfg, field, value)
        return cfg


class Net(ReplayInputMixin, LwfDistillationMixin, nn.Module):
    """Synaptic Intelligence continual learner built on ``ResNet1D``."""

    def __init__(
        self, n_inputs: int, n_outputs: int, n_tasks: int, args: object
    ) -> None:
        super().__init__()
        del n_inputs  # ResNet1D fixes its own receptive field

        assert n_tasks > 0, "SI requires at least one task"

        self.cfg = SiConfig.from_args(args)
        self.n_outputs = n_outputs
        self.n_tasks = n_tasks
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
        self.class_weighted_ce = bool(getattr(args, "class_weighted_ce", True))
        self.incremental_loader_name = getattr(args, "loader", None)
        self.opt = self._build_optimizer()

        self.si_c = float(self.cfg.si_c)
        self.epsilon = float(self.cfg.si_epsilon)
        self.anchor_mode = resolve_anchor_mode(args)
        self.use_proximal_anchor = self.anchor_mode == "proximal"
        self._init_lwf_distillation(args, "si")
        self.clipgrad = self.cfg.clipgrad
        self.cls_lambda = float(self.cfg.cls_lambda)

        self.current_task: Optional[int] = None
        self._param_to_key: Dict[str, str] = {}
        self._initialise_si_state()

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: int, **kwargs) -> torch.Tensor:
        logits = self.net(x)
        if not self.is_task_incremental:
            return logits
        cil = kwargs.get("cil_all_seen_upto_task")
        return misc_utils.apply_task_incremental_logit_mask(
            logits,
            t,
            self.classes_per_task,
            self.n_outputs,
            cil_all_seen_upto_task=cil,
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
            self.opt.zero_grad()
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

            if self.use_proximal_anchor:
                # The anchor is applied in closed form after the optimiser step
                # instead, so it contributes nothing to this backward pass -- and
                # therefore nothing to the global gradient-norm clip budget.
                surrogate = torch.zeros(1, device=self._device())
            else:
                surrogate = self._surrogate_loss()
            loss = self.cls_lambda * loss_ce + self.si_c * surrogate
            if self.lwf_lambda != 0.0:
                # Function-space regulariser running alongside the parameter
                # anchor. Its gradient reaches `param.grad` and therefore the
                # path integral below, which is the existing convention here:
                # the surrogate's own gradient is already integrated too.
                loss = loss + self.lwf_lambda * self._lwf_distillation_loss(
                    cls_logits, x, t
                )

            loss.backward()
            if self.clipgrad is not None and self.clipgrad > 0:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clipgrad)
            self.opt.step()
            if self.use_proximal_anchor:
                self._apply_proximal_anchor()
            # Run after the anchor so the path integral integrates the parameter
            # displacement the network actually kept, matching the loss form
            # where the anchor's pull is likewise already inside the step.
            self._update_path_integral()
            metric_logits = logits_for_loss.detach()

        return float(loss.item()), cls_tr_rec, metric_logits

    # ------------------------------------------------------------------
    def on_task_end(self) -> None:
        """Optional hook to consolidate the final task."""
        self._consolidate_current_task()

    # ------------------------------------------------------------------
    def _build_optimizer(self) -> torch.optim.Optimizer:
        params = self.net.parameters()
        optim = (self.cfg.optimizer or "adam").lower()
        lr = float(self.cfg.lr)

        if optim in {"adam", "adamw"}:
            opt_cls = torch.optim.AdamW if optim == "adamw" else torch.optim.Adam
            return opt_cls(params, lr=lr)
        if optim == "adagrad":
            return torch.optim.Adagrad(params, lr=lr)
        if optim in {"sgd", "sgd_momentum_decay"}:
            return torch.optim.SGD(params, lr=lr, momentum=0.9)
        return torch.optim.Adam(params, lr=lr)

    # ------------------------------------------------------------------
    def _initialise_si_state(self) -> None:
        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue
            key = name.replace(".", "__")
            self._param_to_key[name] = key
            initial = param.detach().clone()
            self.register_buffer(f"{key}_si_prev", initial.clone())
            self.register_buffer(f"{key}_si_omega", torch.zeros_like(param))
            self.register_buffer(f"{key}_si_W", torch.zeros_like(param))
            self.register_buffer(f"{key}_si_p_old", initial.clone())

    # ------------------------------------------------------------------
    def _update_path_integral(self) -> None:
        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue
            grad = param.grad
            if grad is None:
                continue
            key = self._param_to_key[name]
            W_buf = getattr(self, f"{key}_si_W")
            p_old_buf = getattr(self, f"{key}_si_p_old")
            W_buf.add_(-grad * (param.detach() - p_old_buf))
            p_old_buf.copy_(param.detach())

    # ------------------------------------------------------------------
    def _consolidate_current_task(self) -> None:
        if self.current_task is None:
            return
        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue
            key = self._param_to_key[name]
            prev = getattr(self, f"{key}_si_prev")
            omega = getattr(self, f"{key}_si_omega")
            W_buf = getattr(self, f"{key}_si_W")
            delta = param.detach() - prev
            omega.add_(W_buf / (delta.pow(2) + self.epsilon))
            prev.copy_(param.detach())
            W_buf.zero_()
            getattr(self, f"{key}_si_p_old").copy_(param.detach())
        log_importance_summary(
            "si",
            self.current_task,
            (getattr(self, f"{key}_si_omega") for key in self._param_to_key.values()),
        )
        # Freeze the just-finished task as the LwF teacher (no-op when the
        # distillation term is off).
        self._snapshot_lwf_teacher()

    # ------------------------------------------------------------------
    def _surrogate_loss(self) -> torch.Tensor:
        device = self._device()
        loss = torch.zeros(1, device=device)
        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue
            key = self._param_to_key[name]
            omega = getattr(self, f"{key}_si_omega")
            prev = getattr(self, f"{key}_si_prev")
            loss = loss + (omega * (param - prev).pow(2)).sum()
        return loss

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _apply_proximal_anchor(self) -> None:
        """Apply the SI quadratic anchor as a closed-form post-step update.

        SI writes its penalty as ``si_c * sum_i Omega_i (theta_i - prev_i)^2``,
        so the anchor curvature is ``k = 2 * si_c``. See
        ``utils.proximal_anchor`` for the derivation and for why SI's
        unrectified ``Omega`` (``W / (delta^2 + epsilon)`` is not sign
        constrained) is clamped at zero here.

        A no-op on the first task, where ``Omega`` is still all zeros.
        """
        coefficient = proximal_anchor_coefficient(
            optimizer_learning_rate(self.opt), anchor_curvature("si", self.si_c)
        )
        for name, param in self.net.named_parameters():
            key = self._param_to_key.get(name)
            if key is None:
                continue
            apply_proximal_anchor(
                param,
                getattr(self, f"{key}_si_omega"),
                getattr(self, f"{key}_si_prev"),
                coefficient,
            )

    # ------------------------------------------------------------------
    def _compute_offsets(self, task: int) -> Tuple[int, int]:
        offset1, offset2 = misc_utils.compute_offsets(task, self.classes_per_task)
        return offset1, min(self.n_outputs, offset2)

    # ------------------------------------------------------------------
    def _device(self) -> torch.device:
        return next(self.net.parameters()).device


__all__ = ["Net"]
