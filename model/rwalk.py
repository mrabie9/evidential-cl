"""Riemannian Walk (RWalk) learner wired for the repo training loop.

The original script expected to orchestrate its own epochs and data loading.
This version keeps the same Fisher and path-integral bookkeeping but exposes the
``Net``/``observe`` interface so ``main.py`` can drive it batch-by-batch.  A
``ResNet1D`` backbone supplies the feature extractor to stay consistent with the
rest of the repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from model.resnet1d import ResNet1D
from model.detection_replay import (
    DetectionReplayMixin,
    noise_label_from_args,
    signal_mask_exclude_noise,
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
class RWalkConfig:
    """Hyper-parameters harvested from ``args`` with safe fallbacks."""

    inner_steps: int = 1
    lr: float = 0.001
    lamb: float = 1.0
    alpha: float = 0.9
    eps: float = 0.01

    optimizer: str = "sgd"
    clipgrad: Optional[float] = 100.0
    det_lambda: float = 1.0
    cls_lambda: float = 1.0
    det_memories: int = 2000
    det_replay_batch: int = 64

    @staticmethod
    def from_args(args: object | None) -> "RWalkConfig":
        cfg = RWalkConfig()
        if args is None:
            return cfg

        for field in cfg.__dataclass_fields__:
            # `None` means "not set on args" (the parser registers `lamb`,
            # `alpha` and `eps` with None defaults), so the dataclass default
            # stands -- which is what keeps RWalk's `alpha` from picking up
            # UCL's, the two methods sharing the flag name.
            value = getattr(args, field, None)
            if value is not None:
                setattr(cfg, field, value)
        return cfg


class Net(DetectionReplayMixin, LwfDistillationMixin, nn.Module):
    """RWalk continual learner built on top of ``ResNet1D``."""

    def __init__(
        self, n_inputs: int, n_outputs: int, n_tasks: int, args: object | None
    ) -> None:
        super().__init__()
        del n_inputs  # The ResNet1D front-end dictates its own receptive field

        assert n_tasks > 0, "RWalk requires at least one task"

        self.cfg = RWalkConfig.from_args(args)
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
        self.is_task_incremental = getattr(args, "class_incremental", True)
        self.noise_label: int | None = noise_label_from_args(args)
        self.incremental_loader_name: str | None = (
            getattr(args, "loader", None) if args is not None else None
        )

        self.net = ResNet1D(n_outputs, args)
        self.class_weighted_ce = bool(
            getattr(args, "class_weighted_ce", True) if args is not None else True
        )
        self.opt = self._build_optimizer()

        self.lamb = float(self.cfg.lamb)
        self.alpha = float(self.cfg.alpha)
        self.eps = float(self.cfg.eps)
        self.anchor_mode = resolve_anchor_mode(args)
        self.use_proximal_anchor = self.anchor_mode == "proximal"
        self._init_lwf_distillation(args, "rwalk")
        self.clipgrad = self.cfg.clipgrad
        self.det_lambda = float(self.cfg.det_lambda)
        self.cls_lambda = float(self.cfg.cls_lambda)
        self._init_det_replay(
            self.cfg.det_memories,
            self.cfg.det_replay_batch,
            enabled=bool(getattr(args, "use_detector_arch", False)),
        )

        self.current_task: Optional[int] = None
        self.tasks_trained: int = 0

        self.fisher: Dict[str, torch.Tensor] = {}
        self.s: Dict[str, torch.Tensor] = {}
        self.fisher_running: Dict[str, torch.Tensor] = {}
        self.s_running: Dict[str, torch.Tensor] = {}
        self.p_old: Dict[str, torch.Tensor] = {}
        self.param_star: Dict[str, torch.Tensor] = {}

        self._initialise_state()

    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, t: int, **kwargs
    ) -> torch.Tensor:  # pragma: no cover - thin wrapper
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
            global_noise_label=self.noise_label,
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

        # class_counts = getattr(self, "classes_per_task", None)
        # noise_label = None
        # if class_counts is not None:
        #     _, offset2 = misc_utils.compute_offsets(t, class_counts)
        #     noise_label = offset2 - 1
        metric_logits = None
        for _ in range(self.cfg.inner_steps):
            self.opt.zero_grad()
            y_cls = unpack_y_to_class_labels(y)
            cls_logits = self.net.forward_heads(x)[1]
            signal_mask = signal_mask_exclude_noise(y_cls, self.noise_label)
            logits_for_loss = cls_logits
            if self.is_task_incremental:
                logits_for_loss = misc_utils.apply_task_incremental_logit_mask(
                    cls_logits,
                    t,
                    self.classes_per_task,
                    self.n_outputs,
                    cil_all_seen_upto_task=t,
                    global_noise_label=self.noise_label,
                    loader=self.incremental_loader_name,
                )
            targets_for_loss = y_cls.long()
            loss_ce = classification_cross_entropy(
                logits_for_loss,
                targets_for_loss,
                class_weighted_ce=self.class_weighted_ce,
            )

            if signal_mask.any():
                preds = torch.argmax(logits_for_loss[signal_mask], dim=1)
                cls_tr_rec = macro_recall(preds, y_cls[signal_mask].long())
            else:
                cls_tr_rec = 0.0
            # else:
            #     loss_ce = cls_logits.new_zeros(1)
            #     cls_tr_rec = 0.0

            # det_loss = self.det_loss(det_logits, y_det.float())
            # det_replay = self._sample_det_memory()
            # if det_replay is not None:
            #     mem_x, mem_y = det_replay
            #     mem_det_logits, _ = self.net.forward_heads(mem_x)
            #     mem_loss = self.det_loss(mem_det_logits, mem_y.float())
            #     det_loss = 0.5 * (det_loss + mem_loss)

            if self.use_proximal_anchor:
                # The anchor is applied in closed form after the optimiser step
                # instead, so it contributes nothing to this backward pass -- and
                # therefore nothing to the global gradient-norm clip budget.
                regulariser = torch.zeros(1, device=self._device())
            else:
                regulariser = self._regulariser()
            loss = (
                self.cls_lambda * loss_ce
                # + self.det_lambda * det_loss
                + self.lamb * regulariser
            )
            if self.lwf_lambda != 0.0:
                # Function-space regulariser running alongside the parameter
                # anchor. Its gradient reaches `param.grad` and therefore the
                # running Fisher / path integral below, which is the existing
                # convention here: the regulariser's own gradient is already
                # integrated too.
                loss = loss + self.lwf_lambda * self._lwf_distillation_loss(
                    cls_logits, x, t
                )
            loss.backward()

            if self.clipgrad is not None:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clipgrad)
            self.opt.step()
            if self.use_proximal_anchor:
                self._apply_proximal_anchor()
            # Run after the anchor so the running statistics see the parameter
            # displacement the network actually kept, matching the loss form
            # where the anchor's pull is likewise already inside the step.
            self._update_running_statistics()
            metric_logits = logits_for_loss.detach()

        return float(loss.item()), cls_tr_rec, metric_logits

    # ------------------------------------------------------------------
    def on_task_end(self) -> None:
        """Optional hook so callers can flush the last task explicitly."""
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
    def _initialise_state(self) -> None:
        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("det_head"):
                continue
            zero = torch.zeros_like(param)
            device = param.device
            self.fisher[name] = zero.clone().to(device)
            self.s[name] = zero.clone().to(device)
            self.fisher_running[name] = zero.clone().to(device)
            self.s_running[name] = zero.clone().to(device)
            self.p_old[name] = param.detach().clone().to(device)
            self.param_star[name] = param.detach().clone().to(device)

    # ------------------------------------------------------------------
    def _regulariser(self) -> torch.Tensor:
        if self.tasks_trained == 0:
            return torch.zeros(1, device=self._device())

        penalty = torch.zeros(1, device=self._device())
        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("det_head"):
                continue
            self._ensure_state_device(name, param)
            fisher = self.fisher.get(name)
            s_term = self.s.get(name)
            star = self.param_star.get(name)
            if fisher is None or s_term is None or star is None:
                continue
            diff = param - star
            penalty = penalty + ((fisher + s_term) * diff.pow(2)).sum()
        return penalty

    # ------------------------------------------------------------------
    def _update_running_statistics(self) -> None:
        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("det_head"):
                continue
            self._ensure_state_device(name, param)
            grad = param.grad
            if grad is None:
                continue

            fisher_current = grad.detach().pow(2)
            prev_running = self.fisher_running[name]
            self.fisher_running[name] = (
                self.alpha * fisher_current + (1.0 - self.alpha) * prev_running
            )

            delta = param.detach() - self.p_old[name]
            fisher_distance = 0.5 * self.fisher_running[name] * delta.pow(2)
            loss_diff = -grad.detach() * delta
            s_update = loss_diff / (fisher_distance + self.eps)
            self.s_running[name] = self.s_running[name] + s_update.detach()

            self.p_old[name] = param.detach().clone()

    # ------------------------------------------------------------------
    def _consolidate_current_task(self) -> None:
        if self.current_task is None:
            return
        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("det_head"):
                continue
            self._ensure_state_device(name, param)
            self.fisher[name] = self.fisher_running[name].detach().clone()
            s_clone = 0.5 * self.s_running[name].detach().clone()
            self.s[name] = s_clone
            self.s_running[name] = s_clone.clone()
            self.param_star[name] = param.detach().clone()
            self.p_old[name] = param.detach().clone()
        log_importance_summary(
            "rwalk",
            self.current_task,
            (self.fisher[name] + self.s[name] for name in self.fisher),
        )
        # Freeze the just-finished task as the LwF teacher (no-op when the
        # distillation term is off).
        self._snapshot_lwf_teacher()
        self.tasks_trained += 1

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _apply_proximal_anchor(self) -> None:
        """Apply the RWalk quadratic anchor as a closed-form post-step update.

        RWalk writes its penalty as ``lamb * sum_i (F_i + s_i) (theta_i -
        theta_i^*)^2``, so the anchor curvature is ``k = 2 * lamb``. See
        ``utils.proximal_anchor`` for the derivation.

        RWalk's importance ``F + s`` is the one in this repository that most
        often goes negative: ``s`` accumulates ``-grad * delta`` divided by a
        Fisher-weighted distance, which is negative on every step where the loss
        rose. ``apply_proximal_anchor`` clamps it at zero, so such parameters are
        simply left unprotected instead of being actively pushed away from their
        anchor as the loss form does.

        A no-op before the first consolidation, when ``tasks_trained`` is 0.
        """
        if self.tasks_trained == 0:
            return
        coefficient = proximal_anchor_coefficient(
            optimizer_learning_rate(self.opt), anchor_curvature("rwalk", self.lamb)
        )
        for name, param in self.net.named_parameters():
            if not param.requires_grad or name.startswith("det_head"):
                continue
            self._ensure_state_device(name, param)
            fisher = self.fisher.get(name)
            s_term = self.s.get(name)
            star = self.param_star.get(name)
            if fisher is None or s_term is None or star is None:
                continue
            apply_proximal_anchor(param, fisher + s_term, star, coefficient)

    # ------------------------------------------------------------------
    def _ensure_state_device(self, name: str, param: torch.nn.Parameter) -> None:
        """Make sure cached tensors follow the parameter device."""
        device = param.device

        def move_if_needed(t: torch.Tensor) -> torch.Tensor:
            return t.to(device) if t.device != device else t

        if name in self.fisher:
            self.fisher[name] = move_if_needed(self.fisher[name])
        if name in self.s:
            self.s[name] = move_if_needed(self.s[name])
        if name in self.fisher_running:
            self.fisher_running[name] = move_if_needed(self.fisher_running[name])
        if name in self.s_running:
            self.s_running[name] = move_if_needed(self.s_running[name])
        if name in self.p_old:
            self.p_old[name] = move_if_needed(self.p_old[name])
        if name in self.param_star:
            self.param_star[name] = move_if_needed(self.param_star[name])

    # ------------------------------------------------------------------
    def _compute_offsets(self, task: int) -> Tuple[int, int]:
        offset1, offset2 = misc_utils.compute_offsets(task, self.classes_per_task)
        return offset1, min(self.n_outputs, offset2)

    # ------------------------------------------------------------------
    def _device(self) -> torch.device:
        return next(self.net.parameters()).device


__all__ = ["Net"]
