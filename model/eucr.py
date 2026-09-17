"""EUCR: Evidential Uncertainty Channel Regularisation, La-MAML compatible.

EUCR is the evidential analogue of EWC. A single evidential (Dempster-Shafer)
1D-ResNet backbone with per-stage probes (:mod:`model.eucr_backbone`) is trained
with an evidential classification loss plus deep evidential supervision on the
probes. After each task, a MAS-style *evidential importance* is read from the
backbone probe uncertainty (:mod:`model.eucr_consolidation`) and a quadratic
penalty anchors the shared backbone weights while later tasks are learned -- no
pruning and no binary masks.

Like ``model.ewc``, this model keeps a single global evidential head and relies
on :func:`utils.misc_utils.apply_task_incremental_logit_mask` for TIL / CIL
evaluation, so it plugs straight into ``life_experience`` and supports both
incremental loaders. End-of-task importance estimation runs in
``finalize_task_after_training`` (called by the harness with the task loader).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from model import eucr_consolidation as cons
from model.replay_utils import (
    unpack_y_to_class_labels,
)
from model.eucr_backbone import EucrResNet1D
from model.evidential_modules import EvidentialLoss, PignisticNLLLoss
from utils import misc_utils
from utils.class_weighted_loss import classification_cross_entropy
from utils.training_metrics import macro_recall


def _parse_probe_stages(spec) -> tuple:
    if spec is None:
        return (1, 2, 3, 4)
    if isinstance(spec, (list, tuple)):
        return tuple(int(s) for s in spec)
    parts = str(spec).replace(";", ",").split(",")
    stages = tuple(int(p) for p in parts if p.strip())
    return stages or (1, 2, 3, 4)


@dataclass
class EucrConfig:
    """Hyper-parameters pulled from ``args`` with sensible fallbacks."""

    lr: float = 1e-3
    optimizer: str = "adam"
    inner_steps: int = 1
    reg_lambda: float = 1000.0
    probe_loss_weight: float = 0.5
    reg_granularity: str = "channel"
    nu: float = 0.9
    proto_factor: int = 20
    kl_warmup_epochs: int = 35
    importance_batches: Optional[int] = None
    grad_clip_norm: float = 5.0
    eucr_depth: int = 18
    eucr_bn_stats: str = "batch"
    eucr_anchor_mode: str = "loss"
    eucr_ce_aux_weight: float = 0.0
    eucr_ds_detach: bool = False
    eucr_head_lr_scale: float = 0.25

    @staticmethod
    def from_args(args: object) -> "EucrConfig":
        cfg = EucrConfig()
        for field in cfg.__dataclass_fields__:
            if hasattr(args, field) and getattr(args, field) is not None:
                setattr(cfg, field, getattr(args, field))
        return cfg


class Net(nn.Module):
    """EUCR continual learner built on an evidential ResNet-1D backbone."""

    def __init__(
        self, n_inputs: int, n_outputs: int, n_tasks: int, args: object
    ) -> None:
        super().__init__()
        assert n_tasks > 0, "EUCR requires a positive number of tasks"

        self.cfg = EucrConfig.from_args(args)
        self.n_tasks = n_tasks
        self.n_outputs = n_outputs
        self.is_task_incremental = True

        self.classes_per_task = misc_utils.build_task_class_list(
            n_tasks,
            n_outputs,
            nc_per_task=getattr(args, "nc_per_task_list", "")
            or getattr(args, "nc_per_task", None),
            classes_per_task=getattr(args, "classes_per_task", None),
        )
        self.nc_per_task = misc_utils.max_task_class_count(self.classes_per_task)
        self.incremental_loader_name = getattr(args, "loader", None)

        probe_stages = _parse_probe_stages(getattr(args, "probe_stages", "1,2,3,4"))
        num_blocks = (3, 4, 6, 3) if int(self.cfg.eucr_depth) == 34 else (2, 2, 2, 2)
        self.backbone = EucrResNet1D(
            num_classes=n_outputs,
            args=args,
            num_blocks=num_blocks,
            nu=float(self.cfg.nu),
            probe_stages=probe_stages,
            proto_factor=int(self.cfg.proto_factor),
            metric=str(getattr(args, "eucr_distance_metric", "cosine")),
            head=str(getattr(args, "eucr_head", "dm")),
            classes_per_task=self.classes_per_task,
            belief_init=str(getattr(args, "eucr_belief_init", "random")),
            temper=float(getattr(args, "eucr_temper", 0.0) or 0.0),
            activation_norm=str(getattr(args, "eucr_activation_norm", "max")),
            readout_scale=str(getattr(args, "eucr_readout_scale", "untemper")),
            ds_detach=bool(getattr(args, "eucr_ds_detach", False)),
        )

        self.head_mode = str(getattr(args, "eucr_head", "dm")).lower()
        if self.head_mode == "pignistic":
            # Pignistic head emits a proper distribution -> NLL, no KL warm-up.
            self.criterion = PignisticNLLLoss(num_classes=n_outputs)
        else:
            self.criterion = EvidentialLoss(
                num_classes=n_outputs, kl_warmup_epochs=int(self.cfg.kl_warmup_epochs)
            )

        self.reg_lambda = float(self.cfg.reg_lambda)
        self.probe_loss_weight = float(self.cfg.probe_loss_weight)
        self.reg_granularity = str(self.cfg.reg_granularity)
        self.uncertainty_mode = str(getattr(args, "eucr_uncertainty", "both"))
        self.importance_batches = self.cfg.importance_batches
        self.inner_steps = max(1, int(self.cfg.inner_steps))
        self.clipgrad = (
            float(self.cfg.grad_clip_norm)
            if self.cfg.grad_clip_norm and self.cfg.grad_clip_norm > 0
            else None
        )

        self.backbone_params, self.evidential_params = self._split_parameters()
        self.opt = self._build_optimizer()

        self.current_task: Optional[int] = None
        self.importance: Optional[Dict[str, torch.Tensor]] = None
        self.theta_star: Optional[Dict[str, torch.Tensor]] = None

        # BatchNorm running statistics are buffers, so consolidation cannot touch
        # them and each task overwrites them wholesale. Measured (Gate B0): that
        # accounts for ~84% of EUCR's end-of-sequence task-0 forgetting, and it is
        # specific to this head -- an EWC control on the same backbone shows a
        # running-vs-batch gap of <=0.013 F1. The cosine-prototype head reads
        # feature *direction*, so a shift in the normalisation statistics rotates
        # the whole feature cloud onto one prototype.
        #   running  -- shipped behaviour.
        #   freeze   -- stop updating the statistics after the first task.
        #   per_task -- keep one set of statistics per task and select by task id.
        #               Free in TIL, where the task id is given at test time.
        # Direction G: auxiliary linear+CE head on the shared backbone.
        self.ce_aux_weight = float(self.cfg.eucr_ce_aux_weight)
        self.ds_detach = bool(self.cfg.eucr_ds_detach)
        self.class_weighted_ce = bool(getattr(args, "class_weighted_ce", True))

        # Anchor form. Explicit descent on lambda*Omega*(theta-theta*)^2 has
        # curvature k = 2*lambda*Omega and is stable only while lr*k < 2. EUCR's
        # Omega is normalised to unit MEAN but its median is 2.7e-4 and its max
        # 1.7e4, so the mean is set entirely by the tail: at the shipped
        # lambda=0.09 the largest coordinate sits at lr*k = 30.7, and the anchor
        # gradient outweighs the task gradient 60x. Because clip_grad_norm_
        # rescales by one global scalar, that drags the task signal down to 1.6%
        # of its size. "proximal" keeps the anchor out of the backward pass and
        # applies it in closed form after the step, where it cannot overshoot.
        self.anchor_mode = str(self.cfg.eucr_anchor_mode).lower()
        self.use_proximal_anchor = self.anchor_mode == "proximal"

        self.bn_stats_mode = str(self.cfg.eucr_bn_stats).lower()
        self._bn_modules = [
            m
            for m in self.backbone.modules()
            if isinstance(m, nn.modules.batchnorm._BatchNorm)
            and m.running_mean is not None
        ]
        self._bn_bank: Dict[int, list] = {}

    # ------------------------------------------------------------------
    def _split_parameters(self) -> Tuple[list, list]:
        """Partition backbone params into shared-backbone vs evidential-head groups.

        The Dempster-Shafer head / probes (``ds_head``, ``dm_head``, ``probes``,
        which includes the ``DistanceActivation`` gamma/alpha params) produce
        gradients orders of magnitude larger than the convolutional feature
        extractor. Keeping the two groups separate lets the optimizer and the
        gradient clipper treat them independently so the head cannot starve the
        backbone of its gradient budget.
        """
        backbone_params: list[nn.Parameter] = []
        evidential_params: list[nn.Parameter] = []
        for name, param in self.backbone.named_parameters():
            if not param.requires_grad:
                continue
            if any(token in name for token in ("ds_head", "dm_head", "probes")):
                evidential_params.append(param)
            else:
                backbone_params.append(param)
        return backbone_params, evidential_params

    def _build_optimizer(self) -> torch.optim.Optimizer:
        lr = float(self.cfg.lr)
        head_lr = lr * float(self.cfg.eucr_head_lr_scale)
        groups = [
            {"params": self.backbone_params, "lr": lr},
            {"params": self.evidential_params, "lr": head_lr},
        ]
        if str(self.cfg.optimizer).lower() == "sgd":
            return torch.optim.SGD(groups, momentum=0.9)
        return torch.optim.Adam(groups, eps=1e-7, amsgrad=True)

    def _device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    def compute_offsets(self, task: int) -> Tuple[int, int]:
        offset1, offset2 = misc_utils.compute_offsets(task, self.classes_per_task)
        return offset1, min(self.n_outputs, offset2)

    # ------------------------------------------------------------------
    def _mask(
        self, logits: torch.Tensor, t: int, cil_all_seen_upto_task=None
    ) -> torch.Tensor:
        return misc_utils.apply_task_incremental_logit_mask(
            logits,
            t,
            self.classes_per_task,
            self.n_outputs,
            cil_all_seen_upto_task=cil_all_seen_upto_task,
            loader=self.incremental_loader_name,
        )

    # ------------------------------------------------------------------
    def _bn_snapshot(self, task: int) -> None:
        self._bn_bank[int(task)] = [
            (m.running_mean.detach().clone(), m.running_var.detach().clone())
            for m in self._bn_modules
        ]

    def _bn_apply(self, task: int):
        """Swap in task ``task``'s statistics; return the ones displaced."""
        if not self._bn_bank:
            return None
        key = min(int(task), max(self._bn_bank))
        if key not in self._bn_bank:
            return None
        prev = [
            (m.running_mean.detach().clone(), m.running_var.detach().clone())
            for m in self._bn_modules
        ]
        for m, (mean, var) in zip(self._bn_modules, self._bn_bank[key]):
            m.running_mean.copy_(mean)
            m.running_var.copy_(var)
        return prev

    def _bn_restore(self, prev) -> None:
        if prev is None:
            return
        for m, (mean, var) in zip(self._bn_modules, prev):
            m.running_mean.copy_(mean)
            m.running_var.copy_(var)

    def forward(
        self,
        x: torch.Tensor,
        t: int,
        *,
        cil_all_seen_upto_task: int | None = None,
        bn_training: bool | None = None,
    ) -> torch.Tensor:
        if bn_training is None and self.bn_stats_mode == "batch":
            # Match the harness. Every ResNet1D-based learner reaches its backbone
            # through ResNet1D.forward, whose bn_training defaults to True, so the
            # whole benchmark is scored with BATCH statistics. EUCR had no such
            # wrapper and was silently scored with frozen running statistics --
            # a different protocol from every model it is compared against, worth
            # 0.31 final F1 and 0.37 BWT on a 4-task run.
            bn_training = True
        if bn_training is None:
            return self._forward_inner(x, t, cil_all_seen_upto_task)
        prev_mode = self.backbone.training
        self.backbone.train(bn_training)
        try:
            return self._forward_inner(x, t, cil_all_seen_upto_task)
        finally:
            self.backbone.train(prev_mode)

    def _forward_inner(
        self, x: torch.Tensor, t: int, cil_all_seen_upto_task: int | None
    ) -> torch.Tensor:
        if self.bn_stats_mode == "per_task" and not self.backbone.training:
            prev = self._bn_apply(t)
            try:
                eu = self.backbone(x, t, cil_all_seen_upto_task=cil_all_seen_upto_task)
                logits = eu[:, : self.n_outputs]
                return self._mask(
                    logits, t, cil_all_seen_upto_task=cil_all_seen_upto_task
                )
            finally:
                self._bn_restore(prev)
        eu = self.backbone(x, t, cil_all_seen_upto_task=cil_all_seen_upto_task)
        logits = eu[:, : self.n_outputs]
        return self._mask(logits, t, cil_all_seen_upto_task=cil_all_seen_upto_task)

    # ------------------------------------------------------------------
    def observe(
        self, x: torch.Tensor, y: torch.Tensor, t: int
    ) -> Tuple[float, float, torch.Tensor | None]:
        if self.current_task is None:
            self.current_task = t
        elif t != self.current_task:
            self.current_task = t

        self.backbone.train()
        y_cls = unpack_y_to_class_labels(y).long()
        epoch = getattr(self, "real_epoch", None)

        loss_value = 0.0
        cls_tr_rec = 0.0
        metric_logits = None
        device = self._device()
        amp_device = "cuda" if device.type == "cuda" else "cpu"

        for _ in range(self.inner_steps):
            # Keep the evidential head in fp32. Measured, not assumed: forcing
            # autocast on here produces no non-finite loss at all (0 skipped
            # steps over 9 four-task runs), so "unstable" is the wrong word --
            # but it costs accuracy for no speed. Paired per-seed diagonal F1
            # against fp32: bf16 -0.032 +/- 0.031 (not significant), fp16
            # -0.085 +/- 0.012 (significant; fp16 also has no GradScaler here,
            # which is likely the bulk of its penalty). AMP buys ~5% epoch time,
            # because this model is dominated by small kernel launches rather
            # than matmul FLOPs. main.py disables AMP for eucr as well; this
            # guard makes the model safe to call from anywhere.
            with torch.autocast(device_type=amp_device, enabled=False):
                want_ce = self.ce_aux_weight > 0.0
                out = self.backbone(x, t, return_probes=True, return_ce=want_ce)
                eu, _features, _omegas, beliefs, probe_outs = out[:5]
                ce_logits = out[5] if want_ce else None
                head_eu = eu[:, : self.n_outputs].float()
                head_loss = self.criterion(head_eu, y_cls, beliefs, epoch)

                probe_loss = torch.zeros((), device=head_loss.device)
                for eu_p, bel_p, _om_p in probe_outs:
                    probe_loss = probe_loss + self.criterion(
                        eu_p[:, : self.n_outputs].float(), y_cls, bel_p, epoch
                    )
                if probe_outs:
                    probe_loss = probe_loss / len(probe_outs)

                ce_loss = torch.zeros((), device=head_loss.device)
                if ce_logits is not None:
                    offset1, offset2 = self.compute_offsets(t)
                    ce_loss = classification_cross_entropy(
                        ce_logits[:, offset1:offset2],
                        y_cls - offset1,
                        class_weighted_ce=self.class_weighted_ce,
                    )

                reg = (
                    torch.zeros((), device=head_loss.device)
                    if self.use_proximal_anchor
                    else cons.penalty(
                        self.backbone, self.importance, self.theta_star
                    )
                )
                loss = (
                    head_loss
                    + self.probe_loss_weight * probe_loss
                    + self.ce_aux_weight * ce_loss
                    + self.reg_lambda * reg
                )

            if not torch.isfinite(loss).all():
                print(
                    "[WARN] EUCR skipping optimizer step: non-finite loss "
                    f"(task={t}, epoch={epoch})."
                )
                with torch.no_grad():
                    masked = self._mask(head_eu.detach(), t, cil_all_seen_upto_task=t)
                    metric_logits = masked
                loss_value = float("nan")
                break

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.clipgrad is not None:
                # Clip each group against its own budget: a single global clip
                # lets the high-gradient evidential head scale the backbone's
                # update down to near-zero and freeze the feature extractor.
                torch.nn.utils.clip_grad_norm_(self.backbone_params, self.clipgrad)
                torch.nn.utils.clip_grad_norm_(self.evidential_params, self.clipgrad)
            self.opt.step()
            if self.use_proximal_anchor:
                self._apply_proximal_anchor()

            with torch.no_grad():
                masked = self._mask(head_eu.detach(), t, cil_all_seen_upto_task=t)
                metric_logits = masked
                preds = torch.argmax(masked, dim=1)
                cls_tr_rec = macro_recall(preds, y_cls)
            loss_value = float(loss.item())

        return loss_value, float(cls_tr_rec), metric_logits

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _apply_proximal_anchor(self) -> None:
        """Apply the quadratic anchor as a closed-form post-step update.

        Backward-Euler on the anchor evaluates its gradient at the *new* point,
        ``theta_new = theta+ - lr*2*lambda*Omega*(theta_new - theta*)``, which
        solves to a convex combination::

            b = 2 * lr * lambda * Omega
            theta_new = (theta+ + b * theta*) / (1 + b)

        The mixing weight ``b/(1+b)`` saturates at 1 for any Omega, so the update
        can never overshoot: Omega -> 0 leaves the parameter free, Omega -> inf
        pins it to theta*. No-op on the first task, where there is no anchor yet.

        Note the useful lambda is far larger than the loss form's, because b is
        scaled by lr: b = 1 at unit Omega needs lambda = 1/(2*lr) = 50 at lr 0.01.
        """
        if not self.importance or not self.theta_star:
            return
        scale = 2.0 * float(self.opt.param_groups[0]["lr"]) * self.reg_lambda
        if scale == 0.0:
            return
        for name, param in cons._named_consolidatable_params(self.backbone):
            if name not in self.importance or name not in self.theta_star:
                continue
            b = self.importance[name].to(param.device) * scale
            anchor = self.theta_star[name].to(param.device)
            param.copy_((param + b * anchor) / (1.0 + b))

    # ------------------------------------------------------------------
    def finalize_task_after_training(self, train_loader) -> None:
        """Estimate and accumulate evidential importance, then anchor weights."""
        device = self._device()
        new_importance = cons.compute_importance(
            self.backbone,
            train_loader,
            device,
            max_batches=self.importance_batches,
            normalize=True,
            uncertainty_mode=self.uncertainty_mode,
        )
        if self.reg_granularity == "channel":
            new_importance = cons.to_channel(self.backbone, new_importance)
        self.importance = cons.accumulate(self.importance, new_importance)
        self.theta_star = cons.snapshot(self.backbone)

        task = int(self.current_task or 0)
        if self.bn_stats_mode == "per_task":
            self._bn_snapshot(task)
        elif self.bn_stats_mode == "freeze":
            # momentum 0 leaves running = (1-0)*running + 0*batch, i.e. frozen.
            for m in self._bn_modules:
                m.momentum = 0.0


__all__ = ["Net", "EucrConfig"]
