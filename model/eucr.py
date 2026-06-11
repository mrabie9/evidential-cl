"""EUCR: Evidential Uncertainty Channel Regularisation, La-MAML compatible.

EUCR is the evidential analogue of EWC. A single evidential (Dempster-Shafer)
1D-ResNet backbone with per-stage probes (:mod:`model.eucr_backbone`) is trained
with an evidential classification loss plus deep evidential supervision on the
probes. After each task, a Fisher-style *evidential importance* is read from the
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
from model.detection_replay import (
    noise_label_from_args,
    signal_mask_exclude_noise,
    unpack_y_to_class_labels,
)
from model.eucr_backbone import EucrResNet1D
from model.evidential_modules import EvidentialLoss, PignisticNLLLoss
from utils import misc_utils
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
        self.noise_label = noise_label_from_args(args)
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
        if str(self.cfg.optimizer).lower() == "sgd":
            return torch.optim.SGD(
                [
                    {"params": self.backbone_params, "lr": lr},
                    {"params": self.evidential_params, "lr": lr * 0.25},
                ],
                momentum=0.9,
            )
        return torch.optim.Adam(
            [
                {"params": self.backbone_params, "lr": lr},
                {"params": self.evidential_params, "lr": lr * 0.25},
            ],
            eps=1e-7,
            amsgrad=True,
        )

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
            global_noise_label=self.noise_label,
            loader=self.incremental_loader_name,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: int,
        *,
        cil_all_seen_upto_task: int | None = None,
    ) -> torch.Tensor:
        eu = self.backbone(x)
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
            # Evidential heads use exp/normalise chains that are unstable in AMP.
            with torch.autocast(device_type=amp_device, enabled=False):
                eu, _features, _omegas, beliefs, probe_outs = self.backbone(
                    x, return_probes=True
                )
                head_eu = eu[:, : self.n_outputs].float()
                head_loss = self.criterion(head_eu, y_cls, beliefs, epoch)

                probe_loss = torch.zeros((), device=head_loss.device)
                for eu_p, bel_p, _om_p in probe_outs:
                    probe_loss = probe_loss + self.criterion(
                        eu_p[:, : self.n_outputs].float(), y_cls, bel_p, epoch
                    )
                if probe_outs:
                    probe_loss = probe_loss / len(probe_outs)

                reg = cons.penalty(self.backbone, self.importance, self.theta_star)
                loss = (
                    head_loss
                    + self.probe_loss_weight * probe_loss
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

            with torch.no_grad():
                masked = self._mask(head_eu.detach(), t, cil_all_seen_upto_task=t)
                metric_logits = masked
                signal_mask = signal_mask_exclude_noise(y_cls, self.noise_label)
                if signal_mask.any():
                    preds = torch.argmax(masked[signal_mask], dim=1)
                    cls_tr_rec = macro_recall(preds, y_cls[signal_mask])
                else:
                    cls_tr_rec = 0.0
            loss_value = float(loss.item())

        return loss_value, float(cls_tr_rec), metric_logits

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


__all__ = ["Net", "EucrConfig"]
