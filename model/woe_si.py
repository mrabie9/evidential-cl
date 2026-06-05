"""Weight-of-Evidence Synaptic Intelligence (WoE-SI) continual learner.

This module implements "Option C" of a Dempster-Shafer (DS) based importance
estimator for regularisation-based continual learning. It is structurally a
Synaptic Intelligence learner (Zenke et al. 2017), but the tracked scalar that
defines per-parameter importance is **not** the task loss. It is the
Least-Commitment information content ``I_2(m)`` of the DS mass function ``m``
that underlies the classifier's softmax output (Denoeux 2019, "Logistic
Regression, Neural Networks and Dempster-Shafer Theory: A New Perspective",
arXiv:1807.01846v2).

Concretely, for a standard linear readout ``z_k = sum_j beta_jk * phi_j + beta_0k``
on top of penultimate backbone features ``phi``:

* per-feature/per-class weights of evidence (Denoeux Eq 25)::

      w_jk(x) = beta_jk * phi'_j(x) + alpha_jk

  with centered features ``phi'_j = phi_j - mu_j`` and Least-Commitment offsets
  ``alpha_jk = beta_0k / J`` (the centered multi-category generalisation of the
  binary solution in Sec 4.1 Eq 35; ``sum_j alpha_jk = beta_0k``, Eq 29).

* per-class total weights of evidence (Eq 27)::

      w_k_plus  = sum_j relu( w_jk)     # mass supporting {theta_k}
      w_k_minus = sum_j relu(-w_jk)     # mass supporting complement of {theta_k}

* Least-Commitment information content (Eq 10, p=2)::

      I_2(m) = sum_k [ w_k_plus^2 + w_k_minus^2 ]

``I_2(m)`` is a differentiable scalar measuring how *committed* (informative,
far from vacuous) the evidence is. Substituting it for the loss in the SI path
integral yields, per parameter ``theta_i`` and optimiser step::

      h_i      = d I_2(m) / d theta_i           (mean over minibatch)
      delta_i  = theta_i(after step) - theta_i(before step)
      omega_i += h_i * delta_i

so importance is each parameter's accumulated share of the *committed evidence*
built up over the task. Parameters that drove the model from vacuity toward
committed evidence get anchored; the rest stay free.

CAVEAT (documented, not a bug): the DS construction is exact only at the linear
readout (penultimate features -> logits). For the ResNet backbone, ``I_2(m)`` is
a function of the readout weights and the penultimate activations; its gradient
nonetheless flows through the whole backbone via backprop, so *every* parameter
receives an importance. See DESIGN_NOTE.md.

The class mirrors ``model/si.py`` exactly: importance buffers, a per-step
accumulation hook, an end-of-task consolidation hook, and a quadratic penalty
added to the training loss. It is a drop-in CL method (``--model woe_si``)
runnable in both TIL and CIL with no harness changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from model.resnet1d import ResNet1D
from model.detection_replay import (
    DetectionReplayMixin,
    noise_label_from_args,
    signal_mask_exclude_noise,
    unpack_y_to_class_labels,
)
from utils.training_metrics import macro_recall
from utils import misc_utils
from utils.class_weighted_loss import classification_cross_entropy

# ======================================================================
# Pure functional core (imported directly by the unit tests)
# ======================================================================
_CENTERING_MODES = ("centered_uniform", "raw_uniform", "full_lc")


def compute_weights_of_evidence(
    features: torch.Tensor,
    readout_weight: torch.Tensor,
    readout_bias: torch.Tensor,
    feature_mean: torch.Tensor,
    centering_mode: str = "centered_uniform",
) -> torch.Tensor:
    """Compute Denoeux Eq 25 weights of evidence ``w_jk(x)``.

    Args:
        features: Penultimate features ``phi`` with shape ``(batch, J)``.
        readout_weight: Linear readout weight ``beta`` with shape ``(K, J)``
            (``torch.nn.Linear.weight`` convention, ``beta_kj``).
        readout_bias: Linear readout bias ``beta_0`` with shape ``(K,)``.
        feature_mean: Running feature mean ``mu`` with shape ``(J,)`` (the EMA of
            ``phi`` over the current task). Ignored for ``"raw_uniform"``.
        centering_mode: One of ``{"centered_uniform", "raw_uniform", "full_lc"}``.
            ``"centered_uniform"`` (default) centres features and uses the
            Least-Commitment uniform offset ``alpha_jk = beta_0k / J``.
            ``"raw_uniform"`` skips centring (uses raw ``phi``) but keeps the same
            offset. ``"full_lc"`` (exact Sec 4.2 identification) is not
            implemented and raises ``NotImplementedError``.

    Returns:
        Weights of evidence ``w`` with shape ``(batch, K, J)`` where
        ``w[b, k, j] = beta_kj * phi'_bj + beta_0k / J``.

    Usage:
        >>> w = compute_weights_of_evidence(phi, fc.weight, fc.bias, mu)
        >>> w.shape
        torch.Size([batch, K, J])
    """
    if centering_mode not in _CENTERING_MODES:
        raise ValueError(
            f"centering_mode must be one of {_CENTERING_MODES}, got {centering_mode!r}"
        )
    if centering_mode == "full_lc":
        # The exact Least-Commitment identification of Sec 4.2 (solving for the
        # alpha_jk that minimise total information content subject to Eq 29) is
        # intentionally left unimplemented; the centred-uniform default is the
        # documented design choice. See DESIGN_NOTE.md, Design Decision #1.
        raise NotImplementedError(
            "centering_mode='full_lc' (exact Denoeux Sec 4.2 identification) is "
            "not implemented; use 'centered_uniform' (default) or 'raw_uniform'."
        )

    feature_count = features.shape[1]
    if centering_mode == "centered_uniform":
        centered_features = features - feature_mean.unsqueeze(0)
    else:  # raw_uniform
        centered_features = features

    # w[b, k, j] = beta_kj * phi'_bj + alpha_kj, with alpha_kj = beta_0k / J.
    evidence = readout_weight.unsqueeze(0) * centered_features.unsqueeze(1)
    offset = (readout_bias / feature_count).view(1, -1, 1)
    return evidence + offset


def per_class_total_evidence(
    weights_of_evidence: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Denoeux Eq 27 per-class total weights of evidence.

    Args:
        weights_of_evidence: Tensor ``w`` with shape ``(batch, K, J)``.

    Returns:
        Tuple ``(w_plus, w_minus)`` each with shape ``(batch, K)``: the total
        evidence supporting ``{theta_k}`` and its complement, respectively.
    """
    w_plus = torch.relu(weights_of_evidence).sum(dim=2)
    w_minus = torch.relu(-weights_of_evidence).sum(dim=2)
    return w_plus, w_minus


def information_content(
    weights_of_evidence: torch.Tensor,
    conflict_weighting: bool = False,
) -> torch.Tensor:
    """Compute the per-sample Least-Commitment information content ``I_2(m)``.

    Implements Denoeux Eq 10 with ``p = 2`` over the ``2K`` focal sets (the ``K``
    singletons ``{theta_k}`` with weight ``w_k_plus`` and the ``K`` complements
    with weight ``w_k_minus``)::

        I_2(m) = sum_k [ w_k_plus^2 + w_k_minus^2 ]

    Args:
        weights_of_evidence: Tensor ``w`` with shape ``(batch, K, J)``.
        conflict_weighting: If ``True``, multiply each class term by a
            kappa-style conflict factor ``(1 + kappa_k)`` derived from the
            pairwise overlap of positive and negative evidence (Eqs 21/31).
            Default ``False`` (the plain ``I_2`` path). See DESIGN_NOTE.md.

    Returns:
        Per-sample information content with shape ``(batch,)``. Always ``>= 0``.
    """
    w_plus, w_minus = per_class_total_evidence(weights_of_evidence)
    per_class = w_plus.pow(2) + w_minus.pow(2)
    if conflict_weighting:
        per_class = per_class * _conflict_factor(w_plus, w_minus)
    return per_class.sum(dim=1)


def _conflict_factor(w_plus: torch.Tensor, w_minus: torch.Tensor) -> torch.Tensor:
    """Kappa-style conflict factor for the optional ablation (Eqs 21/31).

    The two simple mass functions feeding class ``k`` place belief
    ``1 - exp(-w_k_plus)`` on ``{theta_k}`` and ``1 - exp(-w_k_minus)`` on its
    complement. As ``{theta_k}`` and its complement are disjoint, Dempster's
    combination assigns their product to the empty set, i.e. the degree of
    conflict ``kappa_k``. We return ``1 + kappa_k`` so that classes whose
    evidence is internally conflicting receive *more* importance weight, letting
    the user test whether conflict-awareness changes parameter selection.

    Args:
        w_plus: Positive total evidence ``(batch, K)``.
        w_minus: Negative total evidence ``(batch, K)``.

    Returns:
        Conflict factor with shape ``(batch, K)``, all ``>= 1``.
    """
    belief_plus = 1.0 - torch.exp(-w_plus)
    belief_minus = 1.0 - torch.exp(-w_minus)
    kappa = belief_plus * belief_minus
    return 1.0 + kappa


# ======================================================================
# Configuration
# ======================================================================
@dataclass
class WoeSiConfig:
    """Hyper-parameters with sensible fallbacks pulled from ``args``.

    The continual-learning knobs mirror SI's ``si_c``/``si_epsilon`` naming so
    the YAML config schema extends cleanly:

    * ``woe_lambda`` -- penalty strength ``lambda`` (analogue of SI's ``si_c``).
    * ``woe_xi`` -- damping ``xi`` in the per-task normaliser (default ``1e-3``).
    * ``woe_centering_mode`` -- ``alpha``/centering scheme (Design Decision #1).
    * ``woe_mu_momentum`` -- EMA momentum for feature means (default ``0.9``).
    * ``woe_importance_stride`` -- compute ``h_i`` every ``k`` steps (default 1).
    * ``woe_conflict_weighting`` -- enable the conflict ablation (default False).
    """

    inner_steps: int = 1
    lr: float = 0.001

    woe_lambda: float = 0.1
    woe_xi: float = 1e-3
    woe_centering_mode: str = "centered_uniform"
    woe_mu_momentum: float = 0.9
    woe_importance_stride: int = 1
    woe_conflict_weighting: bool = False

    optimizer: str = "sgd"
    clipgrad: Optional[float] = 100.0
    cls_lambda: float = 1.0
    det_memories: int = 2000
    det_replay_batch: int = 64

    @staticmethod
    def from_args(args: object) -> "WoeSiConfig":
        cfg = WoeSiConfig()
        for field in cfg.__dataclass_fields__:
            if hasattr(args, field):
                setattr(cfg, field, getattr(args, field))
        return cfg


# ======================================================================
# Learner
# ======================================================================
class Net(DetectionReplayMixin, nn.Module):
    """Weight-of-Evidence Synaptic Intelligence learner built on ``ResNet1D``.

    Mirrors ``model.si.Net``: the only behavioural difference is that the
    accumulated importance ``omega`` tracks the path integral of the DS
    information content ``I_2(m)`` instead of the task loss.
    """

    def __init__(
        self, n_inputs: int, n_outputs: int, n_tasks: int, args: object
    ) -> None:
        super().__init__()
        del n_inputs  # ResNet1D fixes its own receptive field

        assert n_tasks > 0, "WoE-SI requires at least one task"

        self.cfg = WoeSiConfig.from_args(args)
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
        self.noise_label: int | None = noise_label_from_args(args)
        self.incremental_loader_name = getattr(args, "loader", None)
        # CIL <-> full shared head; TIL <-> task-masked head. Mirrors the rule in
        # utils.misc_utils._effective_cil_upto_for_loader.
        self.is_cil = self.incremental_loader_name == "class_incremental_loader"
        self.opt = self._build_optimizer()

        if self.cfg.woe_centering_mode not in _CENTERING_MODES:
            raise ValueError(
                f"woe_centering_mode must be one of {_CENTERING_MODES}, "
                f"got {self.cfg.woe_centering_mode!r}"
            )
        self.woe_lambda = float(self.cfg.woe_lambda)
        self.xi = float(self.cfg.woe_xi)
        self.centering_mode = str(self.cfg.woe_centering_mode)
        self.mu_momentum = float(self.cfg.woe_mu_momentum)
        self.importance_stride = max(1, int(self.cfg.woe_importance_stride))
        self.conflict_weighting = bool(self.cfg.woe_conflict_weighting)
        self.clipgrad = self.cfg.clipgrad
        self.cls_lambda = float(self.cfg.cls_lambda)
        self._init_det_replay(
            self.cfg.det_memories,
            self.cfg.det_replay_batch,
            enabled=bool(getattr(args, "use_detector_arch", False)),
        )

        self.feature_dim = int(self.net.feature_dim)
        self.current_task: Optional[int] = None
        self._step_in_task = 0
        self._param_to_key: Dict[str, str] = {}
        self._tracked_names: List[str] = []
        # name -> live Parameter object (same tensors the forward graph uses).
        self._tracked_params: Dict[str, nn.Parameter] = {}
        # Transient per-window scratch (h at window start, params at window start).
        self._window_h: Dict[str, torch.Tensor] = {}
        self._window_p_start: Dict[str, torch.Tensor] = {}
        self._initialise_woe_state()

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
        metric_logits = None
        for _ in range(self.cfg.inner_steps):
            # ----- 1) DS importance gradient h_i = dI_2/dtheta_i -----------
            # Computed at the *pre-step* parameters on a dedicated backward so it
            # never pollutes the CE gradient. Sampled once per importance window.
            window_start = self._step_in_task % self.importance_stride == 0
            if window_start:
                self._capture_importance_gradient(x, t)

            # ----- 2) Standard CE update (drives the parameters) -----------
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

            loss = self.cls_lambda * loss_ce + self.woe_lambda * self._surrogate_loss()

            loss.backward()
            if self.clipgrad is not None:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clipgrad)
            self.opt.step()

            # ----- 3) Accumulate the I_2 path integral over the window -----
            window_end = (
                self._step_in_task % self.importance_stride
                == self.importance_stride - 1
            )
            if window_end:
                self._accumulate_path_integral()

            self._step_in_task += 1
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
    def _is_tracked(self, name: str, param: nn.Parameter) -> bool:
        if not param.requires_grad:
            return False
        if name.startswith("det_head"):
            return False
        return True

    # ------------------------------------------------------------------
    def _initialise_woe_state(self) -> None:
        """Register SI-style buffers plus the running feature mean ``mu``."""
        for name, param in self.net.named_parameters():
            if not self._is_tracked(name, param):
                continue
            key = name.replace(".", "__")
            self._param_to_key[name] = key
            self._tracked_names.append(name)
            self._tracked_params[name] = param
            initial = param.detach().clone()
            # theta^* anchor (theta at the start of the current task).
            self.register_buffer(f"{key}_woe_prev", initial.clone())
            # Cumulative importance across tasks.
            self.register_buffer(f"{key}_woe_omega", torch.zeros_like(param))
            # Per-task path-integral accumulator omega^t.
            self.register_buffer(f"{key}_woe_w", torch.zeros_like(param))
        # Running mean mu_j of penultimate features over the current task.
        self.register_buffer("woe_feature_mean", torch.zeros(self.feature_dim))
        self.register_buffer(
            "woe_feature_mean_initialised", torch.zeros(1, dtype=torch.bool)
        )

    # ------------------------------------------------------------------
    def _compute_information_content(
        self, x: torch.Tensor, t: int, update_feature_mean: bool
    ) -> torch.Tensor:
        """Differentiable per-batch mean ``I_2(m)`` over the active logits.

        The forward uses ``bn_training=False`` so the dedicated importance pass
        does not perturb BatchNorm running statistics (they are owned by the CE
        path). Centred features use the running mean ``mu`` (detached), which is
        EMA-updated here from the current batch when ``update_feature_mean`` is
        set.
        """
        features = self.net.forward_features(x, bn_training=False)

        if update_feature_mean:
            self._update_feature_mean(features.detach())

        active = self._active_class_indices(t, features.device)
        readout = self.net.model.fc
        active_weight = readout.weight[active]
        active_bias = readout.bias[active]

        weights = compute_weights_of_evidence(
            features,
            active_weight,
            active_bias,
            self.woe_feature_mean,
            centering_mode=self.centering_mode,
        )
        i2_per_sample = information_content(
            weights, conflict_weighting=self.conflict_weighting
        )
        return i2_per_sample.mean()

    # ------------------------------------------------------------------
    def _capture_importance_gradient(self, x: torch.Tensor, t: int) -> None:
        """Backward ``I_2(m)`` into a scratch buffer (no CE-gradient pollution).

        Stores ``h_i = dI_2/dtheta_i`` and a snapshot of ``theta_i`` at the start
        of the importance window. ``torch.autograd.grad`` is used so parameter
        ``.grad`` fields (owned by the CE optimiser step) are left untouched.
        """
        info_content = self._compute_information_content(x, t, update_feature_mean=True)
        params = [self._tracked_params[name] for name in self._tracked_names]
        grads = torch.autograd.grad(
            info_content, params, retain_graph=False, allow_unused=True
        )
        for name, param, grad in zip(self._tracked_names, params, grads):
            self._window_h[name] = (
                torch.zeros_like(param) if grad is None else grad.detach().clone()
            )
            self._window_p_start[name] = param.detach().clone()

    # ------------------------------------------------------------------
    def _accumulate_path_integral(self) -> None:
        """omega^t += h_i * delta_i over the window (Eq: SI path integral)."""
        for name in self._tracked_names:
            if name not in self._window_h:
                continue
            param = self._tracked_params[name]
            key = self._param_to_key[name]
            w_buf = getattr(self, f"{key}_woe_w")
            delta = param.detach() - self._window_p_start[name]
            w_buf.add_(self._window_h[name] * delta)
        self._window_h.clear()
        self._window_p_start.clear()

    # ------------------------------------------------------------------
    def _consolidate_current_task(self) -> None:
        """End-of-task consolidation: fold omega^t into the cumulative Omega.

        ``Omega_i^t = relu(omega_i^t) / (delta_total_i^2 + xi)`` -- ``relu``
        because we protect only parameters that *built* committed evidence
        (positive path-integral contribution). The anchor and per-task
        accumulator are reset for the next task, as is the feature mean ``mu``.
        """
        if self.current_task is None:
            return
        for name in self._tracked_names:
            param = self._tracked_params[name]
            key = self._param_to_key[name]
            prev = getattr(self, f"{key}_woe_prev")
            omega = getattr(self, f"{key}_woe_omega")
            w_buf = getattr(self, f"{key}_woe_w")
            delta_total = param.detach() - prev
            omega.add_(torch.relu(w_buf) / (delta_total.pow(2) + self.xi))
            prev.copy_(param.detach())
            w_buf.zero_()
        # Reset running feature stats for the next task.
        self.woe_feature_mean.zero_()
        self.woe_feature_mean_initialised.zero_()
        self._step_in_task = 0
        self._window_h.clear()
        self._window_p_start.clear()

    # ------------------------------------------------------------------
    def _surrogate_loss(self) -> torch.Tensor:
        """Quadratic anchor penalty ``sum_i Omega_i * (theta_i - theta_i^*)^2``.

        The ``lambda/2`` scaling of the spec is folded into ``woe_lambda`` at the
        call site (``self.woe_lambda * self._surrogate_loss()``), matching SI's
        ``si_c`` convention. Returns exactly ``0`` on the first task because
        ``Omega`` is all zeros until the first consolidation.
        """
        device = self._device()
        loss = torch.zeros(1, device=device)
        for name in self._tracked_names:
            param = self._tracked_params[name]
            key = self._param_to_key[name]
            omega = getattr(self, f"{key}_woe_omega")
            prev = getattr(self, f"{key}_woe_prev")
            loss = loss + (omega * (param - prev).pow(2)).sum()
        return loss

    # ------------------------------------------------------------------
    def _update_feature_mean(self, batch_features: torch.Tensor) -> None:
        """EMA-update the per-task running feature mean ``mu_j``."""
        batch_mean = batch_features.mean(dim=0)
        if not bool(self.woe_feature_mean_initialised.item()):
            self.woe_feature_mean.copy_(batch_mean)
            self.woe_feature_mean_initialised.fill_(True)
        else:
            self.woe_feature_mean.mul_(self.mu_momentum).add_(
                batch_mean, alpha=1.0 - self.mu_momentum
            )

    # ------------------------------------------------------------------
    def _active_class_indices(self, t: int, device: torch.device) -> torch.Tensor:
        """Active output columns: cumulative seen classes (CIL) or task slice (TIL).

        Matches ``utils.misc_utils.apply_task_incremental_logit_mask``: the global
        noise label (if any) is always kept active so the DS frame ``Theta``
        spans the same classes the CE head is actually predicting over.
        """
        offset1, offset2 = misc_utils.compute_offsets(t, self.classes_per_task)
        offset2 = min(self.n_outputs, offset2)
        if self.is_cil:
            indices = list(range(0, offset2))
        else:
            indices = list(range(offset1, offset2))
        if self.noise_label is not None:
            noise = int(self.noise_label)
            if 0 <= noise < self.n_outputs and noise not in indices:
                indices.append(noise)
        indices = sorted(set(indices))
        return torch.tensor(indices, dtype=torch.long, device=device)

    # ------------------------------------------------------------------
    def omega_summary(self) -> Dict[str, Dict[str, float]]:
        """Per-group summary of cumulative ``Omega`` (readout vs backbone).

        Returns a dict with ``"readout"`` (the ``fc`` head) and ``"backbone"``
        (everything else) sub-dicts holding mean / max / sum of ``Omega`` so a
        caller can eyeball that importance concentrates sensibly.
        """
        groups: Dict[str, List[torch.Tensor]] = {"readout": [], "backbone": []}
        for name in self._tracked_names:
            key = self._param_to_key[name]
            omega = getattr(self, f"{key}_woe_omega")
            bucket = "readout" if name.startswith("fc.") else "backbone"
            groups[bucket].append(omega.reshape(-1))
        summary: Dict[str, Dict[str, float]] = {}
        for bucket, tensors in groups.items():
            if not tensors:
                summary[bucket] = {"mean": 0.0, "max": 0.0, "sum": 0.0, "count": 0}
                continue
            flat = torch.cat(tensors)
            summary[bucket] = {
                "mean": float(flat.mean().item()),
                "max": float(flat.max().item()),
                "sum": float(flat.sum().item()),
                "count": int(flat.numel()),
            }
        return summary

    # ------------------------------------------------------------------
    def _compute_offsets(self, task: int) -> Tuple[int, int]:
        offset1, offset2 = misc_utils.compute_offsets(task, self.classes_per_task)
        return offset1, min(self.n_outputs, offset2)

    # ------------------------------------------------------------------
    def _device(self) -> torch.device:
        return next(self.net.parameters()).device


__all__ = [
    "Net",
    "WoeSiConfig",
    "compute_weights_of_evidence",
    "per_class_total_evidence",
    "information_content",
]
