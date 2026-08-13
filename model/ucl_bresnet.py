"""Uncertainty-guided Continual Learning with a Bayesian 1D ResNet backbone.

This variant mirrors the behaviour of :mod:`model.ucl` but replaces the
deterministic ``ResNet1D`` feature extractor with a fully Bayesian
counterpart.  All convolutional layers now maintain Gaussian posteriors over
their weights, enabling epistemic uncertainty estimation deeper in the
network while keeping the multi-head Bayesian linear classifiers used for
task-specific outputs.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import os
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from model.resnet1d import AdcIqAdapter, ResNet1D
from model.detection_replay import (
    noise_label_from_args,
    signal_mask_exclude_noise,
    unpack_y_to_class_labels,
)
from utils.iq_features import append_iq_augmented_features
from utils.training_metrics import macro_recall
from utils import misc_utils
from utils.class_weighted_loss import classification_cross_entropy


def _calculate_fan_in_and_fan_out(tensor: torch.Tensor) -> Tuple[int, int]:
    if tensor.dim() < 2:
        raise ValueError("Tensor needs at least 2 dims to compute fan in/out")
    if tensor.dim() == 2:  # Linear layer
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field
        fan_out = num_output_fmaps * receptive_field
    return fan_in, fan_out


class Gaussian:
    """Reparameterised Gaussian for Bayesian layers."""

    def __init__(self, mu: torch.Tensor, rho: torch.Tensor) -> None:
        self.mu = mu
        self.rho = rho
        self._normal = torch.distributions.Normal(0, 1)

    @property
    def sigma(self) -> torch.Tensor:
        return torch.log1p(torch.exp(self.rho))

    def sample(self) -> torch.Tensor:
        eps = self._normal.sample(self.mu.size()).to(self.mu.device)
        return self.mu + self.sigma * eps


class BayesianLayer(nn.Module):
    """Mixin-style base class exposing Bayesian parameters."""

    weight_mu: nn.Parameter
    weight_rho: nn.Parameter

    @property
    def weight_sigma(self) -> torch.Tensor:
        return torch.log1p(torch.exp(self.weight_rho))

    def mu_parameters(self) -> Iterable[nn.Parameter]:
        for attr in ("weight_mu", "bias"):
            param = getattr(self, attr, None)
            if isinstance(param, nn.Parameter):
                yield param

    def rho_parameters(self) -> Iterable[nn.Parameter]:
        yield self.weight_rho


class BayesianLinear(BayesianLayer):
    """Factorised Gaussian linear layer mirroring the UCL implementation."""

    def __init__(self, in_features: int, out_features: int, ratio: float = 0.5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        fan_in, _ = _calculate_fan_in_and_fan_out(self.weight_mu)
        total_var = 2.0 / fan_in
        noise_var = total_var * ratio
        mu_var = total_var - noise_var

        noise_std = noise_var**0.5
        mu_std = mu_var**0.5
        bound = (3.0**0.5) * mu_std
        nn.init.uniform_(self.weight_mu, -bound, bound)

        rho_init = float(math.log(math.expm1(noise_std)))
        self.weight_rho = nn.Parameter(
            torch.full((out_features, in_features), rho_init)
        )

        self.bias = nn.Parameter(torch.zeros(out_features))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)

    def forward(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        weight = self.weight.sample() if sample else self.weight.mu
        return F.linear(x, weight, self.bias)


class BayesianConv1d(BayesianLayer):
    """1D convolution with Gaussian weight posterior."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        ratio: float = 0.5,
    ) -> None:
        super().__init__()
        if bias:
            raise ValueError("BayesianConv1d currently models bias-free convolutions")

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        weight_shape = (out_channels, in_channels // groups, kernel_size)
        self.weight_mu = nn.Parameter(torch.Tensor(*weight_shape))
        fan_in, _ = _calculate_fan_in_and_fan_out(self.weight_mu)
        total_var = 2.0 / fan_in
        noise_var = total_var * ratio
        mu_var = total_var - noise_var

        noise_std = noise_var**0.5
        mu_std = mu_var**0.5
        bound = (3.0**0.5) * mu_std
        nn.init.uniform_(self.weight_mu, -bound, bound)

        rho_init = float(math.log(math.expm1(noise_std)))
        self.weight_rho = nn.Parameter(torch.full(weight_shape, rho_init))

        self.weight = Gaussian(self.weight_mu, self.weight_rho)

    def forward(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        weight = self.weight.sample() if sample else self.weight.mu
        return F.conv1d(
            x,
            weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class BayesianConvBN1d(nn.Module):
    """Convenience wrapper for Conv1d -> BatchNorm1d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        ratio: float,
    ) -> None:
        super().__init__()
        self.conv = BayesianConv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            ratio=ratio,
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        x = self.conv(x, sample=sample)
        x = self.bn(x)
        return x


class BayesianBasicBlock1D(nn.Module):
    """1D version of ResNet BasicBlock with Bayesian convolutions."""

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        *,
        stride: int = 1,
        downsample: nn.Module | None = None,
        ratio: float,
    ) -> None:
        super().__init__()
        self.conv1 = BayesianConv1d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            ratio=ratio,
        )
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = BayesianConv1d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            ratio=ratio,
        )
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        identity = x

        out = self.conv1(x, sample=sample)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, sample=sample)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x, sample=sample)

        out += identity
        out = self.relu(out)
        return out


class BayesianResNet1D(nn.Module):
    """ResNet-18 style network with Bayesian convolutions for IQ data."""

    def __init__(self, in_channels: int = 2, ratio: float = 0.5) -> None:
        super().__init__()
        self.inplanes = 64
        self.ratio = ratio

        self.conv1 = BayesianConv1d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=1,
            ratio=ratio,
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.feature_dim = 512 * BayesianBasicBlock1D.expansion

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.ModuleList:
        downsample = None
        if stride != 1 or self.inplanes != planes * BayesianBasicBlock1D.expansion:
            downsample = BayesianConvBN1d(
                self.inplanes,
                planes * BayesianBasicBlock1D.expansion,
                stride=stride,
                ratio=self.ratio,
            )

        layers = nn.ModuleList()
        layers.append(
            BayesianBasicBlock1D(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                ratio=self.ratio,
            )
        )
        self.inplanes = planes * BayesianBasicBlock1D.expansion
        for _ in range(1, blocks):
            layers.append(
                BayesianBasicBlock1D(
                    self.inplanes,
                    planes,
                    stride=1,
                    downsample=None,
                    ratio=self.ratio,
                )
            )
        return layers

    def _forward_layer(
        self, layer: nn.ModuleList, x: torch.Tensor, sample: bool
    ) -> torch.Tensor:
        for block in layer:
            x = block(x, sample=sample)
        return x

    def forward(
        self, x: torch.Tensor, sample: bool = False, ret_feats: bool = False
    ) -> torch.Tensor:
        x = self.conv1(x, sample=sample)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self._forward_layer(self.layer1, x, sample)
        x = self._forward_layer(self.layer2, x, sample)
        x = self._forward_layer(self.layer3, x, sample)
        x = self._forward_layer(self.layer4, x, sample)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if ret_feats:
            return x
        return x  # feature extractor only; heads perform classification


@dataclass
class UCLConfig:
    inner_steps: int = 1
    lr: float = 1e-3
    lr_rho: float = 1e-2
    beta: float = 0.0002
    alpha: float = 0.3
    ratio: float = 0.125
    det_lambda: float = 1.0

    split: bool = True
    eval_samples: int = 1
    clipgrad: float = 10.0
    class_weighted_ce: bool = True

    @staticmethod
    def from_args(args: object) -> "UCLConfig":
        cfg = UCLConfig()
        for field in cfg.__dataclass_fields__:
            # `None` means "not set on args" (the parser registers `alpha`,
            # `beta`, `ratio` and `lr_rho` with None defaults), so the dataclass
            # default stands -- which is what keeps UCL's `alpha` from picking up
            # RWalk's, the two methods sharing the flag name.
            value = getattr(args, field, None)
            if value is not None:
                setattr(cfg, field, value)
        return cfg


def _infer_ucl_split_from_loader(args: object, cfg: UCLConfig) -> None:
    """Set ``cfg.split`` from ``args.loader`` for CIL vs TIL.

    Class-incremental runs need a single full-width logit vector (concatenated
    heads). Task-incremental runs keep one head per task. Explicit ``split`` on
    ``args`` is applied in :meth:`UCLConfig.from_args` first; this overwrites it
    when the loader name is recognised so YAML defaults stay loader-aligned.

    Args:
        args: Parsed experiment arguments (``loader`` string).
        cfg: Config instance to mutate in place.
    """
    loader_name = str(getattr(args, "loader", "") or "")
    if loader_name == "class_incremental_loader":
        cfg.split = False
    elif loader_name == "task_incremental_loader":
        cfg.split = True


class BayesianClassifier(nn.Module):
    """Bayesian ResNet feature extractor followed by per-task Bayesian heads."""

    def __init__(
        self,
        n_outputs: int,
        n_tasks: int,
        cfg: UCLConfig,
        args: object | None,
        classes_per_task=None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_tasks = n_tasks
        self.n_outputs = n_outputs
        if classes_per_task is None:
            classes_per_task = misc_utils.build_task_class_list(
                n_tasks,
                n_outputs,
                nc_per_task=None,
                classes_per_task=None,
            )

        self.input_adapter = AdcIqAdapter()
        self.use_iq_aug_features = bool(getattr(args, "use_iq_aug_features", False))
        self.iq_aug_scaling_mode = str(getattr(args, "data_scaling", "none"))
        self.iq_aug_feature_type = str(
            getattr(
                args, "iq_aug_feature_type", getattr(args, "iq_aug_feature", "power")
            )
        )
        feature_in_channels = 3 if self.use_iq_aug_features else 2
        self.feature_net = BayesianResNet1D(
            in_channels=feature_in_channels, ratio=cfg.ratio
        )
        self.feature_dim = self.feature_net.feature_dim

        self.heads = nn.ModuleList(
            [
                BayesianLinear(self.feature_dim, c, ratio=cfg.ratio)
                for c in classes_per_task
            ]
        )

        self.split = cfg.split

    def forward(
        self, x: torch.Tensor, sample: bool = False
    ) -> List[torch.Tensor] | torch.Tensor:
        if x.dim() == 3 and x.size(1) == 3:
            if x.size(2) % 2 != 0:
                raise ValueError(
                    "Expected even sequence length for 3-channel interleaved IQ "
                    f"input; got shape {tuple(x.shape)}."
                )
            sequence_length = x.size(2) // 2
            x = x.view(x.size(0), 3, 2, sequence_length)
            x = self.input_adapter(x)
        elif x.dim() == 4 and x.size(1) == 3 and x.size(2) == 2:
            x = self.input_adapter(x)
        if x.dim() == 3 and x.size(1) == 2:
            x = append_iq_augmented_features(
                x,
                enabled=self.use_iq_aug_features,
                scaling_mode=self.iq_aug_scaling_mode,
                feature_type=self.iq_aug_feature_type,
            )
        feats = self.feature_net(x, sample=sample, ret_feats=True)
        outputs = [head(feats, sample=sample) for head in self.heads]
        if self.split:
            return outputs
        return torch.cat(outputs, dim=1)


class Net(nn.Module):
    """UCL learner powered by a Bayesian ResNet-18 backbone."""

    def __init__(
        self, n_inputs: int, n_outputs: int, n_tasks: int, args: object
    ) -> None:
        super().__init__()

        self.cfg = UCLConfig.from_args(args)
        _infer_ucl_split_from_loader(args, self.cfg)
        assert n_tasks > 0, "Number of tasks must be positive for UCL"

        self.args = args
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_tasks = n_tasks
        self.classes_per_task = misc_utils.build_task_class_list(
            n_tasks,
            n_outputs,
            nc_per_task=getattr(args, "nc_per_task_list", "")
            or getattr(args, "nc_per_task", None),
            classes_per_task=getattr(args, "classes_per_task", None),
        )
        self.classes_per_task = self._extend_cil_heads_with_global_noise(
            self.classes_per_task
        )
        self.nc_per_task = misc_utils.max_task_class_count(self.classes_per_task)

        self.model = BayesianClassifier(
            n_outputs, n_tasks, self.cfg, args, self.classes_per_task
        )
        self.split = self.cfg.split
        in_channels = getattr(args, "in_channels", 2)
        self.detector = ResNet1D(num_classes=1, args=args, in_channels=in_channels)
        self.det_loss = nn.BCEWithLogitsLoss()
        self.det_lambda = float(self.cfg.det_lambda)
        self.det_optimizer = torch.optim.SGD(
            self.detector.parameters(),
            lr=self.cfg.lr,
            momentum=0.9,
            weight_decay=0.0,
        )

        mu_params: List[nn.Parameter] = []
        rho_params: List[nn.Parameter] = []

        for module in self._iter_bayesian_modules(self.model):
            mu_params.extend(module.mu_parameters())
            rho_params.extend(module.rho_parameters())
        mu_params.extend(self.model.input_adapter.parameters())

        self.optimizer = torch.optim.SGD(
            [
                {"params": mu_params, "lr": self.cfg.lr},
                {"params": rho_params, "lr": self.cfg.lr_rho},
            ],
            lr=self.cfg.lr,
            momentum=0.9,
            weight_decay=0.0,
        )

        self.current_task: Optional[int] = None
        self.model_old: Optional[BayesianClassifier] = None
        self.saved = False
        self.is_task_incremental: bool = True
        self._debug_step_counter = 0
        self.noise_label: int | None = noise_label_from_args(args)
        self.incremental_loader_name = getattr(args, "loader", None)
        self._use_task_bn_state = bool(
            self.split and self.incremental_loader_name == "task_incremental_loader"
        )
        self._bn_modules: List[_BatchNorm] = [
            module for module in self.model.modules() if isinstance(module, _BatchNorm)
        ]
        self._bn_task_stats: dict[int, List[Tuple[torch.Tensor, torch.Tensor, int]]] = (
            {}
        )
        self._bn_task_affine: dict[
            int, List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]
        ] = {}
        self._bn_finalized_tasks: set[int] = set()

    def _extend_cil_heads_with_global_noise(
        self, classes_per_task: List[int]
    ) -> List[int]:
        """Ensure CIL concatenated heads include a slot for global noise labels.

        In IQ CIL mode, `class_incremental_loader` remaps noise targets to a
        single global class id (typically the last label). UCL uses per-task
        heads and concatenates them when `split=False`; if head widths only sum
        to signal classes, CE receives out-of-range noise targets.

        This method adds one class slot to the final head only when needed.

        Args:
            classes_per_task: Per-task class counts used to size UCL heads.

        Returns:
            Possibly adjusted per-task class counts.
        """

        is_cil = not bool(self.cfg.split)
        if not is_cil:
            return classes_per_task

        total_classes = int(sum(classes_per_task))
        if total_classes >= int(self.n_outputs):
            return classes_per_task

        missing_classes = int(self.n_outputs) - total_classes
        if missing_classes <= 0:
            return classes_per_task

        adjusted = list(classes_per_task)
        adjusted[-1] += missing_classes
        return adjusted

    @contextmanager
    def _temporarily_enable_bn_training(self):
        bn_modules: List[nn.BatchNorm1d] = []
        states: List[bool] = []
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm1d):
                bn_modules.append(module)
                states.append(module.training)
                module.train(True)
        try:
            yield
        finally:
            for module, state in zip(bn_modules, states):
                module.train(state)

    # ------------------------------------------------------------------
    def _reset_bn_stats(self) -> None:
        """Reset BatchNorm running statistics for a fresh TIL task."""
        if not self._use_task_bn_state:
            return
        for batch_norm_module in self._bn_modules:
            batch_norm_module.running_mean.zero_()
            batch_norm_module.running_var.fill_(1.0)
            batch_norm_module.num_batches_tracked.zero_()

    def _snapshot_bn_stats(self, task: int) -> None:
        """Store task-specific BatchNorm state for TIL evaluation.

        Args:
            task: Completed task index whose BatchNorm state should be restored
                for future task-incremental evaluation.
        """
        if not self._use_task_bn_state:
            return
        stats: List[Tuple[torch.Tensor, torch.Tensor, int]] = []
        affine: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = []
        for batch_norm_module in self._bn_modules:
            stats.append(
                (
                    batch_norm_module.running_mean.detach().clone(),
                    batch_norm_module.running_var.detach().clone(),
                    int(batch_norm_module.num_batches_tracked.item()),
                )
            )
            if batch_norm_module.affine:
                affine.append(
                    (
                        batch_norm_module.weight.detach().clone(),
                        batch_norm_module.bias.detach().clone(),
                    )
                )
            else:
                affine.append((None, None))
        self._bn_task_stats[task] = stats
        self._bn_task_affine[task] = affine
        self._bn_finalized_tasks.add(task)

    def _capture_bn_state(
        self,
    ) -> Tuple[
        List[Tuple[torch.Tensor, torch.Tensor, int]],
        List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]],
    ]:
        """Capture currently active BatchNorm state.

        Returns:
            Running-stat and affine snapshots that can be restored after a
            temporary task-specific evaluation forward.
        """
        stats: List[Tuple[torch.Tensor, torch.Tensor, int]] = []
        affine: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = []
        for batch_norm_module in self._bn_modules:
            stats.append(
                (
                    batch_norm_module.running_mean.detach().clone(),
                    batch_norm_module.running_var.detach().clone(),
                    int(batch_norm_module.num_batches_tracked.item()),
                )
            )
            if batch_norm_module.affine:
                affine.append(
                    (
                        batch_norm_module.weight.detach().clone(),
                        batch_norm_module.bias.detach().clone(),
                    )
                )
            else:
                affine.append((None, None))
        return stats, affine

    def _apply_bn_state(
        self,
        stats: List[Tuple[torch.Tensor, torch.Tensor, int]],
        affine: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]],
    ) -> None:
        """Apply a previously captured BatchNorm state.

        Args:
            stats: Running-stat snapshots from :meth:`_capture_bn_state`.
            affine: Affine parameter snapshots from :meth:`_capture_bn_state`.
        """
        if len(stats) != len(self._bn_modules) or len(affine) != len(self._bn_modules):
            raise RuntimeError(
                "BatchNorm state snapshot is out of sync with model BatchNorm modules."
            )
        for batch_norm_module, (running_mean, running_var, num_batches) in zip(
            self._bn_modules, stats
        ):
            batch_norm_module.running_mean.data.copy_(running_mean)
            batch_norm_module.running_var.data.copy_(running_var)
            batch_norm_module.num_batches_tracked.data.fill_(num_batches)
        for batch_norm_module, (weight, bias) in zip(self._bn_modules, affine):
            if weight is not None and batch_norm_module.affine:
                batch_norm_module.weight.data.copy_(weight)
                batch_norm_module.bias.data.copy_(bias)

    def _restore_bn_stats(self, task: int) -> None:
        """Restore saved task BatchNorm state, or reset for unseen TIL tasks.

        Args:
            task: Task index whose BatchNorm state should become active.
        """
        if not self._use_task_bn_state:
            return
        stats = self._bn_task_stats.get(task)
        affine = self._bn_task_affine.get(task)
        if stats is None:
            self._reset_bn_stats()
            return
        if affine is None:
            raise RuntimeError(
                f"BatchNorm affine snapshot missing for task {task} despite saved stats."
            )
        self._apply_bn_state(stats, affine)

    def finalize_task_after_training(
        self,
        train_loader: object | None = None,
        *,
        completed_task_index: int | None = None,
    ) -> None:
        """Snapshot task-specific BatchNorm state after TIL task training.

        Args:
            train_loader: Unused hook argument accepted for compatibility with
                the repository training loop.
            completed_task_index: Completed task index. Defaults to the current
                task tracked by the UCL learner.

        Usage:
            The main training loop calls ``model.finalize_task_after_training(
            train_loader)`` after finishing each task.
        """
        del train_loader
        if not self._use_task_bn_state:
            return
        task = (
            self.current_task if completed_task_index is None else completed_task_index
        )
        if task is None:
            raise RuntimeError("finalize_task_after_training requires a current task.")
        self._snapshot_bn_stats(task)

    # ------------------------------------------------------------------
    def compute_offsets(self, task: int) -> Tuple[int, int]:
        if self.is_task_incremental:
            return misc_utils.compute_offsets(task, self.classes_per_task)
        else:
            return 0, self.n_outputs

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        x: torch.Tensor,
        t: int,
        s: Optional[float] = None,
        *,
        cil_all_seen_upto_task: int | None = None,
    ) -> torch.Tensor:
        """Return task head logits, or concatenated heads when ``split`` is False.

        ``split`` is set from ``args.loader`` in :meth:`Net.__init__` (CIL →
        concatenated heads; TIL → separate heads). With concatenated logits,
        :func:`~utils.misc_utils.apply_task_incremental_logit_mask` applies in
        eval. With ``split=True``, only head ``t`` is returned.
        """
        previous_bn_state = (
            self._capture_bn_state()
            if self._use_task_bn_state and not self.training
            else None
        )
        if self._use_task_bn_state and not self.training:
            self._restore_bn_stats(t)
        try:
            logits = self._forward_with_active_bn(
                x,
                t,
                cil_all_seen_upto_task=cil_all_seen_upto_task,
            )
        finally:
            if previous_bn_state is not None:
                self._apply_bn_state(*previous_bn_state)
        return logits

    def _forward_with_active_bn(
        self,
        x: torch.Tensor,
        t: int,
        *,
        cil_all_seen_upto_task: int | None = None,
    ) -> torch.Tensor:
        """Run the UCL forward pass with the caller-selected BatchNorm state.

        Args:
            x: Input minibatch.
            t: Current task index.
            cil_all_seen_upto_task: Optional CIL mask upper bound.

        Returns:
            Task logits or masked CIL logits.
        """
        if not self.training:
            num_samples = max(1, self.cfg.eval_samples)
            if num_samples == 1:
                outputs = self.model(x, sample=False)
                logits = outputs[t] if self.split else outputs
            else:
                probs_acc: Optional[torch.Tensor] = None
                with torch.no_grad():
                    with self._temporarily_enable_bn_training():
                        for _ in range(num_samples):
                            sampled = self.model(x, sample=True)
                            head_logits = sampled[t] if self.split else sampled
                            head_probs = F.softmax(head_logits, dim=-1)
                            probs_acc = (
                                head_probs
                                if probs_acc is None
                                else probs_acc + head_probs
                            )

                assert probs_acc is not None
                probs_mean = probs_acc / float(num_samples)
                logits = torch.log(probs_mean.clamp_min(1e-8))
        else:
            outputs = self.model(x, sample=False)
            logits = outputs[t] if self.split else outputs

        if (
            not self.training
            and self.is_task_incremental
            and not self.split
            and logits.size(-1) == self.n_outputs
        ):
            logits = misc_utils.apply_task_incremental_logit_mask(
                logits,
                t,
                self.classes_per_task,
                self.n_outputs,
                cil_all_seen_upto_task=cil_all_seen_upto_task,
                global_noise_label=self.noise_label,
                fill_value=-10e10,
                loader=self.incremental_loader_name,
            )
        return logits

    def forward_heads(
        self, x: torch.Tensor, sample: bool = False
    ) -> Tuple[torch.Tensor, List[torch.Tensor] | torch.Tensor]:
        det_logits = self.detector.forward_detection(self.detector.forward_features(x))
        cls_logits = self.model(x, sample=sample)
        return det_logits, cls_logits

    def observe(
        self, x: torch.Tensor, y: torch.Tensor, t: int
    ) -> Tuple[float, float, torch.Tensor | None]:
        y_cls, y_det = self._split_labels(y)
        if not torch.is_tensor(y_cls):
            y_cls = torch.as_tensor(y_cls)
        if y_det is not None and not torch.is_tensor(y_det):
            y_det = torch.as_tensor(y_det)
        y_cls_glob = unpack_y_to_class_labels(
            (y_cls, y_det) if y_det is not None else y_cls
        ).long()
        if (self.current_task is None) or (t != self.current_task):
            if self.current_task is not None:
                if (
                    self._use_task_bn_state
                    and self.current_task not in self._bn_finalized_tasks
                ):
                    self._snapshot_bn_stats(self.current_task)
                self.model_old = self._snapshot_model()
                self.saved = True
            self.current_task = t
            self._restore_bn_stats(t)

        device = self._device()

        x_cls = x
        y_cls_filtered = y_cls_glob
        # if y_det is not None:
        #     signal_mask = (y_det == 1) & (y_cls >= 0)
        #     if not signal_mask.any():
        #         x = x.to(device)
        #         y_det = y_det.to(device)
        #         self.detector.train()
        #         det_logits = self.detector.forward_detection(self.detector.forward_features(x))
        #         det_loss = self.det_loss(det_logits, y_det.float())
        #         self.det_optimizer.zero_grad(set_to_none=True)
        #         det_loss = self.det_lambda * det_loss
        #         det_loss.backward()
        #         if self.cfg.clipgrad > 0:
        #             torch.nn.utils.clip_grad_norm_(self.detector.parameters(), self.cfg.clipgrad)
        #         self.det_optimizer.step()
        #         return float(det_loss.detach().cpu()), 0.0
        #     x_cls = x[signal_mask]
        #     y_cls_filtered = y_cls[signal_mask]

        signal_mask = signal_mask_exclude_noise(y_cls_filtered, self.noise_label)
        if self.split:
            offset1, _ = self.compute_offsets(t)
            y_local = y_cls_filtered.clone() - offset1
            task_classes = self.classes_per_task[t]
            if signal_mask.any():
                y_sig = y_local[signal_mask]
                if (y_sig.min() < 0) or (y_sig.max() >= task_classes):
                    raise ValueError(
                        f"Labels out of range for task {t}: expected in [0, {task_classes - 1}] after offset, got "
                        f"[{int(y_sig.min())}, {int(y_sig.max())}]"
                    )
            y_cls_filtered = y_local

        x = x.to(device)
        x_cls = x_cls.to(device)
        y_cls_filtered = y_cls_filtered.to(device)
        signal_mask = signal_mask.to(device)
        if y_det is not None:
            y_det = y_det.to(device)

        self.train()
        # Let BatchNorm update running buffers so ``model.eval()`` matches training stats.
        metric_logits = None
        for _ in range(self.cfg.inner_steps):
            outputs = self.model(x_cls, sample=True)
            logits = outputs[t] if self.split else outputs

            preds = torch.argmax(logits, dim=1)
            if signal_mask.any():
                cls_tr_rec = macro_recall(
                    preds[signal_mask], y_cls_filtered[signal_mask]
                )
            else:
                cls_tr_rec = 0.0
            metric_logits = logits.detach()
            self._maybe_log_training_debug(
                task_index=t,
                labels=y_cls_filtered,
                predictions=preds,
                logits=logits,
            )
            ce = classification_cross_entropy(
                logits,
                y_cls_filtered,
                class_weighted_ce=bool(self.cfg.class_weighted_ce),
            )
            loss = self._apply_regularisation(ce, y_cls_filtered.size(0))

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.cfg.clipgrad > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.cfg.clipgrad)
            self.optimizer.step()

        # if y_det is not None:
        #     self.detector.train()
        #     det_logits = self.detector.forward_detection(self.detector.forward_features(x))
        #     det_loss = self.det_loss(det_logits, y_det.float())
        #     self.det_optimizer.zero_grad(set_to_none=True)
        #     det_loss = self.det_lambda * det_loss
        #     det_loss.backward()
        #     if self.cfg.clipgrad > 0:
        #         torch.nn.utils.clip_grad_norm_(self.detector.parameters(), self.cfg.clipgrad)
        #     self.det_optimizer.step()

        return float(loss.detach().cpu()), cls_tr_rec, metric_logits

    def _maybe_log_training_debug(
        self,
        task_index: int,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        logits: torch.Tensor,
    ) -> None:
        """Print periodic UCL train diagnostics when enabled via env var.

        Args:
            task_index: Current task id.
            labels: Task-local ground-truth labels for the current minibatch.
            predictions: Argmax predictions for the current minibatch.
            logits: Raw class logits for the current minibatch.

        Usage:
            self._maybe_log_training_debug(task_index, labels, predictions, logits)
        """
        debug_enabled = os.getenv("LA_MAML_UCL_DEBUG", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not debug_enabled:
            return

        debug_every = max(int(os.getenv("LA_MAML_UCL_DEBUG_EVERY", "50")), 1)
        self._debug_step_counter += 1
        if self._debug_step_counter % debug_every != 0:
            return

        labels_cpu = labels.detach().cpu().long()
        predictions_cpu = predictions.detach().cpu().long()
        confidence = (
            torch.softmax(logits.detach(), dim=1).max(dim=1).values.mean().item()
        )
        print(
            "[ucl-debug] step={} task={} saved={} uniq_y={} uniq_pred={} mean_max_softmax={:.4f}".format(
                self._debug_step_counter,
                task_index,
                int(bool(self.saved)),
                labels_cpu.unique(sorted=True).tolist(),
                predictions_cpu.unique(sorted=True).tolist(),
                float(confidence),
            )
        )

    def on_epoch_end(self) -> None:  # pragma: no cover
        pass

    # ------------------------------------------------------------------
    def _snapshot_model(self) -> BayesianClassifier:
        clone = BayesianClassifier(
            self.n_outputs, self.n_tasks, self.cfg, self.args, self.classes_per_task
        )
        clone.load_state_dict(self.model.state_dict())
        clone.to(self._device())
        clone.eval()
        for param in clone.parameters():
            param.requires_grad_(False)
        return clone

    def _compute_layer_regularisation_terms(
        self,
        old_layer: BayesianLayer,
        new_layer: BayesianLayer,
        eps: float,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
    ]:
        """Compute UCL regularisation terms for a Bayesian layer pair.

        Args:
            old_layer: Frozen layer snapshot from the previous task.
            new_layer: Trainable layer for the current task.
            eps: Small constant for numerical stability.

        Returns:
            A tuple containing sigma_weight_reg, sigma_weight_normal_reg,
            mu_weight_reg, mu_bias_reg, l1_mu_weight_reg, and l1_mu_bias_reg.
        """
        trainer_weight_mu = new_layer.weight_mu
        saver_weight_mu = old_layer.weight_mu
        trainer_weight_sigma = new_layer.weight_sigma
        saver_weight_sigma = old_layer.weight_sigma
        safe_saver_weight_sigma = saver_weight_sigma.clamp_min(eps)

        fan_in, _ = _calculate_fan_in_and_fan_out(trainer_weight_mu)
        std_init = math.sqrt((2.0 / fan_in) * self.cfg.ratio)

        curr_strength = std_init / safe_saver_weight_sigma
        saver_strength_flat = curr_strength.view(curr_strength.size(0), -1)
        bias_strength = saver_strength_flat.mean(dim=1)

        mu_weight_reg = (
            (curr_strength * (trainer_weight_mu - saver_weight_mu)) ** 2
        ).sum()
        l1_mu_weight_reg = (
            (saver_weight_mu.pow(2) / safe_saver_weight_sigma.pow(2))
            * (trainer_weight_mu - saver_weight_mu).abs()
        ).sum() * (std_init**2)

        mu_bias_reg = torch.zeros_like(mu_weight_reg)
        l1_mu_bias_reg = torch.zeros_like(mu_weight_reg)
        regularized_parameter_count = (
            trainer_weight_mu.numel() + trainer_weight_sigma.numel()
        )
        trainer_bias = getattr(new_layer, "bias", None)
        saver_bias = getattr(old_layer, "bias", None)
        if trainer_bias is not None and saver_bias is not None:
            mu_bias_reg = ((bias_strength * (trainer_bias - saver_bias)) ** 2).sum()
            saver_sigma_flat = saver_weight_sigma.view(saver_weight_sigma.size(0), -1)
            l1_mu_bias_reg = (
                (saver_bias.pow(2) / saver_sigma_flat.mean(dim=1).clamp_min(eps).pow(2))
                * (trainer_bias - saver_bias).abs()
            ).sum() * (std_init**2)
            regularized_parameter_count += trainer_bias.numel()

        weight_sigma_ratio = trainer_weight_sigma.pow(2) / (
            safe_saver_weight_sigma.pow(2)
        )
        sigma_weight_reg = (
            weight_sigma_ratio - torch.log(weight_sigma_ratio + eps)
        ).sum()
        sigma_weight_normal_reg = (
            trainer_weight_sigma.pow(2) - torch.log(trainer_weight_sigma.pow(2) + eps)
        ).sum()
        return (
            sigma_weight_reg,
            sigma_weight_normal_reg,
            mu_weight_reg,
            mu_bias_reg,
            l1_mu_weight_reg,
            l1_mu_bias_reg,
            regularized_parameter_count,
        )

    def _apply_regularisation(
        self, base_loss: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """Apply UCL regularisation against the previous-task posterior snapshot.

        Args:
            base_loss: Current task classification loss.
            batch_size: Current minibatch size.

        Returns:
            The total loss including UCL regularisation.
        """
        if not self.saved or self.model_old is None:
            return base_loss

        sigma_weight_reg = torch.zeros_like(base_loss)
        sigma_weight_normal_reg = torch.zeros_like(base_loss)
        mu_weight_reg = torch.zeros_like(base_loss)
        mu_bias_reg = torch.zeros_like(base_loss)
        l1_mu_weight_reg = torch.zeros_like(base_loss)
        l1_mu_bias_reg = torch.zeros_like(base_loss)
        regularized_parameter_count = 0
        eps = 1e-8

        for old_layer, new_layer in zip(
            self._iter_bayesian_modules(self.model_old.feature_net),
            self._iter_bayesian_modules(self.model.feature_net),
        ):
            (
                sigma_term,
                sigma_normal_term,
                mu_weight_term,
                mu_bias_term,
                l1_weight_term,
                l1_bias_term,
                param_count,
            ) = self._compute_layer_regularisation_terms(old_layer, new_layer, eps)
            sigma_weight_reg = sigma_weight_reg + sigma_term
            sigma_weight_normal_reg = sigma_weight_normal_reg + sigma_normal_term
            mu_weight_reg = mu_weight_reg + mu_weight_term
            mu_bias_reg = mu_bias_reg + mu_bias_term
            l1_mu_weight_reg = l1_mu_weight_reg + l1_weight_term
            l1_mu_bias_reg = l1_mu_bias_reg + l1_bias_term
            regularized_parameter_count += param_count

        current_task_index = (
            int(self.current_task) if self.current_task is not None else 0
        )
        for head_index in range(min(current_task_index, len(self.model.heads))):
            old_head = self.model_old.heads[head_index]
            new_head = self.model.heads[head_index]
            (
                sigma_term,
                sigma_normal_term,
                mu_weight_term,
                mu_bias_term,
                l1_weight_term,
                l1_bias_term,
                param_count,
            ) = self._compute_layer_regularisation_terms(old_head, new_head, eps)
            sigma_weight_reg = sigma_weight_reg + sigma_term
            sigma_weight_normal_reg = sigma_weight_normal_reg + sigma_normal_term
            mu_weight_reg = mu_weight_reg + mu_weight_term
            mu_bias_reg = mu_bias_reg + mu_bias_term
            l1_mu_weight_reg = l1_mu_weight_reg + l1_weight_term
            l1_mu_bias_reg = l1_mu_bias_reg + l1_bias_term
            regularized_parameter_count += param_count

        normaliser = max(1, regularized_parameter_count)
        sigma_weight_reg = sigma_weight_reg / normaliser
        sigma_weight_normal_reg = sigma_weight_normal_reg / normaliser
        mu_weight_reg = mu_weight_reg / normaliser
        mu_bias_reg = mu_bias_reg / normaliser
        l1_mu_weight_reg = l1_mu_weight_reg / normaliser
        l1_mu_bias_reg = l1_mu_bias_reg / normaliser

        loss = base_loss
        loss = loss + self.cfg.alpha * (mu_weight_reg + mu_bias_reg) / (2 * batch_size)
        loss = loss + self.saved * (l1_mu_weight_reg + l1_mu_bias_reg) / batch_size
        loss = loss + self.cfg.beta * (sigma_weight_reg + sigma_weight_normal_reg) / (
            2 * batch_size
        )
        return loss

    def _iter_bayesian_modules(self, module: nn.Module) -> Iterable[BayesianLayer]:
        for sub in module.modules():
            if isinstance(sub, BayesianLayer):
                yield sub

    def _split_labels(
        self, y: torch.Tensor | Tuple[torch.Tensor, torch.Tensor] | dict
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(y, (tuple, list)) and len(y) == 2:
            return y[0], y[1]
        if isinstance(y, dict):
            y_cls = y.get("y_cls", y.get("y"))
            return y_cls, y.get("y_det")
        return y, None

    @torch.no_grad()
    def mc_epistemic_classification(self, x, t, S=20, temperature=1.0, clamp_eps=1e-8):
        """Monte-Carlo epistemic uncertainty for classification."""

        model = self.model
        model.eval()
        probs_accum = []

        for _ in range(S):
            logits = model(x, sample=True)[t] / temperature
            probs = F.softmax(logits, dim=-1)
            probs_accum.append(probs)

        probs_stack = torch.stack(probs_accum, dim=0)
        p_mean = probs_stack.mean(dim=0)

        p_mean_clamped = p_mean.clamp(min=clamp_eps, max=1.0)
        H_pred = -(p_mean_clamped * p_mean_clamped.log()).sum(dim=-1)

        probs_clamped = probs_stack.clamp(min=clamp_eps, max=1.0)
        entropies = -(probs_clamped * probs_clamped.log()).sum(dim=-1)
        EH = entropies.mean(dim=0)

        MI = H_pred - EH

        return p_mean, H_pred, EH, MI


__all__ = ["Net"]
