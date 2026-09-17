"""Uncertainty-guided Continual Learning with a Bayesian 1D ResNet backbone.

This variant mirrors the behaviour of :mod:`model.ucl` but replaces the
deterministic ``ResNet1D`` feature extractor with a fully Bayesian
counterpart.  All convolutional layers now maintain Gaussian posteriors over
their weights, enabling epistemic uncertainty estimation deeper in the
network. Task heads are deterministic linear layers, and the regulariser follows
the reference UCL ``custom_regularization``.
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
from model.resnet1d import AdcIqAdapter
from model.replay_utils import unpack_y_to_class_labels
from model import task_bn
from model.task_bn import frozen_running_stats
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
        # Tie sigma per output node: the incoming weights to node ``i`` share one
        # rho_i (paper Sec. 3.3). Shape (out, 1) broadcasts across ``in_features``.
        self.weight_rho = nn.Parameter(torch.full((out_features, 1), rho_init))

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
        # Tie sigma per output channel (filter): one rho per filter, shared across
        # input channels and kernel taps (paper Supp. Sec. 5.1.2). Shape
        # (out_channels, 1, 1) broadcasts over the full weight tensor.
        self.weight_rho = nn.Parameter(torch.full((out_channels, 1, 1), rho_init))

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

    def ucl_regularisation_chain(self) -> List[Tuple[BayesianConv1d, Optional[int]]]:
        """List every Bayesian conv with the index of the conv feeding its input.

        UCL's node-wise L2 strength takes, per weight, the max of its output
        node's strength and its input node's strength (the feeding layer's output
        node). The reference implementation used a plain sequential net; here a
        block's input is taken from the previous block's main-path ``conv2``, and
        the downsample conv shares its block's input. The stem conv has no
        feeding layer (``None``), matching the reference's zero initial strength.

        Returns:
            ``(layer, feeding_index)`` pairs in forward order.
        """
        chain: List[Tuple[BayesianConv1d, Optional[int]]] = [(self.conv1, None)]
        block_input = 0
        for stage in (self.layer1, self.layer2, self.layer3, self.layer4):
            for block in stage:
                chain.append((block.conv1, block_input))
                chain.append((block.conv2, len(chain) - 1))
                main_path_output = len(chain) - 1
                if block.downsample is not None:
                    chain.append((block.downsample.conv, block_input))
                block_input = main_path_output
        return chain

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

    split: bool = True
    eval_samples: int = 1
    clipgrad: float = 0.0
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
        print(f"Configs: {cfg}")
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
    """Bayesian ResNet feature extractor with a split or single-head classifier.

    Mirrors the reference UCL networks: ``split`` uses deterministic per-task
    heads, otherwise one Bayesian output layer spans all classes and is part of
    the regularised network.
    """

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

        self.split = cfg.split
        if self.split:
            # Deterministic task heads, as in the reference split network (``self.last``).
            self.heads = nn.ModuleList(
                [nn.Linear(self.feature_dim, c) for c in classes_per_task]
            )
        else:
            self.output = BayesianLinear(self.feature_dim, n_outputs, ratio=cfg.ratio)

    def ucl_regularisation_chain(self) -> List[Tuple[BayesianLayer, Optional[int]]]:
        """Regularised layers in forward order with their feeding-layer indices.

        Returns:
            The feature-net chain, plus the single-head output layer (fed by the
            last main-path conv) when not ``split``.
        """
        chain: List[Tuple[BayesianLayer, Optional[int]]] = list(
            self.feature_net.ucl_regularisation_chain()
        )
        if not self.split:
            last_conv = self.feature_net.layer4[-1].conv2
            feeding = next(i for i, (layer, _) in enumerate(chain) if layer is last_conv)
            chain.append((self.output, feeding))
        return chain

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
        if self.split:
            return [head(feats) for head in self.heads]
        return self.output(feats, sample=sample)


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
        self.nc_per_task = misc_utils.max_task_class_count(self.classes_per_task)

        self.model = BayesianClassifier(
            n_outputs, n_tasks, self.cfg, args, self.classes_per_task
        )
        self.split = self.cfg.split

        mu_params: List[nn.Parameter] = []
        rho_params: List[nn.Parameter] = []
        for name, param in self.model.named_parameters():
            (rho_params if name.endswith("weight_rho") else mu_params).append(param)

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
        self.incremental_loader_name = getattr(args, "loader", None)
        # Per-task BatchNorm running statistics are handled centrally by
        # :mod:`model.task_bn`, installed from ``main`` once the model is built.

    @contextmanager
    def _temporarily_enable_bn_training(self):
        """Put BatchNorm in train mode for multi-sample Bayesian evaluation.

        Running-statistic updates stay frozen throughout: this is an evaluation
        path, so it must read each task's statistics without writing them.
        """
        bn_modules: List[nn.BatchNorm1d] = []
        states: List[bool] = []
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm1d):
                bn_modules.append(module)
                states.append(module.training)
                module.train(True)
        try:
            with frozen_running_stats(self):
                yield
        finally:
            for module, state in zip(bn_modules, states):
                module.train(state)

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

        Per-task BatchNorm statistics are selected by the caller (see
        :func:`utils.training_forward.model_forward_for_metric_loop`), so this
        forward no longer swaps BatchNorm state itself.
        """
        return self._forward_with_active_bn(
            x,
            t,
            cil_all_seen_upto_task=cil_all_seen_upto_task,
        )

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
                fill_value=-10e10,
                loader=self.incremental_loader_name,
            )
        return logits

    def observe(
        self, x: torch.Tensor, y: torch.Tensor, t: int
    ) -> Tuple[float, float, torch.Tensor | None]:
        y_cls_glob = unpack_y_to_class_labels(y).long()
        if self.current_task is None:
            # The reference regularises the first task towards the initial weights.
            self.model_old = self._snapshot_model()
            self.current_task = t
        elif t != self.current_task:
            self.model_old = self._snapshot_model()
            self.saved = True
            self.current_task = t

        device = self._device()

        x_cls = x
        y_cls_filtered = y_cls_glob
        if self.split:
            offset1, _ = self.compute_offsets(t)
            y_local = y_cls_filtered.clone() - offset1
            task_classes = self.classes_per_task[t]
            if y_local.numel() and (
                (y_local.min() < 0) or (y_local.max() >= task_classes)
            ):
                raise ValueError(
                    f"Labels out of range for task {t}: expected in [0, {task_classes - 1}] after offset, got "
                    f"[{int(y_local.min())}, {int(y_local.max())}]"
                )
            y_cls_filtered = y_local

        x = x.to(device)
        x_cls = x_cls.to(device)
        y_cls_filtered = y_cls_filtered.to(device)

        self.train()
        # Let BatchNorm update running buffers so ``model.eval()`` matches training stats.
        metric_logits = None
        for _ in range(self.cfg.inner_steps):
            outputs = self.model(x_cls, sample=True)
            if self.split:
                logits = outputs[t]
            else:
                # Not in the reference (its single head shares labels across tasks):
                # unseen-class rows would otherwise be pushed down, then anchored there.
                logits = misc_utils.apply_task_incremental_logit_mask(
                    outputs,
                    t,
                    self.classes_per_task,
                    self.n_outputs,
                    cil_all_seen_upto_task=t,
                    loader=self.incremental_loader_name,
                )

            preds = torch.argmax(logits, dim=1)
            cls_tr_rec = macro_recall(preds, y_cls_filtered)
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
        # ``self.model`` may have had its BatchNorm layers converted in place by
        # :mod:`model.task_bn` after construction, which adds per-task buffers to
        # its state dict. Give the freshly built clone the same layer types
        # before loading, or the load fails on unexpected keys.
        task_bn_layers = task_bn.task_bn_layers(self.model)
        if task_bn_layers:
            task_bn.convert_batchnorm_to_task_specific(
                clone, task_bn_layers[0].num_tasks
            )
        clone.load_state_dict(self.model.state_dict())
        clone.to(self._device())
        clone.eval()
        for param in clone.parameters():
            param.requires_grad_(False)
        return clone

    def _apply_regularisation(
        self, base_loss: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """Add the UCL regulariser, following the reference ``custom_regularization``.

        During the first task ``model_old`` holds the initial weights and ``alpha``
        is ``cfg.alpha``; once a task has been consolidated ``alpha`` is 1 and the
        L1 term switches on. Terms are summed over layers and scaled only by the
        minibatch size. Split task heads are deterministic and not regularised;
        the single-head output layer is.

        Args:
            base_loss: Mean classification loss of the current minibatch.
            batch_size: Current minibatch size.

        Returns:
            The total loss including UCL regularisation.
        """
        alpha = 1.0 if self.saved else float(self.cfg.alpha)

        mu_reg = base_loss.new_zeros(())
        l1_mu_reg = base_loss.new_zeros(())
        sigma_weight_reg = base_loss.new_zeros(())
        sigma_weight_normal_reg = base_loss.new_zeros(())

        saver_strengths: List[torch.Tensor] = []
        for (saver_layer, feeding_index), (trainer_layer, _) in zip(
            self.model_old.ucl_regularisation_chain(),
            self.model.ucl_regularisation_chain(),
        ):
            trainer_weight_mu = trainer_layer.weight_mu
            saver_weight_mu = saver_layer.weight_mu
            trainer_weight_sigma = trainer_layer.weight_sigma
            saver_weight_sigma = saver_layer.weight_sigma

            # The reference uses fan_in for linear layers and fan_out for convolutions.
            fan_in, fan_out = _calculate_fan_in_and_fan_out(trainer_weight_mu)
            is_linear = isinstance(trainer_layer, BayesianLinear)
            std_init = math.sqrt(
                (2.0 / (fan_in if is_linear else fan_out)) * self.cfg.ratio
            )

            saver_weight_strength = std_init / saver_weight_sigma
            saver_strengths.append(saver_weight_strength)
            l2_strength = saver_weight_strength
            if feeding_index is not None:
                # One strength per feeding output node, repeated over the input
                # features it produces (the reference's conv -> linear flatten).
                feeding_strength = saver_strengths[feeding_index]
                in_features = trainer_weight_mu.size(1)
                prev_strength = (
                    feeding_strength.reshape(feeding_strength.size(0), 1)
                    .expand(-1, in_features // feeding_strength.size(0))
                    .reshape(1, in_features, *([1] * (trainer_weight_mu.dim() - 2)))
                )
                l2_strength = torch.max(saver_weight_strength, prev_strength)

            delta_mu = trainer_weight_mu - saver_weight_mu
            mu_reg = mu_reg + (l2_strength * delta_mu).pow(2).sum()
            l1_mu_reg = l1_mu_reg + (
                (saver_weight_mu.pow(2) / saver_weight_sigma.pow(2)) * delta_mu
            ).abs().sum() * (std_init**2)

            trainer_bias = getattr(trainer_layer, "bias", None)
            if trainer_bias is not None:
                saver_bias = saver_layer.bias
                delta_bias = trainer_bias - saver_bias
                bias_strength = saver_weight_strength.reshape(-1)
                bias_sigma = saver_weight_sigma.reshape(-1)
                mu_reg = mu_reg + (bias_strength * delta_bias).pow(2).sum()
                l1_mu_reg = l1_mu_reg + (
                    (saver_bias.pow(2) / bias_sigma.pow(2)) * delta_bias
                ).abs().sum() * (std_init**2)

            weight_sigma = trainer_weight_sigma.pow(2) / saver_weight_sigma.pow(2)
            normal_weight_sigma = trainer_weight_sigma.pow(2)
            sigma_weight_reg = sigma_weight_reg + (
                weight_sigma - torch.log(weight_sigma)
            ).sum()
            sigma_weight_normal_reg = sigma_weight_normal_reg + (
                normal_weight_sigma - torch.log(normal_weight_sigma)
            ).sum()

        loss = base_loss
        loss = loss + alpha * mu_reg / (2 * batch_size)
        loss = loss + float(self.saved) * l1_mu_reg / batch_size
        loss = loss + self.cfg.beta * (sigma_weight_reg + sigma_weight_normal_reg) / (
            2 * batch_size
        )
        return loss

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
