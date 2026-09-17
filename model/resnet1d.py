"""1D ResNet-18 architecture for raw IQ data.

This module implements a lightweight 1D variant of the popular ResNet-18
architecture.  The network operates on two input channels representing the
in-phase and quadrature components of complex signals and is thus suitable for
raw IQ sample classification tasks.
"""

from __future__ import annotations


import torch
import torch.nn as nn
from torch.func import functional_call
from model.adab1n import AdaB1N
from utils.iq_features import append_iq_augmented_features

# Ceiling for AdaB1N's per-task concentration logits, not an exact task count.
ADAB1N_MAX_TASKS = 20


class BasicBlock1D(nn.Module):
    """1D version of the standard ResNet ``BasicBlock``."""

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        norm_layer=None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.conv1 = nn.Conv1d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover -
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AdcIqAdapter(nn.Module):
    """Reduce 3 (or more) channels into 2 IQ channels.

    Accepts either (B, 3, 2, L) [4D] or (B, 3, L) [3D] and returns (B, 2, L).
    """

    def __init__(self) -> None:
        super().__init__()
        # Single ADC mixing vector shared by both the I and Q channels: the
        # same linear combination of ADC0/ADC1/ADC2 is applied to whichever
        # channel it sees, rather than learning separate per-channel mixes.
        self.weight = nn.Parameter(torch.ones(3))
        # The adapter's effective bias is always forced to 0 in `forward`.
        self.bias = nn.Parameter(torch.zeros(2), requires_grad=False)
        # Used by the 3D path: (B, 3, L) -> (B, 2, L). The 4D path uses
        # `self.weight` with einsum instead.
        self.proj_3ch = nn.Conv1d(3, 2, kernel_size=1, bias=False)
        # Keep adapter parameters trainable by default:
        # - `self.weight` controls the 4D path (B, 3, 2, L), which is the main
        #   path for 3-ADC IQ tensors.
        # - `proj_3ch` controls the 3D fallback path (B, 3, L).
        self.weight.requires_grad = True
        for param in self.proj_3ch.parameters():
            param.requires_grad = True

    def set_initial_parameters(
        self,
        *,
        weight_4d: torch.Tensor | None = None,
        bias_4d: torch.Tensor | None = None,
        weight_3d: torch.Tensor | None = None,
        bias_3d: torch.Tensor | None = None,
        freeze: bool = False,
    ) -> None:
        """Optionally override the adapter's initial parameters.

        This helper lets you seed the adapter with known linear mappings
        (e.g. from a previous run or domain knowledge) instead of relying
        on the default random initialization.

        Args:
            weight_4d: Optional weight for the 4D path with shape (3,), shared
                across the I and Q channels.
            bias_4d: Optional bias for the 4D path with shape (2,).
            weight_3d: Optional weight for the 3D Conv1d path. Accepts either
                a tensor of shape (2, 3) or (2, 3, 1); the latter will be
                used directly as ``proj_3ch.weight``.
            bias_3d: Optional bias for the 3D Conv1d path with shape (2,).
            freeze: If True, disables gradient updates for the adapter
                parameters after initialization.

        Usage:
            >>> adapter = AdcIqAdapter()
            >>> adapter.set_initial_parameters(
            ...     weight_4d=my_w4d, bias_4d=my_b4d,
            ...     weight_3d=my_w3d, bias_3d=my_b3d,
            ...     freeze=False,
            ... )
        """
        print("[WARNING] Setting initial parameters for AdcIqAdapter")
        if weight_4d is not None:
            if weight_4d.shape != self.weight.shape:
                raise ValueError(
                    f"weight_4d must have shape {tuple(self.weight.shape)}, got {tuple(weight_4d.shape)}"
                )
            with torch.no_grad():
                self.weight.copy_(weight_4d)
        if bias_4d is not None:
            if bias_4d.shape != self.bias.shape:
                raise ValueError(
                    f"bias_4d must have shape {tuple(self.bias.shape)}, got {tuple(bias_4d.shape)}"
                )
            with torch.no_grad():
                self.bias.copy_(bias_4d)

        if weight_3d is not None:
            if weight_3d.dim() == 2:
                if weight_3d.shape != (2, 3):
                    raise ValueError(
                        f"weight_3d must have shape (2, 3) or (2, 3, 1); got {tuple(weight_3d.shape)}"
                    )
                w3 = weight_3d.view(2, 3, 1)
            elif weight_3d.dim() == 3:
                if weight_3d.shape != (2, 3, 1):
                    raise ValueError(
                        f"weight_3d must have shape (2, 3, 1); got {tuple(weight_3d.shape)}"
                    )
                w3 = weight_3d
            else:
                raise ValueError(
                    f"weight_3d must be rank 2 or 3 tensor; got dim={weight_3d.dim()}"
                )
            with torch.no_grad():
                self.proj_3ch.weight.copy_(w3)

        if bias_3d is not None:
            if bias_3d.shape != self.proj_3ch.bias.shape:
                raise ValueError(
                    f"bias_3d must have shape {tuple(self.proj_3ch.bias.shape)}, got {tuple(bias_3d.shape)}"
                )
            with torch.no_grad():
                self.proj_3ch.bias.copy_(bias_3d)

        if freeze:
            for param in (self.weight, self.bias, *self.proj_3ch.parameters()):
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Enforce a stochastic mix: the (shared) weight vector sums to 1.
        weight_sum = self.weight.sum()
        zero_sum_mask = weight_sum.abs() <= 1e-12
        denom = torch.where(zero_sum_mask, torch.ones_like(weight_sum), weight_sum)
        normalized_weight = self.weight / denom  # (3,)
        uniform = torch.full_like(self.weight, 1.0 / self.weight.size(0))
        normalized_weight = torch.where(zero_sum_mask, uniform, normalized_weight)

        # Bias is always 0 (and does not receive gradients).
        self.bias.data.zero_()

        if x.dim() == 3 and x.size(1) == 3:
            # Use proj_3ch parameters for the 3D path so gradients flow there.
            # Enforce row-stochastic mixing for the effective conv weights.
            conv_weight = self.proj_3ch.weight.squeeze(-1)  # (2, 3)
            conv_row_sums = conv_weight.sum(dim=1, keepdim=True)
            conv_zero_row_mask = conv_row_sums.abs() <= 1e-12
            conv_denom = torch.where(
                conv_zero_row_mask, torch.ones_like(conv_row_sums), conv_row_sums
            )
            normalized_conv_weight = conv_weight / conv_denom
            conv_uniform = torch.full_like(
                conv_weight, 1.0 / conv_weight.size(1)  # 1/3
            )
            normalized_conv_weight = torch.where(
                conv_zero_row_mask.expand_as(normalized_conv_weight),
                conv_uniform,
                normalized_conv_weight,
            )
            y = torch.einsum("bcl,oc->bol", x, normalized_conv_weight)
            return y + self.bias.view(1, 2, 1)
        if x.dim() != 4 or x.size(1) != 3 or x.size(2) != 2:
            raise ValueError(
                f"ADC adapter expects (B, 3, 2, L) or (B, 3, L); got shape {tuple(x.shape)}."
            )
        # Rows whose ADC1/ADC2 are padded with exact zeros (e.g. a 2-channel
        # dataset batched alongside a 3-channel one) keep ADC0's I/Q channels
        # instead of the learned 3->2 mixing, which would merely rescale them.
        #
        # This is decided per row. Deciding it for the whole batch made a
        # sample's output depend on which other samples shared its batch: one
        # genuine 3-ADC row switched every padded row in the batch onto the
        # mixing path, scaling it by the ADC0 weight (~0.5 in trained runs).
        # Training and batch-statistic evaluation hid this, because BatchNorm
        # cancels a uniform scale, but running-statistic evaluation did not.
        zero_adc_rows = x[:, 1:, :, :].abs().amax(dim=(1, 2, 3)) == 0
        # (B, 3, 2, L) -> (B, 2, 3, L)
        permuted = x.permute(0, 2, 1, 3)
        # Mix ADCs with the same weights for both I and Q: (B, 2, 3, L) x (3,) -> (B, 2, L)
        y = torch.einsum("bial,a->bil", permuted, normalized_weight)
        y = y + self.bias.view(1, 2, 1)
        return torch.where(zero_adc_rows.view(-1, 1, 1), x[:, 0, :, :], y)


class _ResNet1D(nn.Module):
    """Internal utility that mirrors ``torchvision``'s ``ResNet`` logic."""

    def __init__(
        self,
        block: type[nn.Module],
        layers: list[int],
        num_classes: int,
        in_channels: int = 2,
        norm_layer=None,
        input_adapter: nn.Module | None = None,
        use_iq_aug_features: bool = False,
        iq_aug_scaling_mode: str = "none",
        iq_aug_feature_type: str = "power",
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.input_adapter = input_adapter or nn.Identity()
        self.use_iq_aug_features = bool(use_iq_aug_features)
        self.iq_aug_scaling_mode = iq_aug_scaling_mode
        self.iq_aug_feature_type = str(iq_aug_feature_type)

        self.conv1 = nn.Conv1d(
            in_channels, 64, kernel_size=7, stride=2, padding=1, bias=False
        )
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        out_dim = 512 * block.expansion
        self.fc = nn.Linear(out_dim, num_classes)

    def _make_layer(
        self, block: type[nn.Module], planes: int, blocks: int, stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                self._norm_layer(planes * block.expansion),
            )

        layers = [
            block(
                self.inplanes, planes, stride, downsample, norm_layer=self._norm_layer
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=self._norm_layer))
        return nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        return_features=False,
        classify_feats=False,
        return_h4=False,
    ) -> torch.Tensor:  # pragma: no cover -

        if classify_feats:
            return self.fc(x)
        else:
            if x.dim() == 4:
                if not isinstance(self.input_adapter, nn.Identity):
                    x = self.input_adapter(x)
                else:
                    raise ValueError("Received 4D input but no adapter is configured.")
            elif (
                x.dim() == 3
                and x.size(1) != 2
                and not isinstance(self.input_adapter, nn.Identity)
            ):
                # When augmentation is enabled, we may already have augmented
                # 3-channel input. Don't run the 3->2 adapter in that case.
                if self.use_iq_aug_features and x.size(1) == 3:
                    pass
                else:
                    x = self.input_adapter(x)
            if x.dim() == 3 and x.size(1) == 2:
                x = append_iq_augmented_features(
                    x,
                    enabled=self.use_iq_aug_features,
                    scaling_mode=self.iq_aug_scaling_mode,
                    feature_type=self.iq_aug_feature_type,
                )

            x = self.bn1(self.conv1(x))
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            if return_h4:
                return x

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            if return_features:
                return x
            x = self.fc(x)
            return x


class ResNet1D(nn.Module):
    """ResNet-18 style network using 1D convolutions for IQ data."""

    def __init__(self, num_classes: int, args=None, in_channels: int = 2) -> None:
        super().__init__()
        self.args = args
        self.use_iq_aug_features = bool(getattr(args, "use_iq_aug_features", False))
        self.iq_aug_scaling_mode = str(getattr(args, "data_scaling", "none"))
        self.iq_aug_feature_type = str(
            getattr(
                args, "iq_aug_feature_type", getattr(args, "iq_aug_feature", "power")
            )
        )
        effective_in_channels = 3 if self.use_iq_aug_features else in_channels
        norm_layer = self._build_norm_factory(args)
        self.input_adapter = AdcIqAdapter()
        self.model = _ResNet1D(
            BasicBlock1D,
            [2, 2, 2, 2],
            num_classes,
            in_channels=effective_in_channels,
            norm_layer=norm_layer,
            input_adapter=self.input_adapter,
            use_iq_aug_features=self.use_iq_aug_features,
            iq_aug_scaling_mode=self.iq_aug_scaling_mode,
            iq_aug_feature_type=self.iq_aug_feature_type,
        )
        self.feature_dim = self.model.fc.in_features

        # Ordered names for mapping fast weights
        self.param_names = [n for n, _ in self.model.named_parameters()]
        # Learner-compat convenience (list, not registered)
        self.vars = list(self.model.parameters())
        alpha_init = getattr(args, "alpha_init", 1e-3) if args is not None else 1e-3
        self.alpha_lr = nn.ParameterList(
            [
                nn.Parameter(torch.ones_like(p) * alpha_init)
                for p in self.model.parameters()
            ]
        )

    # def classify_feats(self, x: torch.Tensor) -> torch.Tensor:
    #     logits = self.model.fc(x)
    #     return logits

    def forward(
        self,
        x: torch.Tensor,
        vars=None,
        bn_training: bool | None = None,
        classify_feats=False,
        ret_feats=False,
    ) -> torch.Tensor:
        """Run the backbone.

        Args:
            bn_training: Overrides the normalisation layers' train/eval mode for
                this call, for meta-learning inner loops that need to suppress or
                force running-stat updates. ``None`` (the default) leaves the
                ambient mode alone, so ``eval()`` normalises with the tracked
                running statistics instead of the current batch's.
        """
        prev = self.model.training
        if bn_training is not None:
            self.model.train(bn_training)
        try:
            if not classify_feats:
                # print(f"Input shape: {tuple(x.shape)}")
                x = self._prepare_input(x)
                # print(f"Prepared input shape: {tuple(x.shape)}")
            if vars is None:
                out = self.model(
                    x, return_features=ret_feats, classify_feats=classify_feats
                )
            else:
                assert len(vars) == len(
                    self.param_names
                ), f"len(vars)={len(vars)} vs params={len(self.param_names)}"
                param_dict = {n: p for n, p in zip(self.param_names, vars)}
                out = functional_call(
                    self.model,
                    param_dict,
                    (x,),
                    {"return_features": ret_feats, "classify_feats": classify_feats},
                )
        finally:
            if bn_training is not None:
                self.model.train(prev)
        return out

    def forward_features(
        self, x: torch.Tensor, vars=None, bn_training: bool | None = None
    ) -> torch.Tensor:
        return self.forward(x, vars=vars, bn_training=bn_training, ret_feats=True)

    def forward_classifier(
        self, feats: torch.Tensor, vars=None, bn_training: bool | None = None
    ) -> torch.Tensor:
        return self.forward(
            feats, vars=vars, bn_training=bn_training, classify_feats=True
        )

    # Expose only the underlying model parameters, excluding alpha lrs
    def parameters(self, recurse: bool = True):
        return self.model.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        return self.model.named_parameters(prefix=prefix, recurse=recurse)

    def define_task_lr_params(self, alpha_init: float = 1e-3) -> None:
        self.alpha_lr = nn.ParameterList(
            [
                nn.Parameter(torch.ones_like(p) * alpha_init)
                for p in self.model.parameters()
            ]
        )

    # ------------------------------------------------------------------
    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize inputs into channel-first IQ tensors.

        Accepted formats:
        - (B, F): flat samples. If F is divisible by 2 or 3, reshape to
          (B, 2, L) or (B, 3, 2, L) respectively. Ambiguous shapes (divisible
          by both 2 and 3) raise an error.
        - (B, C, L): channel-first. C can be 1 or 2 (passed through) or 3
          (interpreted as 3 ADC channels with interleaved IQ and reshaped to
          (B, 3, 2, L)).
        - (C, B, L): channel-first but transposed; will be permuted to (B, C, L).
        - (B, 3, 2, L): explicit ADC + IQ layout (passed through).
        """
        if x.dim() == 2:
            batch, features = x.shape
            if features % 2 == 0 and features % 3 == 0:
                raise ValueError(
                    f"Ambiguous flat input shape: features={features} divisible by both 2 and 3."
                )
            if features % 2 == 0 and features % 3 != 0:
                seq_len = features // 2
                x = x.view(batch, 2, seq_len)
            else:
                x = x.unsqueeze(1)
        elif x.dim() == 3:
            if x.shape[1] not in (1, 2, 3) and x.shape[0] in (1, 2, 3):
                x = x.permute(1, 0, 2).contiguous()
            if x.shape[1] == 3:
                if x.shape[2] % 2 != 0:
                    raise ValueError(
                        f"Expected even length for 3-ADC IQ input; got shape {tuple(x.shape)}."
                    )
                seq_len = x.shape[2] // 2
                x = x.view(x.shape[0], 3, 2, seq_len)
                return x
            if x.shape[1] not in (1, 2):
                raise ValueError(
                    f"Unexpected channel dimension (expected 1, 2, or 3); got shape {tuple(x.shape)}."
                )
        elif x.dim() == 4:
            if x.shape[1] == 3 and x.shape[2] == 2:
                return x
            raise ValueError(
                f"Unexpected 4D input shape {tuple(x.shape)}; expected (B, 3, 2, L)."
            )
        else:
            raise ValueError(
                f"Unexpected input shape {tuple(x.shape)}; expected 2D, 3D, or 4D tensor."
            )
        return x

    # ------------------------------------------------------------------
    def _build_norm_factory(self, args):
        """Build the per-channel normalization layer factory for this backbone.

        Selected via ``args.norm_type`` (``"batchnorm"`` (default),
        ``"groupnorm"``, or ``"adab1n"``); ``args.use_groupnorm`` remains
        supported as a legacy alias for ``norm_type="groupnorm"``.

        Args:
            args: Experiment arguments, or ``None`` for the BatchNorm1d default.

        Returns:
            A callable mapping a channel count to a fresh normalization module.
        """
        if args is None:
            return lambda channels: nn.BatchNorm1d(channels)

        use_groupnorm = bool(getattr(args, "use_groupnorm", False))
        norm_type = str(getattr(args, "norm_type", "batchnorm")).lower()
        if norm_type in {"groupnorm", "group_norm", "gn"}:
            use_groupnorm = True

        if use_groupnorm:
            min_channels_per_group = max(
                1, int(getattr(args, "groupnorm_min_channels", 64))
            )

            def gn_factory(channels: int):
                groups = max(1, channels // min_channels_per_group)
                while groups > 1 and channels % groups != 0:
                    groups -= 1
                return nn.GroupNorm(groups, channels)

            return gn_factory

        if norm_type in {"adab1n", "ada_b1n", "adab2n"}:
            return self._build_adab1n_factory(args)

        return lambda c: nn.BatchNorm1d(c)

    def _build_adab1n_factory(self, args):
        """Build an :class:`~model.adab1n.AdaB1N` factory sized from ``args``.

        Args:
            args: Experiment arguments; reads ``n_tasks``, ``kappa`` and
                ``adab1n_init_weight`` when present.

        Returns:
            A callable mapping a channel count to a fresh ``AdaB1N`` module.
        """
        # num_tasks only needs to be a ceiling: unused ``task_weight`` entries are
        # sliced out of the forward and never receive gradient, so over-allocating
        # is numerically inert, while under-allocating makes end_task() raise.
        num_tasks = max(ADAB1N_MAX_TASKS, int(getattr(args, "n_tasks", 1) or 1))
        kappa = float(getattr(args, "kappa", 1.0) or 1.0)
        init_weight = float(getattr(args, "adab1n_init_weight", 0.0) or 0.0)

        def adab1n_factory(channels: int):
            return AdaB1N(
                channels,
                num_tasks=num_tasks,
                kappa=kappa,
                init_weight=init_weight,
            )

        return adab1n_factory


__all__ = ["ResNet1D"]
