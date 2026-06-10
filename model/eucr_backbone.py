"""Evidential 1D ResNet backbone for EUCR.

This mirrors La-MAML's :class:`model.resnet1d.ResNet1D` feature extractor (same
IQ input handling, ``AdcIqAdapter`` and optional augmented-feature channel) but
replaces the linear ``fc`` head with a Dempster-Shafer evidential head and
attaches a lightweight evidential *probe* to the output of every ResNet stage
(``layer1..layer4``).

Each probe global-average-pools a stage feature map to a per-channel descriptor
and emits Dempster-Shafer class belief masses plus an ignorance mass ``omega``.
Training the probes with deep evidential supervision forces every backbone stage
to produce calibrated, low-ignorance evidence; the resulting per-stage ``omega``
is the signal EUCR uses to derive per-channel / per-parameter consolidation
importance (see :mod:`model.eucr_consolidation`).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.resnet1d import AdcIqAdapter, BasicBlock1D
from model.evidential_modules import DM, Dempster_Shafer_module
from utils.iq_features import append_iq_augmented_features


class EvidentialProbe(nn.Module):
    """A small Dempster-Shafer readout attached to one backbone stage."""

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        nu: float = 0.9,
        proto_factor: int = 20,
        metric: str = "cosine",
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.norm = nn.LayerNorm(n_channels)
        self.ds_module = Dempster_Shafer_module(
            n_feature_maps=n_channels,
            n_classes=n_classes,
            n_prototypes=max(1, n_classes) * proto_factor,
            metric=metric,
        )
        self.dm_layer = DM(num_class=n_classes, nu=float(nu))

    def forward(
        self, feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled = F.adaptive_avg_pool1d(feat, 1).flatten(1)
        pooled = self.norm(pooled)
        mass = self.ds_module(pooled)
        eu = self.dm_layer(mass)
        omega = mass[:, -1]
        beliefs = mass[:, :-1]
        return eu, beliefs, omega


class EucrResNet1D(nn.Module):
    """Evidential ResNet-1D whose stages each carry an evidential probe.

    ``forward`` returns the final head expected utilities ``[B, n_classes + 1]``
    (last column is the ignorance mass). With ``return_features`` / ``return_probes``
    it additionally exposes the pooled feature vector, the head omega / beliefs,
    and the per-stage probe outputs.
    """

    def __init__(
        self,
        num_classes: int,
        args=None,
        num_blocks: Sequence[int] = (2, 2, 2, 2),
        nu: float = 0.9,
        probe_stages: Sequence[int] = (1, 2, 3, 4),
        proto_factor: int = 20,
        in_channels: int = 2,
        metric: str = "cosine",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.probe_stages = tuple(sorted(set(int(s) for s in probe_stages)))
        self.in_planes = 64
        self.metric = str(metric).lower()

        self.use_iq_aug_features = bool(getattr(args, "use_iq_aug_features", False))
        self.iq_aug_scaling_mode = str(getattr(args, "data_scaling", "none"))
        self.iq_aug_feature_type = str(
            getattr(
                args, "iq_aug_feature_type", getattr(args, "iq_aug_feature", "power")
            )
        )
        effective_in_channels = 3 if self.use_iq_aug_features else in_channels

        self.input_adapter = AdcIqAdapter()

        self.conv1 = nn.Conv1d(
            effective_in_channels, 64, kernel_size=7, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        self.do = nn.Dropout(p=0.2)

        feat_dim = 512 * BasicBlock1D.expansion
        self.feat_norm = nn.LayerNorm(feat_dim)

        self.ds_head = Dempster_Shafer_module(
            n_feature_maps=feat_dim,
            n_classes=num_classes,
            n_prototypes=num_classes * proto_factor,
            metric=self.metric,
        )
        self.dm_head = DM(num_class=num_classes, nu=float(nu))

        stage_channels = {
            1: 64 * BasicBlock1D.expansion,
            2: 128 * BasicBlock1D.expansion,
            3: 256 * BasicBlock1D.expansion,
            4: 512 * BasicBlock1D.expansion,
        }
        self.probes = nn.ModuleDict(
            {
                str(stage): EvidentialProbe(
                    stage_channels[stage],
                    num_classes,
                    nu=nu,
                    proto_factor=proto_factor,
                    metric=self.metric,
                )
                for stage in self.probe_stages
            }
        )

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        norm_layer = nn.BatchNorm1d
        downsample = None
        if stride != 1 or self.in_planes != planes * BasicBlock1D.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.in_planes,
                    planes * BasicBlock1D.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                norm_layer(planes * BasicBlock1D.expansion),
            )
        layers = [
            BasicBlock1D(
                self.in_planes, planes, stride, downsample, norm_layer=norm_layer
            )
        ]
        self.in_planes = planes * BasicBlock1D.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(self.in_planes, planes, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    # ------------------------------------------------------------------
    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise inputs into channel-first IQ tensors (mirrors ResNet1D)."""
        if x.dim() == 2:
            batch, features = x.shape
            if features % 2 == 0 and features % 3 == 0:
                raise ValueError(
                    f"Ambiguous flat input shape: features={features} divisible by both 2 and 3."
                )
            if features % 2 == 0:
                x = x.view(batch, 2, features // 2)
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
                x = x.view(x.shape[0], 3, 2, x.shape[2] // 2)
        elif x.dim() == 4:
            if not (x.shape[1] == 3 and x.shape[2] == 2):
                raise ValueError(
                    f"Unexpected 4D input shape {tuple(x.shape)}; expected (B, 3, 2, L)."
                )
        else:
            raise ValueError(
                f"Unexpected input shape {tuple(x.shape)}; expected 2D, 3D, or 4D tensor."
            )

        if x.dim() == 4:
            x = self.input_adapter(x)
        elif x.dim() == 3 and x.shape[1] == 3 and not self.use_iq_aug_features:
            x = self.input_adapter(x)

        if x.dim() == 3 and x.size(1) == 2:
            x = append_iq_augmented_features(
                x,
                enabled=self.use_iq_aug_features,
                scaling_mode=self.iq_aug_scaling_mode,
                feature_type=self.iq_aug_feature_type,
            )
        return x

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_probes: bool = False,
    ):
        x = self._prepare_input(x)
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))

        s1 = self.layer1(out)
        out = self.do(s1)
        s2 = self.layer2(out)
        out = self.do(s2)
        s3 = self.layer3(out)
        out = self.do(s3)
        s4 = self.layer4(out)
        out = self.do(s4)

        out = F.adaptive_avg_pool1d(out, 1).flatten(1)
        features = self.feat_norm(self.do(out))

        mass = self.ds_head(features)
        eu = self.dm_head(mass)
        omegas = mass[:, -1]
        beliefs = mass[:, :-1]

        if return_probes:
            stage_feats = {1: s1, 2: s2, 3: s3, 4: s4}
            probe_outs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            for stage in self.probe_stages:
                probe_outs.append(self.probes[str(stage)](stage_feats[stage]))
            return eu, features, omegas, beliefs, probe_outs

        if return_features:
            return eu, features, omegas, beliefs
        return eu


__all__ = ["EucrResNet1D", "EvidentialProbe"]
