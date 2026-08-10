# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
from torch.nn.functional import relu, normalize
from itertools import chain
from model.resnet1d import _ResNet1D, BasicBlock1D, AdcIqAdapter


class ContextInputAdapter(nn.Module):
    """Normalize 3-ADC IQ inputs before adapting to 2 channels."""

    def __init__(self) -> None:
        super().__init__()
        self.adapter = AdcIqAdapter()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.size(1) == 3:
            if x.size(2) % 2 != 0:
                raise ValueError(
                    "Expected even sequence length for 3-channel interleaved IQ "
                    f"input; got shape {tuple(x.shape)}."
                )
            sequence_length = x.size(2) // 2
            x = x.view(x.size(0), 3, 2, sequence_length)
            return self.adapter(x)
        if x.dim() == 4 and x.size(1) == 3 and x.size(2) == 2:
            return self.adapter(x)
        return x


def Xavier(m):
    if m.__class__.__name__ == "Linear":
        fan_in, fan_out = m.weight.data.size(1), m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        m.bias.data.fill_(0.0)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def Flatten(x):
    return x.view(x.size(0), -1)


class noReLUBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(noReLUBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return out


class ContextNet(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channels=2,
        task_emb=64,
        n_tasks=17,
        use_iq_aug_features: bool = False,
        iq_aug_scaling_mode: str = "none",
        iq_aug_feature_type: str = "power",
        use_film: bool = True,
    ):
        super(ContextNet, self).__init__()
        self.use_film = use_film
        self.in_planes = nf = 64
        # self.conv1 = conv3x3(3, nf * 1)
        # self.bn1 = nn.BatchNorm2d(nf * 1)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        # self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.linear = nn.Linear(nf * 8 * block.expansion, int(num_classes))
        self.model = _ResNet1D(
            BasicBlock1D,
            [2, 2, 2, 2],
            num_classes,
            in_channels=in_channels,
            input_adapter=ContextInputAdapter(),
            use_iq_aug_features=use_iq_aug_features,
            iq_aug_scaling_mode=iq_aug_scaling_mode,
            iq_aug_feature_type=iq_aug_feature_type,
        )
        self.feature_dim = self.model.fc.in_features
        self.det_head = nn.Linear(self.feature_dim, 1)

        # self.film1 = nn.Linear(task_emb, nf * 1 * 2)
        # self.film2 = nn.Linear(task_emb, nf * 2 * 2)
        # self.film3 = nn.Linear(task_emb, nf * 4 * 2)
        self.film4 = nn.Linear(task_emb, nf * 8 * 2)
        self.nf = nf
        self.emb = torch.nn.Embedding(n_tasks, task_emb)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def base_param(self):
        base_iter = chain(
            self.model.parameters(),
            self.det_head.parameters(),
        )
        for param in base_iter:
            if param.requires_grad:
                yield param

    def context_param(self):
        film_iter = chain(self.emb.parameters(), self.film4.parameters())
        for param in film_iter:
            yield param

    def forward_h4(self, x):
        return self.model.forward(x, return_h4=True)

    def _apply_film(self, h4, t, use_all=True):
        device = h4.device
        B, C, _ = h4.shape

        if not self.use_film:
            # Ablation: drop the task-embedding->FiLM modulation entirely.
            return relu(h4)

        if t is None:
            gamma4 = torch.zeros((B, C, 1), device=device, dtype=h4.dtype)
            beta4 = torch.zeros((B, C, 1), device=device, dtype=h4.dtype)
        elif isinstance(t, int):
            t = torch.full((B,), t, dtype=torch.long, device=device)
        else:
            t = torch.as_tensor(t, dtype=torch.long, device=device)
            if t.ndim == 0 or t.numel() == 1:
                t = t.view(1).expand(B)
            else:
                t = t.view(-1)
                if t.numel() != B:
                    raise ValueError(f"Expected {B} task ids, got {t.numel()}.")
        if t is None:
            h4_new = gamma4 * h4 + beta4
        else:
            t = self.emb(t)
            film4 = self.film4(t)
            gamma4, beta4 = film4.split(C, 1)
            gamma4 = normalize(gamma4, p=2, dim=1).view(B, C, 1)
            beta4 = normalize(beta4, p=2, dim=1).view(B, C, 1)
            h4_new = gamma4 * h4 + beta4

        if use_all:
            return relu(h4_new) + relu(h4)
        return relu(h4_new)

    def _pool_features(self, h4):
        out = self.model.avgpool(h4)
        return out.view(out.size(0), -1)

    def forward_features(self, x, t=None, use_all=True):
        h4 = self.forward_h4(x)
        h4 = self._apply_film(h4, t, use_all=use_all)
        return self._pool_features(h4)

    def forward_heads(self, x, t=None, use_all=True):
        h4 = self.forward_h4(x)
        det_feat = self._pool_features(h4)
        det_logits = self.det_head(det_feat).squeeze(1)
        cls_h4 = self._apply_film(h4, t, use_all=use_all)
        cls_feat = self._pool_features(cls_h4)
        cls_logits = self.model.fc(cls_feat)
        return det_logits, cls_logits

    def forward_det_agnostic(self, x, use_all=True):
        h4 = self.forward_h4(x)
        feat = self._pool_features(h4)
        return self.det_head(feat).squeeze(1)

    def forward(self, x, t, use_all=True):
        h4 = self.forward_h4(x)
        h4 = self._apply_film(h4, t, use_all=use_all)
        feat = self._pool_features(h4)
        y = self.model.fc(feat)
        return y


def ContextNet18(
    num_classes,
    in_channels=2,
    n_tasks=17,
    task_emb=64,
    use_iq_aug_features: bool = False,
    iq_aug_scaling_mode: str = "none",
    iq_aug_feature_type: str = "power",
    use_film: bool = True,
):
    return ContextNet(
        num_classes,
        in_channels=in_channels,
        n_tasks=n_tasks,
        task_emb=task_emb,
        use_iq_aug_features=use_iq_aug_features,
        iq_aug_scaling_mode=iq_aug_scaling_mode,
        iq_aug_feature_type=iq_aug_feature_type,
        use_film=use_film,
    )
