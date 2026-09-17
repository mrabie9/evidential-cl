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

from model.resnet1d import AdcIqAdapter, BasicBlock1D, build_norm_factory
from model.evidential_modules import DM, Dempster_Shafer_module, pignistic_probability
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
        head: str = "dm",
        belief_init: str = "random",
        temper: float = 0.0,
        activation_norm: str = "max",
        readout_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.readout_scale = float(readout_scale)
        self.n_classes = n_classes
        self.head = str(head).lower()
        self.norm = nn.LayerNorm(n_channels)
        self.ds_module = Dempster_Shafer_module(
            n_feature_maps=n_channels,
            n_classes=n_classes,
            n_prototypes=max(1, n_classes) * proto_factor,
            metric=metric,
            belief_init=belief_init,
            temper=temper,
            activation_norm=activation_norm,
        )
        self.dm_layer = DM(num_class=n_classes, nu=float(nu))

    def forward(
        self, feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled = F.adaptive_avg_pool1d(feat, 1).flatten(1)
        pooled = self.norm(pooled)
        mass = self.ds_module(pooled)
        readout = (
            pignistic_probability(mass, scale=self.readout_scale)
            if self.head == "pignistic"
            else self.dm_layer(mass)
        )
        omega = mass[:, -1]
        beliefs = mass[:, :-1]
        return readout, beliefs, omega


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
        head: str = "dm",
        classes_per_task: Sequence[int] | None = None,
        belief_init: str = "random",
        temper: float = 0.0,
        activation_norm: str = "max",
        readout_scale: str | float = "untemper",
        ds_detach: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        # Direction G: with ds_detach, the evidential path sees a detached feature,
        # so cross-entropy alone shapes the backbone and the DS head/probes are a
        # pure readout on top of it.
        self.ds_detach = bool(ds_detach)
        self.belief_init = str(belief_init).lower()
        self.temper = float(temper)
        self.activation_norm = str(activation_norm).lower()
        self._readout_scale_spec = readout_scale
        self.probe_stages = tuple(sorted(set(int(s) for s in probe_stages)))
        self.in_planes = 64
        self.metric = str(metric).lower()
        self.head = str(head).lower()
        self.proto_factor = int(proto_factor)

        # Per-task evidential heads: each task owns a Dempster-Shafer head over
        # only its own classes. Routing forward(x, t) through head t decouples
        # stability from plasticity -- head t is only trained during task t (its
        # params get no gradient otherwise, so it is naturally frozen afterwards),
        # so new tasks keep full plasticity and old tasks suffer no head drift.
        # The shared backbone is what consolidation protects.
        self.classes_per_task = (
            [int(c) for c in classes_per_task]
            if classes_per_task
            else [int(num_classes)]
        )
        offsets, running = [], 0
        for count in self.classes_per_task:
            offsets.append((running, running + count))
            running += count
        self._task_offsets = offsets

        self.use_iq_aug_features = bool(getattr(args, "use_iq_aug_features", False))
        self.iq_aug_scaling_mode = str(getattr(args, "data_scaling", "none"))
        self.iq_aug_feature_type = str(
            getattr(
                args, "iq_aug_feature_type", getattr(args, "iq_aug_feature", "power")
            )
        )
        effective_in_channels = 3 if self.use_iq_aug_features else in_channels

        self.input_adapter = AdcIqAdapter()

        # Honour --use_groupnorm / --norm_type like ResNet1D. The evidential head
        # reads feature *direction* (cosine distance to prototypes), so it is
        # unusually sensitive to the train/eval gap BatchNorm's running statistics
        # introduce: measured directional coherence of the pooled features swings
        # 0.16-0.42 under running stats against a steady 0.07 under batch stats,
        # which is enough to sweep every sample onto one prototype at eval time.
        self._norm_factory = build_norm_factory(args)
        self.conv1 = nn.Conv1d(
            effective_in_channels, 64, kernel_size=7, stride=2, padding=1, bias=False
        )
        self.bn1 = self._norm_factory(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        self.do = nn.Dropout(p=0.2)

        feat_dim = 512 * BasicBlock1D.expansion
        self.feat_norm = nn.LayerNorm(feat_dim)

        self.ds_heads = nn.ModuleDict(
            {
                str(task_index): Dempster_Shafer_module(
                    n_feature_maps=feat_dim,
                    n_classes=count,
                    n_prototypes=count * proto_factor,
                    metric=self.metric,
                    belief_init=self.belief_init,
                    temper=self.temper,
                    activation_norm=self.activation_norm,
                )
                for task_index, count in enumerate(self.classes_per_task)
            }
        )
        # Auxiliary per-task LINEAR heads. The DS head needs ~4x the steps of a
        # linear+CE head to reach the same accuracy, which at a fixed epoch budget
        # is indistinguishable from catastrophic forgetting. These give the shared
        # backbone a fast route to good features; the DS head then forms a readout
        # on top of them instead of having to shape them itself. Predictions at
        # evaluation still come from the DS head, so any gain is a statement about
        # the DS head, not about the linear one.
        self.ce_heads = nn.ModuleDict(
            {
                str(task_index): nn.Linear(feat_dim, count)
                for task_index, count in enumerate(self.classes_per_task)
            }
        )
        self.dm_heads = nn.ModuleDict(
            {
                str(task_index): DM(num_class=count, nu=float(nu))
                for task_index, count in enumerate(self.classes_per_task)
            }
        )

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
                    head=self.head,
                    belief_init=self.belief_init,
                    temper=self.temper,
                    activation_norm=self.activation_norm,
                    readout_scale=self._scale_for(num_classes),
                )
                for stage in self.probe_stages
            }
        )

    def _scale_for(self, n_classes: int) -> float:
        """Readout rescale that cancels tempering for the decision only.

        ``"untemper"`` returns ``P**temper`` for this head's prototype count, so
        temper=0 is a no-op and any temper>0 keeps the logits at their untempered
        scale. A float overrides it; ``"none"`` disables it.
        """
        spec = self._readout_scale_spec
        if isinstance(spec, (int, float)):
            return float(spec)
        if str(spec).lower() != "untemper" or self.temper == 0.0:
            return 1.0
        return float(max(1, n_classes) * self.proto_factor) ** self.temper

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        norm_layer = self._norm_factory
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
    def _task_readout(self, features: torch.Tensor, task: int):
        """Run task ``task``'s head; return (local class scores, mass)."""
        mass = self.ds_heads[str(task)](features)
        if self.head == "pignistic":
            scores = pignistic_probability(
                mass, scale=self._scale_for(mass.size(-1) - 1)
            )
        else:
            n_classes = mass.size(-1) - 1
            scores = self.dm_heads[str(task)](mass)[:, :n_classes]
        return scores, mass

    def ce_logits(self, features: torch.Tensor, task: int) -> torch.Tensor:
        """Auxiliary linear-head logits for ``task``, scattered to global width."""
        scores = self.ce_heads[str(int(task))](features)
        start, end = self._task_offsets[int(task)]
        out = features.new_full((features.size(0), self.num_classes), -1e9)
        out[:, start:end] = scores
        return out

    def forward(
        self,
        x: torch.Tensor,
        task: int | None = None,
        *,
        cil_all_seen_upto_task: int | None = None,
        return_features: bool = False,
        return_probes: bool = False,
        return_ce: bool = False,
        bn_training: bool | None = None,
    ):
        """``bn_training`` mirrors :meth:`model.resnet1d.ResNet1D.forward`.

        Every ResNet1D-based learner in this repo reaches its backbone with that
        method's default ``bn_training=True``, so the harness scores them with
        *batch* statistics. EUCR had no such parameter and therefore inherited
        whatever ``model.eval()`` set, i.e. frozen running statistics -- a
        different evaluation protocol from every model it is compared against.
        Passing it explicitly puts EUCR on whichever protocol the caller wants.
        ``None`` keeps the module's current mode (the previous behaviour).
        """
        if bn_training is not None:
            prev_mode = self.training
            self.train(bn_training)
            try:
                return self.forward(
                    x,
                    task,
                    cil_all_seen_upto_task=cil_all_seen_upto_task,
                    return_features=return_features,
                    return_probes=return_probes,
                    return_ce=return_ce,
                )
            finally:
                self.train(prev_mode)

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

        # Per-task head routing. ``task=None`` (e.g. importance estimation, which
        # only needs the probes) skips the head entirely. Each task's local scores
        # are scattered into a global ``[B, num_classes]`` tensor at the task's
        # class offsets, so callers keep working in the global label space. CIL
        # runs every seen head; TIL runs only the queried task's head.
        ds_features = features.detach() if self.ds_detach else features

        eu = omegas = beliefs = None
        if task is not None:
            if cil_all_seen_upto_task is not None:
                tasks_to_run = range(0, int(cil_all_seen_upto_task) + 1)
            else:
                tasks_to_run = [int(task)]
            eu = features.new_zeros((features.size(0), self.num_classes))
            for task_index in tasks_to_run:
                scores, mass = self._task_readout(ds_features, task_index)
                start, end = self._task_offsets[task_index]
                eu[:, start:end] = scores
                if task_index == int(task):
                    omegas = mass[:, -1]
                    beliefs = mass[:, :-1]

        if return_probes:
            stage_feats = (
                {1: s1.detach(), 2: s2.detach(), 3: s3.detach(), 4: s4.detach()}
                if self.ds_detach
                else {1: s1, 2: s2, 3: s3, 4: s4}
            )
            probe_outs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            for stage in self.probe_stages:
                probe_outs.append(self.probes[str(stage)](stage_feats[stage]))
            if return_ce:
                # Appended last only when asked, so compute_importance's out[-1]
                # (which requests probes alone) keeps meaning probe_outs.
                return (
                    eu,
                    features,
                    omegas,
                    beliefs,
                    probe_outs,
                    self.ce_logits(features, task) if task is not None else None,
                )
            return eu, features, omegas, beliefs, probe_outs

        if return_features:
            return eu, features, omegas, beliefs
        return eu


__all__ = ["EucrResNet1D", "EvidentialProbe"]
