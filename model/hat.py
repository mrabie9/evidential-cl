"""Hard Attention to the Task (HAT) for the La-MAML codebase.

This implementation mirrors the original HAT design: every convolutional
and fully-connected layer receives a task-specific gate that modulates its
activations.  Gradients are masked using the familiar min(pre, post) rule
so previously allocated units remain untouched when new tasks arrive.
The embedding compensation and clamping heuristics from the reference
implementation are also preserved.
"""

from __future__ import annotations

from collections import defaultdict
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from model.replay_utils import unpack_y_to_class_labels
from utils.training_metrics import macro_recall
from utils import misc_utils
from utils.class_weighted_loss import classification_cross_entropy
from model.resnet1d import AdcIqAdapter
from utils.iq_features import append_iq_augmented_features


class HatInputAdapter(nn.Module):
    """Map 3-ADC input into 2 IQ channels via ``AdcIqAdapter``.

    For 3D tensors with shape ``(B, 3, L)``, this adapter first deinterleaves
    the last axis into explicit IQ channels and reshapes to ``(B, 3, 2, L/2)``.
    The ADC adapter then reduces ADC channels to a canonical ``(B, 2, L/2)``
    tensor expected by the HAT backbone.
    """

    def __init__(self) -> None:
        super().__init__()
        self._adapter = AdcIqAdapter()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.size(1) == 3:
            if x.size(2) % 2 != 0:
                raise ValueError(
                    "Expected even sequence length for 3-channel interleaved IQ "
                    f"input; got shape {tuple(x.shape)}."
                )
            x = misc_utils.deinterleave_iq_last_axis(x)
            return self._adapter(x)
        if x.dim() == 4 and x.size(1) == 3 and x.size(2) == 2:
            return self._adapter(x)
        return x


@dataclass
class HatConfig:
    inner_steps: int = 1
    lr: float = 1e-4
    gamma: float = 0.75
    smax: float = 50
    grad_clip_norm: float = 0.0
    anneal_schedule: str = "linear"  # "linear" or "geometric" (original paper)

    cuda: bool = True
    optimizer: str = "sgd"
    arch: str = "resnet1d"
    dataset: str = "iq"
    input_channels: int = 2

    @staticmethod
    def from_args(args: object) -> "HatConfig":
        cfg = HatConfig()
        # Override defaults with any args attributes that match. `None` means the
        # argument was registered but never set -- the parser gives `gamma` a None
        # default because MER uses the same flag name for a different quantity --
        # so the dataclass default stands.
        for field in cfg.__dataclass_fields__:
            value = getattr(args, field, None)
            if value is not None:
                setattr(cfg, field, value)
        if getattr(args, "clipgrad", None) is not None and not hasattr(
            args, "grad_clip_norm"
        ):
            cfg.grad_clip_norm = getattr(args, "clipgrad")
        return cfg


class GateRegistry:
    """Tracks gate metadata and pre/post relationships for masking."""

    def __init__(self) -> None:
        self.specs: List[Dict[str, Optional[str]]] = []

    def register(self, name: str, size: int, input_gate: Optional[str]) -> str:
        self.specs.append({"name": name, "size": size, "input": input_gate})
        return name


class HatBasicBlock1D(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int,
        prev_gate: Optional[str],
        prefix: str,
        registry: GateRegistry,
        param_register,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(planes)

        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv1d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes),
            )
        else:
            self.downsample = None

        self.gate_conv1 = registry.register(f"{prefix}.conv1", planes, prev_gate)
        param_register(
            f"{prefix}.conv1", self.conv1, post_gate=self.gate_conv1, pre_gate=prev_gate
        )
        param_register(f"{prefix}.bn1", self.bn1, post_gate=self.gate_conv1)

        self.gate_conv2 = registry.register(f"{prefix}.conv2", planes, self.gate_conv1)
        param_register(
            f"{prefix}.conv2",
            self.conv2,
            post_gate=self.gate_conv2,
            pre_gate=self.gate_conv1,
        )
        param_register(f"{prefix}.bn2", self.bn2, post_gate=self.gate_conv2)

        if self.downsample is not None:
            param_register(
                f"{prefix}.downsample.0",
                self.downsample[0],
                post_gate=self.gate_conv2,
                pre_gate=prev_gate,
            )
            param_register(
                f"{prefix}.downsample.1", self.downsample[1], post_gate=self.gate_conv2
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(
        self, x: torch.Tensor, mask_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        gate1 = mask_dict[self.gate_conv1].view(1, -1, 1)
        out = out * gate1.expand_as(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        gate2 = mask_dict[self.gate_conv2].view(1, -1, 1)
        out = out * gate2.expand_as(out)
        return out


class HatResNet1D(nn.Module):
    """ResNet-18 backbone instrumented with per-layer HAT gates."""

    def __init__(
        self, in_channels: int, num_classes: int, registry: GateRegistry, param_register
    ) -> None:
        super().__init__()
        self.registry = registry
        self.param_register = param_register

        self.inplanes = 64
        self.conv1 = nn.Conv1d(
            in_channels, 64, kernel_size=7, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

        self.param_gate_map: Dict[str, Dict[str, Optional[str]]] = {}

        conv1_gate = self.registry.register("conv1", 64, None)
        self.param_register("conv1", self.conv1, post_gate=conv1_gate, pre_gate=None)
        self.param_register("bn1", self.bn1, post_gate=conv1_gate)

        prev_gate = conv1_gate
        self.layer1, prev_gate = self._make_layer(
            64, 2, stride=1, layer_id=1, prev_gate=prev_gate
        )
        self.layer2, prev_gate = self._make_layer(
            128, 2, stride=2, layer_id=2, prev_gate=prev_gate
        )
        self.layer3, prev_gate = self._make_layer(
            256, 2, stride=2, layer_id=3, prev_gate=prev_gate
        )
        self.layer4, prev_gate = self._make_layer(
            512, 2, stride=2, layer_id=4, prev_gate=prev_gate
        )

        fc_gate = self.registry.register("fc", 512, prev_gate)
        self.param_register("fc", self.fc, post_gate=fc_gate, pre_gate=prev_gate)
        self.final_gate = fc_gate

    # ------------------------------------------------------------------
    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int,
        layer_id: int,
        prev_gate: Optional[str],
    ) -> Tuple[nn.ModuleList, Optional[str]]:
        layers = nn.ModuleList()
        inplanes = self.inplanes

        for block_idx in range(blocks):
            block_stride = stride if block_idx == 0 else 1
            block = HatBasicBlock1D(
                inplanes,
                planes,
                block_stride,
                prev_gate,
                prefix=f"layer{layer_id}.{block_idx}",
                registry=self.registry,
                param_register=self.param_register,
            )
            layers.append(block)
            prev_gate = block.gate_conv2
            inplanes = planes * block.expansion

        self.inplanes = planes * HatBasicBlock1D.expansion
        return layers, prev_gate

    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, mask_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        gate = mask_dict["conv1"].view(1, -1, 1)
        out = out * gate.expand_as(out)
        out = self.maxpool(out)

        for block in self.layer1:
            out = block(out, mask_dict)
        for block in self.layer2:
            out = block(out, mask_dict)
        for block in self.layer3:
            out = block(out, mask_dict)
        for block in self.layer4:
            out = block(out, mask_dict)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        fc_gate = mask_dict[self.final_gate].view(1, -1)
        out = out * fc_gate.expand_as(out)
        logits = self.fc(out)
        return logits


class HatBackbone(nn.Module):
    """Wraps the gated ResNet1D and exposes mask utilities."""

    def __init__(
        self, n_inputs: int, n_tasks: int, n_outputs: int, cfg: HatConfig, args: object
    ) -> None:
        super().__init__()
        if cfg.arch.lower() != "resnet1d":
            raise NotImplementedError(
                "HAT is currently implemented only for ResNet1D in this repository"
            )

        self.cfg = cfg
        self.n_tasks = n_tasks
        self.n_outputs = n_outputs

        if cfg.dataset.lower() == "iq" or cfg.input_channels == 2:
            if n_inputs % 2 != 0:
                raise ValueError("IQ inputs must have an even number of features")
            self.use_iq_aug_features = bool(getattr(args, "use_iq_aug_features", False))
            self.iq_aug_scaling_mode = str(getattr(args, "data_scaling", "none"))
            self.iq_aug_feature_type = str(
                getattr(
                    args,
                    "iq_aug_feature_type",
                    getattr(args, "iq_aug_feature", "power"),
                )
            )
            self.in_channels = 3 if self.use_iq_aug_features else 2
            self.seq_len = n_inputs // 2
        else:
            raise NotImplementedError

        # Adapter maps 3-channel (e.g. (B, 3, 1024)) to 2-channel; 2-channel input unchanged
        self.input_adapter = HatInputAdapter()

        registry = GateRegistry()

        def param_register(
            module_path: str,
            module: nn.Module,
            post_gate: str,
            pre_gate: Optional[str] = None,
        ) -> None:
            key_prefix = module_path
            if isinstance(module, nn.Conv1d):
                self.param_gate_map[f"{key_prefix}.weight"] = {
                    "type": "conv_weight",
                    "post": post_gate,
                    "pre": pre_gate,
                }
                if module.bias is not None:
                    self.param_gate_map[f"{key_prefix}.bias"] = {
                        "type": "bias",
                        "post": post_gate,
                    }
            elif isinstance(module, nn.BatchNorm1d):
                self.param_gate_map[f"{key_prefix}.weight"] = {
                    "type": "bn_weight",
                    "post": post_gate,
                }
                if module.bias is not None:
                    self.param_gate_map[f"{key_prefix}.bias"] = {
                        "type": "bn_bias",
                        "post": post_gate,
                    }
            elif isinstance(module, nn.Linear):
                self.param_gate_map[f"{key_prefix}.weight"] = {
                    "type": "linear_weight",
                    "post": post_gate,
                    "pre": pre_gate,
                }

        self.param_gate_map: Dict[str, Dict[str, Optional[str]]] = {}
        self.model = HatResNet1D(self.in_channels, n_outputs, registry, param_register)

        self.gate_specs = registry.specs
        self.gate_to_idx = {
            spec["name"]: idx for idx, spec in enumerate(self.gate_specs)
        }
        self.embeddings = nn.ModuleList(
            [nn.Embedding(n_tasks, spec["size"]) for spec in self.gate_specs]
        )

        for emb in self.embeddings:
            nn.init.uniform_(emb.weight, -0.1, 0.1)

        # Cache shapes to simplify masking logic
        self.param_shapes = {
            name: tuple(param.size()) for name, param in self.model.named_parameters()
        }

        self.freeze_bn_stats = True

    def set_bn_eval(self, flag: bool) -> None:
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm1d):
                # keep BN affine params trainable, just freeze running stats
                m.track_running_stats = True
                m.running_mean = m.running_mean  # no-op, clarifies intent
                m.running_var = m.running_var
                if flag:
                    m.eval()  # freezes running stats updates
                else:
                    m.train()  # re-enables running stats updates

    # ------------------------------------------------------------------
    def _ensure_iq_shape(self, x: torch.Tensor) -> torch.Tensor:
        if self.in_channels == 1:
            return x
        if x.dim() == 2:
            if self.in_channels == 2:
                return misc_utils.deinterleave_iq_last_axis(x)
            return x.view(x.size(0), self.in_channels, self.seq_len)
        return x

    def mask(self, task: torch.LongTensor, s: float) -> List[torch.Tensor]:
        masks: List[torch.Tensor] = []
        for emb in self.embeddings:
            masks.append(torch.sigmoid(s * emb(task)))
        return masks

    def forward(
        self,
        task: torch.LongTensor,
        x: torch.Tensor,
        s: float,
        return_masks: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]] | torch.Tensor:
        x = self._ensure_iq_shape(x)
        x = self.input_adapter(x)
        if x.dim() == 3 and x.size(1) == 2:
            x = append_iq_augmented_features(
                x,
                enabled=self.use_iq_aug_features,
                scaling_mode=self.iq_aug_scaling_mode,
                feature_type=self.iq_aug_feature_type,
            )
        masks = self.mask(task, s)  # mask for each stage
        mask_dict = {  # dict of masks per layer
            spec["name"]: masks[idx].view(-1)
            for idx, spec in enumerate(self.gate_specs)
        }
        logits = self.model(x, mask_dict)
        if return_masks:
            return logits, masks
        return logits

    # ------------------------------------------------------------------
    def compensate_embedding_grads(
        self, s: float, smax: float, thres_cosh: float = 50.0
    ) -> None:
        # if s <= 0:
        #     return
        for emb in self.embeddings:
            weight = emb.weight
            if weight.grad is None:
                continue
            num = torch.cosh(torch.clamp(s * weight.data, -thres_cosh, thres_cosh)) + 1
            den = torch.cosh(weight.data) + 1
            weight.grad.data *= smax / s * (num / den)

    def clamp_embeddings(self, thres_emb: float = 6.0) -> None:
        for emb in self.embeddings:
            emb.weight.data.clamp_(-thres_emb, thres_emb)

    # ------------------------------------------------------------------
    def get_view_for(
        self, name: str, masks: List[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        key = name
        if key.startswith("model."):
            key = key[len("model.") :]

        info = self.param_gate_map.get(key)
        if info is None:
            return None

        mask_dict = {
            spec["name"]: masks[idx].view(-1)
            for idx, spec in enumerate(self.gate_specs)
        }
        post = mask_dict[info["post"]]
        shape = self.param_shapes[key]
        device = post.device

        if info["type"] == "conv_weight":
            post_tensor = post.view(-1, 1, 1).expand(shape)
            if info["pre"] is None:
                pre_tensor = torch.ones(1, shape[1], 1, device=device).expand(shape)
            else:
                pre = mask_dict[info["pre"]]
                pre_tensor = pre.view(1, -1, 1).expand(shape)
            return torch.min(post_tensor, pre_tensor)

        if info["type"] == "linear_weight":
            if post.numel() == shape[0]:
                post_tensor = post.view(-1, 1).expand(shape)
            elif post.numel() == shape[1]:
                post_tensor = post.view(1, -1).expand(shape)
            else:
                post_tensor = torch.ones(shape, device=device)

            if info["pre"] is None:
                pre_tensor = torch.ones(shape, device=device)
            else:
                pre = mask_dict[info["pre"]]
                if pre.numel() == shape[1]:
                    pre_tensor = pre.view(1, -1).expand(shape)
                elif pre.numel() == shape[0]:
                    pre_tensor = pre.view(-1, 1).expand(shape)
                else:
                    pre_tensor = torch.ones(shape, device=device)
            return torch.min(post_tensor, pre_tensor)

        if info["type"] in {"bn_weight", "bn_bias", "bias"}:
            return post.view(-1)

        return post.view(-1)


class Net(nn.Module):
    """HAT learner compatible with the repository training harness."""

    def __init__(
        self, n_inputs: int, n_outputs: int, n_tasks: int, args: object
    ) -> None:
        super().__init__()

        self.cfg = HatConfig.from_args(args)
        self.n_tasks = n_tasks
        self.n_outputs = n_outputs
        self.classes_per_task = misc_utils.build_task_class_list(
            n_tasks,
            n_outputs,
            nc_per_task=getattr(args, "nc_per_task_list", "")
            or getattr(args, "nc_per_task", None),
            classes_per_task=getattr(args, "classes_per_task", None),
        )
        self.nc_per_task = misc_utils.max_task_class_count(self.classes_per_task)
        self.real_epoch = 0

        self.bridge = HatBackbone(n_inputs, n_tasks, n_outputs, self.cfg, args)

        self.class_weighted_ce = bool(getattr(args, "class_weighted_ce", True))
        self.incremental_loader_name = getattr(args, "loader", None)

        params: Iterable[nn.Parameter] = self.bridge.parameters()
        if self.cfg.optimizer.lower() == "adam":
            self.opt = torch.optim.Adam(params, lr=self.cfg.lr)
        else:
            self.opt = torch.optim.SGD(params, lr=self.cfg.lr, momentum=0.9)

        self.smax = float(self.cfg.smax)
        self.lamb = float(self.cfg.gamma)
        self.grad_clip = (
            float(self.cfg.grad_clip_norm) if self.cfg.grad_clip_norm > 0 else None
        )

        self.current_task: Optional[int] = None
        self.mask_pre: Optional[List[torch.Tensor]] = None
        self.mask_back: Dict[str, torch.Tensor] = {}

        self._epoch_counts: Dict[int, int] = defaultdict(int)
        self._epoch_sizes: Dict[int, Optional[int]] = defaultdict(lambda: None)
        self._last_epoch: Dict[int, int] = defaultdict(lambda: -1)

        self.samples_per_task_resolver = getattr(args, "get_samples_per_task", None)
        self.samples_per_task = int(getattr(args, "samples_per_task", -1))
        self.batch_size = int(getattr(args, "batch_size", 1))
        if self.samples_per_task > 0:
            self.num_batches = max(int(self.samples_per_task) // self.batch_size, 1)
        else:
            self.num_batches = None
        self.batch_idx = 0
        self._finalized_tasks: set[int] = set()
        # Per-task BatchNorm running statistics are handled centrally by
        # :mod:`model.task_bn`, installed from ``main`` once the model is built.
        # Unlike the in-memory snapshots this replaces, those buffers are part of
        # ``state_dict`` and so survive into the per-task checkpoints.

    # ------------------------------------------------------------------
    def _device(self) -> torch.device:
        return next(self.bridge.parameters()).device

    def _task_tensor(self, t: int, device: torch.device) -> torch.LongTensor:
        return torch.tensor([t], device=device, dtype=torch.long)

    def _update_epoch_counters(self, t: int) -> Tuple[int, int]:
        epoch = getattr(self, "real_epoch", 0)
        if self._last_epoch[t] != epoch:
            if self._last_epoch[t] >= 0:
                prev = self._epoch_counts[t]
                if prev > 0:
                    self._epoch_sizes[t] = prev
            self._epoch_counts[t] = 0
            self._last_epoch[t] = epoch

        batch_idx = self._epoch_counts[t]
        total = self._epoch_sizes[t]
        if total is None or total <= 1:
            total = max(batch_idx + 1, 1)
        return batch_idx, total

    def _schedule_s(self, batch_idx: int, total_batches: int) -> float:
        if self.smax <= 0:
            return 1.0
        progress = 0.0 if total_batches <= 1 else batch_idx / max(1, total_batches - 1)
        progress = float(max(0.0, min(1.0, progress)))
        if self.cfg.anneal_schedule == "geometric":
            # Original paper schedule: log-linear from 1/smax to smax.
            # s = smax^(2*progress - 1), passing through 1 at the midpoint.
            return self.smax ** (2.0 * progress - 1.0)
        # Default: linear interpolation from 1/smax to smax.
        base = 1.0 / self.smax
        return base + progress * (self.smax - base)

    def _total_batches_for_task(self, t: int) -> Optional[int]:
        if self.samples_per_task_resolver is None:
            if self.samples_per_task > 0:
                return max(int(math.ceil(self.samples_per_task / self.batch_size)), 1)
            return None
        samples = int(self.samples_per_task_resolver(t))
        if samples <= 0:
            return None
        return max(int(math.ceil(samples / self.batch_size)), 1)

    def _log_final_mask_stats(self, task: int, masks: List[torch.Tensor]) -> None:
        """Print compact end-of-task mask saturation diagnostics.

        Args:
            task: Completed task index.
            masks: HAT masks evaluated at ``smax`` for the completed task.
        """
        if not masks:
            return
        flat_masks = torch.cat([mask.detach().view(-1).cpu() for mask in masks])
        high_fraction = float((flat_masks > 0.95).float().mean().item())
        low_fraction = float((flat_masks < 0.05).float().mean().item())
        print(
            "HAT mask stats task {}: mean={:.4f} hi>0.95={:.4f} lo<0.05={:.4f}".format(
                task,
                float(flat_masks.mean().item()),
                high_fraction,
                low_fraction,
            )
        )

    def _finalise_task(self, t: Optional[int]) -> None:
        if t is None:
            return
        if t in self._finalized_tasks:
            return
        device = self._device()
        with torch.no_grad():
            masks = self.bridge.mask(self._task_tensor(t, device), self.smax)
            masks = [m.detach() for m in masks]
        self._log_final_mask_stats(t, masks)
        if self.mask_pre is None:
            self.mask_pre = masks
        else:
            self.mask_pre = [torch.max(mp, m) for mp, m in zip(self.mask_pre, masks)]

        self.mask_back.clear()
        if self.mask_pre is None:
            return
        for name, param in self.bridge.named_parameters():
            view = self.bridge.get_view_for(name, self.mask_pre)
            if view is not None:
                if view.shape != param.shape:
                    if view.t().shape == param.shape:
                        view = view.t()
                    elif view.numel() == param.numel():
                        view = view.reshape(param.shape)
                    else:
                        view = view.expand_as(param)
                self.mask_back[name] = (1 - view).to(param.device)
        self._finalized_tasks.add(t)

    def finalize_task_after_training(
        self,
        train_loader: object | None = None,
        *,
        completed_task_index: int | None = None,
    ) -> None:
        """Finalize HAT masks for the completed task.

        Args:
            train_loader: Unused hook argument accepted for compatibility with
                the repository training loop.
            completed_task_index: Completed task index. Defaults to the current
                task tracked by the HAT learner.
        """
        del train_loader
        task = (
            self.current_task if completed_task_index is None else completed_task_index
        )
        if task is None:
            raise RuntimeError("finalize_task_after_training requires a current task.")
        self._finalise_task(task)

    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, t: int, s: Optional[float] = None
    ) -> torch.Tensor:
        device = x.device if x.is_cuda else self._device()
        logits = self.bridge.forward(
            self._task_tensor(t, device), x, s or self.smax, return_masks=False
        )
        offset1, offset2 = misc_utils.compute_offsets(t, self.classes_per_task)
        masked = logits.clone()
        if offset1 > 0:
            masked[:, :offset1] = -1e9
        if offset2 < self.n_outputs:
            masked[:, offset2:] = -1e9
        return masked

    def log_gate_stats(self, task_id, epoch, batch, masks):
        lines = [f"[task={task_id} epoch={epoch} iter={batch}]"]
        for idx, mask in enumerate(masks):
            m = mask.detach().view(-1)
            hi = (m > 0.95).float().mean().item()
            lo = (m < 0.05).float().mean().item()
            lines.append(
                f"  layer {idx:02d} | mean={m.mean():.3f} std={m.std():.3f} hi={hi:.3f} lo={lo:.3f}"
            )
        print("\n".join(lines))

    def observe(
        self, x: torch.Tensor, y: torch.Tensor, t: int
    ) -> Tuple[float, float, torch.Tensor | None]:

        if self.current_task is None:
            self.current_task = t
        elif t != self.current_task:
            if self.current_task not in self._finalized_tasks:
                self._finalise_task(self.current_task)
            self.current_task = t

        device = x.device if x.is_cuda else self._device()
        batch_idx, total_batches = self._update_epoch_counters(t)
        self.batch_idx += 1
        resolved_total = self._total_batches_for_task(t)
        if resolved_total is None or resolved_total <= 0:
            effective_total = total_batches
        else:
            effective_total = resolved_total
        s = self._schedule_s(batch_idx, effective_total)
        # print(s, self.num_batches)

        self.bridge.set_bn_eval(False)

        metric_logits = None
        for _ in range(self.cfg.inner_steps):
            self.opt.zero_grad(set_to_none=True)
            logits, masks = self.bridge.forward(
                self._task_tensor(t, device), x, s, return_masks=True
            )
            # with torch.no_grad():
            #     self.log_gate_stats(t, self.real_epoch, batch_idx, masks)

            # for mask in masks:
            #     print(mask.mean(), mask.min())
            y_cls = unpack_y_to_class_labels(y)
            logits_for_loss = misc_utils.apply_task_incremental_logit_mask(
                logits,
                t,
                self.classes_per_task,
                self.n_outputs,
                cil_all_seen_upto_task=t,
                loader=self.incremental_loader_name,
            )
            targets = y_cls.long()
            loss, _ = self._criterion(logits_for_loss, targets, masks)
            preds = torch.argmax(logits_for_loss, dim=1)
            cls_tr_rec = macro_recall(preds, targets)
            loss.backward()

            if self.mask_back:
                for name, param in self.bridge.named_parameters():
                    if param.grad is None:
                        continue
                    if name in self.mask_back:
                        param.grad.data *= self.mask_back[name]

            self.bridge.compensate_embedding_grads(s, self.smax)

            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.bridge.parameters(), self.grad_clip)

            self.opt.step()
            self.bridge.clamp_embeddings()
            metric_logits = logits_for_loss.detach()

        self._epoch_counts[t] += 1
        return float(loss.detach().cpu()), cls_tr_rec, metric_logits

    def on_epoch_end(self) -> None:
        pass

    # ------------------------------------------------------------------
    def _criterion(
        self, outputs: torch.Tensor, targets: torch.Tensor, masks: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not masks:
            return (
                classification_cross_entropy(
                    outputs,
                    targets,
                    class_weighted_ce=self.class_weighted_ce,
                ),
                torch.tensor(0.0, device=outputs.device),
            )

        reg = torch.zeros(1, device=outputs.device)
        count = torch.zeros(1, device=outputs.device)
        if self.mask_pre is not None:
            for m, mp in zip(masks, self.mask_pre):
                aux = 1 - mp
                reg += (m * aux).sum()
                count += aux.sum()
        else:
            for m in masks:
                reg += m.sum()
                count += torch.tensor(float(m.numel()), device=outputs.device)
        reg = reg / torch.clamp(count, min=1.0)
        return (
            classification_cross_entropy(
                outputs,
                targets,
                class_weighted_ce=self.class_weighted_ce,
            )
            + self.lamb * reg,
            reg,
        )


__all__ = ["Net"]
