"""GEM-R: Gradient Episodic Memory with active replay descent and EMA eval.

GEM-R extends GEM (Lopez-Paz & Ranzato, 2017) with two additions aimed at
the one-shot (single-epoch) regime:

1. Active replay descent: the per-task memory gradients that GEM computes
   for its inequality constraints are additionally mixed into the update
   direction (``g = g_current + lambda * mean(g_past)``), so past-task loss
   is actively decreased instead of merely being prevented from increasing.
   The usual GEM projection is still applied to the mixed gradient.
2. EMA evaluation: an exponential moving average of the network weights is
   maintained during training and swapped in at evaluation time, damping
   the parameter noise of single-epoch streaming updates.

With ``memory_loss_lambda == 0`` and ``ema_decay == 0`` the algorithm is
exactly GEM.
"""

from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from model.gem import overwrite_grad, project2cone2, store_grad
from model.replay_utils import (
    ReplayInputMixin,
    classification_loss_zero_stub,
    unpack_y_to_class_labels,
)
from model.resnet1d import ResNet1D
from utils import misc_utils
from utils.class_weighted_loss import classification_cross_entropy
from utils.training_metrics import macro_recall


@dataclass
class GemRConfig:
    """Hyperparameters for GEM-R, populated from the global args namespace."""

    gamma: float = (
        0.0  # margin added to the dual QP constraint (gamma in the GEM paper)
    )
    memory_loss_lambda: float = 1.0
    ema_decay: float = 0.0
    inner_steps: int = 1
    lr: float = 1e-3
    n_memories: int = 0
    arch: str = "resnet1d"
    dataset: str = "iq"
    cuda: bool = True
    alpha_init: float = 1e-3
    grad_clip_norm: Optional[float] = 0.0
    input_channels: int = 2

    @staticmethod
    def from_args(args: object) -> "GemRConfig":
        """Build a config by copying matching attributes from ``args``.

        Args:
            args: Parsed experiment arguments namespace.

        Returns:
            A populated :class:`GemRConfig` instance.
        """
        cfg = GemRConfig()
        for field in cfg.__dataclass_fields__:
            value = getattr(args, field, None)
            # `None` means the argument was registered but never set (the parser
            # gives shared names like `beta` and `gamma` a None default so each
            # model keeps its own), so the dataclass default stands.
            if value is not None:
                setattr(cfg, field, value)
        return cfg


class Net(ReplayInputMixin, nn.Module):
    """GEM-R continual learner over the shared ResNet1D backbone."""

    def __init__(self, n_inputs: int, n_outputs: int, n_tasks: int, args: object):
        super(Net, self).__init__()
        self.cfg = GemRConfig.from_args(args)
        self.margin = self.cfg.gamma
        self.replay_lambda = float(self.cfg.memory_loss_lambda)
        self.ema_decay = float(self.cfg.ema_decay)

        self.input_channels = self.cfg.input_channels
        self.is_iq = (self.cfg.dataset == "iq") or (self.input_channels == 2)

        if self.cfg.arch != "resnet1d":
            raise ValueError(
                f"Unsupported arch {self.cfg.arch}; only resnet1d is available now."
            )
        self.net = ResNet1D(n_outputs, args)
        self.net.define_task_lr_params(alpha_init=self.cfg.alpha_init)
        self.netforward = self.net.forward
        self.class_weighted_ce = bool(getattr(args, "class_weighted_ce", True))
        self.n_outputs = n_outputs
        self.inner_steps = self.cfg.inner_steps

        self.opt = optim.SGD(self._ll_params(), self.cfg.lr, momentum=0.9)

        self.n_memories = int(self.cfg.n_memories)
        self.task_memory_capacities = self._build_task_memory_capacities(
            self.n_memories,
            n_tasks,
        )
        self.max_task_memories = max(self.task_memory_capacities, default=0)
        self.gpu = self.cfg.cuda

        if self.is_iq:
            assert n_inputs % 2 == 0, f"n_inputs={n_inputs} must be 2*L for IQ."
            self.seq_len = n_inputs // 2
            self.memory_data = torch.FloatTensor(
                n_tasks, self.max_task_memories, 2, self.seq_len
            )
        else:
            self.memory_data = torch.FloatTensor(
                n_tasks, self.max_task_memories, n_inputs
            )

        self.memory_labs = torch.LongTensor(n_tasks, self.max_task_memories)
        if self.gpu:
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()

        self.task_mem_filled = torch.zeros(n_tasks, dtype=torch.long)
        self.task_mem_ptr = torch.zeros(n_tasks, dtype=torch.long)
        if self.gpu:
            self.task_mem_filled = self.task_mem_filled.cuda()
            self.task_mem_ptr = self.task_mem_ptr.cuda()

        self.grad_dims = [p.data.numel() for p in self._ll_params()]
        self.grads = torch.Tensor(sum(self.grad_dims), n_tasks)
        if self.gpu:
            self.grads = self.grads.cuda()

        self.observed_tasks: list[int] = []
        self.old_task = -1
        self.classes_per_task = misc_utils.build_task_class_list(
            n_tasks,
            n_outputs,
            nc_per_task=getattr(args, "nc_per_task_list", "")
            or getattr(args, "nc_per_task", None),
            classes_per_task=getattr(args, "classes_per_task", None),
        )
        self.nc_per_task = misc_utils.max_task_class_count(self.classes_per_task)
        self.incremental_loader_name = getattr(args, "loader", None)

        if self.gpu:
            self.cuda()

        self.ema_enabled = self.ema_decay > 0.0
        self._ema_active = False
        self._ema_state: Dict[str, torch.Tensor] = {}
        self._live_state: Dict[str, torch.Tensor] = {}
        if self.ema_enabled:
            self._ema_state = {
                name: tensor.detach().clone()
                for name, tensor in self._named_net_tensors()
            }

    def _named_net_tensors(self) -> Iterator[Tuple[str, torch.Tensor]]:
        """Yield all named parameters and buffers of the backbone network."""
        yield from self.net.named_parameters()
        yield from self.net.named_buffers()

    def _update_ema(self) -> None:
        """Blend current network tensors into the EMA shadow copy."""
        with torch.no_grad():
            for name, tensor in self._named_net_tensors():
                shadow = self._ema_state[name]
                if tensor.dtype.is_floating_point:
                    shadow.mul_(self.ema_decay).add_(
                        tensor.detach(), alpha=1.0 - self.ema_decay
                    )
                else:
                    shadow.copy_(tensor)

    def _swap_in_ema_weights(self) -> None:
        """Back up live weights and load the EMA shadow into the network."""
        with torch.no_grad():
            self._live_state = {
                name: tensor.detach().clone()
                for name, tensor in self._named_net_tensors()
            }
            for name, tensor in self._named_net_tensors():
                tensor.copy_(self._ema_state[name])
        self._ema_active = True

    def _swap_out_ema_weights(self) -> None:
        """Restore live training weights after an EMA evaluation pass."""
        with torch.no_grad():
            for name, tensor in self._named_net_tensors():
                tensor.copy_(self._live_state[name])
        self._live_state = {}
        self._ema_active = False

    def train(self, mode: bool = True) -> "Net":
        """Switch train/eval mode, swapping EMA weights in for evaluation."""
        if self.ema_enabled:
            if mode and self._ema_active:
                self._swap_out_ema_weights()
            elif not mode and not self._ema_active:
                self._swap_in_ema_weights()
        return super().train(mode)

    def _build_task_memory_capacities(
        self, total_memories: int, n_tasks: int
    ) -> list[int]:
        """Split a total memory budget across tasks.

        Args:
            total_memories: Total replay-buffer capacity requested by config.
            n_tasks: Number of tasks in the stream.

        Returns:
            A list of per-task capacities whose sum equals ``total_memories``.
        """
        if n_tasks <= 0:
            return []
        base_capacity = total_memories // n_tasks
        remainder_capacity = total_memories % n_tasks
        return [
            base_capacity + (1 if task_index < remainder_capacity else 0)
            for task_index in range(n_tasks)
        ]

    def _ensure_iq_shape(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure ``x`` is shaped ``(B, 2, L)`` for IQ mode."""
        if x.dim() == 4 and x.size(1) == 3 and x.size(2) == 2:
            # 3-ADC layout; ResNet1D._prepare_input passes it through and the
            # ADC adapter reduces it to 2 channels.
            return x
        if x.dim() == 3:
            return x
        if x.dim() == 2:
            batch_size, feature_dim = x.shape
            assert (
                feature_dim % 2 == 0
            ), f"Feature dim {feature_dim} not divisible by 2 for (2, L) reshape."
            return misc_utils.deinterleave_iq_last_axis(x)
        raise ValueError(
            f"Unexpected IQ input shape {tuple(x.shape)}; expected (B, 2, L) or (B, 2L)."
        )

    def _adapt_for_memory(self, x: torch.Tensor) -> torch.Tensor:
        """Convert IQ inputs to the canonical replay-memory shape.

        Args:
            x: Input batch in one of the supported IQ layouts.

        Returns:
            Tensor with shape ``(B, 2, L_memory)`` where ``L_memory`` matches
            the model replay buffer sequence length.
        """
        adapted_x = x
        if adapted_x.dim() == 4 and adapted_x.size(1) == 3 and adapted_x.size(2) == 2:
            adapted_x = self.net.model.input_adapter(adapted_x)
        elif adapted_x.dim() == 3 and adapted_x.size(1) == 3:
            if adapted_x.size(2) % 2 != 0:
                raise ValueError(
                    f"Expected even length for 3-ADC IQ input; got shape {tuple(adapted_x.shape)}."
                )
            adapted_x = misc_utils.deinterleave_iq_last_axis(adapted_x)
            adapted_x = self.net.model.input_adapter(adapted_x)
        else:
            adapted_x = self._ensure_iq_shape(adapted_x)

        if adapted_x.size(1) != 2:
            raise ValueError(
                f"Replay memory expects 2 IQ channels; got shape {tuple(adapted_x.shape)}."
            )
        if adapted_x.size(2) > self.seq_len:
            adapted_x = adapted_x[:, :, : self.seq_len]
        elif adapted_x.size(2) < self.seq_len:
            pad_amount = self.seq_len - adapted_x.size(2)
            adapted_x = torch.nn.functional.pad(adapted_x, (0, pad_amount))
        return adapted_x

    def _ll_params(self) -> Iterator[torch.Tensor]:
        """Yield lifelong-learning parameters (backbone without det head)."""
        for name, param in self.net.named_parameters():
            yield param

    def forward(
        self, x: torch.Tensor, t: int, *, cil_all_seen_upto_task: int | None = None
    ) -> torch.Tensor:
        """Forward pass with task-incremental logit masking."""
        if self.is_iq:
            x = self._ensure_iq_shape(x)

        output = self.netforward(x)
        output = misc_utils.apply_task_incremental_logit_mask(
            output,
            t,
            self.classes_per_task,
            self.n_outputs,
            cil_all_seen_upto_task=cil_all_seen_upto_task,
            fill_value=-10e10,
            loader=self.incremental_loader_name,
        )
        return output

    def _store_current_batch(self, x: torch.Tensor, y_work: torch.Tensor, t: int):
        """Write the current batch into task ``t``'s ring-buffer memory."""
        task_capacity = int(self.task_memory_capacities[t])
        if task_capacity <= 0:
            return
        write_pointer = int(self.task_mem_ptr[t].item())
        bsz = y_work.size(0)
        endcnt = min(write_pointer + bsz, task_capacity)
        effbsz = endcnt - write_pointer
        mem_x = self._input_for_replay(x.data[:effbsz])
        self.memory_data[t, write_pointer:endcnt].copy_(mem_x)

        if bsz == 1:
            self.memory_labs[t, write_pointer] = y_work.data[0]
        else:
            self.memory_labs[t, write_pointer:endcnt].copy_(y_work.data[:effbsz])

        if effbsz > 0:
            filled_before_update = int(self.task_mem_filled[t].item())
            self.task_mem_filled[t] = min(task_capacity, filled_before_update + effbsz)
        self.task_mem_ptr[t] = 0 if endcnt == task_capacity else endcnt

    def _compute_past_task_grads(self, current_task: int) -> None:
        """Compute and store replay gradients for every previously seen task.

        Under CIL the memory loss covers every class of tasks
        ``0..current_task`` with global labels; under TIL it is the past task's
        own class block.
        """
        cil = self.incremental_loader_name == "class_incremental_loader"
        for tt in range(len(self.observed_tasks) - 1):
            self.zero_grad()
            past_task = self.observed_tasks[tt]
            offset1, offset2 = misc_utils.compute_offsets(
                past_task, self.classes_per_task
            )
            filled = int(self.task_mem_filled[past_task].item())
            if filled == 0:
                continue

            mem_x = self.memory_data[past_task, :filled]
            mem_y_flat = self.memory_labs[past_task, :filled]
            if filled > 0 and cil:
                logits_replay = self.forward(
                    mem_x, current_task, cil_all_seen_upto_task=current_task
                )
                ptloss = classification_cross_entropy(
                    logits_replay,
                    mem_y_flat,
                    class_weighted_ce=self.class_weighted_ce,
                )
            elif filled > 0:
                logits_replay = self.forward(mem_x, past_task)[:, offset1:offset2]
                targets_replay = mem_y_flat - offset1
                ptloss = classification_cross_entropy(
                    logits_replay,
                    targets_replay,
                    class_weighted_ce=self.class_weighted_ce,
                )
            else:
                logits_replay = self.forward(mem_x[:1], past_task)[:, offset1:offset2]
                ptloss = classification_loss_zero_stub(logits_replay)
            ptloss.backward()
            if self.cfg.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), self.cfg.grad_clip_norm
                )
            store_grad(self._ll_params, self.grads, self.grad_dims, past_task)

    def _mix_and_project_grads(self, t: int) -> None:
        """Mix replay descent into the current gradient and GEM-project it.

        The mixed update is ``g = g_t + lambda * mean(g_past)``. If the mixed
        gradient still conflicts with any past-task gradient it is projected
        onto the GEM feasibility cone. The (possibly projected) mixed
        gradient overwrites the parameter gradients in-place.

        Args:
            t: Current task index.
        """
        store_grad(self._ll_params, self.grads, self.grad_dims, t)
        device = torch.device("cuda") if self.gpu else torch.device("cpu")
        indx = torch.tensor(self.observed_tasks[:-1], dtype=torch.long, device=device)
        past_grads = self.grads.index_select(1, indx)
        mixed_grad = self.grads[:, t] + self.replay_lambda * past_grads.mean(dim=1)
        dotp = torch.mm(mixed_grad.unsqueeze(0), past_grads)
        if (dotp < 0).sum() != 0:
            mixed_grad = mixed_grad.contiguous().clone()
            project2cone2(mixed_grad.unsqueeze(1), past_grads, self.margin)
        overwrite_grad(self._ll_params, mixed_grad, self.grad_dims)

    def observe(
        self, x: torch.Tensor, y: torch.Tensor, t: int
    ) -> Tuple[float, float, torch.Tensor | None]:
        """One optimization step on batch ``(x, y, t)`` with GEM-R updates.

        Args:
            x: Input batch.
            y: Labels (possibly packed with detection targets).
            t: Task index of the incoming batch.

        Returns:
            Tuple of (loss value, mean train macro recall, detached logits).
        """
        if self.is_iq:
            x = self._ensure_iq_shape(x)
        else:
            x = x.view(x.size(0), -1)
        y_work = unpack_y_to_class_labels(y)

        if t != self.old_task:
            if t not in self.observed_tasks:
                self.observed_tasks.append(t)
            self.old_task = t

        cls_tr_rec = []
        metric_logits = None

        for pass_itr in range(self.inner_steps):
            if pass_itr == 0:
                self._store_current_batch(x, y_work, t)

            if len(self.observed_tasks) > 1:
                self._compute_past_task_grads(t)

            self.zero_grad()
            logits_full = self.forward(x, t, cil_all_seen_upto_task=t)
            targets = y_work.long()
            preds = torch.argmax(logits_full, dim=1)
            cls_tr_rec.append(macro_recall(preds, targets))
            loss = classification_cross_entropy(
                logits_full, targets, class_weighted_ce=self.class_weighted_ce
            )
            loss.backward()
            if self.cfg.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), self.cfg.grad_clip_norm
                )

            if len(self.observed_tasks) > 1:
                self._mix_and_project_grads(t)

            self.opt.step()
            if self.ema_enabled:
                self._update_ema()
            metric_logits = logits_full.detach()

        avg_cls_tr_rec = sum(cls_tr_rec) / len(cls_tr_rec) if cls_tr_rec else 0.0
        return loss.item(), avg_cls_tr_rec, metric_logits
