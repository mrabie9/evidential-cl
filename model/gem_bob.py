# GEM-BoB ("best of best"): GEM kept structurally intact (ResNet1D trunk, per-task episodic
# buffer, current-task CE, QP gradient-projection constraint) plus the four mechanisms the
# leave-one-out grid in docs/ablation_studies.tex found load-bearing, each independently gated:
#
#   1. --gembob_dynamic_ring   fully-utilised ring buffer          (E2, +2.1 F1 on Res-ER)
#   2. --gembob_distill        KL distillation on frozen soft targets (B1 -1.2, T2 -4.7)
#   3. --gembob_bilevel        inner/outer round + Reptile interp   (B5b -1.4 at matched budget)
#   4. --gembob_meta_batches   meta-batch averaging, K>1            (M3 -1.2)
#
# With every gate off this file is plain GEM: that identity is the correctness gate for the
# add-one study. Each mechanism is ported from the parent method that established it --
# (1) er_ring.py, (2) gem_distill.py, (3) bcl_dual.py, (4) lamaml_cifar.py -- so a null result
# here is evidence the mechanism does not transfer to GEM, not that it was reimplemented wrong.
#
# BUDGET. docs/ablation_studies.tex finds raw step count is itself one of the largest levers
# ("budget masquerades as mechanism": B5a vs B5b = 3.4 F1 of pure compute), so mechanisms 3 and 4
# are written to be step-matchable against the baseline. Bilevel takes 2 SGD steps per round, so
# it is budget-matched at --inner_steps 1 against a baseline at --inner_steps 2. Meta-batch
# averaging is budget-neutral by construction: it accumulates gradients over K chunks of the
# incoming batch and still takes exactly one projected step per round.

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import quadprog

from model.resnet1d import ResNet1D
from model.detection_replay import (
    DetectionReplayMixin,
    classification_loss_zero_stub,
    noise_label_from_args,
    signal_mask_exclude_noise,
    unpack_y_to_class_labels,
)
from utils.training_metrics import macro_recall
from utils import misc_utils
from utils.class_weighted_loss import classification_cross_entropy


@dataclass
class GemBobConfig:
    memory_strength: float = 0.0  # lambda in the paper (QP margin)
    gem_disable_qp: bool = False  # ablation: skip QP projection + replay-gradient pass
    # --- mechanism gates (all default off => plain GEM) ---
    gembob_dynamic_ring: bool = False
    gembob_distill: bool = False
    gembob_bilevel: bool = False
    gembob_meta_batches: int = 1  # 1 = no meta-batch averaging
    # --- mechanism hyperparameters (inert unless the matching gate is on) ---
    distill_lambda: float = 1.0
    temperature: float = 5.0
    beta: float = 1.0  # Reptile interpolation coefficient for the bilevel round
    gembob_val_memories: int = 512  # total held-out validation budget (bilevel outer step)
    inner_steps: int = 1
    lr: float = 1e-3
    n_memories: int = 0
    replay_batch_size: int = 128
    arch: str = "resnet1d"
    dataset: str = "tinyimagenet"
    cuda: bool = True
    alpha_init: float = 1e-3
    grad_clip_norm: Optional[float] = 100.0
    input_channels: int = 2
    det_lambda: float = 1.0
    cls_lambda: float = 1.0
    det_memories: int = 2000
    det_replay_batch: int = 64

    @staticmethod
    def from_args(args: object) -> "GemBobConfig":
        cfg = GemBobConfig()
        for field in cfg.__dataclass_fields__:
            # `None` means the argument was registered but never set (the parser
            # gives shared names like `beta` and `distill_lambda` a None default
            # so each model keeps its own), so the dataclass default stands.
            value = getattr(args, field, None)
            if value is not None:
                setattr(cfg, field, value)
        return cfg


# Auxiliary functions useful for GEM's inner optimization.


def compute_offsets(task, nc_per_task, is_cifar):
    """
    Compute offsets for cifar to determine which
    outputs to select for a given task.
    """
    return misc_utils.compute_offsets(task, nc_per_task)


def store_grad(pp, grads, grad_dims, tid):
    """
    This stores parameter gradients of past tasks.
    pp: parameters
    grads: gradients
    grad_dims: list with number of parameters per layers
    tid: task id
    """
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[: cnt + 1])
            grads[beg:en, tid].copy_(param.grad.data.view(-1))
        cnt += 1


def overwrite_grad(pp, newgrad, grad_dims):
    """
    This is used to overwrite the gradients with a new gradient
    vector, whenever violations occur.
    pp: parameters
    newgrad: corrected gradient
    grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[: cnt + 1])
            this_grad = newgrad[beg:en].contiguous().view(param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
    Solves the GEM dual QP described in the paper given a proposed
    gradient "gradient", and a memory of task gradients "memories".
    Overwrites "gradient" with the final projected update.
    input:  gradient, p-vector
    input:  memories, (t * p)-vector
    output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))


class Net(DetectionReplayMixin, nn.Module):
    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        super(Net, self).__init__()
        self.cfg = GemBobConfig.from_args(args)
        self.margin = self.cfg.memory_strength
        self.use_qp = not bool(self.cfg.gem_disable_qp)
        self.is_cifar = (self.cfg.dataset == "cifar100") or (
            self.cfg.dataset == "tinyimagenet"
        )

        # --- mechanism gates ---
        self.dynamic_ring = bool(self.cfg.gembob_dynamic_ring)
        self.use_distill = bool(self.cfg.gembob_distill)
        self.use_bilevel = bool(self.cfg.gembob_bilevel)
        self.meta_batches = max(1, int(self.cfg.gembob_meta_batches))
        # distill_lambda/temperature/beta carry parser-level defaults that are non-zero, so
        # they are only ever read through their gate -- see the module docstring.
        self.distill_lambda = float(self.cfg.distill_lambda) if self.use_distill else 0.0
        self.temp = float(self.cfg.temperature)
        self.beta = float(self.cfg.beta)
        self.kl = nn.KLDivLoss(reduction="batchmean")

        # --- IQ mode toggle ---
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
        self.n_tasks = n_tasks
        self.inner_steps = self.cfg.inner_steps
        self.replay_batch_size = int(self.cfg.replay_batch_size)
        self.det_lambda = float(self.cfg.det_lambda)
        self.cls_lambda = float(self.cfg.cls_lambda)
        self._init_det_replay(
            self.cfg.det_memories,
            self.cfg.det_replay_batch,
            enabled=bool(getattr(args, "use_detector_arch", False)),
        )

        self.opt = optim.SGD(self._ll_params(), self.cfg.lr, momentum=0.9)
        self.det_opt = optim.SGD(
            self.net.det_head.parameters(), self.cfg.lr, momentum=0.9
        )

        self.n_memories = int(self.cfg.n_memories)
        if self.dynamic_ring:
            # Dynamic ring: the budget is re-split across only the tasks seen so far, so a
            # single task (task 0) may transiently occupy the entire buffer. Storage must
            # therefore hold n_memories rows per task. Ported from er_ring.py.
            self.task_memory_capacities = self._dynamic_task_capacities(num_seen=1)
            self.max_task_memories = self.n_memories
        else:
            self.task_memory_capacities = self._build_task_memory_capacities(
                self.n_memories, n_tasks
            )
            self.max_task_memories = max(self.task_memory_capacities, default=0)
        self.gpu = self.cfg.cuda

        # --- Episodic memory allocation ---
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

        # track how many exemplars each task has actually written
        self.task_mem_filled = torch.zeros(n_tasks, dtype=torch.long)
        self.task_mem_ptr = torch.zeros(n_tasks, dtype=torch.long)
        if self.gpu:
            self.task_mem_filled = self.task_mem_filled.cuda()
            self.task_mem_ptr = self.task_mem_ptr.cuda()

        # --- Held-out validation buffer (bilevel outer step only) ---
        self.val_capacities = self._build_task_memory_capacities(
            int(self.cfg.gembob_val_memories) if self.use_bilevel else 0, n_tasks
        )
        self.max_val_memories = max(self.val_capacities, default=0)
        if self.use_bilevel and self.max_val_memories > 0:
            self.valx = torch.FloatTensor(
                n_tasks, self.max_val_memories, 2, self.seq_len
            ).fill_(0)
            self.valy = torch.LongTensor(n_tasks, self.max_val_memories).fill_(-1)
            self.task_val_filled = torch.zeros(n_tasks, dtype=torch.long)
            self.task_val_ptr = torch.zeros(n_tasks, dtype=torch.long)
            if self.gpu:
                self.valx = self.valx.cuda()
                self.valy = self.valy.cuda()
                self.task_val_filled = self.task_val_filled.cuda()
                self.task_val_ptr = self.task_val_ptr.cuda()

        # --- GEM gradient buffers ---
        self.grad_dims = [p.data.numel() for p in self._ll_params()]
        self.grads = torch.Tensor(sum(self.grad_dims), n_tasks)
        if self.gpu:
            self.grads = self.grads.cuda()

        # --- counters / bookkeeping ---
        self.observed_tasks = []
        self.old_task = -1
        self.classes_per_task = misc_utils.build_task_class_list(
            n_tasks,
            n_outputs,
            nc_per_task=getattr(args, "nc_per_task_list", "")
            or getattr(args, "nc_per_task", None),
            classes_per_task=getattr(args, "classes_per_task", None),
        )
        self.nc_per_task = misc_utils.max_task_class_count(self.classes_per_task)
        self.noise_label: int | None = noise_label_from_args(args)
        self.incremental_loader_name = getattr(args, "loader", None)

        # --- Distillation soft targets (frozen per task at its boundary) ---
        if self.use_distill:
            self.memory_feat = torch.zeros(
                n_tasks, self.max_task_memories, self.nc_per_task
            )
            if self.gpu:
                self.memory_feat = self.memory_feat.cuda()
        self.distill_ready = [False] * n_tasks

        if self.gpu:
            self.cuda()

    def _build_task_memory_capacities(
        self, total_memories: int, n_tasks: int
    ) -> list[int]:
        """Split a total memory budget across tasks.

        Args:
            total_memories: Total replay-buffer capacity requested by config.
            n_tasks: Number of tasks in the stream.

        Returns:
            A list of per-task capacities whose sum equals `total_memories`.
        """
        if n_tasks <= 0:
            return []
        base_capacity = total_memories // n_tasks
        remainder_capacity = total_memories % n_tasks
        return [
            base_capacity + (1 if task_index < remainder_capacity else 0)
            for task_index in range(n_tasks)
        ]

    def _dynamic_task_capacities(self, num_seen: int) -> list[int]:
        """Split the whole budget equally across the first ``num_seen`` tasks.

        Tasks not yet seen get capacity 0. The remainder from an uneven division is
        handed to the earliest seen tasks, mirroring ``_build_task_memory_capacities``
        so the final split (num_seen == n_tasks) is identical to the static allocation.

        Args:
            num_seen: Number of tasks encountered so far (>= 1).

        Returns:
            Per-task capacities of length ``n_tasks`` summing to ``n_memories``.
        """
        num_seen = max(1, min(int(num_seen), self.n_tasks))
        base_capacity = self.n_memories // num_seen
        remainder_capacity = self.n_memories % num_seen
        return [
            (
                (base_capacity + (1 if task_index < remainder_capacity else 0))
                if task_index < num_seen
                else 0
            )
            for task_index in range(self.n_tasks)
        ]

    def _reallocate_dynamic_ring(self, num_seen: int) -> None:
        """Re-split the buffer across ``num_seen`` tasks and shrink prior tasks.

        Called at each task boundary. Every already-seen task whose stored count now
        exceeds its reduced capacity is truncated to the first ``capacity`` slots (the
        ring's current occupancy), freeing room for the new task. Truncation keeps slot
        indices stable, so the frozen distillation soft targets in ``memory_feat`` stay
        aligned with their samples and need no recompute.
        """
        self.task_memory_capacities = self._dynamic_task_capacities(num_seen)
        for task_index in range(min(num_seen, self.n_tasks)):
            capacity = self.task_memory_capacities[task_index]
            filled = int(self.task_mem_filled[task_index].item())
            if filled > capacity:
                self.task_mem_filled[task_index] = capacity
                # Occupancy is now exactly `capacity` (full); wrap the write pointer so
                # any further writes overwrite from the start.
                self.task_mem_ptr[task_index] = 0

    def _ensure_iq_shape(self, x):
        """
        Ensure x is (B, 2, L) for IQ mode.
        Accepts (B, 2, L) or (B, 2L).
        """
        if x.dim() == 3:
            return x
        elif x.dim() == 2:
            B, F = x.shape
            assert F % 2 == 0, f"Feature dim {F} not divisible by 2 for (2, L) reshape."
            L = F // 2
            return x.view(B, 2, L)
        else:
            raise ValueError(
                f"Unexpected IQ input shape {tuple(x.shape)}; expected (B, 2, L) or (B, 2L)."
            )

    def _adapt_for_memory(self, x: torch.Tensor) -> torch.Tensor:
        """Convert IQ inputs to the canonical replay-memory shape.

        Args:
            x: Input batch in one of the supported IQ layouts.

        Returns:
            Tensor with shape ``(B, 2, L_memory)`` where ``L_memory`` matches the model
            replay buffer sequence length.
        """
        adapted_x = x
        if adapted_x.dim() == 4 and adapted_x.size(1) == 3 and adapted_x.size(2) == 2:
            adapted_x = self.net.model.input_adapter(adapted_x)
        elif adapted_x.dim() == 3 and adapted_x.size(1) == 3:
            if adapted_x.size(2) % 2 != 0:
                raise ValueError(
                    f"Expected even length for 3-ADC IQ input; got shape {tuple(adapted_x.shape)}."
                )
            sequence_length = adapted_x.size(2) // 2
            adapted_x = adapted_x.view(adapted_x.size(0), 3, 2, sequence_length)
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

    def _ll_params(self):
        for name, param in self.net.named_parameters():
            if name.startswith("det_head"):
                continue
            yield param

    def forward(self, x, t, *, cil_all_seen_upto_task=None):
        if self.cfg.dataset == "tinyimagenet":
            x = x.view(-1, 3, 64, 64)
        elif self.cfg.dataset == "cifar100":
            x = x.view(-1, 3, 32, 32)
        elif self.is_iq:
            x = self._ensure_iq_shape(x)  # (B, 2, L)

        output = self.netforward(x)

        # TIL slice masking, or CIL eval: keep all logits for tasks seen so far.
        output = misc_utils.apply_task_incremental_logit_mask(
            output,
            t,
            self.classes_per_task,
            self.n_outputs,
            cil_all_seen_upto_task=cil_all_seen_upto_task,
            global_noise_label=self.noise_label,
            fill_value=-10e10,
            loader=self.incremental_loader_name,
        )
        return output

    # ------------------------------------------------------------------
    # Distillation (mechanism 2, ported from gem_distill.py)
    # ------------------------------------------------------------------

    def _store_soft_targets(self, task):
        """Freeze per-task soft targets on `task`'s buffer (teacher = current model)."""
        if self.distill_lambda <= 0:
            return
        filled = int(self.task_mem_filled[task].item())
        if filled == 0:
            return
        offset1, offset2 = compute_offsets(task, self.classes_per_task, self.is_cifar)
        cls_size = int(offset2 - offset1)
        with torch.no_grad():
            logits = self.forward(self.memory_data[task, :filled], task)[
                :, offset1:offset2
            ]
            soft = torch.softmax(logits / self.temp, dim=1)
        self.memory_feat[task, :filled].zero_()
        self.memory_feat[task, :filled, :cls_size].copy_(soft)
        self.distill_ready[task] = True

    def _distillation_loss(self):
        """Mean KL distillation against frozen soft targets over past-task buffers."""
        if len(self.observed_tasks) <= 1:
            return None
        kl_terms = []
        for past_task in self.observed_tasks[:-1]:
            if not self.distill_ready[past_task]:
                continue
            filled = int(self.task_mem_filled[past_task].item())
            if filled == 0:
                continue
            offset1, offset2 = compute_offsets(
                past_task, self.classes_per_task, self.is_cifar
            )
            cls_size = int(offset2 - offset1)
            logits = self.forward(self.memory_data[past_task, :filled], past_task)[
                :, offset1:offset2
            ]
            soft = self.memory_feat[past_task, :filled, :cls_size]
            kl_terms.append(self.kl(torch.log_softmax(logits / self.temp, dim=1), soft))
        if not kl_terms:
            return None
        return torch.stack(kl_terms).mean()

    # ------------------------------------------------------------------
    # Bilevel outer step (mechanism 3, ported from bcl_dual.py)
    # ------------------------------------------------------------------

    def _store_validation(self, task_id: int, x_row: torch.Tensor, y_row: torch.Tensor):
        """Write one held-out row into ``task_id``'s validation ring buffer."""
        capacity = int(self.val_capacities[task_id])
        if capacity <= 0:
            return
        write_pointer = int(self.task_val_ptr[task_id].item())
        self.valx[task_id, write_pointer].copy_(x_row)
        self.valy[task_id, write_pointer] = y_row
        self.task_val_filled[task_id] = min(
            capacity, int(self.task_val_filled[task_id].item()) + 1
        )
        self.task_val_ptr[task_id] = (
            0 if (write_pointer + 1) == capacity else (write_pointer + 1)
        )

    def _sample_validation_rows(self, upto_task: int):
        """Sample stored validation rows across all tasks seen so far.

        Returns:
            ``(x, y_global, task_index)`` or None when nothing is stored yet. Rows are
            drawn uniformly over the occupied (task, slot) pairs; the enumeration is
            vectorized on-device so it costs one kernel rather than one GPU sync per
            occupied slot.
        """
        if self.max_val_memories == 0:
            return None
        device = self.valx.device
        filled = self.task_val_filled[: upto_task + 1]
        if int(filled.sum().item()) == 0:
            return None
        slots = torch.arange(self.max_val_memories, device=device)
        occupied = slots.unsqueeze(0) < filled.unsqueeze(1)  # (tasks, max_val)
        pairs = occupied.nonzero(as_tuple=False)  # (N, 2) -> [task, slot]
        sample_size = min(pairs.size(0), self.replay_batch_size)
        chosen = torch.randperm(pairs.size(0), device=device)[:sample_size]
        t_idx = pairs[chosen, 0]
        s_idx = pairs[chosen, 1]
        return self.valx[t_idx, s_idx], self.valy[t_idx, s_idx], t_idx

    def _masked_global_loss(self, xx, yy_global, t_idx):
        """CE on mixed-task rows, each row masked to its own task's logits.

        Mirrors gem.py/er_ring: raw logits + per-sample-task TIL/CIL mask with GLOBAL
        labels, so a mixed-task batch is scored exactly as it would be at eval time.
        """
        raw = self.netforward(self._ensure_iq_shape(xx) if self.is_iq else xx)
        masked = raw.clone()
        for task_id in torch.unique(t_idx).tolist():
            rows = t_idx == int(task_id)
            masked[rows] = misc_utils.apply_task_incremental_logit_mask(
                raw[rows],
                int(task_id),
                self.classes_per_task,
                self.n_outputs,
                cil_all_seen_upto_task=int(task_id),
                global_noise_label=self.noise_label,
                fill_value=-10e10,
                loader=self.incremental_loader_name,
            )
        return classification_cross_entropy(
            masked, yy_global, class_weighted_ce=self.class_weighted_ce
        )

    # ------------------------------------------------------------------

    def _store_replay(self, task_id: int, x_data: torch.Tensor, y_data: torch.Tensor):
        """Write the current batch into ``task_id``'s episodic buffer (FIFO ring)."""
        capacity = int(self.task_memory_capacities[task_id])
        if capacity <= 0:
            return
        # Store adapter output (e.g. 3-ADC -> 2-channel) so replay matches forward.
        mem_x = self._input_for_replay(x_data)
        write_pointer = int(self.task_mem_ptr[task_id].item())
        endcnt = min(write_pointer + mem_x.size(0), capacity)
        effbsz = endcnt - write_pointer
        if effbsz > 0:
            self.memory_data[task_id, write_pointer:endcnt].copy_(mem_x[:effbsz])
            self.memory_labs[task_id, write_pointer:endcnt].copy_(y_data[:effbsz])
            filled_before_update = int(self.task_mem_filled[task_id].item())
            self.task_mem_filled[task_id] = min(capacity, filled_before_update + effbsz)
        self.task_mem_ptr[task_id] = 0 if endcnt == capacity else endcnt

    def _past_task_gradients(self, t):
        """Populate ``self.grads`` with one gradient per past task (GEM's constraints)."""
        for tt in range(len(self.observed_tasks) - 1):
            self.zero_grad()
            past_task = self.observed_tasks[tt]
            offset1, offset2 = compute_offsets(
                past_task, self.classes_per_task, self.is_cifar
            )
            filled = int(self.task_mem_filled[past_task].item())
            if filled == 0:
                continue  # nothing stored for this task yet

            mem_x = Variable(self.memory_data[past_task, :filled])
            mem_y_flat = self.memory_labs[past_task, :filled]
            replay_mask = signal_mask_exclude_noise(mem_y_flat, self.noise_label)
            if replay_mask.any():
                mem_x_sub = mem_x[replay_mask]
                logits_replay = self.forward(mem_x_sub, past_task)[:, offset1:offset2]
                targets_replay = mem_y_flat[replay_mask] - offset1
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

    def _project_and_step(self, t):
        """Apply GEM's QP projection to the accumulated gradient, then step."""
        if self.cfg.grad_clip_norm:
            torch.nn.utils.clip_grad_norm_(
                self.net.parameters(), self.cfg.grad_clip_norm
            )
        if self.use_qp and len(self.observed_tasks) > 1:
            store_grad(self._ll_params, self.grads, self.grad_dims, t)
            device = torch.device("cuda") if self.gpu else torch.device("cpu")
            indx = torch.tensor(
                self.observed_tasks[:-1], dtype=torch.long, device=device
            )
            dotp = torch.mm(
                self.grads[:, t].unsqueeze(0), self.grads.index_select(1, indx)
            )
            if (dotp < 0).sum() != 0:
                project2cone2(
                    self.grads[:, t].unsqueeze(1),
                    self.grads.index_select(1, indx),
                    self.margin,
                )
                overwrite_grad(self._ll_params, self.grads[:, t], self.grad_dims)
        self.opt.step()

    def observe(self, x, y, t):
        """
        One optimization step on batch (x,y,t), with GEM constraints and inner_steps.
        """
        # --- shape handling ---
        if self.is_iq:
            x = self._ensure_iq_shape(x)  # keep (B, 2, L)
        else:
            x = x.view(x.size(0), -1)  # legacy: flatten non-IQ inputs
        y_work = unpack_y_to_class_labels(y)

        # track tasks
        if t != self.old_task:
            # Freeze soft targets for the task that just finished (teacher = current model).
            if self.use_distill and self.old_task >= 0:
                self._store_soft_targets(self.old_task)
            if t not in self.observed_tasks:
                self.observed_tasks.append(t)
            # Re-split the buffer across the tasks seen so far. Ordered AFTER the soft
            # targets are frozen so distillation sees the pre-shrink occupancy, and the
            # rows that survive truncation keep targets that were computed for them.
            if self.dynamic_ring:
                self._reallocate_dynamic_ring(num_seen=len(self.observed_tasks))
            self.old_task = t

        # Bilevel: hold out the first row of the batch for the outer step. Removing it
        # (rather than copying) keeps the validation buffer disjoint from the training
        # stream, as in bcl_dual.
        held_out_row = False
        if self.use_bilevel and self.max_val_memories > 0 and x.size(0) > 1:
            x_for_val = self._input_for_replay(x[:1].data)
            self._store_validation(t, x_for_val[0], y_work[0].data)
            x, y_work = x[1:], y_work[1:]
            held_out_row = True

        cls_tr_rec = []
        metric_logits = None
        loss = None

        for pass_itr in range(self.inner_steps):
            # push current batch once per batch (not each glance)
            if pass_itr == 0:
                self._store_replay(t, x.data, y_work.data)

            # gradients on past tasks (GEM's inequality constraints)
            if self.use_qp and len(self.observed_tasks) > 1:
                self._past_task_gradients(t)

            # --- inner (training) step ---
            self.zero_grad()
            targets = y_work.long()

            # Meta-batch averaging: accumulate CE gradients over K chunks of the incoming
            # batch and still take ONE step, so the mechanism costs no extra budget. K=1
            # is exactly the single full-batch pass (baseline).
            chunk_size = max(1, (x.size(0) + self.meta_batches - 1) // self.meta_batches)
            bounds = [
                (lo, min(lo + chunk_size, x.size(0)))
                for lo in range(0, x.size(0), chunk_size)
            ]
            # Divide by the chunks that actually contribute, not by meta_batches: a batch
            # smaller than K yields fewer chunks, and counting the empty ones would silently
            # shrink the gradient. Each chunk's MEAN loss gets equal weight (an average of
            # per-chunk means, as in C-MAML's meta-batching), which differs slightly from a
            # single batch mean when the final chunk is short.
            n_chunks = len(bounds)
            logits_chunks = []
            loss_value = 0.0
            for lo, hi in bounds:
                logits_chunk = self.forward(x[lo:hi], t, cil_all_seen_upto_task=t)
                chunk_loss = classification_cross_entropy(
                    logits_chunk,
                    targets[lo:hi],
                    class_weighted_ce=self.class_weighted_ce,
                )
                (chunk_loss / n_chunks).backward()
                loss_value += float(chunk_loss.item()) / n_chunks
                logits_chunks.append(logits_chunk.detach())

            # Distillation replay: pull the model toward frozen per-task soft targets.
            # Computed once per round (not once per chunk) -- the term does not depend on
            # the incoming batch, so chunking it would only multiply its cost.
            if self.distill_lambda > 0:
                distill_loss = self._distillation_loss()
                if distill_loss is not None:
                    scaled = self.distill_lambda * distill_loss
                    scaled.backward()
                    loss_value += float(scaled.item())

            logits_full = torch.cat(logits_chunks, dim=0) if logits_chunks else None
            if logits_full is not None:
                signal_mask = signal_mask_exclude_noise(y_work, self.noise_label)
                if signal_mask.any():
                    preds = torch.argmax(logits_full[signal_mask], dim=1)
                    cls_tr_rec.append(macro_recall(preds, targets[signal_mask]))
                else:
                    cls_tr_rec.append(0.0)
                metric_logits = logits_full
            loss = loss_value

            # At beta == 1 the Reptile interpolation is an identity copy of the round (the
            # tuned value in the write-up), so the snapshot is only taken when it can matter.
            weights_before = None
            if self.use_bilevel and self.beta != 1.0:
                weights_before = {
                    name: param.detach().clone()
                    for name, param in self.net.state_dict().items()
                }
            self._project_and_step(t)

            # --- outer (validation) step + Reptile interpolation ---
            # The outer gradient is NOT QP-projected: GEM's constraints are defined on the
            # training objective, and BCL-Dual's outer step is likewise unconstrained, so
            # projecting here would confound the two mechanisms.
            if self.use_bilevel:
                self.zero_grad()
                sampled = self._sample_validation_rows(t)
                if sampled is not None:
                    xval, yval, val_t_idx = sampled
                    outer_loss = self._masked_global_loss(xval, yval, val_t_idx)
                    outer_loss.backward()
                    if self.cfg.grad_clip_norm:
                        torch.nn.utils.clip_grad_norm_(
                            self.net.parameters(), self.cfg.grad_clip_norm
                        )
                    self.opt.step()
                self.zero_grad()
                if self.beta != 1.0:
                    weights_after = self.net.state_dict()
                    self.net.load_state_dict(
                        {
                            name: weights_before[name]
                            + (weights_after[name] - weights_before[name]) * self.beta
                            for name in weights_before
                        }
                    )

        avg_cls_tr_rec = sum(cls_tr_rec) / len(cls_tr_rec) if cls_tr_rec else 0.0
        if held_out_row:
            # The bilevel arm trains on one row fewer than the caller's batch, so these
            # logits no longer line up with main.py's y_cls (it would index a 256-row mask
            # into 255 rows). Returning None makes main.py recompute the progress-bar
            # metrics with its own full-batch forward, as bcl_dual does for the same reason.
            # Final evaluation and the reported Final F1 are unaffected either way.
            metric_logits = None
        return (loss if loss is not None else 0.0), avg_cls_tr_rec, metric_logits
