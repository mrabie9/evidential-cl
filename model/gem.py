# TODO: update mmemory buffer to store IQ values together

### This is a copy of GEM from https://github.com/facebookresearch/GradientEpisodicMemory.
### In order to ensure complete reproducability, we do not change the file and treat it as a baseline.

# Copyright 2019-present, IBM Research
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import quadprog

from model.resnet1d import ResNet1D
from model.replay_utils import (
    ReplayInputMixin,
    unpack_y_to_class_labels,
)
from model.task_bn import frozen_running_stats
from utils.training_metrics import macro_recall
from utils import misc_utils
from utils.class_weighted_loss import classification_cross_entropy


@dataclass
class GemConfig:
    gamma: float = 0.0  # margin added to the dual QP constraint (gamma in the paper)
    gem_disable_qp: bool = False  # ablation: skip QP projection + replay-gradient pass
    gem_replay: bool = False  # add ER-style replay CE on buffer samples to the loss
    gem_replay_lambda: float = 1.0  # weight of the replay CE term
    replay_batch_size: float = 20  # rows sampled per step for the replay CE
    inner_steps: int = 1
    lr: float = 1e-3
    n_memories: int = 0
    mem_sampling: str = "ring"
    arch: str = "resnet1d"
    dataset: str = "tinyimagenet"
    cuda: bool = True
    alpha_init: float = 1e-3
    grad_clip_norm: Optional[float] = 0.0
    input_channels: int = 2
    cls_lambda: float = 1.0

    @staticmethod
    def from_args(args: object) -> "GemConfig":
        cfg = GemConfig()
        for field in cfg.__dataclass_fields__:
            # `None` means "not set on args" (the parser registers `gamma`, which
            # HAT and MER share, with a None default), so the dataclass default
            # stands.
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
    # store the gradients
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

    The O(parameters) algebra (building ``P``/``q`` and reconstructing the
    update) runs on ``gradient``'s device in float64, matching the original
    NumPy double precision. Only the tiny ``t x t`` QP is shipped to quadprog
    on CPU, so the parameter-sized buffer never crosses the host<->device
    boundary -- avoiding a per-step sync that dominates on large models.
    """
    memories_d = memories.to(dtype=torch.float64)  # (p, t)
    gradient_d = gradient.contiguous().view(-1).to(dtype=torch.float64)  # (p,)
    t = memories_d.size(1)

    P = memories_d.t().mm(memories_d)  # (t, t)
    P = 0.5 * (P + P.t()) + torch.eye(t, dtype=P.dtype, device=P.device) * eps
    q = memories_d.t().mv(gradient_d).neg()  # (t,)

    # Solve the t x t dual QP on CPU (quadprog is CPU-only); t is tiny.
    P_np = P.cpu().numpy()
    q_np = q.cpu().numpy()
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P_np, q_np, G, h)[0]

    v_d = torch.as_tensor(v, dtype=memories_d.dtype, device=memories_d.device)
    x = memories_d.mv(v_d) + gradient_d  # (p,)
    gradient.copy_(x.view(-1, 1))


class Net(ReplayInputMixin, nn.Module):
    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        super(Net, self).__init__()
        self.cfg = GemConfig.from_args(args)
        self.margin = self.cfg.gamma
        # Ablation toggle: when False, skip both the past-task replay-gradient pass and the
        # QP projection, reducing GEM to plain fine-tuning at matched buffer size.
        self.use_qp = not bool(self.cfg.gem_disable_qp)
        self.is_cifar = (self.cfg.dataset == "cifar100") or (
            self.cfg.dataset == "tinyimagenet"
        )

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
        self.inner_steps = self.cfg.inner_steps
        self.cls_lambda = float(self.cfg.cls_lambda)

        self.opt = optim.SGD(self._ll_params(), self.cfg.lr, momentum=0.9)

        self.n_memories = int(self.cfg.n_memories)
        self.task_memory_capacities = self._build_task_memory_capacities(
            self.n_memories,
            n_tasks,
        )
        self.max_task_memories = max(self.task_memory_capacities, default=0)
        self.gpu = self.cfg.cuda

        # --- Episodic memory allocation ---
        if self.is_iq:
            assert n_inputs % 2 == 0, f"n_inputs={n_inputs} must be 2*L for IQ."
            self.seq_len = n_inputs // 2
            # (task, mem, C=2, L)
            self.memory_data = torch.FloatTensor(
                n_tasks, self.max_task_memories, 2, self.seq_len
            )
        else:
            # (task, mem, F)
            self.memory_data = torch.FloatTensor(
                n_tasks, self.max_task_memories, n_inputs
            )

        self.memory_labs = torch.LongTensor(n_tasks, self.max_task_memories)
        if self.gpu:
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()

        self.mem_sampling = self.cfg.mem_sampling
        if self.mem_sampling not in misc_utils.MEM_SAMPLING_MODES:
            raise ValueError(
                f"mem_sampling must be one of {misc_utils.MEM_SAMPLING_MODES}, "
                f"got {self.mem_sampling!r}"
            )

        # track how many exemplars each task has actually written
        self.task_mem_filled = torch.zeros(n_tasks, dtype=torch.long)
        self.task_mem_ptr = torch.zeros(n_tasks, dtype=torch.long)
        # total stream items seen per task (reservoir sampling only)
        self.task_mem_seen = torch.zeros(n_tasks, dtype=torch.long)
        if self.gpu:
            self.task_mem_filled = self.task_mem_filled.cuda()
            self.task_mem_ptr = self.task_mem_ptr.cuda()
            self.task_mem_seen = self.task_mem_seen.cuda()

        # --- GEM gradient buffers ---
        self.grad_dims = [p.data.numel() for p in self._ll_params()]
        self.grads = torch.Tensor(sum(self.grad_dims), n_tasks)
        if self.gpu:
            self.grads = self.grads.cuda()

        # --- optional ER-style replay CE on buffer samples ---
        self.gem_replay = bool(self.cfg.gem_replay)
        self.gem_replay_lambda = float(self.cfg.gem_replay_lambda)
        self.replay_batch_size = int(self.cfg.replay_batch_size)

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
        self.incremental_loader_name = getattr(args, "loader", None)

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

    def _ensure_iq_shape(self, x):
        """
        Ensure x is (B, 2, L) for IQ mode.
        Accepts (B, 2, L) or (B, 2L).
        """
        if x.dim() == 4 and x.size(1) == 3 and x.size(2) == 2:
            # 3-ADC layout; ResNet1D._prepare_input passes it through and the
            # ADC adapter reduces it to 2 channels.
            return x
        if x.dim() == 3:
            # (B, 2, L) already
            return x
        elif x.dim() == 2:
            # (B, 2L) -> (B, 2, L)
            B, F = x.shape
            assert F % 2 == 0, f"Feature dim {F} not divisible by 2 for (2, L) reshape."
            return misc_utils.deinterleave_iq_last_axis(x)
        else:
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

    def _ll_params(self):
        for name, param in self.net.named_parameters():
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
            fill_value=-10e10,
            loader=self.incremental_loader_name,
        )
        return output
    
    def _sample_replay_rows(self, current_task):
        """Uniformly sample stored rows from past tasks for the replay CE."""
        pairs = [
            (past_task, row)
            for past_task in self.observed_tasks
            if past_task != current_task
            for row in range(int(self.task_mem_filled[past_task].item()))
        ]
        if not pairs:
            return None
        sample_size = min(len(pairs), self.replay_batch_size)
        chosen = np.random.choice(len(pairs), sample_size, replace=False)
        device = self.memory_data.device
        t_idx = torch.tensor([pairs[i][0] for i in chosen], dtype=torch.long, device=device)
        s_idx = torch.tensor([pairs[i][1] for i in chosen], dtype=torch.long, device=device)
        return self.memory_data[t_idx, s_idx], self.memory_labs[t_idx, s_idx], t_idx

    def _masked_global_replay_loss(self, xx, yy_global, t_idx):
        """CE on replay rows, each row masked to its own task's logits.

        Mirrors er_ring/lamaml_cifar: raw logits + per-sample-task TIL/CIL mask
        with GLOBAL labels, so mixed-task replay batches are scored exactly as
        at eval time.
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
                fill_value=-10e10,
                loader=self.incremental_loader_name,
            )
        return classification_cross_entropy(
            masked, yy_global, class_weighted_ce=self.class_weighted_ce
        )

    def _store_replay(self, task_id: int, x_data: torch.Tensor, y_data: torch.Tensor):
        """Write the current batch into task_id's episodic buffer.

        Dispatches on ``self.mem_sampling``: ``"ring"`` overwrites the oldest
        samples via a wrapping write pointer; ``"reservoir"`` keeps a uniform
        random sample of the whole task stream (see
        :func:`utils.misc_utils.reservoir_slots`). Both keep occupied slots dense
        in ``[0, task_mem_filled[task_id])`` so replay slicing is unchanged.
        """
        capacity = int(self.task_memory_capacities[task_id])
        if capacity <= 0:
            return
        # Store adapter output (e.g. 3-ADC -> 2-channel) so replay matches forward.
        mem_x = self._input_for_replay(x_data)

        if self.mem_sampling == "reservoir":
            slots, filled, seen = misc_utils.reservoir_slots(
                mem_x.size(0),
                int(self.task_mem_filled[task_id].item()),
                int(self.task_mem_seen[task_id].item()),
                capacity,
            )
            for i, slot in enumerate(slots):
                if slot >= 0:
                    self.memory_data[task_id, slot].copy_(mem_x[i])
                    self.memory_labs[task_id, slot] = y_data[i]
            self.task_mem_filled[task_id] = filled
            self.task_mem_seen[task_id] = seen
            return

        write_pointer = int(self.task_mem_ptr[task_id].item())
        endcnt = min(write_pointer + mem_x.size(0), capacity)
        effbsz = endcnt - write_pointer
        if effbsz > 0:
            self.memory_data[task_id, write_pointer:endcnt].copy_(mem_x[:effbsz])
            self.memory_labs[task_id, write_pointer:endcnt].copy_(y_data[:effbsz])
            filled_before_update = int(self.task_mem_filled[task_id].item())
            self.task_mem_filled[task_id] = min(capacity, filled_before_update + effbsz)
        self.task_mem_ptr[task_id] = 0 if endcnt == capacity else endcnt

    def _store_past_task_gradients(self, current_task: int) -> None:
        """Backprop each previously observed task's memory and store its gradient.

        Fills ``self.grads`` with one column per past task, which the GEM
        projection below then constrains the current gradient against.

        Under CIL the memory loss is scored over every class of tasks
        ``0..current_task`` with global labels, the space the current batch is
        trained in; scoring it on the past task's own block would never
        constrain old samples against newer classes. Under TIL it is the past
        task's own block, as before.

        These forwards carry old-task data while the current task is active, so
        the whole loop runs under :func:`~model.task_bn.frozen_running_stats`:
        the replay batches are normalized with their own statistics but must not
        be folded into the current task's per-task BatchNorm running statistics.
        """
        with frozen_running_stats(self):
            for tt in range(len(self.observed_tasks) - 1):
                self.zero_grad()
                past_task = self.observed_tasks[tt]
                offset1, offset2 = compute_offsets(
                    past_task, self.classes_per_task, self.is_cifar
                )
                filled = int(self.task_mem_filled[past_task].item())
                if filled == 0:
                    continue  # nothing stored for this task yet

                # replay batch (shape already in memory)
                mem_x = Variable(
                    self.memory_data[past_task, :filled]
                )  # (mem, F) or (mem, 2, L)
                mem_y_flat = self.memory_labs[past_task, :filled]
                if self.incremental_loader_name == "class_incremental_loader":
                    logits_replay = self.forward(
                        mem_x, current_task, cil_all_seen_upto_task=current_task
                    )
                    targets_replay = mem_y_flat
                else:
                    logits_replay = self.forward(mem_x, past_task)[:, offset1:offset2]
                    targets_replay = mem_y_flat - offset1
                ptloss = classification_cross_entropy(
                    logits_replay,
                    targets_replay,
                    class_weighted_ce=self.class_weighted_ce,
                )
                ptloss.backward()
                if self.cfg.grad_clip_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.net.parameters(), self.cfg.grad_clip_norm
                    )
                store_grad(self._ll_params, self.grads, self.grad_dims, past_task)

    def observe(self, x, y, t):
        """
        One optimization step on batch (x,y,t), with GEM constraints and inner_steps.
        """
        # --- shape handling ---
        if self.is_iq:
            # keep (B, 2, L)
            x = self._ensure_iq_shape(x)
        else:
            # legacy: flatten non-IQ inputs
            x = x.view(x.size(0), -1)
        y_work = unpack_y_to_class_labels(y)

        # track tasks
        if t != self.old_task:
            if t not in self.observed_tasks:
                self.observed_tasks.append(t)
            self.old_task = t

        cls_tr_rec = []
        metric_logits = None

        for pass_itr in range(self.inner_steps):
            # push current batch once per batch (not each glance)
            if pass_itr == 0:
                self._store_replay(t, x.data, y_work.data)

            # gradients on past tasks (replay)
            if self.use_qp and len(self.observed_tasks) > 1:
                self._store_past_task_gradients(t)

            # current batch
            self.zero_grad()
            logits_full = self.forward(x, t, cil_all_seen_upto_task=t)
            targets = y_work.long()
            preds = torch.argmax(logits_full, dim=1)
            cls_tr_rec.append(macro_recall(preds, targets))
            loss = classification_cross_entropy(
                logits_full, targets, class_weighted_ce=self.class_weighted_ce
            )
            # Optional ER-style replay CE: trains directly on buffer samples, in
            # addition to the QP constraints those samples define. Added before
            # backward so the projection acts on the combined gradient.
            if self.gem_replay and len(self.observed_tasks) > 1:
                replay = self._sample_replay_rows(t)
                if replay is not None:
                    xx, yy_global, t_idx = replay
                    loss = loss + self.gem_replay_lambda * self._masked_global_replay_loss(
                        xx, yy_global, t_idx
                    )
            loss.backward()
            if self.cfg.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), self.cfg.grad_clip_norm
                )

            # GEM projection if needed
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
            metric_logits = logits_full.detach()
        avg_cls_tr_rec = sum(cls_tr_rec) / len(cls_tr_rec) if cls_tr_rec else 0.0
        return loss.item(), avg_cls_tr_rec, metric_logits
