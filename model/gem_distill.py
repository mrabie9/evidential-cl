# GEM-Distill: GEM kept intact (ResNet1D trunk, reservoir buffer, current-task-only CE, QP
# gradient-projection constraint) plus a KL-distillation replay term added to the stepped loss.
# Soft targets are frozen per task at its boundary (teacher = model right after training that task),
# mirroring CTN's distillation (loss3) but WITHOUT FiLM and WITHOUT changing GEM's structure.
# Rationale (docs/ablation_findings.md): distillation is the CTN ingredient that helps BWT; FiLM is
# only a stabiliser GEM does not need. This avoids both failure modes of the GEM x CTN grafts
# (B1 gem_ctn plasticity collapse; B2 ctn_gem instability). distill_lambda=0 recovers plain GEM.

import copy
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
from utils.training_metrics import macro_recall
from utils import misc_utils
from utils.class_weighted_loss import classification_cross_entropy


@dataclass
class GemDistillConfig:
    memory_strength: float = 0.0  # lambda in the paper (QP margin)
    gem_disable_qp: bool = False  # ablation: skip QP projection + replay-gradient pass
    distill_lambda: float = (
        1.0  # weight of the KL-distillation replay term (0 = pure GEM)
    )
    gem_lwf: bool = False  # add a LwF term (KL on current batch vs frozen teacher)
    gem_lwf_lambda: float = 1.0  # weight of the LwF term when gem_lwf is on
    temperature: float = 5.0  # softmax temperature for distillation soft targets
    balanced_replay: bool = (
        False  # class-balanced reservoir sampling (A1) instead of FIFO ring
    )
    inner_steps: int = 1
    lr: float = 1e-3
    n_memories: int = 0
    arch: str = "resnet1d"
    dataset: str = "tinyimagenet"
    cuda: bool = True
    alpha_init: float = 1e-3
    grad_clip_norm: Optional[float] = 100.0
    input_channels: int = 2
    cls_lambda: float = 1.0

    @staticmethod
    def from_args(args: object) -> "GemDistillConfig":
        cfg = GemDistillConfig()
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


class Net(ReplayInputMixin, nn.Module):
    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        super(Net, self).__init__()
        self.cfg = GemDistillConfig.from_args(args)
        self.margin = self.cfg.memory_strength
        self.distill_lambda = float(self.cfg.distill_lambda)
        self.use_lwf = bool(self.cfg.gem_lwf)
        self.gem_lwf_lambda = float(self.cfg.gem_lwf_lambda)
        self.teacher = None  # frozen model snapshot for LwF (current-data distillation)
        self.temp = float(self.cfg.temperature)
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.balanced_replay = bool(self.cfg.balanced_replay)
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

        # track how many exemplars each task has actually written
        self.task_mem_filled = torch.zeros(n_tasks, dtype=torch.long)
        self.task_mem_ptr = torch.zeros(n_tasks, dtype=torch.long)
        if self.gpu:
            self.task_mem_filled = self.task_mem_filled.cuda()
            self.task_mem_ptr = self.task_mem_ptr.cuda()

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
        self.incremental_loader_name = getattr(args, "loader", None)

        # --- Distillation soft targets (frozen per task at its boundary) ---
        self.memory_feat = torch.zeros(
            n_tasks, self.max_task_memories, self.nc_per_task
        )
        # Tracks which tasks have had their soft targets stored.
        self.distill_ready = [False] * n_tasks
        # --- Class-balanced reservoir bookkeeping (A1) ---
        self.cls_stored = torch.zeros(n_tasks, n_outputs, dtype=torch.long)
        self.cls_seen = torch.zeros(n_tasks, n_outputs, dtype=torch.long)
        if self.gpu:
            self.memory_feat = self.memory_feat.cuda()
            self.cls_stored = self.cls_stored.cuda()
            self.cls_seen = self.cls_seen.cuda()

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
        if x.dim() == 3:
            # (B, 2, L) already
            return x
        elif x.dim() == 2:
            # (B, 2L) -> (B, 2, L)
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

    def _update_ring_buffer(self, t, task_capacity, mem_x, y_all):
        """Original GEM per-task FIFO ring admission (default)."""
        write_pointer = int(self.task_mem_ptr[t].item())
        bsz = y_all.size(0)
        endcnt = min(write_pointer + bsz, task_capacity)
        effbsz = endcnt - write_pointer
        if effbsz <= 0:
            return
        self.memory_data[t, write_pointer:endcnt].copy_(mem_x[:effbsz])
        self.memory_labs[t, write_pointer:endcnt].copy_(y_all[:effbsz])
        filled_before = int(self.task_mem_filled[t].item())
        self.task_mem_filled[t] = min(task_capacity, filled_before + effbsz)
        self.task_mem_ptr[t] = 0 if endcnt == task_capacity else endcnt

    def _random_slot_of_class(self, t, c, task_capacity):
        labs = self.memory_labs[t, :task_capacity]
        slots = (labs == c).nonzero(as_tuple=True)[0]
        if slots.numel() == 0:
            return None
        pick = torch.randint(slots.numel(), (1,), device=slots.device)
        return int(slots[pick].item())

    def _reservoir_replace(self, t, c, x_i, task_capacity):
        """Standard reservoir replacement within class `c` (label unchanged)."""
        m_c = int(self.cls_stored[t, c].item())
        n_c = int(self.cls_seen[t, c].item())
        if m_c > 0 and torch.rand(1).item() < (m_c / n_c):
            evict = self._random_slot_of_class(t, c, task_capacity)
            if evict is not None:
                self.memory_data[t, evict].copy_(x_i)

    def _update_balanced_buffer(self, t, task_capacity, mem_x, y_all):
        """Class-balanced reservoir sampling (Chrysakis & Moens, 2020), per task.

        Balances admission across observed classes with no a-priori class stats, so
        replay and the distillation soft targets are not dominated by the frequent
        classes.
        """
        for i in range(y_all.size(0)):
            y_i = int(y_all[i].item())
            self.cls_seen[t, y_i] += 1
            filled = int(self.task_mem_filled[t].item())
            if filled < task_capacity:
                self.memory_data[t, filled].copy_(mem_x[i])
                self.memory_labs[t, filled] = y_i
                self.task_mem_filled[t] = filled + 1
                self.cls_stored[t, y_i] += 1
                continue
            stored_row = self.cls_stored[t]
            largest_c = int(torch.argmax(stored_row).item())
            if int(stored_row[y_i].item()) < int(stored_row[largest_c].item()):
                # Under-represented class: evict a random sample of the largest class.
                evict = self._random_slot_of_class(t, largest_c, task_capacity)
                if evict is None:
                    continue
                self.memory_data[t, evict].copy_(mem_x[i])
                self.memory_labs[t, evict] = y_i
                self.cls_stored[t, largest_c] -= 1
                self.cls_stored[t, y_i] += 1
            else:
                # Incoming is (one of) the largest classes: reservoir within it.
                self._reservoir_replace(t, y_i, mem_x[i], task_capacity)

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
            # Freeze soft targets for the task that just finished (teacher = current model).
            if self.old_task >= 0:
                self._store_soft_targets(self.old_task)
            # LwF: snapshot the just-finished model as a frozen teacher. Unlike the
            # distill term (frozen soft targets on buffer samples), LwF distills on the
            # CURRENT task's incoming data against this teacher.
            if self.use_lwf:
                self.teacher = copy.deepcopy(self.net)
                self.teacher.eval()
                for param in self.teacher.parameters():
                    param.requires_grad = False
            if t not in self.observed_tasks:
                self.observed_tasks.append(t)
            self.old_task = t

        cls_tr_rec = []
        metric_logits = None

        for pass_itr in range(self.inner_steps):
            # push current batch once per batch (not each glance)
            if pass_itr == 0:
                task_capacity = int(self.task_memory_capacities[t])
                if task_capacity > 0:
                    # Store adapter output (e.g. 3-ADC -> 2-channel) so replay matches forward.
                    mem_x = self._input_for_replay(x.data)
                    if self.balanced_replay:
                        self._update_balanced_buffer(
                            t, task_capacity, mem_x, y_work.data
                        )
                    else:
                        self._update_ring_buffer(t, task_capacity, mem_x, y_work.data)

            # gradients on past tasks (replay)
            if self.use_qp and len(self.observed_tasks) > 1:
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

            # current batch
            self.zero_grad()
            logits_full = self.forward(x, t, cil_all_seen_upto_task=t)
            targets = y_work.long()
            preds = torch.argmax(logits_full, dim=1)
            cls_tr_rec.append(macro_recall(preds, targets))
            loss = classification_cross_entropy(
                logits_full, targets, class_weighted_ce=self.class_weighted_ce
            )
            # Distillation replay: pull the model toward frozen per-task soft targets.
            if self.distill_lambda > 0:
                distill_loss = self._distillation_loss()
                if distill_loss is not None:
                    loss = loss + self.distill_lambda * distill_loss
            # LwF: KL over previous-task classes [0, offset1) on the CURRENT batch x
            # against the frozen teacher. Added to the current-task loss (so it is
            # subject to GEM's QP projection, like the distill term).
            if self.use_lwf and self.teacher is not None:
                lwf_offset1, _ = compute_offsets(
                    t, self.classes_per_task, self.is_cifar
                )
                if lwf_offset1 > 0:
                    student_prev = self.net(x)[:, :lwf_offset1]
                    with torch.no_grad():
                        teacher_prev = self.teacher(x)[:, :lwf_offset1]
                        teacher_probs = torch.softmax(teacher_prev / self.temp, dim=1)
                    lwf_loss = self.kl(
                        torch.log_softmax(student_prev / self.temp, dim=1),
                        teacher_probs,
                    )
                    loss = loss + self.gem_lwf_lambda * lwf_loss
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
