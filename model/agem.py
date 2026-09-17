### This is a pytorch implementation of AGEM based on https://github.com/facebookresearch/agem.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

import random

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
class AgemConfig:
    ## AGEM-specific hyperparameters
    lr: float = 1e-3
    inner_steps: int = 1
    memories: int = 5120

    ## Generic hyperparameters
    arch: str = "resnet1d"
    n_layers: int = 2
    n_hiddens: int = 100
    dataset: str = "tinyimagenet"
    cuda: bool = True
    grad_clip_norm: Optional[float] = 0.0
    input_channels: int = 1
    cls_lambda: float = 1.0
    memory_loss_lambda: float = 1.0

    @staticmethod
    def from_args(args: object) -> "AgemConfig":
        cfg = AgemConfig()
        for field in cfg.__dataclass_fields__:
            if hasattr(args, field):
                setattr(cfg, field, getattr(args, field))
        return cfg


# Auxiliary functions useful for AGEM's inner optimization.


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


def projectgrad(gradient, memories, margin=0.5, eps=1e-3, oiter=0):
    """
    Solves the GEM dual QP described in the paper given a proposed
    gradient "gradient", and a memory of task gradients "memories".
    Overwrites "gradient" with the final projected update.
    input:  gradient, p-vector
    input:  memories, (t * p)-vector
    output: x, p-vector
    """

    # Keep projection math in Torch to avoid GPU<->CPU sync overhead.
    # Use float64 to match previous NumPy double precision behavior.
    memories_t = memories.t().to(dtype=torch.float64)
    gradient_t = gradient.contiguous().view(-1).to(dtype=torch.float64)

    # merge memories
    memories_t2 = memories_t.mean(dim=0)
    ref_mag = torch.dot(memories_t2, memories_t2)
    dotp = torch.dot(gradient_t, memories_t2)

    # if(oiter%100==0):
    #     print('similarity : ', similarity.item())
    #     print('dotp:', dotp.item())

    if dotp.item() < 0:
        proj = gradient_t - ((dotp / ref_mag) * memories_t2)
        gradient.copy_(proj.to(dtype=gradient.dtype).view(-1, 1))


class Net(ReplayInputMixin, nn.Module):
    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        super(Net, self).__init__()
        self.cfg = AgemConfig.from_args(args)

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

        self.class_weighted_ce = bool(getattr(args, "class_weighted_ce", True))
        self.n_outputs = n_outputs
        self.inner_steps = self.cfg.inner_steps
        self.cls_lambda = float(self.cfg.cls_lambda)
        self.memory_loss_lambda = float(self.cfg.memory_loss_lambda)

        self.opt = optim.SGD(self._ll_params(), self.cfg.lr, momentum=0.9)

        self.n_memories = int(self.cfg.memories / n_tasks)
        self.gpu = self.cfg.cuda

        self.age = 0
        self.M = []
        self.memories = self.cfg.memories
        self.grad_align = []
        self.grad_task_align = {}
        self.current_task = None

        # --- Episodic memory allocation ---
        if self.is_iq:
            assert n_inputs % 2 == 0, f"n_inputs={n_inputs} must be 2*L for IQ."
            self.seq_len = n_inputs // 2
            # (task, mem, C=2, L)
            self.memory_data = torch.FloatTensor(
                n_tasks, self.n_memories, 2, self.seq_len
            )
        else:
            # (task, mem, F)
            self.memory_data = torch.FloatTensor(n_tasks, self.n_memories, n_inputs)

        self.memory_labs = torch.LongTensor(n_tasks, self.n_memories)
        if self.gpu:
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()

        # Track how many exemplars each task has actually written.
        self.task_mem_filled = torch.zeros(n_tasks, dtype=torch.long)
        if self.gpu:
            self.task_mem_filled = self.task_mem_filled.cuda()

        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self._ll_params():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), n_tasks)
        # Single reference gradient buffer for the averaged A-GEM constraint,
        # computed from one mixed-buffer batch (not one column per past task).
        self.ref_grad = torch.Tensor(sum(self.grad_dims))
        if self.gpu:
            self.grads = self.grads.cuda()
            self.ref_grad = self.ref_grad.cuda()

        # allocate counters
        self.observed_tasks = []
        self.mem_cnt = 0
        self.classes_per_task = misc_utils.build_task_class_list(
            n_tasks,
            n_outputs,
            nc_per_task=getattr(args, "nc_per_task_list", "")
            or getattr(args, "nc_per_task", None),
            classes_per_task=getattr(args, "classes_per_task", None),
        )
        self.nc_per_task = misc_utils.max_task_class_count(self.classes_per_task)

        # Per-task signal-class offsets, precomputed for vectorised per-sample
        # logit masking of mixed-task reference batches.
        task_offset1, task_offset2 = [], []
        for task_index in range(n_tasks):
            off1, off2 = compute_offsets(
                task_index, self.classes_per_task, self.is_cifar
            )
            task_offset1.append(off1)
            task_offset2.append(off2)
        self._task_offset1 = torch.tensor(task_offset1, dtype=torch.long)
        self._task_offset2 = torch.tensor(task_offset2, dtype=torch.long)
        if self.gpu:
            self._task_offset1 = self._task_offset1.cuda()
            self._task_offset2 = self._task_offset2.cuda()
        self.incremental_loader_name = getattr(args, "loader", None)

        if self.gpu:
            self.cuda()

        self.iter = 0

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
        """Ensure inputs stored in memory are (B, 2, L)."""
        if x.dim() == 4 and x.size(1) == 3 and x.size(2) == 2:
            return self.net.model.input_adapter(x)
        if x.dim() == 3 and x.size(1) == 3:
            if x.size(2) % 2 != 0:
                raise ValueError(
                    f"Expected even length for 3-ADC IQ input; got shape {tuple(x.shape)}."
                )
            seq_len = x.size(2) // 2
            x4 = x.view(x.size(0), 3, 2, seq_len)
            return self.net.model.input_adapter(x4)
        if x.dim() == 2:
            features = x.size(1)
            if features % 6 == 0:
                seq_len = features // 6
                x4 = x.view(x.size(0), 3, 2, seq_len)
                return self.net.model.input_adapter(x4)
        return self._ensure_iq_shape(x)

    def forward(self, x, t, *, cil_all_seen_upto_task=None):
        if self.cfg.dataset == "tinyimagenet":
            x = x.view(-1, 3, 64, 64)
        elif self.cfg.dataset == "cifar100":
            x = x.view(-1, 3, 32, 32)
        elif self.is_iq:
            x = self._ensure_iq_shape(x)  # (B, 2, L)

        output = self.net.forward(x)

        return misc_utils.apply_task_incremental_logit_mask(
            output,
            t,
            self.classes_per_task,
            self.n_outputs,
            cil_all_seen_upto_task=cil_all_seen_upto_task,
            fill_value=-10e10,
            loader=self.incremental_loader_name,
        )

    def _ll_params(self):
        for name, param in self.net.named_parameters():
            yield param

    def _forward_features_no_mask(self, x):
        """Run the backbone with the same reshape as ``forward`` but no logit mask.

        Mixed-task reference batches span several tasks at once, so the single
        ``task_index`` TIL mask in :meth:`forward` does not apply; per-sample
        masking is done separately by :meth:`_mask_logits_per_sample`. For the
        same reason the batch must not update per-task BatchNorm running
        statistics -- it belongs to no single task.
        """
        if self.cfg.dataset == "tinyimagenet":
            x = x.view(-1, 3, 64, 64)
        elif self.cfg.dataset == "cifar100":
            x = x.view(-1, 3, 32, 32)
        elif self.is_iq:
            x = self._ensure_iq_shape(x)
        with frozen_running_stats(self):
            return self.net.forward(x)

    def _mask_logits_per_sample(self, logits, task_ids, current_task):
        """Mask a mixed-task reference batch for the memory loss.

        Under CIL every row sees all classes of tasks ``0..current_task``, the
        same space the current batch is trained in (see
        :func:`utils.misc_utils.mask_replay_logits`). Under TIL each row is
        restricted to its own task's signal-class block, mirroring the
        ``[:, offset1:offset2]`` slice the per-task path used, vectorised across
        a batch whose rows belong to different tasks.
        """
        if self.incremental_loader_name == "class_incremental_loader":
            return misc_utils.mask_replay_logits(
                logits,
                task_ids,
                current_task,
                self.classes_per_task,
                self.n_outputs,
                loader=self.incremental_loader_name,
                fill_value=-10e10,
            )
        offset1 = self._task_offset1[task_ids].unsqueeze(1)
        offset2 = self._task_offset2[task_ids].unsqueeze(1)
        cols = torch.arange(logits.size(1), device=logits.device).unsqueeze(0)
        valid = (cols >= offset1) & (cols < offset2)
        return logits.masked_fill(~valid, -10e10)

    def _sample_reference_batch(self, current_task, batch_size):
        """Sample one mixed batch uniformly over all stored past-task exemplars.

        This is the A-GEM reference batch: a single draw from the union of every
        past task's memory, weighted by how many exemplars each task holds so
        every stored example is equally likely. Returns ``(x, y_global, tasks)``
        with noise-labelled exemplars removed, or ``None`` if nothing is stored.
        """
        device = self.memory_data.device
        valid_tasks, counts = [], []
        for past_task in self.observed_tasks:
            if past_task == current_task:
                continue
            filled = int(self.task_mem_filled[past_task].item())
            if filled > 0:
                valid_tasks.append(past_task)
                counts.append(filled)
        if not valid_tasks:
            return None

        valid_tasks_t = torch.tensor(valid_tasks, device=device, dtype=torch.long)
        counts_t = torch.tensor(counts, device=device, dtype=torch.float)
        # P(task) ∝ exemplars stored, so P(any single exemplar) is uniform.
        task_pick = torch.multinomial(
            counts_t / counts_t.sum(), batch_size, replacement=True
        )
        task_ids = valid_tasks_t[task_pick]
        counts_per_pick = counts_t[task_pick]
        slot_ids = (torch.rand(batch_size, device=device) * counts_per_pick).long()
        slot_ids = torch.minimum(slot_ids, counts_per_pick.long() - 1)

        ref_x = self.memory_data[task_ids, slot_ids]
        ref_y = unpack_y_to_class_labels(self.memory_labs[task_ids, slot_ids]).long()
        return ref_x, ref_y, task_ids

    def observe(self, x, y, t):

        self.iter += 1

        # --- shape handling ---
        if self.is_iq:
            # keep (B, 2, L)
            x = self._ensure_iq_shape(x)
        else:
            # legacy: flatten non-IQ inputs
            x = x.view(x.size(0), -1)

        y_work = unpack_y_to_class_labels(y).long()

        # update memory
        if t != self.current_task:
            # finalize previous task's filled count
            if self.current_task is not None:
                self.task_mem_filled[self.current_task] = min(
                    self.mem_cnt, self.n_memories
                )
            self.observed_tasks.append(t)
            self.current_task = t
            self.grad_align.append([])
            # start writing this task from the beginning
            self.mem_cnt = 0

        cls_tr_rec = []
        metric_logits = None
        for pass_itr in range(self.inner_steps):
            # copy x into memory with matching shape

            if pass_itr == 0:
                # Update ring buffer storing examples from current task
                bsz = y_work.data.size(0)
                endcnt = min(self.mem_cnt + bsz, self.n_memories)
                effbsz = endcnt - self.mem_cnt
                # self.memory_data[t, self.mem_cnt: endcnt].copy_(
                #     x.data[: effbsz])
                # if bsz == 1:
                #     self.memory_labs[t, self.mem_cnt] = y.data[0]
                # else:
                #     self.memory_labs[t, self.mem_cnt: endcnt].copy_(
                #         y.data[: effbsz])
                # self.mem_cnt += effbsz
                # if self.mem_cnt == self.n_memories:
                #     self.mem_cnt = 0

                if effbsz > 0:
                    mem_x = self._input_for_replay(x.data[:effbsz])
                    self.memory_data[t, self.mem_cnt : endcnt].copy_(mem_x)

                    if bsz == 1:
                        self.memory_labs[t, self.mem_cnt] = y_work.data[0]
                    else:
                        self.memory_labs[t, self.mem_cnt : endcnt].copy_(
                            y_work.data[:effbsz]
                        )

                    self.mem_cnt += effbsz
                    if self.mem_cnt == self.n_memories:
                        self.task_mem_filled[t] = self.n_memories  # full before wrap
                        self.mem_cnt = 0
                    else:
                        self.task_mem_filled[t] = self.mem_cnt

            # compute the single A-GEM reference gradient on one mixed batch
            # sampled across all past tasks (not one gradient per past task)
            have_reference_grad = False
            if len(self.observed_tasks) > 1:
                reference_batch = self._sample_reference_batch(
                    current_task=t, batch_size=y_work.data.size(0)
                )
                if reference_batch is not None:
                    ref_x, ref_y, ref_tasks = reference_batch
                    self.zero_grad()
                    ref_logits = self._mask_logits_per_sample(
                        self._forward_features_no_mask(ref_x), ref_tasks, t
                    )
                    memory_loss = classification_cross_entropy(
                        ref_logits,
                        ref_y,
                        class_weighted_ce=self.class_weighted_ce,
                    )
                    ptloss = self.memory_loss_lambda * memory_loss
                    ptloss.backward()
                    if self.cfg.grad_clip_norm:
                        torch.nn.utils.clip_grad_norm_(
                            self.net.parameters(), self.cfg.grad_clip_norm
                        )
                    store_grad(
                        self._ll_params,
                        self.ref_grad.unsqueeze(1),
                        self.grad_dims,
                        0,
                    )
                    have_reference_grad = True

            # now compute the grad on the current minibatch
            self.zero_grad()
            logits_full = self.forward(x, t, cil_all_seen_upto_task=t)
            y_cls = y_work.long()
            pb = torch.argmax(logits_full, dim=1)
            targets = y_cls
            cls_tr_rec.append(macro_recall(pb, targets))
            loss = classification_cross_entropy(
                logits_full,
                y_cls,
                class_weighted_ce=self.class_weighted_ce,
            )
            loss.backward()
            if self.cfg.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), self.cfg.grad_clip_norm
                )

            # project the current gradient against the single reference gradient
            # if it violates the averaged A-GEM constraint (g · g_ref < 0)
            if have_reference_grad:
                store_grad(self._ll_params, self.grads, self.grad_dims, t)
                projectgrad(
                    self.grads[:, t].unsqueeze(1),
                    self.ref_grad.unsqueeze(1),
                    oiter=self.iter,
                )
                # copy gradients back
                overwrite_grad(self._ll_params, self.grads[:, t], self.grad_dims)

            self.opt.step()
            metric_logits = logits_full.detach()

        x_for_storage = self._input_for_replay(x)
        xi = x_for_storage.data.cpu().numpy()
        yi = y_work.data.cpu().numpy()
        for i in range(0, x.size()[0]):
            self.age += 1
            # Reservoir sampling memory update:
            if len(self.M) < self.memories:
                self.M.append([xi[i], yi[i], t])

            else:
                p = random.randint(0, self.age)
                if p < self.memories:
                    self.M[p] = [xi[i], yi[i], t]

        avg_cls_tr_rec = sum(cls_tr_rec) / len(cls_tr_rec) if cls_tr_rec else 0.0
        return loss.item(), avg_cls_tr_rec, metric_logits
