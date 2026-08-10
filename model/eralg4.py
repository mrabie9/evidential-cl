# An implementation of Experience Replay (ER) with reservoir sampling and without using tasks from Algorithm 4 of https://openreview.net/pdf?id=B1gTShAct7

# Copyright 2019-present, IBM Research
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

import copy
import os
import random
import warnings
import math

from model.resnet1d import ResNet1D
from model.detection_replay import (
    DetectionReplayMixin,
    noise_label_from_args,
    unpack_y_to_class_labels,
)
from utils.training_metrics import macro_recall
from utils import misc_utils
from utils.class_weighted_loss import classification_cross_entropy

warnings.filterwarnings("ignore")


@dataclass
class ErAlgConfig:
    alpha_init: float = 1e-3
    lr: float = 1e-3
    opt_lr: float = 1e-1
    learn_lr: bool = False
    inner_steps: int = 1
    memories: int = 5120
    replay_batch_size: int = 20
    grad_clip_norm: Optional[float] = 2.0
    second_order: bool = False
    meta_batches: int = 3
    eralg4_masked_loss: bool = True
    # PROBE: sister-repo two-forward joint ER loop (see --eralg4_joint_er).
    eralg4_joint_er: bool = False
    # PROBE: average the ER loss over K independent stochastic forward passes of
    # the SAME batch before one optimizer step -- the twin of C-MAML's
    # ``meta_batches``. resnet1d carries four Dropout(p=0.2) layers that
    # ResNet1D.forward keeps active, so a single-forward gradient aligns only
    # ~0.69 with the noise-free gradient (K=3 reaches ~0.85) and its inflated
    # norm trips grad_clip_norm every step. K=1 is the historical behaviour.
    eralg4_grad_avg: int = 1
    # Res-ER + distillation: KL on replay samples against a teacher frozen at each
    # task boundary (LwF-style, matching er_ring's er_distill / gem_distill's loss3).
    # memory_strength is the KL weight, temperature the softmax temperature.
    er_distill: bool = False
    memory_strength: float = 1.0
    temperature: float = 5.0

    arch: str = "resnet1d"
    dataset: str = "tinyimagenet"
    cuda: bool = True
    det_lambda: float = 1.0
    cls_lambda: float = 1.0
    det_memories: int = 2000
    det_replay_batch: int = 64
    memory_loss_lambda: float = 1.0

    @staticmethod
    def from_args(args: object) -> "ErAlgConfig":
        cfg = ErAlgConfig()
        for field in cfg.__dataclass_fields__:
            if hasattr(args, field):
                setattr(cfg, field, getattr(args, field))
        return cfg


class Net(DetectionReplayMixin, nn.Module):
    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        super(Net, self).__init__()

        self.cfg = ErAlgConfig.from_args(args)
        self.class_weighted_ce = bool(getattr(args, "class_weighted_ce", True))

        if self.cfg.arch != "resnet1d":
            raise ValueError(
                f"Unsupported arch {self.cfg.arch}; only resnet1d is available now."
            )
        self.net = ResNet1D(n_outputs, args)
        self.net.define_task_lr_params(alpha_init=self.cfg.alpha_init)

        self.opt_wt = optim.SGD(self._ll_params(), lr=self.cfg.lr, momentum=0.9)
        self.det_opt = optim.SGD(
            self.net.det_head.parameters(), lr=self.cfg.lr, momentum=0.9
        )

        if self.cfg.learn_lr:
            self.opt_lr = torch.optim.SGD(
                list(self.net.alpha_lr.parameters()), lr=self.cfg.opt_lr, momentum=0.9
            )

        self.is_cifar = (self.cfg.dataset == "cifar100") or (
            self.cfg.dataset == "tinyimagenet"
        )
        self.inner_steps = self.cfg.inner_steps
        self.det_lambda = float(self.cfg.det_lambda)
        self.cls_lambda = float(self.cfg.cls_lambda)
        self.memory_loss_lambda = float(self.cfg.memory_loss_lambda)
        # Distillation (er_distill): teacher snapshot frozen at each task boundary,
        # KL over replay rows within their per-sample task class slices.
        self.use_distill = bool(self.cfg.er_distill)
        self.reg = float(self.cfg.memory_strength)
        self.temp = float(self.cfg.temperature)
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.teacher = None  # frozen model snapshot, set at each task boundary
        self._init_det_replay(
            self.cfg.det_memories,
            self.cfg.det_replay_batch,
            enabled=bool(getattr(args, "use_detector_arch", False)),
        )

        self.current_task = 0
        self.memories = self.cfg.memories
        self.batchSize = int(self.cfg.replay_batch_size)

        # allocate buffer
        self.M = []
        self.age = 0

        # handle gpus if specified
        self.use_cuda = self.cfg.cuda
        if self.use_cuda:
            self.net = self.net.cuda()

        self.n_outputs = n_outputs
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
        # if self.is_cifar:
        #     self.nc_per_task = int(n_outputs / n_tasks)
        # else:
        #     self.nc_per_task = n_outputs

    def compute_offsets(self, task):
        return misc_utils.compute_offsets(task, self.classes_per_task)

    def _ll_params(self):
        for name, param in self.net.named_parameters():
            if name.startswith("det_head"):
                continue
            yield param

    def take_multitask_loss(self, bt, logits, y):
        """Batched CE over global labels, per-sample task-masked by default.

        Masking (``eralg4_masked_loss``, default on) confines each row's softmax
        to its own task's classes, matching er_ring / lamaml_cifar; the unmasked
        global softmax is kept only as an ablation (`--eralg4_unmasked_loss`).
        Class weights are inverse-frequency over this batch — the per-row loop
        this replaces silently collapsed them to 1.0 (single-row batches), so
        weighting only takes effect with the batched call.
        """
        if logits.size(0) == 0:
            return torch.zeros((), device=logits.device, dtype=logits.dtype)
        if self.cfg.eralg4_masked_loss:
            logits = self._mask_logits_for_sample_tasks(logits, bt)
        return classification_cross_entropy(
            logits,
            y.long(),
            class_weighted_ce=self.class_weighted_ce,
        )

    def forward(self, x, t, *, cil_all_seen_upto_task=None):
        output = self.net.forward(x)
        if True:  # self.is_cifar:
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

    def getBatch(self, x, y, t):
        if x is not None:
            mxi = np.array(x)
            myi = np.array(y)
            mti = np.ones(x.shape[0], dtype=int) * t
        else:
            mxi = np.empty(shape=(0, 0))
            myi = np.empty(shape=(0, 0))
            mti = np.empty(shape=(0, 0))

        replay_x = []
        replay_y = []
        replay_t = []
        current_x = []
        current_y = []
        current_t = []

        if len(self.M) > 0:
            osize = min(self.batchSize, len(self.M))
            # The original loop reshuffled the full index list once per draw and
            # took position ``j`` — uniform sampling-with-replacement, which
            # ``random.choices`` reproduces without O(N * osize) shuffling.
            for k in random.choices(range(len(self.M)), k=osize):
                x, y, t = self.M[k]
                xi = np.array(x)
                yi_scalar = int(torch.as_tensor(y).long().flatten()[0].item())
                ti = np.array(t)
                if self.noise_label is not None and yi_scalar == self.noise_label:
                    continue

                replay_x.append(xi)
                replay_y.append(yi_scalar)
                replay_t.append(ti)

        for i in range(len(myi)):
            current_x.append(mxi[i])
            current_y.append(myi[i])
            current_t.append(mti[i])

        bxs = replay_x + current_x
        bys = replay_y + current_y
        bts = replay_t + current_t
        replay_count = len(replay_x)

        bxs = Variable(torch.from_numpy(np.array(bxs))).float()
        bys = Variable(torch.from_numpy(np.array(bys))).long().view(-1)
        bts = Variable(torch.from_numpy(np.array(bts))).long().view(-1)

        # handle gpus if specified
        if self.use_cuda:
            bxs = bxs.cuda()
            bys = bys.cuda()
            bts = bts.cuda()

        return bxs, bys, bts, replay_count

    def _mask_logits_for_sample_tasks(
        self, raw_logits: torch.Tensor, sample_task_indices: torch.Tensor
    ) -> torch.Tensor:
        """Apply TIL masking per sample task id (mirrors lamaml_cifar.meta_loss)."""
        if sample_task_indices.numel() == 0:
            return raw_logits
        masked_logits = raw_logits.clone()
        for task_id in torch.unique(sample_task_indices).tolist():
            row_selector = sample_task_indices == int(task_id)
            if not torch.any(row_selector):
                continue
            masked_logits[row_selector] = misc_utils.apply_task_incremental_logit_mask(
                raw_logits[row_selector],
                int(task_id),
                self.classes_per_task,
                self.n_outputs,
                cil_all_seen_upto_task=int(task_id),
                global_noise_label=self.noise_label,
                fill_value=-10e10,
                loader=self.incremental_loader_name,
            )
        return masked_logits

    def _weighted_multitask_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        tasks: torch.Tensor,
        replay_count: int,
    ) -> torch.Tensor:
        replay_count = max(0, min(int(replay_count), logits.size(0)))
        replay_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
        if replay_count > 0:
            replay_loss = self.take_multitask_loss(
                tasks[:replay_count], logits[:replay_count], labels[:replay_count]
            )
        current_loss = self.take_multitask_loss(
            tasks[replay_count:], logits[replay_count:], labels[replay_count:]
        )
        return current_loss + (self.memory_loss_lambda * replay_loss)

    def observe(self, x, y, t):
        ### step through elements of x

        # noise_label = None
        # if class_counts is not None:
        #     _, offset2 = misc_utils.compute_offsets(t, class_counts)
        #     noise_label = offset2 - 1
        # y_cls, y_det = self._unpack_labels(
        #     y,
        #     noise_label=noise_label,
        #     use_detector_arch=bool(getattr(self, "det_enabled", False)),
        # )
        # if y_det is not None and self.det_memories > 0:
        #     self._update_det_memory(x, y_det)
        # x_det = x
        # signal_mask = (y_det == 1) & (y_cls >= 0)
        # if not signal_mask.any():
        #     if not getattr(self, "det_enabled", True):
        #         return 0.0, 0.0
        #     self.det_opt.zero_grad()
        #     det_logits, _ = self.net.forward_heads(x_det)
        #     det_loss = self.det_loss(det_logits, y_det.float())
        #     det_replay = self._sample_det_memory()
        #     if det_replay is not None:
        #         mem_x, mem_y = det_replay
        #         mem_det_logits, _ = self.net.forward_heads(mem_x)
        #         mem_loss = self.det_loss(mem_det_logits, mem_y.float())
        #         det_loss = 0.5 * (det_loss + mem_loss)
        #     det_loss = self.det_lambda * det_loss
        #     det_loss.backward()
        #     self.det_opt.step()
        #     return float(det_loss.item()), 0.0

        # x = x[signal_mask]
        # y = y_cls[signal_mask]
        # Keep a detached leaf copy so each inner/meta step can rebuild a fresh
        # canonicalized graph for 3-channel adapter inputs.
        raw_x_train = x.detach().requires_grad_(True)
        x_for_storage = self._input_for_replay(x)
        xi = x_for_storage.data.cpu().numpy()
        y_work = unpack_y_to_class_labels(y).long()
        yi = y_work.data.cpu().numpy()

        if t != self.current_task:
            # Distillation: freeze the just-finished model as a teacher, mirroring
            # er_ring / gem_distill / BCL-Dual. Replay-sample KL below distills the
            # student toward this snapshot within each row's task class slice.
            if self.use_distill:
                self.teacher = copy.deepcopy(self.net)
                self.teacher.eval()
                for param in self.teacher.parameters():
                    param.requires_grad = False
            self.current_task = t

        metric_logits = None
        if self.cfg.learn_lr:
            loss, cls_tr_rec = self.la_ER(raw_x_train, y, t)
        elif self.cfg.eralg4_joint_er:
            # Sister-repo path: current live batch + replay in separate forwards,
            # adapter co-trained in-graph. No decoupled adapter step below.
            loss, cls_tr_rec, metric_logits = self.ER_joint(raw_x_train, y_work, t)
        else:
            loss, cls_tr_rec = self.ER(xi, yi, t)

        # Ensure the adapter is explicitly trained on the current differentiable
        # 3-ADC/4D batch even when replay path uses detached storage tensors.
        if not self.cfg.eralg4_joint_er and (
            (x.dim() == 3 and x.size(1) == 3)
            or (x.dim() == 4 and x.size(1) == 3 and x.size(2) == 2)
        ):
            self.net.zero_grad(set_to_none=True)
            live_x_train = self._canonicalize_input(x.detach(), detach=False)
            live_logits = misc_utils.apply_task_incremental_logit_mask(
                self.net.forward(live_x_train),
                t,
                self.classes_per_task,
                self.n_outputs,
                cil_all_seen_upto_task=t,
                global_noise_label=self.noise_label,
                loader=self.incremental_loader_name,
            )
            targets = y_work.long()
            live_loss = classification_cross_entropy(
                live_logits,
                targets,
                class_weighted_ce=self.class_weighted_ce,
            )
            live_loss.backward()
            adapter_module = getattr(self.net.model, "input_adapter", None)
            if adapter_module is not None:
                with torch.no_grad():
                    for parameter in adapter_module.parameters():
                        if parameter.grad is None:
                            continue
                        parameter.add_(parameter.grad, alpha=-self.cfg.lr)
            self.net.zero_grad(set_to_none=True)
            metric_logits = live_logits.detach()

        for i in range(0, x.size()[0]):
            self.age += 1
            # Reservoir sampling memory update:
            if len(self.M) < self.memories:
                self.M.append([xi[i], yi[i], t])

            else:
                p = random.randint(0, self.age)
                if p < self.memories:
                    self.M[p] = [xi[i], yi[i], t]

        # if getattr(self, "det_enabled", True):
        #     self.det_opt.zero_grad()
        #     det_logits, _ = self.net.forward_heads(x_det)
        #     det_loss = self.det_loss(det_logits, y_det.float())
        #     det_replay = self._sample_det_memory()
        #     if det_replay is not None:
        #         mem_x, mem_y = det_replay
        #         mem_det_logits, _ = self.net.forward_heads(mem_x)
        #         mem_loss = self.det_loss(mem_det_logits, mem_y.float())
        #         det_loss = 0.5 * (det_loss + mem_loss)
        #     det_loss = self.det_lambda * det_loss
        #     det_loss.backward()
        #     self.det_opt.step()

        return loss.item(), cls_tr_rec, metric_logits

    def _batch_accuracy(self, bt, logits, labels):
        if len(bt) == 0:
            return 0.0
        with torch.no_grad():
            # Vectorized per-sample argmax within each row's task class slice
            # [offset1, offset2). Masking non-task columns to -inf makes a single
            # batched argmax exact, avoiding one GPU sync per sample.
            labels_dev = labels.long().view(-1)
            bt_list = bt.long().view(-1).tolist()
            offsets = {tid: self.compute_offsets(tid) for tid in set(bt_list)}
            o1 = torch.tensor(
                [offsets[tid][0] for tid in bt_list], device=logits.device
            )
            o2 = torch.tensor(
                [offsets[tid][1] for tid in bt_list], device=logits.device
            )
            cols = torch.arange(logits.size(1), device=logits.device)
            valid = (cols.unsqueeze(0) >= o1.unsqueeze(1)) & (
                cols.unsqueeze(0) < o2.unsqueeze(1)
            )
            masked = logits.masked_fill(~valid, float("-inf"))
            preds = masked.argmax(dim=1) - o1
            targets = labels_dev - o1
            keep = torch.ones_like(labels_dev, dtype=torch.bool)
            if self.noise_label is not None:
                keep &= labels_dev != self.noise_label
            if not keep.any():
                return 0.0
            return macro_recall(preds[keep], targets[keep])

    def ER(self, x, y, t):
        cls_tr_rec = []
        for pass_itr in range(self.inner_steps):

            self.net.zero_grad()

            # Draw batch from buffer:
            bx, by, bt, replay_count = self.getBatch(x, y, t)

            bx = bx.squeeze()
            # Raw logits; per-sample task masking happens inside
            # ``take_multitask_loss`` (global CE targets index ``n_outputs``).
            prediction = self.net.forward(bx)
            loss = self._weighted_multitask_loss(prediction, by, bt, replay_count)
            cls_tr_rec.append(self._batch_accuracy(bt, prediction, by))
            self._dbg("BASE", pass_itr, t, bx, prediction, by, bt, replay_count, loss)

            # Distillation on replay rows: KL(student || frozen teacher) within each
            # row's task class slice. Per-sample masking (student and teacher alike)
            # confines the softmax to that row's task, so masked columns contribute ~0.
            rc = max(0, min(int(replay_count), prediction.size(0)))
            if self.use_distill and self.teacher is not None and rc > 0:
                rt = bt[:rc]
                student_masked = self._mask_logits_for_sample_tasks(prediction[:rc], rt)
                with torch.no_grad():
                    teacher_masked = self._mask_logits_for_sample_tasks(
                        self.teacher.forward(bx[:rc]), rt
                    )
                    teacher_probs = F.softmax(teacher_masked / self.temp, dim=1)
                loss = loss + self.reg * self.kl(
                    F.log_softmax(student_masked / self.temp, dim=1), teacher_probs
                )

            loss.backward()
            if self.cfg.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), self.cfg.grad_clip_norm
                )

            self.opt_wt.step()

        avg_cls_tr_rec = sum(cls_tr_rec) / len(cls_tr_rec) if cls_tr_rec else 0.0
        return loss, avg_cls_tr_rec

    def _dbg(self, tag, pass_itr, t, bx, prediction, by, bt, replay_count, loss):
        """PROBE-only per-step diagnostics (enable with ERALG4_DEBUG=1)."""
        if os.environ.get("ERALG4_DEBUG") != "1":
            return
        n = getattr(self, "_dbg_steps", 0)
        if n >= 8:
            return
        self._dbg_steps = n + 1
        rc = max(0, min(int(replay_count), prediction.size(0)))
        with torch.no_grad():
            cur = self.take_multitask_loss(bt[rc:], prediction[rc:], by[rc:])
            rep = (
                self.take_multitask_loss(bt[:rc], prediction[:rc], by[:rc])
                if rc > 0
                else torch.zeros((), device=prediction.device)
            )
        print(
            f"[DBG {tag}] step={n} pass={pass_itr} t={t} "
            f"fwd_batch={tuple(bx.shape)} replay_n={rc} cur_n={prediction.size(0) - rc} "
            f"cur_loss={cur.item():.4f} rep_loss={rep.item():.4f} total={loss.item():.4f} "
            f"x_mean={bx.mean().item():.5f} x_std={bx.std().item():.5f}",
            flush=True,
        )

    def _sample_replay(self, device):
        """Sample a noise-excluded replay minibatch from reservoir ``M``.

        Rows in ``M`` are pre-canonicalized (2, L) tensors, so no adapter grad
        flows for replayed samples (mirrors the sister-repo ``_sample_replay``).
        Returns ``(bx, by, bt)`` or ``None`` when no eligible samples exist.
        """
        if len(self.M) == 0:
            return None
        osize = min(self.batchSize, len(self.M))
        replay_x, replay_y, replay_t = [], [], []
        for k in random.choices(range(len(self.M)), k=osize):
            xi, yi, ti = self.M[k]
            yi_scalar = int(torch.as_tensor(yi).long().flatten()[0].item())
            if self.noise_label is not None and yi_scalar == self.noise_label:
                continue
            replay_x.append(torch.as_tensor(np.array(xi)))
            replay_y.append(yi_scalar)
            replay_t.append(int(ti))
        if not replay_x:
            return None
        bx = torch.stack(replay_x).float().to(device, non_blocking=True)
        by = torch.tensor(replay_y, dtype=torch.long, device=device)
        bt = torch.tensor(replay_t, dtype=torch.long, device=device)
        return bx, by, bt

    def ER_joint(self, raw_x, y, t):
        """Sister-repo two-forward ER: current live batch + replay in separate
        forwards, adapter and backbone co-trained in one ``opt_wt`` step.

        ``raw_x`` is the detached leaf current batch; each inner step rebuilds a
        fresh canonicalized (adapter-differentiable) graph so the adapter and
        backbone receive gradients together. Replay rows are pre-canonicalized
        buffer tensors (no adapter grad).
        """
        cls_tr_rec = []
        metric_logits = None
        y = y.long()
        current_t = torch.full(
            (raw_x.size(0),), int(t), dtype=torch.long, device=raw_x.device
        )
        k_avg = max(1, int(self.cfg.eralg4_grad_avg))
        for _pass_itr in range(self.inner_steps):
            self.net.zero_grad()

            # Draw the replay rows once per optimizer step, then evaluate the loss
            # on them k_avg times. Dropout resamples on every forward, so this
            # averages out gradient noise exactly as C-MAML's meta_batches does;
            # k_avg=1 is the historical single-forward path.
            replay = self._sample_replay(raw_x.device)
            k_losses = []
            for _k in range(k_avg):
                live_x = self._canonicalize_input(raw_x, detach=False)
                current_logits = self.net.forward(live_x)
                current_loss = self.take_multitask_loss(current_t, current_logits, y)

                if replay is not None:
                    replay_x, replay_y, replay_t = replay
                    replay_logits = self.net.forward(replay_x)
                    replay_loss = self.take_multitask_loss(
                        replay_t, replay_logits, replay_y
                    )
                else:
                    replay_loss = torch.zeros(
                        (), device=current_logits.device, dtype=current_logits.dtype
                    )
                k_losses.append(current_loss + (self.memory_loss_lambda * replay_loss))

            loss = k_losses[0] if k_avg == 1 else sum(k_losses) / k_avg
            cls_tr_rec.append(self._batch_accuracy(current_t, current_logits, y))
            if replay is not None:
                self._dbg(
                    "JOINT",
                    _pass_itr,
                    t,
                    torch.cat([replay_x, live_x.detach()]),
                    torch.cat([replay_logits.detach(), current_logits.detach()]),
                    torch.cat([replay_y, y]),
                    torch.cat([replay_t, current_t]),
                    replay_x.size(0),
                    loss,
                )
            else:
                self._dbg(
                    "JOINT",
                    _pass_itr,
                    t,
                    live_x.detach(),
                    current_logits.detach(),
                    y,
                    current_t,
                    0,
                    loss,
                )
            metric_logits = misc_utils.apply_task_incremental_logit_mask(
                current_logits.detach(),
                t,
                self.classes_per_task,
                self.n_outputs,
                cil_all_seen_upto_task=t,
                global_noise_label=self.noise_label,
                loader=self.incremental_loader_name,
            )

            loss.backward()
            if self.cfg.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), self.cfg.grad_clip_norm
                )
            self.opt_wt.step()

        avg_cls_tr_rec = sum(cls_tr_rec) / len(cls_tr_rec) if cls_tr_rec else 0.0
        return loss, avg_cls_tr_rec, metric_logits

    def inner_update(self, x, fast_weights, y, t):
        """
        Update the fast weights using the current samples and return the updated fast
        """

        # if self.is_cifar:
        #     offset1, offset2 = self.compute_offsets(t)
        #     logits = self.net.forward(x, fast_weights)[:, :offset2]
        #     loss = self.loss(logits[:, offset1:offset2], y-offset1)
        # else:
        #     logits = self.net.forward(x, fast_weights)
        #     loss = self.loss(logits, y)

        logits = misc_utils.apply_task_incremental_logit_mask(
            self.net.forward(x, fast_weights)[:, : self.n_outputs],
            t,
            self.classes_per_task,
            self.n_outputs,
            cil_all_seen_upto_task=t,
            global_noise_label=self.noise_label,
            loader=self.incremental_loader_name,
        )
        y_cls = unpack_y_to_class_labels(y).long()
        targets = y_cls
        loss = classification_cross_entropy(
            logits,
            targets,
            class_weighted_ce=self.class_weighted_ce,
        )

        if fast_weights is None:
            # fast_weights = self.net.parameters()
            fast_weights = list(self.net.parameters())

        graph_required = self.cfg.second_order

        # Some model parameters are intentionally frozen (e.g. adapter biases with
        # `requires_grad=False`). `torch.autograd.grad` errors if any of the
        # differentiation targets do not require grad, so we differentiate only
        # w.r.t. tensors that are set to require gradients and then reconstruct the
        # full gradient list aligned with `fast_weights`.
        require_grad_targets = [w for w in fast_weights if w.requires_grad]
        if require_grad_targets:
            raw_gradients_subset = torch.autograd.grad(
                loss,
                require_grad_targets,
                create_graph=graph_required,
                retain_graph=graph_required,
                allow_unused=True,
            )
            subset_iter = iter(raw_gradients_subset)
            raw_gradients = [
                next(subset_iter) if w.requires_grad else None for w in fast_weights
            ]
        else:
            raw_gradients = [None for _ in fast_weights]

        grads = [
            grad if grad is not None else torch.zeros_like(weight)
            for grad, weight in zip(raw_gradients, fast_weights)
        ]

        for i in range(len(grads)):
            if self.cfg.grad_clip_norm:
                clip_val = self.cfg.grad_clip_norm
                grads[i] = torch.clamp(grads[i], min=-clip_val, max=clip_val)

        updated_fast_weights = []
        for grad, weight, alpha_lr in zip(grads, fast_weights, self.net.alpha_lr):
            # Preserve frozen tensors exactly; only update tensors that are
            # intended to participate in gradient-based inner updates.
            if not weight.requires_grad:
                updated_fast_weights.append(weight)
                continue
            updated_fast_weights.append(weight - grad * alpha_lr)
        fast_weights = updated_fast_weights
        return fast_weights, loss.item()

    def la_ER(self, raw_x, y, t):
        """
        this ablation tests whether it suffices to just do the learning rate modulation
        guided by gradient alignment + clipping (that La-MAML does implciitly through autodiff)
        and use it with ER (therefore no meta-learning for the weights)

        """
        cls_tr_rec = []
        for pass_itr in range(self.inner_steps):
            # Rebuild a fresh canonicalized tensor each round; previous
            # autograd.grad/backward calls free the old graph.
            x = self._canonicalize_input(raw_x, detach=False)

            perm = torch.randperm(x.size(0))
            x = x[perm]
            if isinstance(y, (list, tuple)):
                y = tuple(yi[perm] if yi is not None else None for yi in y)
            else:
                y = y[perm]

            batch_sz = x.shape[0]
            n_batches = self.cfg.meta_batches
            rough_sz = math.ceil(batch_sz / n_batches)
            fast_weights = None
            meta_losses = [0 for _ in range(n_batches)]

            y_pack = unpack_y_to_class_labels(y)
            bx, by, bt, replay_count = self.getBatch(
                x.detach().cpu().numpy(),
                y_pack.detach().cpu().numpy(),
                t,
            )
            bx = bx.squeeze()

            for i in range(n_batches):

                batch_x = x[i * rough_sz : (i + 1) * rough_sz]
                if isinstance(y, (list, tuple)):
                    batch_y = tuple(
                        (
                            yi[i * rough_sz : (i + 1) * rough_sz]
                            if yi is not None
                            else None
                        )
                        for yi in y
                    )
                else:
                    batch_y = y[i * rough_sz : (i + 1) * rough_sz]

                # assuming labels for inner update are from the same
                fast_weights, inner_loss = self.inner_update(
                    batch_x, fast_weights, batch_y, t
                )

                prediction = self.net.forward(bx, fast_weights)
                meta_loss = self._weighted_multitask_loss(
                    prediction, by, bt, replay_count
                )
                meta_losses[i] += meta_loss

            # update alphas
            self.net.zero_grad()
            self.opt_lr.zero_grad()

            meta_loss = meta_losses[-1]  # sum(meta_losses)/len(meta_losses)
            meta_loss.backward()

            if self.cfg.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), self.cfg.grad_clip_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.net.alpha_lr.parameters(), self.cfg.grad_clip_norm
                )

            # update the LRs (guided by meta-loss, but not the weights)
            self.opt_lr.step()

            # update weights
            self.net.zero_grad()

            # compute ER loss for network weights
            prediction = self.net.forward(bx)
            loss = self._weighted_multitask_loss(prediction, by, bt, replay_count)
            cls_tr_rec.append(self._batch_accuracy(bt, prediction, by))

            loss.backward()

            if self.cfg.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), self.cfg.grad_clip_norm
                )

            # update weights with grad from simple ER loss
            # and LRs obtained from meta-loss guided by old and new tasks
            for i, p in enumerate(self.net.parameters()):
                if p.grad is None:
                    continue
                p.data = p.data - (p.grad * nn.functional.relu(self.net.alpha_lr[i]))
            self.net.zero_grad()
            self.net.alpha_lr.zero_grad()

        avg_cls_tr_rec = sum(cls_tr_rec) / len(cls_tr_rec) if cls_tr_rec else 0.0
        return loss, avg_cls_tr_rec
