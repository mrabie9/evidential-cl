# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
from model.resnet1d import ResNet1D
from model.replay_utils import (
    ReplayInputMixin,
    unpack_y_to_class_labels,
)
from model.task_bn import (
    forward_replay_and_current,
    frozen_running_stats,
    replay_forward_is_joint,
)
import torch.nn as nn
import numpy as np
from utils.training_metrics import macro_recall
from utils import misc_utils
from utils.class_weighted_loss import classification_cross_entropy


@dataclass
class ErRingConfig:
    memory_strength: float = 1.0
    lr: float = 1e-3
    n_memories: int = 2000
    replay_batch_size: int = 20
    inner_steps: int = 5

    batch_size: int = 128
    cuda: bool = True
    # temperature: float = 2.0
    cls_lambda: float = 1.0
    memory_loss_lambda: float = 1.0

    @staticmethod
    def from_args(args: object) -> "ErRingConfig":
        cfg = ErRingConfig()
        for field in cfg.__dataclass_fields__:
            if hasattr(args, field):
                setattr(cfg, field, getattr(args, field))
        return cfg


class Net(ReplayInputMixin, torch.nn.Module):

    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        super(Net, self).__init__()
        self.cfg = ErRingConfig.from_args(args)
        self.reg = self.cfg.memory_strength
        # self.temp = self.cfg.temperature
        # setup network
        self.is_task_incremental = True
        self.net = ResNet1D(n_outputs, args)
        # setup optimizer
        self.lr = self.cfg.lr
        # if self.is_task_incremental:
        #    self.opt = torch.optim.Adam(self.net.parameters(), lr='self.lr)
        # else:
        self.opt = torch.optim.SGD(self._ll_params(), lr=self.lr, momentum=0.9)

        self.class_weighted_ce = bool(getattr(args, "class_weighted_ce", True))
        self.cls_lambda = float(self.cfg.cls_lambda)
        self.memory_loss_lambda = float(self.cfg.memory_loss_lambda)

        self.classes_per_task = misc_utils.build_task_class_list(
            n_tasks,
            n_outputs,
            nc_per_task=getattr(args, "nc_per_task_list", "")
            or getattr(args, "nc_per_task", None),
            classes_per_task=getattr(args, "classes_per_task", None),
        )
        if self.is_task_incremental:
            self.nc_per_task = misc_utils.max_task_class_count(self.classes_per_task)
        else:
            self.nc_per_task = n_outputs
        self.incremental_loader_name = getattr(args, "loader", None)
        # setup memories
        self.current_task = 0
        self.fisher = {}
        self.optpar = {}
        self.n_memories = int(self.cfg.n_memories)
        self.task_memory_capacities = self._build_task_memory_capacities(
            self.n_memories,
            n_tasks,
        )
        self.max_task_memories = max(self.task_memory_capacities, default=0)

        # Replay buffer stores canonical shape (2, 512) from _input_for_replay (2-channel or adapter output).
        seq_len = n_inputs // 2
        self.memx = torch.FloatTensor(
            n_tasks, self.max_task_memories, 2, seq_len
        ).fill_(0)
        self.memy = torch.LongTensor(n_tasks, self.max_task_memories).fill_(-1)
        self.mem_feat = torch.FloatTensor(
            n_tasks, self.max_task_memories, self.nc_per_task
        ).fill_(0)
        self.mem = {}
        if self.cfg.cuda:
            self.memx = self.memx.cuda()
            self.memy = self.memy.cuda()
            self.mem_feat = self.mem_feat.cuda()
        self.bsz = self.cfg.batch_size
        self.task_mem_filled = torch.zeros(n_tasks, dtype=torch.long)
        self.task_mem_ptr = torch.zeros(n_tasks, dtype=torch.long)
        if self.cfg.cuda:
            self.task_mem_filled = self.task_mem_filled.cuda()
            self.task_mem_ptr = self.task_mem_ptr.cuda()

        self.n_outputs = n_outputs

        self.mse = nn.MSELoss()
        # Use batchmean to align with KL math and silence PyTorch warning
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.samples_seen = 0
        self.sz = int(self.cfg.replay_batch_size)
        self.inner_steps = self.cfg.inner_steps

    def on_epoch_end(self):
        pass

    def _build_task_memory_capacities(
        self, total_memories: int, n_tasks: int
    ) -> list[int]:
        """Split a total replay budget across tasks.

        Args:
            total_memories: Total replay-buffer capacity.
            n_tasks: Number of tasks in the stream.

        Returns:
            Per-task capacities whose sum equals `total_memories`.
        """
        if n_tasks <= 0:
            return []
        base_capacity = total_memories // n_tasks
        remainder_capacity = total_memories % n_tasks
        return [
            base_capacity + (1 if task_index < remainder_capacity else 0)
            for task_index in range(n_tasks)
        ]

    def compute_offsets(self, task):
        if self.is_task_incremental:
            return misc_utils.compute_offsets(task, self.classes_per_task)
        else:
            return 0, self.n_outputs

    def _ll_params(self):
        for name, param in self.net.named_parameters():
            yield param

    def forward(self, x, t, return_feat=False, *, cil_all_seen_upto_task=None):
        output = self.net(x)

        if self.is_task_incremental:
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

    def memory_sampling(self, t):
        # Vectorized construction of valid (task, slot) replay indices. The
        # row-major ``nonzero`` ordering (task-major, then slot) is identical to
        # the original nested Python loop, so ``np.random.choice`` selects the
        # same rows — but without one GPU sync per filled slot.
        device = self.memx.device
        filled = self.task_mem_filled[:t]
        if int(filled.sum().item()) == 0:
            return None
        labs = self.memy[:t]
        slot_ids = torch.arange(labs.size(1), device=device).unsqueeze(0)
        valid = (slot_ids < filled.unsqueeze(1)) & (labs >= 0)
        tk, sm = torch.nonzero(valid, as_tuple=True)
        n_valid = int(tk.numel())
        if n_valid == 0:
            return None
        sz = int(min(n_valid, self.sz))
        flat_indices = np.random.choice(n_valid, sz, replace=False)
        sel = torch.as_tensor(flat_indices, device=device, dtype=torch.long)
        t_idx = tk[sel]
        s_idx = sm[sel]

        offsets = torch.tensor(
            [self.compute_offsets(int(i)) for i in t_idx.tolist()],
            device=self.memx.device,
        )
        xx = self.memx[t_idx, s_idx]
        yy_global = self.memy[t_idx, s_idx]
        yy = yy_global - offsets[:, 0]
        feat = self.mem_feat[t_idx, s_idx]
        mask = torch.zeros(xx.size(0), self.nc_per_task, device=self.memx.device)
        for j in range(mask.size(0)):
            cls_size = offsets[j][1] - offsets[j][0]
            mask[j, :cls_size] = torch.arange(
                offsets[j][0], offsets[j][1], device=self.memx.device
            )
        sizes = (offsets[:, 1] - offsets[:, 0]).long()
        return xx, yy, feat, mask.long(), sizes, t_idx, yy_global

    def _forward_current_and_replay(
        self, x: torch.Tensor, t: int, replay_rows: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward the current batch and, if drawn, the replay rows.

        Under CIL the two blocks share one forward so BatchNorm normalizes the
        current task with the mixed-batch statistics it meets at evaluation
        (:func:`model.task_bn.replay_forward_is_joint`). Otherwise the replay
        rows, which come from earlier tasks, are normalized with their own batch
        statistics without writing task ``t``'s running buffers.

        Args:
            x: Current minibatch (raw input format).
            t: Task being trained.
            replay_rows: Canonical replay inputs, or ``None`` when none were drawn.

        Returns:
            ``(current_logits, replay_raw)``: current logits masked to tasks
            ``0..t``, and unmasked replay logits (``None`` without replay).
        """
        if replay_rows is None:
            return self.forward(x, t, True, cil_all_seen_upto_task=t), None
        if not replay_forward_is_joint(self.incremental_loader_name):
            current_logits = self.forward(x, t, True, cil_all_seen_upto_task=t)
            with frozen_running_stats(self):
                replay_raw = self.net(replay_rows)
            return current_logits, replay_raw
        replay_raw, current_raw = forward_replay_and_current(
            self,
            self.net,
            replay_rows,
            self._canonicalize_input(x, detach=False),
            joint=True,
        )
        current_logits = misc_utils.apply_task_incremental_logit_mask(
            current_raw,
            t,
            self.classes_per_task,
            self.n_outputs,
            cil_all_seen_upto_task=t,
            fill_value=-10e10,
            loader=self.incremental_loader_name,
        )
        return current_logits, replay_raw

    def observe(self, x, y, t):
        # t = info[0]
        # idx = info[1]
        self.net.train()

        y_work = unpack_y_to_class_labels(y).long()
        task_capacity = int(self.task_memory_capacities[t])
        if task_capacity > 0:
            write_pointer = int(self.task_mem_ptr[t].item())
            bsz = y_work.data.size(0)
            endcnt = min(write_pointer + bsz, task_capacity)
            effbsz = endcnt - write_pointer
            if effbsz > 0:
                x_for_storage = self._input_for_replay(x)
                self.memx[t, write_pointer:endcnt].copy_(x_for_storage.data[:effbsz])
                self.memy[t, write_pointer:endcnt].copy_(y_work.data[:effbsz])
                filled_before_update = int(self.task_mem_filled[t].item())
                self.task_mem_filled[t] = min(
                    task_capacity, filled_before_update + effbsz
                )
            self.task_mem_ptr[t] = 0 if endcnt == task_capacity else endcnt

        if t != self.current_task:
            tt = self.current_task
            offset1, offset2 = self.compute_offsets(tt)
            # out = self.forward(self.memx[tt],tt, True)
            # self.mem_feat[tt] = F.softmax(out[:, offset1:offset2] / self.temp, dim=1 ).data.clone()
            self.current_task = t

        cls_tr_rec = []
        metric_logits = None

        for _ in range(self.inner_steps):
            self.net.zero_grad()
            loss1 = torch.tensor(0.0).cuda()
            loss2 = torch.tensor(0.0).cuda()

            offset1, offset2 = self.compute_offsets(t)
            sampled = self.memory_sampling(t) if t > 0 else None
            replay_rows = None if sampled is None else sampled[0]
            logits, replay_raw = self._forward_current_and_replay(x, t, replay_rows)
            targets = y_work.long()
            preds = torch.argmax(logits, dim=1)
            cls_tr_rec.append(macro_recall(preds, targets))
            loss1 = classification_cross_entropy(
                logits,
                targets,
                class_weighted_ce=self.class_weighted_ce,
            )
            if t > 0:
                if sampled is not None:
                    xx, yy, target, mask, class_sizes, t_idx, yy_global = sampled
                    pred_ = replay_raw
                    if self.incremental_loader_name == "class_incremental_loader":
                        # CIL: replayed rows compete with every class seen so
                        # far, as the current batch does; a per-task block would
                        # never push them away from newer classes.
                        pred = misc_utils.mask_replay_logits(
                            pred_,
                            t_idx,
                            t,
                            self.classes_per_task,
                            self.n_outputs,
                            loader=self.incremental_loader_name,
                        )
                        yy = yy_global
                    else:
                        pred = torch.gather(pred_, 1, mask)
                        for row, size in enumerate(class_sizes):
                            if size < pred.size(1):
                                pred[row, size:] = -1e9
                    if yy.min() < 0 or yy.max() >= pred.size(1):
                        raise ValueError(
                            f"Replay target out of range: min={int(yy.min())}, max={int(yy.max())}, "
                            f"class_count={pred.size(1)}, sizes={class_sizes.tolist()}"
                        )
                    loss2 += classification_cross_entropy(
                        pred, yy, class_weighted_ce=self.class_weighted_ce
                    )

            loss = loss1 + (self.memory_loss_lambda * loss2)
            loss.backward()
            self.opt.step()
            metric_logits = logits.detach()

        avg_cls_tr_rec = sum(cls_tr_rec) / len(cls_tr_rec) if cls_tr_rec else 0.0
        return loss.item(), avg_cls_tr_rec, metric_logits
