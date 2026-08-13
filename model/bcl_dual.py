# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
from model.resnet1d import ResNet1D
from model.detection_replay import (
    DetectionReplayMixin,
    noise_label_from_args,
    signal_mask_exclude_noise,
    unpack_y_to_class_labels,
)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from utils.training_metrics import macro_recall
from utils import misc_utils
from utils.class_weighted_loss import classification_cross_entropy


@dataclass
class BclDualConfig:
    lr: float = 1e-3
    beta: float = 1.0
    memory_strength: float = 1.0
    temperature: float = 5.0
    n_memories: int = 2000
    mem_sampling: str = "ring"
    inner_steps: int = 5
    adapt_inner_steps: int = 5

    cuda: bool = True
    replay_batch_size: int = 20
    det_lambda: float = 10.0
    cls_lambda: float = 1.0
    det_memories: int = 2000
    det_replay_batch: int = 64

    @staticmethod
    def from_args(args: object) -> "BclDualConfig":
        """Build config from CLI / merged YAML.

        ``inner_steps`` counts alternating fast (inner optimizer) / meta rounds per
        ``observe``. Legacy YAML that set both ``inner_steps`` and ``n_meta`` is
        folded into ``inner_steps = inner_steps * n_meta``. ``adapt_inner_steps``
        defaults to the pre-merge ``inner_steps`` so ``adapt()`` SGD depth stays
        stable when only the product changes.
        """
        cfg = BclDualConfig()
        inner_raw = int(getattr(args, "inner_steps", cfg.inner_steps) or 1)
        legacy_n_meta = int(getattr(args, "n_meta", 1) or 1)
        merged_rounds = max(1, inner_raw * legacy_n_meta)
        for field_name in cfg.__dataclass_fields__:
            if field_name == "inner_steps":
                cfg.inner_steps = merged_rounds
                continue
            if field_name == "adapt_inner_steps":
                if hasattr(args, "adapt_inner_steps"):
                    cfg.adapt_inner_steps = max(
                        1, int(getattr(args, "adapt_inner_steps") or 1)
                    )
                else:
                    cfg.adapt_inner_steps = max(1, inner_raw)
                continue
            value = getattr(args, field_name, None)
            # `None` means the argument was registered but never set (the
            # parser gives shared names like `beta` a None default so each
            # model keeps its own), so the dataclass default stands.
            if value is not None:
                setattr(cfg, field_name, value)
        return cfg


class Net(DetectionReplayMixin, torch.nn.Module):

    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        super(Net, self).__init__()
        self.cfg = BclDualConfig.from_args(args)
        self.n_tasks = n_tasks
        self.reg = self.cfg.memory_strength
        self.temp = self.cfg.temperature
        # setup network
        self.is_task_incremental = True
        self.net = ResNet1D(n_outputs, args)

        # setup optimizer
        self.inner_lr = self.cfg.lr
        self.beta = self.cfg.beta
        # self.outer_opt = torch.optim.SGD(self.net.parameters(), lr=self.outer_lr)
        self.inner_opt = torch.optim.SGD(
            self.net.parameters(), lr=self.inner_lr, momentum=0.9
        )
        self.class_weighted_ce = bool(getattr(args, "class_weighted_ce", True))
        self.det_lambda = float(self.cfg.det_lambda)
        self.cls_lambda = float(self.cfg.cls_lambda)
        self._init_det_replay(
            self.cfg.det_memories,
            self.cfg.det_replay_batch,
            enabled=bool(getattr(args, "use_detector_arch", False)),
        )

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
        self.noise_label = noise_label_from_args(args)
        self.incremental_loader_name = getattr(args, "loader", None)
        # setup memories: n_memories = total buffer size, split across tasks
        self.current_task = 0
        self.fisher = {}
        self.optpar = {}
        total_memories = int(self.cfg.n_memories)
        self.task_total_capacities = self._build_task_memory_capacities(
            total_memories,
            n_tasks,
        )
        val_fraction = 0.2
        self.task_val_capacities = [
            max(0, int(cap * val_fraction)) for cap in self.task_total_capacities
        ]
        self.task_replay_capacities = [
            cap - val_cap
            for cap, val_cap in zip(
                self.task_total_capacities, self.task_val_capacities
            )
        ]
        self.max_task_replay_capacity = max(self.task_replay_capacities, default=0)
        self.max_task_val_capacity = max(self.task_val_capacities, default=0)

        self.memx = torch.FloatTensor(
            n_tasks, self.max_task_replay_capacity, 2, n_inputs // 2
        )
        self.valx = torch.FloatTensor(
            n_tasks, self.max_task_val_capacity, 2, n_inputs // 2
        )
        self.memy = torch.LongTensor(n_tasks, self.max_task_replay_capacity)
        self.mem_feat = torch.FloatTensor(
            n_tasks, self.max_task_replay_capacity, self.nc_per_task
        ).fill_(0)
        self.valy = torch.LongTensor(n_tasks, self.max_task_val_capacity)
        self.mem = {}
        if self.cfg.cuda:
            self.memy = self.memy.cuda().fill_(0)
            self.memx = self.memx.cuda().fill_(0)
            self.mem_feat = self.mem_feat.cuda()
            self.valx = self.valx.cuda().fill_(0)
            self.valy = self.valy.cuda().fill_(0)

        self.task_mem_ptr = torch.zeros(
            n_tasks, dtype=torch.long, device=self.memx.device
        )
        self.task_mem_filled = torch.zeros(
            n_tasks, dtype=torch.long, device=self.memx.device
        )
        # total stream items seen per task (reservoir sampling only)
        self.task_mem_seen = torch.zeros(
            n_tasks, dtype=torch.long, device=self.memx.device
        )
        self.mem_sampling = self.cfg.mem_sampling
        if self.mem_sampling not in misc_utils.MEM_SAMPLING_MODES:
            raise ValueError(
                f"mem_sampling must be one of {misc_utils.MEM_SAMPLING_MODES}, "
                f"got {self.mem_sampling!r}"
            )
        self.task_val_ptr = torch.zeros(
            n_tasks, dtype=torch.long, device=self.valx.device
        )
        self.task_val_filled = torch.zeros(
            n_tasks, dtype=torch.long, device=self.valx.device
        )
        self.bsz = args.batch_size
        self.valid_id = []
        self.n_outputs = n_outputs
        self.n_memories = total_memories  # total buffer size (for logging/config)

        self.mse = nn.MSELoss()
        # Use batchmean to match KL definition and remove PyTorch warning
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.samples_seen = 0
        self.sz = int(self.cfg.replay_batch_size)
        self.inner_steps = self.cfg.inner_steps
        self.adapt_inner_steps = self.cfg.adapt_inner_steps
        self.adapt_ = False  # args.adapt
        self.adapt_lr = self.cfg.lr
        self.models = {}
        # Per-task ``adapt()`` checkpoints are TIL-only; class-incremental runs
        # should use the shared backbone at inference.
        self.use_cil_inference: bool = (
            getattr(args, "loader", "") == "class_incremental_loader"
        )

        # print(f"task_val_capacities: {self.task_val_capacities}")
        # print(f"task_replay_capacities: {self.task_replay_capacities}")
        # print(f"max_task_replay_capacity: {self.max_task_replay_capacity}")
        # print(f"max_task_val_capacity: {self.max_task_val_capacity}")
        # print(f"task_total_capacities: {self.task_total_capacities}")

    def on_epoch_end(self):
        pass

    def _build_task_memory_capacities(
        self, total_memories: int, n_tasks: int
    ) -> list[int]:
        """Split total replay budget across tasks so sum(capacities) == total_memories."""
        if n_tasks <= 0:
            return []
        base = total_memories // n_tasks
        remainder = total_memories % n_tasks
        return [base + (1 if i < remainder else 0) for i in range(n_tasks)]

    def adapt(self):
        print("Adapting")
        for t in range(self.n_tasks):
            model = deepcopy(self.net)
            if t > self.current_task:
                self.models[t] = model
                continue
            filled = int(self.task_mem_filled[t].item())
            if filled <= 0:
                self.models[t] = model
                continue
            xx = self.memx[t, :filled]
            yy = self.memy[t, :filled]
            opt = torch.optim.SGD(model.parameters(), self.adapt_lr, momentum=0.9)
            for _ in range(self.adapt_inner_steps):
                model.zero_grad()
                pred = model.forward(xx)
                loss = classification_cross_entropy(
                    pred, yy, class_weighted_ce=self.class_weighted_ce
                )
                loss.backward()
                opt.step()
            self.models[t] = model

    def compute_offsets(self, task):
        if self.is_task_incremental:
            return misc_utils.compute_offsets(task, self.classes_per_task)
        else:
            return 0, self.n_outputs

    def forward(self, x, t, return_feat=False, *, cil_all_seen_upto_task=None):
        use_task_adapted = (
            self.adapt_
            and not self.net.training
            and not self.use_cil_inference
            and t in self.models
        )
        if use_task_adapted:
            output = self.models[t](x)
        else:
            output = self.net(x)

        if self.is_task_incremental:
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

    def memory_sampling(self, t: int, valid: bool = False):
        """Sample from validation (valid=True) or replay (valid=False) buffers.

        Uses per-task filled counts so only valid slots are sampled.
        """
        n_tasks = t
        if valid:
            filled = [int(self.task_val_filled[i].item()) for i in range(n_tasks)]
            mem_x = self.valx[:n_tasks]
            mem_y = self.valy[:n_tasks]
            mem_feat = self.mem_feat[:n_tasks]
        else:
            filled = [int(self.task_mem_filled[i].item()) for i in range(n_tasks)]
            mem_x = self.memx[:n_tasks]
            mem_y = self.memy[:n_tasks]
            mem_feat = self.mem_feat[:n_tasks]

        if sum(filled) == 0:
            return None

        # Build flat index -> (task_idx, slot_idx); skip padding / global noise for CE
        flat_to_task_slot = []
        for task_idx in range(n_tasks):
            for slot in range(filled[task_idx]):
                lab = int(mem_y[task_idx, slot].item())
                if lab < 0:
                    continue
                if self.noise_label is not None and lab == self.noise_label:
                    continue
                flat_to_task_slot.append((task_idx, slot))
        flat_to_task_slot = np.array(flat_to_task_slot)
        total_filled = len(flat_to_task_slot)
        if total_filled == 0:
            return None

        sz = min(total_filled, self.sz)
        chosen = np.random.choice(total_filled, size=sz, replace=False)
        t_idx_np = flat_to_task_slot[chosen, 0]
        s_idx_np = flat_to_task_slot[chosen, 1]
        if valid:
            self.valid_id = chosen.tolist()

        device = mem_x.device
        t_idx = torch.from_numpy(t_idx_np).to(device)
        s_idx = torch.from_numpy(s_idx_np).to(device)

        offsets = torch.tensor(
            [self.compute_offsets(int(i)) for i in t_idx.tolist()],
            device=mem_x.device,
            dtype=torch.long,
        )
        xx = mem_x[t_idx, s_idx]
        yy = mem_y[t_idx, s_idx] - offsets[:, 0]
        feat = mem_feat[t_idx, s_idx]
        mask = torch.zeros(xx.size(0), self.nc_per_task, device=xx.device)
        for j in range(mask.size(0)):
            cls_size = offsets[j][1] - offsets[j][0]
            mask[j, :cls_size] = torch.arange(
                offsets[j][0], offsets[j][1], device=xx.device
            )
        mask = mask.long()
        sizes = (offsets[:, 1] - offsets[:, 0]).long()
        return xx, yy, feat, mask, t_idx.tolist(), sizes

    def _store_replay(
        self,
        task_id: int,
        x_src: torch.Tensor,
        y_src: torch.Tensor,
        capacity: int,
    ):
        """Write already-adapted new samples into task_id's replay buffer.

        Dispatches on ``self.mem_sampling``: ``"ring"`` overwrites the oldest
        samples via a wrapping write pointer; ``"reservoir"`` keeps a uniform
        random sample of the whole task stream (see
        :func:`utils.misc_utils.reservoir_slots`). Occupied slots stay dense in
        ``[0, task_mem_filled[task_id])`` so distillation/replay slicing is
        unchanged; the matching ``mem_feat`` targets are refreshed at the next
        task boundary as before.
        """
        if self.mem_sampling == "reservoir":
            slots, filled, seen = misc_utils.reservoir_slots(
                x_src.size(0),
                int(self.task_mem_filled[task_id].item()),
                int(self.task_mem_seen[task_id].item()),
                capacity,
            )
            for i, slot in enumerate(slots):
                if slot >= 0:
                    self.memx[task_id, slot].copy_(x_src[i])
                    self.memy[task_id, slot] = y_src[i]
            self.task_mem_filled[task_id] = filled
            self.task_mem_seen[task_id] = seen
            return

        write_pointer = int(self.task_mem_ptr[task_id].item())
        endcnt = min(write_pointer + x_src.size(0), capacity)
        effbsz = endcnt - write_pointer
        if effbsz > 0:
            self.memx[task_id, write_pointer:endcnt].copy_(x_src[:effbsz])
            self.memy[task_id, write_pointer:endcnt].copy_(y_src[:effbsz])
            self.task_mem_filled[task_id] = min(
                capacity, int(self.task_mem_filled[task_id].item()) + effbsz
            )
        self.task_mem_ptr[task_id] = 0 if endcnt == capacity else endcnt

    def observe(self, x, y, t):
        # noise_label = None
        # if class_counts is not None:
        #     _, offset2 = misc_utils.compute_offsets(t, class_counts)
        #     noise_label = offset2 - 1
        # y_cls, y_det = self._unpack_labels(
        #     y,
        #     noise_label=noise_label,
        #     use_detector_arch=bool(getattr(self, "det_enabled", False)),
        # )
        if self.current_task is None:
            self.current_task = t
        # if (
        #     self.current_task == t
        #     and int(self.task_mem_filled[t].item()) == 0
        #     and y_det is not None
        # ):
        #     det_mean = float(y_det.float().mean().item())
        #     if det_mean >= 0.99:
        #         print("[WARN][BCL_Dual] y_det mean ~1.0; detection may be missing negatives.")
        # if y_det is not None and self.det_memories > 0:
        #     self._update_det_memory(x, y_det)
        # x_det = x
        # signal_mask = (y_det == 1) & (y_cls >= 0)
        # if not signal_mask.any():
        #     print("[WARN][BCL_Dual] No signal mask; skipping detection.")
        #     if not getattr(self, "det_enabled", True):
        #         return 0.0, 0.0
        #     det_logits, _ = self.net.forward_heads(x_det)
        #     det_loss = self.det_loss(det_logits, y_det.float())
        #     det_replay = self._sample_det_memory()
        #     if det_replay is not None:
        #         mem_x, mem_y = det_replay
        #         mem_det_logits, _ = self.net.forward_heads(mem_x)
        #         mem_loss = self.det_loss(mem_det_logits, mem_y.float())
        #         det_loss = 0.5 * (det_loss + mem_loss)
        #     self.zero_grad()
        #     det_loss.backward()
        #     self.inner_opt.step()
        #     return float(det_loss.item()), 0.0
        # x = x[signal_mask]
        # y = y_cls[signal_mask]
        raw_x_train = x.detach().requires_grad_(True)
        x_train = self._canonicalize_input(raw_x_train, detach=False)
        x_for_storage = self._input_for_replay(x)
        y_work = unpack_y_to_class_labels(y).long()
        if t != self.current_task:
            tt = self.current_task
            previous_filled = int(self.task_mem_filled[tt].item())
            if previous_filled > 0:
                offset1, offset2 = self.compute_offsets(tt)
                out = self.forward(self.memx[tt, :previous_filled], tt, True)
                cls_size = int(offset2 - offset1)
                feat = self.mem_feat[tt, :previous_filled]
                feat.zero_()
                feat[:, :cls_size] = F.softmax(
                    out[:, offset1:offset2] / self.temp, dim=1
                ).data.clone()
            self.current_task = t

        # Validation set (ring buffer per task); store adapted input in val buffer
        n_val_taken = 0
        n_rotated_in = 0  # old val sample cat'd back into batch; not in x_for_storage
        rotated_validation_sample_for_meta = None
        task_val_capacity = int(self.task_val_capacities[t])
        if task_val_capacity > 0 and x_train.size(0) > 0:
            n_val_taken = 1
            _, valy = x_train[0], y_work[0]
            raw_x_train = raw_x_train[1:]
            x_train, y_work = x_train[1:], y_work[1:]
            val_write = int(self.task_val_ptr[t].item())
            # Only rotate in when we're overwriting a slot that has valid data (buffer full)
            val_filled = int(self.task_val_filled[t].item())
            if val_filled >= task_val_capacity:
                n_rotated_in = 1
                rotated_validation_sample_for_meta = self.valx[t, val_write].unsqueeze(
                    0
                )
                x_train = torch.cat([x_train, rotated_validation_sample_for_meta])
                y_work = torch.cat([y_work, self.valy[t, val_write].unsqueeze(0)])
            self.valx[t, val_write].copy_(x_for_storage[0])
            self.valy[t, val_write].copy_(valy)
            self.task_val_filled[t] = min(
                task_val_capacity,
                int(self.task_val_filled[t].item()) + 1,
            )
            self.task_val_ptr[t] = (
                0 if (val_write + 1) == task_val_capacity else (val_write + 1)
            )
            if x_train.size(0) == 0:
                x_train = x_for_storage[0].unsqueeze(0)
                y_work = valy.unsqueeze(0)

        # Replay memory (ring buffer per task); store adapted input
        # Only the "new" samples go to replay; rotated-in val sample is already in val buffer
        self.net.train()
        task_replay_capacity = int(self.task_replay_capacities[t])
        if task_replay_capacity > 0 and y_work.size(0) > 0:
            # Exclude the rotated-in val sample; it already lives in the val buffer.
            n_new = y_work.size(0) - n_rotated_in
            if n_new > 0:
                replay_start = n_val_taken
                self._store_replay(
                    t,
                    x_for_storage[replay_start : replay_start + n_new],
                    y_work[:n_new],
                    task_replay_capacity,
                )

        self.zero_grad()
        tt = t + 1
        cls_tr_rec = []
        for _ in range(self.inner_steps):
            x_train = self._canonicalize_input(raw_x_train, detach=False)
            if rotated_validation_sample_for_meta is not None:
                x_train = torch.cat(
                    [x_train, rotated_validation_sample_for_meta], dim=0
                )
            weights_before = deepcopy(self.net.state_dict())
            pred = self.forward(x_train, t)
            signal_mask = signal_mask_exclude_noise(y_work, self.noise_label)
            logits_for_loss = pred
            if self.is_task_incremental:
                logits_for_loss = misc_utils.apply_task_incremental_logit_mask(
                    pred,
                    t,
                    self.nc_per_task,
                    self.n_outputs,
                    cil_all_seen_upto_task=t,
                    global_noise_label=self.noise_label,
                )
            targets = y_work.long()
            preds = torch.argmax(logits_for_loss, dim=1)
            if signal_mask.any():
                cls_tr_rec.append(
                    macro_recall(preds[signal_mask], targets[signal_mask])
                )
            else:
                cls_tr_rec.append(0.0)
            loss1 = classification_cross_entropy(
                logits_for_loss,
                targets,
                class_weighted_ce=self.class_weighted_ce,
            )
            if t > 0:
                sampled = self.memory_sampling(t)
                if sampled is not None:
                    xx, yy, feat, mask, list_t, class_sizes = sampled
                    pred_ = self.net(xx)
                    pred = torch.gather(pred_, 1, mask)
                    for row, size in enumerate(class_sizes):
                        if size < pred.size(1):
                            pred[row, size:] = -1e9
                    loss2 = classification_cross_entropy(
                        pred, yy, class_weighted_ce=self.class_weighted_ce
                    )
                    loss3 = self.reg * self.kl(
                        F.log_softmax(pred / self.temp, dim=1), feat
                    )
                    loss = self.cls_lambda * loss1 + loss2 + loss3
                else:
                    loss = self.cls_lambda * loss1
            else:
                loss = self.cls_lambda * loss1
            loss.backward()
            self.inner_opt.step()
            sampled_validation = self.memory_sampling(tt, valid=True)
            if sampled_validation is not None:
                xval, yval, _, mask_val, list_t, class_sizes_val = sampled_validation
                pred_ = self.net(xval)
                pred = torch.gather(pred_, 1, mask_val)
                for row, size in enumerate(class_sizes_val):
                    if size < pred.size(1):
                        pred[row, size:] = -1e9
                outer_loss = classification_cross_entropy(
                    pred, yval, class_weighted_ce=self.class_weighted_ce
                )
            else:
                x_train = self._canonicalize_input(raw_x_train, detach=False)
                if rotated_validation_sample_for_meta is not None:
                    x_train = torch.cat(
                        [x_train, rotated_validation_sample_for_meta], dim=0
                    )
                pred = self.forward(x_train, t)
                logits_outer_for_loss = pred
                if self.is_task_incremental:
                    logits_outer_for_loss = (
                        misc_utils.apply_task_incremental_logit_mask(
                            pred,
                            t,
                            self.nc_per_task,
                            self.n_outputs,
                            cil_all_seen_upto_task=t,
                            global_noise_label=self.noise_label,
                        )
                    )
                outer_loss = classification_cross_entropy(
                    logits_outer_for_loss,
                    targets,
                    class_weighted_ce=self.class_weighted_ce,
                )
            outer_loss.backward()
            self.inner_opt.step()
            self.zero_grad()
            weights_after = self.net.state_dict()
            new_params = {
                name: weights_before[name]
                + ((weights_after[name] - weights_before[name]) * self.beta)
                for name in weights_before.keys()
            }
            self.net.load_state_dict(new_params)
        avg_cls_tr_rec = sum(cls_tr_rec) / len(cls_tr_rec) if cls_tr_rec else 0.0
        return outer_loss.item(), avg_cls_tr_rec, None
