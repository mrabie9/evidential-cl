# CTN-GEM (flagship B2): the full CTN learner (meta-learned task-embedding->FiLM head, KL
# distillation, validation rotation) with GEM's QP gradient-projection constraint applied to the
# *shared trunk* update only (net.base_param() in the fast step). FiLM/context meta-learning is
# left untouched. Rationale (docs/ablation_findings.md): CTN already supplies high F1 + positive
# BWT via replay/distillation + FiLM stability; GEM's QP adds trunk-drift protection. This is the
# principled sibling of B1 (model/gem_ctn.py), which failed by QP-projecting FiLM under a single
# joint optimizer.

from dataclasses import dataclass

import torch

# from .common import ContextMLP, ContextNet18
# from .resnet import ResNet18 as ResNet18Full
from model.ctn_base import ContextNet18
from model.gem import project2cone2
from model.detection_replay import (
    DetectionReplayMixin,
    noise_label_from_args,
    signal_mask_exclude_noise,
    unpack_y_to_class_labels,
)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.training_metrics import macro_recall
from utils import misc_utils
from utils.class_weighted_loss import classification_cross_entropy


@dataclass
class CtnConfig:
    memory_strength: float = 0.5
    temperature: float = 5.0
    ctn_disable_distill: bool = False  # ablation: drop the KL-distillation replay term
    gem_margin: float = 0.5  # QP margin for the GEM trunk constraint (B2)
    gem_disable_qp: bool = False  # ablation: disable the GEM trunk projection
    task_emb: int = 64
    lr: float = 0.01
    ctx_lr: float = 0.05
    n_memories: int = 50
    validation: float = 0.0
    replay_batch_size: int = 20
    inner_steps: int = 2
    arch: str = "resnet1d"
    cuda: bool = True
    batch_size: int = 128
    det_lambda: float = 1.0
    cls_lambda: float = 1.0
    det_memories: int = 2000
    det_replay_batch: int = 64

    @staticmethod
    def from_args(args: object) -> "CtnConfig":
        """Build config from CLI / merged YAML.

        ``inner_steps`` is the number of alternating rounds per ``observe`` call
        (one fast/base gradient step, then one meta/context step). For YAML that
        still sets the deprecated pair ``inner_steps`` × ``n_meta`` (nested
        loops), the effective count is the product so total fast updates match
        the old schedule.
        """
        cfg = CtnConfig()
        inner_raw = int(getattr(args, "inner_steps", cfg.inner_steps) or 1)
        legacy_n_meta = int(getattr(args, "n_meta", 1) or 1)
        merged_rounds = max(1, inner_raw * legacy_n_meta)
        for field_name in cfg.__dataclass_fields__:
            if field_name == "inner_steps":
                cfg.inner_steps = merged_rounds
                continue
            if hasattr(args, field_name):
                setattr(cfg, field_name, getattr(args, field_name))
        return cfg


class Net(DetectionReplayMixin, torch.nn.Module):

    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        super(Net, self).__init__()
        self.cfg = CtnConfig.from_args(args)
        self.reg = self.cfg.memory_strength
        self.temp = self.cfg.temperature
        # Ablation toggle: when False, skip the KL-distillation replay term (loss3) while
        # keeping plain replay CE (loss2), isolating distillation's contribution to BWT.
        self.use_distill = not bool(self.cfg.ctn_disable_distill)
        # setup network
        if self.cfg.arch == "resnet1d":
            # self.net = ResNet1D(n_outputs, args)
            use_iq_aug_features = bool(getattr(args, "use_iq_aug_features", False))
            iq_aug_scaling_mode = str(getattr(args, "data_scaling", "none"))
            iq_aug_feature_type = str(
                getattr(
                    args,
                    "iq_aug_feature_type",
                    getattr(args, "iq_aug_feature", "power"),
                )
            )
            in_channels = 3 if use_iq_aug_features else 2
            use_film = not bool(getattr(args, "ctn_disable_film", False))
            self.net = ContextNet18(
                n_outputs,
                in_channels=in_channels,
                n_tasks=n_tasks,
                task_emb=self.cfg.task_emb,
                use_iq_aug_features=use_iq_aug_features,
                iq_aug_scaling_mode=iq_aug_scaling_mode,
                iq_aug_feature_type=iq_aug_feature_type,
                use_film=use_film,
            )
        # self.net.define_task_lr_params(alpha_init=args.alpha_init)
        else:
            raise NotImplementedError(
                f"Unsupported arch {self.cfg.arch}; only resnet1d is available now."
            )

        self.is_task_incremental = True
        self.inner_lr = self.cfg.lr
        self.outer_lr = self.cfg.ctx_lr
        self.opt = torch.optim.SGD(
            self.net.parameters(), lr=self.outer_lr, momentum=0.9
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
        # setup memories
        self.current_task = 0
        self.fisher = {}
        self.optpar = {}
        self.n_memories = int(self.cfg.n_memories)
        self.task_total_capacities = self._build_task_memory_capacities(
            self.n_memories,
            n_tasks,
        )
        self.task_val_capacities = [
            int(task_capacity * self.cfg.validation)
            for task_capacity in self.task_total_capacities
        ]
        self.task_replay_capacities = [
            task_capacity - task_val
            for task_capacity, task_val in zip(
                self.task_total_capacities, self.task_val_capacities
            )
        ]
        self.max_task_replay_capacity = max(self.task_replay_capacities, default=0)
        self.max_task_val_capacity = max(self.task_val_capacities, default=0)

        # set up the semantic memory
        self.full_val = True  # avoid OOM when using too large memory

        # if 'cub' in args.data_file:
        #     self.memx = torch.FloatTensor(n_tasks, self.n_memories, 3, 224, 224)
        #     self.valx = torch.FloatTensor(n_tasks, self.n_val, 3, 224, 224)
        # elif 'mini' in args.data_file or 'core' in args.data_file:
        #     self.memx = torch.FloatTensor(n_tasks, self.n_memories, 3, 84, 84)
        #     self.valx = torch.FloatTensor(n_tasks, self.n_val , 3, 84, 84)
        #     if self.n_memories > 75:
        #         self.full_val = False
        # else:
        self.memx = torch.FloatTensor(
            n_tasks, self.max_task_replay_capacity, 2, n_inputs // 2
        )
        self.valx = torch.FloatTensor(
            n_tasks, self.max_task_val_capacity, 2, n_inputs // 2
        )

        self.memy = torch.LongTensor(n_tasks, self.max_task_replay_capacity)
        self.valy = torch.LongTensor(n_tasks, self.max_task_val_capacity)
        self.mem_feat = torch.FloatTensor(
            n_tasks, self.max_task_replay_capacity, self.nc_per_task
        )
        self.mem = {}
        if self.cfg.cuda:
            self.valx = self.valx.cuda().fill_(0)
            self.memx = self.memx.cuda().fill_(0)
            self.memy = self.memy.cuda().fill_(0)
            self.mem_feat = self.mem_feat.cuda().fill_(0)
            self.valy = self.valy.cuda().fill_(0)
            # self.valy.data.fill_(0)

        self.task_mem_ptr = torch.zeros(
            n_tasks, dtype=torch.long, device=self.memx.device
        )
        self.task_mem_filled = torch.zeros(
            n_tasks, dtype=torch.long, device=self.memx.device
        )
        self.task_val_ptr = torch.zeros(
            n_tasks, dtype=torch.long, device=self.memx.device
        )
        self.task_val_filled = torch.zeros(
            n_tasks, dtype=torch.long, device=self.memx.device
        )
        self.bsz = self.cfg.batch_size

        self.n_outputs = n_outputs

        self.mse = nn.MSELoss()
        # Use batchmean to align with KL definition and silence PyTorch deprecation warning
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.samples_seen = 0
        self.sz = int(self.cfg.replay_batch_size)
        self.inner_steps = self.cfg.inner_steps
        self.counter = 0

        # --- GEM trunk-projection state (B2) ---
        # QP constrains only the shared trunk (base_param); FiLM/context params are excluded so
        # their meta-learning is untouched.
        self.margin = float(self.cfg.gem_margin)
        self.use_qp = not bool(self.cfg.gem_disable_qp)
        self.n_tasks = n_tasks
        base_params = list(self.net.base_param())
        self.base_grad_dims = [p.numel() for p in base_params]
        self.grads = torch.zeros(sum(self.base_grad_dims), n_tasks)
        if self.cfg.cuda:
            self.grads = self.grads.cuda()

    def on_epoch_end(self):
        self.counter += 1
        pass

    def _build_task_memory_capacities(
        self, total_memories: int, n_tasks: int
    ) -> list[int]:
        """Split a total replay budget across tasks.

        Args:
            total_memories: Total replay capacity configured through `n_memories`.
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

    def forward(self, x, t, return_feat=False, *, cil_all_seen_upto_task=None):
        output = self.net(x, t)

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
        """Sample replay or validation examples from buffered tasks.

        Args:
            t: Number of tasks to sample from, starting from task index `0`.
            valid: Whether to sample validation memory.

        Returns:
            Tuple of sampled tensors, or `None` when no memory exists.
        """
        if valid:
            filled_counts = [int(self.task_val_filled[i].item()) for i in range(t)]
        else:
            filled_counts = [int(self.task_mem_filled[i].item()) for i in range(t)]

        total_samples = sum(filled_counts)
        if total_samples <= 0:
            return None

        if valid and self.full_val:
            flat_sample_indices = np.arange(total_samples)
        else:
            sample_size = (
                min(total_samples, 64) if valid else int(min(total_samples, self.sz))
            )
            if sample_size <= 0:
                return None
            flat_sample_indices = np.random.choice(
                total_samples, sample_size, replace=False
            )

        cumulative_counts = np.cumsum([0] + filled_counts)
        task_indices_np = (
            np.searchsorted(cumulative_counts, flat_sample_indices, side="right") - 1
        )
        sample_indices_np = flat_sample_indices - cumulative_counts[task_indices_np]
        t_idx = torch.from_numpy(task_indices_np).to(
            device=self.memx.device, dtype=torch.long
        )
        s_idx = torch.from_numpy(sample_indices_np).to(
            device=self.memx.device, dtype=torch.long
        )

        if valid:
            yy_global = self.valy[t_idx, s_idx]
        else:
            yy_global = self.memy[t_idx, s_idx]
        signal_rows = signal_mask_exclude_noise(yy_global, self.noise_label)
        if not signal_rows.any():
            return None
        t_idx = t_idx[signal_rows]
        s_idx = s_idx[signal_rows]

        offsets = torch.tensor(
            [self.compute_offsets(int(task_index)) for task_index in t_idx.tolist()],
            device=self.memx.device,
            dtype=torch.long,
        )
        if valid:
            xx = self.valx[t_idx, s_idx]
            yy = self.valy[t_idx, s_idx] - offsets[:, 0]
            feat = torch.zeros(xx.size(0), self.nc_per_task, device=self.memx.device)
        else:
            xx = self.memx[t_idx, s_idx]
            yy = self.memy[t_idx, s_idx] - offsets[:, 0]
            feat = self.mem_feat[t_idx, s_idx]
        mask = torch.zeros(xx.size(0), self.nc_per_task, device=self.memx.device)
        for row_index in range(mask.size(0)):
            class_size = offsets[row_index][1] - offsets[row_index][0]
            mask[row_index, :class_size] = torch.arange(
                offsets[row_index][0],
                offsets[row_index][1],
                device=self.memx.device,
            )
        sizes = (offsets[:, 1] - offsets[:, 0]).long()
        return xx, yy, feat, mask.long(), t_idx.tolist(), sizes

    # --- GEM trunk-projection helpers (B2) ---

    def _flatten_grads(self, grad_list, params):
        parts = []
        for grad, param in zip(grad_list, params):
            parts.append(
                (grad if grad is not None else torch.zeros_like(param)).reshape(-1)
            )
        return torch.cat(parts)

    def _unflatten_grads(self, flat, params):
        out = []
        idx = 0
        for param in params:
            count = param.numel()
            out.append(flat[idx : idx + count].view_as(param))
            idx += count
        return out

    def _past_task_base_grad(self, past_task, base_params):
        """Flattened gradient of past task's replay CE wrt the trunk, or None."""
        filled = int(self.task_mem_filled[past_task].item())
        if filled == 0:
            return None
        mem_y = self.memy[past_task, :filled]
        signal_rows = signal_mask_exclude_noise(mem_y, self.noise_label)
        if not signal_rows.any():
            return None
        offset1, offset2 = self.compute_offsets(past_task)
        mem_x = self.memx[past_task, :filled][signal_rows]
        logits = self.forward(mem_x, past_task)[:, offset1:offset2]
        targets = (mem_y[signal_rows] - offset1).long()
        loss = classification_cross_entropy(
            logits, targets, class_weighted_ce=self.class_weighted_ce
        )
        grad = torch.autograd.grad(
            loss, base_params, create_graph=False, allow_unused=True
        )
        return self._flatten_grads(grad, base_params)

    def _project_base_grads(self, grad_list, base_params, t):
        """GEM-project the trunk gradient against per-past-task trunk gradients.

        Mirrors GEM's QP step (model/gem.py) but only over net.base_param(): the
        FiLM/context parameters are excluded and keep their meta-learned updates.
        """
        if not self.use_qp or t <= 0:
            return grad_list
        constrained_tasks = []
        for past_task in range(t):
            past_grad = self._past_task_base_grad(past_task, base_params)
            if past_grad is None:
                continue
            self.grads[:, past_task].copy_(past_grad)
            constrained_tasks.append(past_task)
        if not constrained_tasks:
            return grad_list
        g_vec = self._flatten_grads(grad_list, base_params)
        idx = torch.tensor(
            constrained_tasks, dtype=torch.long, device=self.grads.device
        )
        dotp = torch.mm(g_vec.unsqueeze(0), self.grads.index_select(1, idx))
        if (dotp < 0).sum() == 0:
            return grad_list
        g_col = g_vec.clone().unsqueeze(1)
        project2cone2(g_col, self.grads.index_select(1, idx), self.margin)
        return self._unflatten_grads(g_col.view(-1).to(g_vec.device), base_params)

    def observe(self, x, y, t):
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
        #     det_logits = self.net.forward_det_agnostic(x_det)
        #     det_loss = self.det_loss(det_logits, y_det.float())
        #     det_replay = self._sample_det_memory()
        #     if det_replay is not None:
        #         mem_x, mem_y = det_replay
        #         mem_det_logits = self.net.forward_det_agnostic(mem_x)
        #         mem_loss = self.det_loss(mem_det_logits, mem_y.float())
        #         det_loss = 0.5 * (det_loss + mem_loss)
        #     self.zero_grad()
        #     grads = torch.autograd.grad(
        #         det_loss,
        #         self.net.base_param(),
        #         create_graph=False,
        #         allow_unused=True,
        #     )
        #     for param, grad in zip(self.net.base_param(), grads):
        #         if grad is None:
        #             continue
        #         with torch.no_grad():
        #             param.add_(grad, alpha=-self.inner_lr)
        #     return float(det_loss.item()), 0.0
        # x = x[signal_mask]
        # y = y_cls[signal_mask]
        raw_x_train = x
        x_train = self._canonicalize_input(raw_x_train, detach=False)
        x_for_storage = self._input_for_replay(x)
        y_work = unpack_y_to_class_labels(y).long()

        # if task has changed, run model on val set of previous task to get soft targets
        if t != self.current_task:
            tt = self.current_task
            previous_task_filled = int(self.task_mem_filled[tt].item())
            if previous_task_filled > 0:
                offset1, offset2 = self.compute_offsets(tt)
                out = self.forward(self.memx[tt, :previous_task_filled], tt, True)
                cls_size = int(offset2 - offset1)
                feat = self.mem_feat[tt, :previous_task_filled]
                feat.zero_()
                feat[:, :cls_size] = F.softmax(
                    out[:, offset1:offset2] / self.temp, dim=1
                ).data.clone()  # store soft targets
            self.current_task = t
            self.memy[t] = 0

        # maintain validation set (store adapted input in val buffer)
        n_val_taken = 0
        n_rotated_in = 0
        rotated_validation_sample_for_meta = None
        task_val_capacity = int(self.task_val_capacities[t])
        if task_val_capacity > 0 and x_train.size(0) > 0:
            n_val_taken = 1
            incoming_val_y = y_work[0]
            raw_x_train = raw_x_train[1:]
            x_train = x_train[1:]
            y_work = y_work[1:]
            val_write_pointer = int(self.task_val_ptr[t].item())
            val_filled = int(self.task_val_filled[t].item())
            # Only rotate in when overwriting a slot that has valid data (buffer full)
            if val_filled >= task_val_capacity:
                n_rotated_in = 1
                rotated_validation_sample_for_meta = self.valx[
                    t, val_write_pointer
                ].unsqueeze(0)
                x_train = torch.cat([x_train, rotated_validation_sample_for_meta])
                y_work = torch.cat(
                    [y_work, self.valy[t, val_write_pointer].unsqueeze(0)]
                )
            self.valx[t, val_write_pointer].copy_(x_for_storage[0])
            self.valy[t, val_write_pointer].copy_(incoming_val_y)
            filled_val_before_update = int(self.task_val_filled[t].item())
            self.task_val_filled[t] = min(
                task_val_capacity, filled_val_before_update + 1
            )
            self.task_val_ptr[t] = (
                0
                if (val_write_pointer + 1) == task_val_capacity
                else (val_write_pointer + 1)
            )
            if x_train.size(0) == 0:
                x_train = x_for_storage[0].unsqueeze(0)
                y_work = incoming_val_y.unsqueeze(0)
        # memory set: only write "new" samples to replay; rotated-in sample is already in val buffer
        self.net.train()
        task_replay_capacity = int(self.task_replay_capacities[t])
        if task_replay_capacity > 0 and y_work.size(0) > 0:
            replay_write_pointer = int(self.task_mem_ptr[t].item())
            batch_size = y_work.size(0)
            n_new = batch_size - n_rotated_in
            endcnt = min(replay_write_pointer + n_new, task_replay_capacity)
            effbsz = endcnt - replay_write_pointer
            if effbsz > 0:
                replay_start = n_val_taken
                self.memx[t, replay_write_pointer:endcnt].copy_(
                    x_for_storage[replay_start : replay_start + effbsz]
                )
                self.memy[t, replay_write_pointer:endcnt].copy_(y_work[:effbsz])
                filled_mem_before_update = int(self.task_mem_filled[t].item())
                self.task_mem_filled[t] = min(
                    task_replay_capacity, filled_mem_before_update + effbsz
                )
            self.task_mem_ptr[t] = 0 if endcnt == task_replay_capacity else endcnt

        # if getattr(self, "det_enabled", True):
        #     det_logits = self.net.forward_det_agnostic(x_det)
        #     det_loss = self.det_loss(det_logits, y_det.float())
        #     det_replay = self._sample_det_memory()
        #     if det_replay is not None:
        #         mem_x, mem_y = det_replay
        #         mem_det_logits = self.net.forward_det_agnostic(mem_x)
        #         mem_loss = self.det_loss(mem_det_logits, mem_y.float())
        #         det_loss = 0.5 * (det_loss + mem_loss)
        #     det_loss_value = det_loss.detach()
        #     det_loss = self.det_lambda * det_loss
        #     det_grads = torch.autograd.grad(
        #         det_loss,
        #         self.net.base_param(),
        #         create_graph=False,
        #         allow_unused=True,
        #     )
        #     for param, grad in zip(self.net.base_param(), det_grads):
        #         if grad is None:
        #             continue
        #         with torch.no_grad():
        #             param.add_(grad, alpha=-self.inner_lr)
        # else:
        if True:
            det_loss_value = torch.zeros((), device=x_train.device, dtype=torch.float32)

        # Slicing (e.g. `raw_x_train = raw_x_train[1:]`) creates a non-leaf view.
        # After inner-loop autograd calls free graphs, reuse of that view can hit
        # "backward through the graph a second time". Re-leaf once per observe.
        raw_x_train = raw_x_train.detach().requires_grad_(True)

        # Rebuild canonicalized train input each inner SGD step from raw inputs to avoid
        # reusing a freed autograd graph while still allowing adapter gradients.
        self.zero_grad()
        cls_tr_rec = []
        context_parameters = list(self.net.context_param())
        targets = y_work.long()
        for _ in range(self.inner_steps):
            # Each fast step needs a fresh canonicalized input: the first
            # `autograd.grad(loss, base_param)` frees the graph that produced the
            # previous `x_train` (e.g. via `input_adapter`), so reusing it on the
            # next iteration triggers "backward through the graph a second time".
            x_train = self._canonicalize_input(raw_x_train, detach=False)
            if rotated_validation_sample_for_meta is not None:
                x_train = torch.cat(
                    [x_train, rotated_validation_sample_for_meta], dim=0
                )
            pred = self.forward(x_train, t, cil_all_seen_upto_task=t)
            logits = pred
            signal_mask_for_metric = signal_mask_exclude_noise(y_work, self.noise_label)
            if signal_mask_for_metric.any():
                preds = torch.argmax(logits[signal_mask_for_metric], dim=1)
                cls_tr_rec.append(macro_recall(preds, targets[signal_mask_for_metric]))
            else:
                cls_tr_rec.append(0.0)

            loss1 = classification_cross_entropy(
                logits,
                targets,
                class_weighted_ce=self.class_weighted_ce,
            )
            loss2 = torch.tensor(0.0, device=x_train.device)
            loss3 = torch.tensor(0.0, device=x_train.device)
            if t > 0:
                sampled = self.memory_sampling(t)
                if sampled is not None:
                    xx, yy, feat, mask, list_t, class_sizes = sampled
                    pred_ = self.net(xx, list_t)
                    replay_pred = torch.gather(pred_, 1, mask)
                    for row, size in enumerate(class_sizes):
                        if size < replay_pred.size(1):
                            replay_pred[row, size:] = -1e9
                    loss2 = classification_cross_entropy(
                        replay_pred, yy, class_weighted_ce=self.class_weighted_ce
                    )
                    if self.use_distill:
                        loss3 = self.reg * self.kl(
                            F.log_softmax(replay_pred / self.temp, dim=1), feat
                        )
                loss = (
                    self.cls_lambda * loss1
                    + self.det_lambda * det_loss_value
                    + loss2
                    + loss3
                )
            else:
                loss = self.cls_lambda * loss1 + self.det_lambda * det_loss_value

            base_params = list(self.net.base_param())
            grads = torch.autograd.grad(
                loss,
                base_params,
                create_graph=False,
                allow_unused=True,
            )
            # GEM trunk projection (B2): constrain the trunk update so it does not
            # increase any past task's replay loss. FiLM/context params are untouched.
            grads = self._project_base_grads(grads, base_params, t)

            # SGD update only the BASE NETWORK
            for param, grad in zip(base_params, grads):
                if grad is None:
                    continue
                with torch.no_grad():
                    param.add_(grad, alpha=-self.inner_lr)

            # Fast-step `autograd.grad` freed the graph; rebuild before meta forward.
            x_train = self._canonicalize_input(raw_x_train, detach=False)
            if rotated_validation_sample_for_meta is not None:
                x_train = torch.cat(
                    [x_train, rotated_validation_sample_for_meta], dim=0
                )
            logits = self.forward(x_train, t, cil_all_seen_upto_task=t)

            sampled_validation = self.memory_sampling(t + 1, valid=True)
            if sampled_validation is None:
                outer_loss = classification_cross_entropy(
                    logits,
                    targets,
                    class_weighted_ce=self.class_weighted_ce,
                )
            else:
                xval, yval, feat, mask, list_t, class_sizes_val = sampled_validation
                pred_ = self.net(xval, list_t)
                pred = torch.gather(pred_, 1, mask)
                for row, size in enumerate(class_sizes_val):
                    if size < pred.size(1):
                        pred[row, size:] = -1e9
                outer_loss = classification_cross_entropy(
                    pred, yval, class_weighted_ce=self.class_weighted_ce
                )
            outer_grad = torch.autograd.grad(
                outer_loss,
                context_parameters,
                create_graph=False,
                allow_unused=True,
            )

            self.opt.zero_grad()
            for param, grad in zip(context_parameters, outer_grad):
                if grad is None:
                    continue
                param.grad = grad.detach().clamp(-1, 1)
            self.opt.step()
            # SGD update the CONTROLLER
            self.zero_grad()

        avg_cls_tr_rec = sum(cls_tr_rec) / len(cls_tr_rec) if cls_tr_rec else 0.0
        return loss.item(), avg_cls_tr_rec, None
