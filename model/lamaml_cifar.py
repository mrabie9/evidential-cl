import math
import torch
from model.lamaml_base import *  # noqa: F403
from model.detection_replay import (
    DetectionReplayMixin,
    unpack_y_to_class_labels,
)
from utils.training_metrics import macro_recall
from utils import misc_utils
from utils.class_weighted_loss import classification_cross_entropy


class Net(DetectionReplayMixin, BaseNet):  # noqa: F405

    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        super(Net, self).__init__(n_inputs, n_outputs, n_tasks, args)
        self.nc_per_task = misc_utils.max_task_class_count(self.classes_per_task)
        self.det_lambda = float(getattr(args, "det_lambda", 1.0))
        self.cls_lambda = float(getattr(args, "cls_lambda", 1.0))
        self._init_det_replay(
            getattr(args, "det_memories", 2000),
            getattr(args, "det_replay_batch", 64),
            enabled=bool(getattr(args, "use_detector_arch", False)),
        )

    def take_loss(self, _t, logits, y):
        # Full CIL logits (including global noise class); targets are global indices.
        y_cls = unpack_y_to_class_labels(y).long()
        return self._classification_loss(logits, y_cls)

    def take_multitask_loss(self, bt, t, logits, y):
        """Batched CE over global labels on per-sample task-masked logits.

        ``meta_loss`` masks each row to its own task's classes before calling
        this, so one batched call over the mixed replay+current batch is exact.
        The per-row loop this replaces fed single-element batches through the
        weighted CE, where inverse-frequency weights collapse to 1.0 — it
        silently trained unweighted. The batched call makes
        ``class_weighted_ce`` effective and removes ~|batch| Python-level CE
        calls per meta step.
        """
        if logits.size(0) == 0:
            return torch.zeros((), device=logits.device, dtype=logits.dtype)
        return self._classification_loss(logits, y.long())

    def forward(self, x, t, *, cil_all_seen_upto_task=None):
        output = self.net.forward(x)
        return misc_utils.apply_task_incremental_logit_mask(
            output,
            t,
            self.classes_per_task,
            self.n_outputs,
            cil_all_seen_upto_task=cil_all_seen_upto_task,
            global_noise_label=self.noise_label,
            fill_value=-10e10,
            loader=self.incremental_loader_name,
        )

    def meta_loss(self, x, fast_weights, y, bt, t, replay_count=None):
        """
        differentiate the loss through the network updates wrt alpha

        With ``--cmaml_joint_er`` (PROBE, twin of eralg4's ``--eralg4_joint_er``)
        the concatenated getBatch batch ``x`` is forwarded in two separate
        passes -- replay rows (``x[:replay_count]``) then current rows
        (``x[replay_count:]``) -- so backbone BatchNorm normalizes each with its
        own statistics instead of the pooled replay+current mixture. The two
        logit blocks are concatenated (preserving replay-first alignment with
        ``bt``/``y``) and scored with the identical mask + single CE, so the ONLY
        change vs the default single forward is the BN statistics split. This
        isolates the concat BN-mixing retention effect eralg4's probe revealed.

        ``--cmaml_replay_loss_mode`` controls how the replay and current blocks
        are combined *after* the forward; it defaults to ``split``, matching
        eralg4's ``_weighted_multitask_loss``.
        """

        if (
            self.cfg.cmaml_joint_er
            and replay_count is not None
            and 0 < int(replay_count) < x.size(0)
        ):
            rc = int(replay_count)
            replay_raw = self.net.forward(x[:rc], fast_weights)
            current_raw = self.net.forward(x[rc:], fast_weights)
            raw = torch.cat([replay_raw, current_raw], dim=0)
        else:
            raw = self.net.forward(x, fast_weights)
        logits = self._mask_logits_for_sample_tasks(raw, bt)
        loss_q = self._combine_replay_current_loss(bt, t, logits, y, replay_count)

        return loss_q, logits

    def _combine_replay_current_loss(self, bt, t, logits, y, replay_count):
        """Meta-loss reduction over the replay-then-current getBatch layout.

        ``split`` (default) mirrors eralg4's
        ``current + memory_loss_lambda * replay``: each block gets its own mean,
        pinning replay's share of the loss at 1:1. ``split_norm`` renormalizes by
        ``1 + memory_loss_lambda`` to hold the total loss scale fixed as well.

        ``pooled`` is the legacy single CE over every row. It looks like a
        neutral bookkeeping choice but is not: the inverse-frequency class
        weights are derived from the pooled batch, where the ~256 current rows
        span one task's classes while the ~128 replay rows spread over every task
        seen so far. Old-task classes are therefore rare and collect much larger
        per-row weights, so replay's share of the loss escalates with task count
        -- 0.34 at task 0 rising to 0.85 by task 9, at which point the current
        task receives 15% of the gradient. Worth ~3 F1 of plasticity in
        single-epoch TIL (docs/cmaml_vs_reser_til.md); retained only to reproduce
        runs logged before 2026-07-25.
        """
        mode = self.cfg.cmaml_replay_loss_mode
        rc = (
            0
            if replay_count is None
            else max(0, min(int(replay_count), logits.size(0)))
        )
        if mode == "pooled" or rc == 0 or rc == logits.size(0):
            return self.take_multitask_loss(bt, t, logits, y)

        replay_loss = self.take_multitask_loss(bt[:rc], t, logits[:rc], y[:rc])
        current_loss = self.take_multitask_loss(bt[rc:], t, logits[rc:], y[rc:])
        mll = float(self.cfg.memory_loss_lambda)
        total = current_loss + mll * replay_loss
        if mode == "split_norm":
            total = total / (1.0 + mll)
        return total

    def _mask_logits_for_sample_tasks(
        self, raw_logits: torch.Tensor, sample_task_indices: torch.Tensor
    ) -> torch.Tensor:
        """Apply TIL/CIL masking per replay sample task id.

        Args:
            raw_logits: Unmasked logits for the replay/meta batch.
            sample_task_indices: Task index per sample in ``raw_logits``.

        Returns:
            Logits masked according to each sample's task id.
        """
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
                loader=self.incremental_loader_name,
            )
        return masked_logits

    def inner_update(self, x, fast_weights, y, t):
        # Ensure we have a concrete, non-empty list of tensors
        if not fast_weights:  # handles None or []
            fast_weights = [p for p in self.net.parameters()]
        else:
            # if it might be an iterator, force-list it once
            fast_weights = list(fast_weights)

        # Forward using fast weights
        raw = self.net.forward(x, vars=fast_weights)
        logits = misc_utils.apply_task_incremental_logit_mask(
            raw,
            t,
            self.classes_per_task,
            self.n_outputs,
            cil_all_seen_upto_task=t,
            global_noise_label=self.noise_label,
            loader=self.incremental_loader_name,
        )
        loss = self.take_loss(t, logits, y)

        graph_required = bool(self.cfg.second_order)

        # All inputs to grad must require grad and be used in loss
        for p in fast_weights:
            p.requires_grad_(True)

        raw_gradients = torch.autograd.grad(
            loss,
            fast_weights,
            create_graph=graph_required,
            retain_graph=graph_required,
            allow_unused=True,
        )
        grads = [
            grad if grad is not None else torch.zeros_like(weight)
            for grad, weight in zip(raw_gradients, fast_weights)
        ]

        # Clip
        grads = (
            [
                g.clamp(min=-self.cfg.grad_clip_norm, max=self.cfg.grad_clip_norm)
                for g in grads
            ]
            if self.cfg.grad_clip_norm
            else grads
        )

        # Inner step: w' = w - alpha * g
        # (zip three lists directly; avoid nested zip to prevent iterator surprises)
        fast_weights = [
            w - a * g for (g, w, a) in zip(grads, fast_weights, self.net.alpha_lr)
        ]

        return fast_weights

    def observe(self, x, y, t):
        self.net.train()
        cls_tr_rec = []
        # Original batch order for this observe call; reset each pass so
        # permutations do not compound across inner_steps.
        x_base = x
        y_base = y
        for pass_itr in range(self.inner_steps):
            self.pass_itr = pass_itr
            x, y = x_base, y_base
            perm = torch.randperm(x.size(0))
            x = x[perm]
            if isinstance(y, (list, tuple)):
                y = tuple(yi[perm] if yi is not None else None for yi in y)
            else:
                y = y[perm]
            # noise_label = None
            # if class_counts is not None:
            #     _, offset2 = misc_utils.compute_offsets(t, class_counts)
            #     noise_label = offset2 - 1
            # y_cls, y_det = self._unpack_labels(
            #     y,
            #     noise_label=noise_label,
            #     use_detector_arch=bool(getattr(self, "det_enabled", False)),
            # )
            # y_cls = y_cls[perm]
            # y_det = y_det[perm]
            # if y_det is not None and self.det_memories > 0:
            #     self._update_det_memory(x, y_det)
            # signal_mask = (y_det == 1) & (y_cls >= 0)
            # if not signal_mask.any():
            #     if not getattr(self, "det_enabled", True):
            #         return 0.0, 0.0
            #     self.zero_grads()
            #     det_logits, _ = self.net.forward_heads(x)
            #     det_loss = self.det_loss(det_logits, y_det.float())
            #     det_replay = self._sample_det_memory()
            #     if det_replay is not None:
            #         mem_x, mem_y = det_replay
            #         mem_det_logits, _ = self.net.forward_heads(mem_x)
            #         mem_loss = self.det_loss(mem_det_logits, mem_y.float())
            #         det_loss = 0.5 * (det_loss + mem_loss)
            #     det_loss = self.det_lambda * det_loss
            #     det_loss.backward()
            #     self.opt_wt.step()
            #     return float(det_loss.item()), 0.0

            # x_det = x
            # x = x[signal_mask]
            # y = y_cls[signal_mask]
            raw_x = x
            input_was_3adc = (x.dim() == 3 and x.size(1) == 3) or (
                x.dim() == 4 and x.size(1) == 3 and x.size(2) == 2
            )
            # Train with differentiable canonicalized inputs; use detached tensors for replay writes.
            x_train = self._canonicalize_input(x, detach=False)
            x_for_storage = self._input_for_replay(x)
            x = x_train

            self.epoch += 1
            self.zero_grads()

            if t != self.current_task:
                self.M = self.M_new.copy()
                self.current_task = t
                self._reset_velocity()

            batch_sz = x.shape[0]
            n_batches = self.cfg.meta_batches
            rough_sz = math.ceil(batch_sz / n_batches)
            fast_weights = None
            meta_losses = [0 for _ in range(n_batches)]

            # get a batch by augmented incming data with old task data, used for
            # computing meta-loss
            y_np = unpack_y_to_class_labels(y).long().cpu().numpy()
            bx, by, bt = self.getBatch(x.detach().cpu().numpy(), y_np, t)
            # getBatch appends the current batch (``x.size(0)`` rows) after the
            # replay rows, so the replay block is the leading remainder. Used by
            # the --cmaml_joint_er probe to split the meta-loss forward.
            replay_count = max(0, bx.size(0) - x.size(0))

            for i in range(n_batches):

                batch_x = x[i * rough_sz : (i + 1) * rough_sz]
                batch_y = y[i * rough_sz : (i + 1) * rough_sz]

                # assuming labels for inner update are from the same
                fast_weights = self.inner_update(batch_x, fast_weights, batch_y, t)
                # only sample and push to replay buffer once for each task's stream
                # instead of pushing every epoch
                if self.real_epoch == 0:
                    self.push_to_mem(
                        x_for_storage[i * rough_sz : (i + 1) * rough_sz],
                        batch_y,
                        torch.tensor(t),
                    )
                meta_loss, logits = self.meta_loss(
                    bx, fast_weights, by, bt, t, replay_count=replay_count
                )
                with torch.no_grad():
                    # Vectorized equivalent of the per-sample argmax loop: for
                    # each row, argmax within its own task's class slice
                    # [offset1, offset2) and record the local prediction/target.
                    # Masking non-task columns to -inf makes a single batched
                    # argmax exact, avoiding one GPU sync per sample.
                    by_dev = by.long().view(-1)
                    bt_list = bt.long().view(-1).tolist()
                    offsets = {tid: self.compute_offsets(tid) for tid in set(bt_list)}
                    o1 = torch.tensor(
                        [offsets[tid][0] for tid in bt_list],
                        device=logits.device,
                    )
                    o2 = torch.tensor(
                        [offsets[tid][1] for tid in bt_list],
                        device=logits.device,
                    )
                    cols = torch.arange(logits.size(1), device=logits.device)
                    valid = (cols.unsqueeze(0) >= o1.unsqueeze(1)) & (
                        cols.unsqueeze(0) < o2.unsqueeze(1)
                    )
                    masked = logits.masked_fill(~valid, float("-inf"))
                    preds = masked.argmax(dim=1) - o1
                    targets = by_dev - o1
                    keep = torch.ones_like(by_dev, dtype=torch.bool)
                    if self.noise_label is not None:
                        keep &= by_dev != self.noise_label
                    if keep.any():
                        cls_tr_rec.append(macro_recall(preds[keep], targets[keep]))

                meta_losses[i] += meta_loss

            # Taking the meta gradient step (will update the learning rates)
            self.zero_grads()

            meta_loss = sum(meta_losses) / len(meta_losses)
            meta_loss.backward()

            if self.cfg.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.net.alpha_lr.parameters(), self.cfg.grad_clip_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), self.cfg.grad_clip_norm
                )
            if self.cfg.learn_lr:
                self.opt_lr.step()

            # if sync-update is being carried out (as in sync-maml) then update the weights using the optimiser
            # otherwise update the weights with sgd using updated LRs as step sizes
            if self.cfg.sync_update:
                self.opt_wt.step()
            else:
                self._async_weight_update()

            # Ensure the input adapter is explicitly optimized on the current
            # differentiable 3-ADC batch.
            if input_was_3adc and x_train.dim() == 3 and x_train.size(1) == 2:
                _offset1, _offset2 = self.compute_offsets(t)
                self.net.zero_grad(set_to_none=True)
                live_logits = misc_utils.apply_task_incremental_logit_mask(
                    self.net.forward(raw_x.detach()),
                    t,
                    self.classes_per_task,
                    self.n_outputs,
                    cil_all_seen_upto_task=t,
                    global_noise_label=self.noise_label,
                    loader=self.incremental_loader_name,
                )
                y_live = unpack_y_to_class_labels(y).long()
                live_loss = classification_cross_entropy(
                    live_logits,
                    y_live,
                    class_weighted_ce=self.class_weighted_ce,
                )
                live_loss.backward()
                adapter_module = getattr(self.net.model, "input_adapter", None)
                if adapter_module is not None:
                    with torch.no_grad():
                        for parameter in adapter_module.parameters():
                            if parameter.grad is None:
                                continue
                            parameter.add_(parameter.grad, alpha=-self.cfg.opt_wt)
            self.net.zero_grad()
            self.net.alpha_lr.zero_grad()

            # if getattr(self, "det_enabled", True):
            #     det_logits, _ = self.net.forward_heads(x_det)
            #     det_loss = self.det_loss(det_logits, y_det.float())
            #     det_replay = self._sample_det_memory()
            #     if det_replay is not None:
            #         mem_x, mem_y = det_replay
            #         mem_det_logits, _ = self.net.forward_heads(mem_x)
            #         mem_loss = self.det_loss(mem_det_logits, mem_y.float())
            #         det_loss = 0.5 * (det_loss + mem_loss)
            #     self.opt_wt.zero_grad()
            #     det_loss = self.det_lambda * det_loss
            #     det_loss.backward()
            #     self.opt_wt.step()

        avg_cls_tr_rec = sum(cls_tr_rec) / len(cls_tr_rec) if cls_tr_rec else 0.0
        return meta_loss.item(), avg_cls_tr_rec, None
