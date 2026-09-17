import math
import torch
from model.lamaml_base import *  # noqa: F403
from model.replay_utils import (
    ReplayInputMixin,
    unpack_y_to_class_labels,
)
from model.task_bn import frozen_running_stats
from utils.training_metrics import macro_recall
from utils import misc_utils


class Net(ReplayInputMixin, BaseNet):  # noqa: F405

    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        super(Net, self).__init__(n_inputs, n_outputs, n_tasks, args)
        self.nc_per_task = misc_utils.max_task_class_count(self.classes_per_task)
        self.cls_lambda = float(getattr(args, "cls_lambda", 1.0))

    def take_loss(self, _t, logits, y):
        # Full CIL logits (including global noise class); targets are global indices.
        y_cls = unpack_y_to_class_labels(y).long()
        return self._classification_loss(logits, y_cls)

    def take_multitask_loss(self, bt, t, logits, y):
        """Batched CE over global labels on per-sample task-masked logits.

        ``meta_loss`` masks the rows first (own task's classes under TIL, tasks
        ``0..t`` under CIL; see :func:`utils.misc_utils.mask_replay_logits`), so
        one batched call over the mixed replay+current batch is exact.
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
            fill_value=-10e10,
            loader=self.incremental_loader_name,
        )

    def meta_loss(self, x, fast_weights, y, bt, t, replay_count=None):
        """
        differentiate the loss through the network updates wrt alpha

        The meta batch ``x`` is the replay rows followed by the current-task rows
        (see ``getBatch`` and the live-batch splice in ``observe``). Forwarding it
        in a single pass makes backbone BatchNorm normalize the replay rows with
        statistics pooled over the current-task mixture, which corrupts the
        retention (replay) term of the meta loss while leaving the current-task
        term unaffected -- the deficit grows with task count. Forward the replay
        and current blocks in separate passes so each is normalized with its own
        statistics, then concatenate the logits (replay-first, keeping alignment
        with ``bt``/``y``) and score with the identical mask.

        The two blocks are also *scored* separately, as eralg4 does -- see
        ``_combine_replay_current_loss``.

        The replay block additionally runs under ``frozen_running_stats``: its
        rows span several old tasks, so they must not be folded into the current
        task's per-task BatchNorm running statistics.
        """

        rc = None if replay_count is None else int(replay_count)
        if rc is not None and 0 < rc < x.size(0):
            with frozen_running_stats(self):
                replay_raw = self.net.forward(x[:rc], fast_weights)
            current_raw = self.net.forward(x[rc:], fast_weights)
            raw = torch.cat([replay_raw, current_raw], dim=0)
        elif rc is not None and rc >= x.size(0) > 0:
            # Whole meta batch is replay.
            with frozen_running_stats(self):
                raw = self.net.forward(x, fast_weights)
        else:
            raw = self.net.forward(x, fast_weights)
        logits = misc_utils.mask_replay_logits(
            raw,
            bt,
            t,
            self.classes_per_task,
            self.n_outputs,
            loader=self.incremental_loader_name,
        )
        loss_q = self._combine_replay_current_loss(bt, t, logits, y, replay_count)

        return loss_q, logits

    def _combine_replay_current_loss(self, bt, t, logits, y, replay_count):
        """Meta-loss reduction over the replay-then-current getBatch layout.

        ``--cmaml_replay_loss_mode`` selects it:

        - ``split`` (default) scores the replay and current blocks separately, as
          eralg4 does, and returns ``current_loss + memory_loss_lambda *
          replay_loss``. This pins replay's share of the loss at 1:1.
        - ``split_norm`` divides ``split`` by ``1 + memory_loss_lambda`` to hold
          the total loss scale fixed as well.
        - ``pooled`` is the legacy single CE over every row, kept only to
          reproduce runs logged before 2026-07-25.

        The pooled CE looked like a neutral bookkeeping choice but was not: the
        inverse-frequency class weights are derived from the pooled batch, where
        the current rows span one task's classes while the replay rows spread
        over every task seen so far. Old-task classes are therefore rare and
        collect much larger per-row weights, so replay's share of the loss
        escalated with task count -- measured 0.34 at task 0 rising to 0.85 by
        task 9, at which point the current task received 15% of the gradient.
        Pinning it recovers ~3 F1 of plasticity in single-epoch TIL (n=6, every
        seed; docs/cmaml_vs_reser_til.md).
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
            # ``getBatch`` round-trips through detached NumPy, which severs the
            # input adapter's graph for the current rows. Re-attach the live,
            # adapter-differentiable current batch (the trailing ``x.size(0)``
            # rows ``getBatch`` appends after the replay rows) so the single
            # ``meta_loss.backward()`` below also trains the adapter via the
            # standard La-MAML weight update -- no separate adapter-only forward
            # and manual SGD step needed. ``inner_update`` uses
            # ``autograd.grad(..., inputs=fast_weights)``, which prunes the
            # adapter subgraph (off the output->inputs path), so it survives for
            # the meta backward.
            n_current = x.size(0)
            if n_current > 0:
                bx = torch.cat([bx[:-n_current], x], dim=0)
            # Replay rows are the leading block of ``bx`` (getBatch appends the
            # current rows after them); used to split the meta-loss forward so
            # BatchNorm does not mix replay and current statistics.
            replay_count = max(0, bx.size(0) - n_current)

            for i in range(n_batches):

                # Detach the inner-update input: its grad w.r.t. ``fast_weights``
                # depends only on ``batch_x``'s values, not its graph. Keeping the
                # adapter graph out of ``inner_update``'s ``autograd.grad`` (which
                # frees it with ``retain_graph=False``) leaves ``x_train``'s live
                # graph solely in the spliced meta batch, consumed by exactly one
                # ``meta_loss.backward()`` -- so the adapter trains without a
                # double backward.
                batch_x = x[i * rough_sz : (i + 1) * rough_sz].detach()
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
                    cls_tr_rec.append(macro_recall(preds, targets))

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

            self.net.zero_grad()
            self.net.alpha_lr.zero_grad()

        avg_cls_tr_rec = sum(cls_tr_rec) / len(cls_tr_rec) if cls_tr_rec else 0.0
        return meta_loss.item(), avg_cls_tr_rec, None
