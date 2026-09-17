from dataclasses import dataclass
import sys

import torch

from model.adab1n import adab1n_layers, end_task_all
from model.resnet1d import ResNet1D
from model.replay_utils import unpack_y_to_class_labels
from utils.training_metrics import macro_recall
from utils import misc_utils
from utils.class_weighted_loss import classification_cross_entropy

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("once")


@dataclass
class IidConfig:
    """Configuration shim for IID (non-LL) runs.

    This mirrors the basic argument harvesting pattern used by other models
    while keeping the training behaviour strictly non-lifelong.

    Attributes:
        arch: Backbone architecture identifier (only ``\"resnet1d\"`` is supported).
        lr: Learning rate for the SGD optimizer.
        cuda: Whether to place the model on GPU.

    Usage:
        cfg = IidConfig.from_args(args)
    """

    arch: str = "resnet1d"
    inner_steps: int = 1
    lr: float = 1e-3
    cuda: bool = True

    @staticmethod
    def from_args(args: object) -> "IidConfig":
        cfg = IidConfig()
        for field in cfg.__dataclass_fields__:
            if hasattr(args, field):
                setattr(cfg, field, getattr(args, field))
        return cfg


class Net(torch.nn.Module):
    """Maximal-replay upper bound: plain ResNet1D trained on all data seen so far.

    Runs through the ordinary continual schedule in ``main.life_experience``,
    but ``cumulative_replay`` makes that loop train task ``t`` on the shuffled
    union of the training sets of tasks ``0..t``. There is no other
    continual-learning machinery (no memory budget, no regularisers).

    Logits are masked the same way as the other baselines: under the
    task-incremental loader each training row only competes within its own
    task's class block (derived from its global label, since batches mix
    tasks) and inference masks to task ``t``; under the class-incremental
    loader every class seen up to task ``t`` stays active.

    Usage:
        model = Net(n_inputs, n_outputs, n_tasks, args)
        logits = model(x, t)
        loss, rec, logits = model.observe(x, y, t)
    """

    # Read by ``main.life_experience``: train each task on tasks 0..t combined.
    cumulative_replay = True

    def __init__(
        self, n_inputs: int, n_outputs: int, n_tasks: int, args: object
    ) -> None:
        super().__init__()
        del n_inputs  # ResNet1D determines its own front-end shape

        if n_tasks <= 0:
            raise ValueError("IID2 requires a positive number of tasks")

        self.cfg = IidConfig.from_args(args)
        self.class_weighted_ce = bool(getattr(args, "class_weighted_ce", True))
        self.n_outputs = n_outputs
        self.n_tasks = n_tasks
        self.classes_per_task = misc_utils.build_task_class_list(
            n_tasks,
            n_outputs,
            nc_per_task=getattr(args, "nc_per_task_list", "")
            or getattr(args, "nc_per_task", None),
            classes_per_task=getattr(args, "classes_per_task", None),
        )
        self.nc_per_task = misc_utils.max_task_class_count(self.classes_per_task)
        self.incremental_loader_name = getattr(args, "loader", None)
        # Global label -> owning task, for per-row TIL masking of mixed batches.
        self.register_buffer(
            "_label_to_task",
            torch.repeat_interleave(
                torch.arange(len(self.classes_per_task)),
                torch.as_tensor(self.classes_per_task, dtype=torch.long),
            ),
            persistent=False,
        )

        if self.cfg.arch != "resnet1d":
            raise ValueError(
                f"Unsupported arch {self.cfg.arch}; only resnet1d is available now."
            )

        # Shared backbone with other models.
        self.net = ResNet1D(n_outputs, args)

        # Optimiser and loss.
        self.opt = torch.optim.SGD(self.parameters(), lr=self.cfg.lr, momentum=0.9)

        # Empty unless --norm_type adab1n. No per-row task counts are set, so
        # AdaB1N's cross-task reweighting stays off even though batches mix
        # tasks, and the layer reduces to BatchNorm1d with a kappa-scheduled
        # running-stat momentum; only the task counter is advanced, which keeps
        # this a clean control arm.
        self._adab1n = adab1n_layers(self.net)
        self._steps_since_boundary = 0

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | int,
        **kwargs,
    ) -> torch.Tensor:  # pragma: no cover - thin wrapper
        """Return masked logits for task ``t``.

        TIL masks to task ``t``'s class block; for class-incremental evaluation
        ``cil_all_seen_upto_task`` keeps every class introduced so far.
        """
        return misc_utils.apply_task_incremental_logit_mask(
            self.net(x),
            int(t),
            self.classes_per_task,
            self.n_outputs,
            cil_all_seen_upto_task=kwargs.get("cil_all_seen_upto_task"),
            loader=self.incremental_loader_name,
        )

    def _mask_training_logits(
        self, logits: torch.Tensor, targets: torch.Tensor, t: int
    ) -> torch.Tensor:
        """Mask training logits for a batch drawn from tasks ``0..t``.

        Args:
            logits: Unmasked logits ``(batch, n_outputs)``.
            targets: Global class labels for each row.
            t: Task currently being trained.

        Returns:
            Logits where, under CIL, classes after task ``t`` are masked and,
            otherwise (TIL), each row keeps only its own task's class block.
        """
        if self.incremental_loader_name == "class_incremental_loader":
            return misc_utils.apply_task_incremental_logit_mask(
                logits,
                t,
                self.classes_per_task,
                self.n_outputs,
                cil_all_seen_upto_task=t,
                loader=self.incremental_loader_name,
            )
        row_tasks = self._label_to_task[targets]
        col_tasks = self._label_to_task[: logits.size(1)]
        own_block = row_tasks.unsqueeze(1) == col_tasks.unsqueeze(0)
        return logits.masked_fill(~own_block, -1e9)

    def observe(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor | int,
    ) -> tuple[float, float, torch.Tensor | None]:
        """Perform SGD step(s) on a batch drawn from tasks ``0..t``.

        Args:
            x: Input batch.
            y: Ground-truth global class labels (0..n_outputs-1).
            t: Task currently being trained; batches may contain earlier tasks.

        Returns:
            Tuple of (loss_value, training_recall, masked_logits).

        Usage:
            loss, rec, logits = model.observe(x, y, t)
        """
        self._steps_since_boundary += 1
        self.train()
        metric_logits = None
        targets = unpack_y_to_class_labels(y).long()
        for _ in range(self.cfg.inner_steps):
            self.opt.zero_grad()

            logits = self._mask_training_logits(self.net(x), targets, int(t))
            loss_tensor = classification_cross_entropy(
                logits,
                targets,
                class_weighted_ce=self.class_weighted_ce,
            )
            loss_tensor.backward()
            self.opt.step()
            metric_logits = logits.detach()

            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                cls_tr_rec = macro_recall(
                    preds.detach().cpu(),
                    targets.detach().cpu(),
                )
        return float(loss_tensor.item()), float(cls_tr_rec), metric_logits

    def finalize_task_after_training(self, train_loader=None) -> None:
        """Advance AdaB1N's task counter at the end of a task (no-op otherwise).

        Idempotent: a repeat call with no training in between does nothing, since
        double-advancing would misalign later batches' task metadata.
        """
        if not self._adab1n or self._steps_since_boundary == 0:
            return
        end_task_all(self._adab1n)
        self._steps_since_boundary = 0


__all__ = ["Net"]
