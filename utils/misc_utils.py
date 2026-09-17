import datetime
import glob
import json
import os
import random
from typing import Final

import numpy as np
import torch

# Valid strategies for choosing which stream samples to keep in a per-task
# episodic memory buffer. Must stay in sync with the ``--mem_sampling`` choices
# in ``parser.py``. ``"ring"`` keeps the most recent samples per task;
# ``"reservoir"`` keeps a uniform random sample of the whole task stream.
MEM_SAMPLING_MODES: Final = ("ring", "reservoir")


def reservoir_slots(batch_size, filled, seen, capacity):
    """Assign destination slots for an incoming batch under reservoir sampling.

    Textbook Vitter Algorithm R over a per-task buffer of ``capacity`` slots that
    already holds ``filled`` items (dense in ``[0, filled)``) and has observed
    ``seen`` stream items so far. Each incoming item is either written to a slot or
    rejected, such that the buffer remains a uniform random sample of the whole
    task stream and occupied slots stay dense in ``[0, filled)`` (so replay/
    distillation slicing is unchanged).

    Args:
        batch_size: number of incoming items to place.
        filled: currently occupied slot count (``0 <= filled <= capacity``).
        seen: total stream items observed by this task's buffer so far.
        capacity: total slots available for this task.

    Returns:
        ``(slots, filled, seen)`` where ``slots`` is a list of length
        ``batch_size`` giving each item's destination index, or ``-1`` if the item
        is not admitted; ``filled`` and ``seen`` are advanced past this batch.
    """
    slots = []
    for _ in range(batch_size):
        if filled < capacity:
            # Buffer not yet full: fill the next dense slot deterministically.
            slots.append(filled)
            filled += 1
        else:
            # Buffer full: the (seen+1)-th item replaces a uniform slot with
            # probability capacity/(seen+1), else it is dropped.
            j = random.randint(0, seen)  # inclusive: seen+1 equally likely outcomes
            slots.append(j if j < capacity else -1)
        seen += 1
    return slots, filled, seen


def _parse_class_list(value):
    """Convert string/list/tuple values into a list of ints."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if len(value) == 0:
            return None
        parts = value.replace(";", ",").split(",")
        return [int(p) for p in parts if len(p.strip()) > 0]
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    if isinstance(value, np.ndarray):
        return [int(v) for v in value.tolist()]
    return None


def build_task_class_list(n_tasks, n_outputs, nc_per_task=None, classes_per_task=None):
    """
    Return a per-task class-count list.
    Priority:
        1. Explicit classes_per_task (list or comma-separated str)
        2. Explicit nc_per_task_list (via nc_per_task when given as list/str)
        3. Scalar nc_per_task replicated per task
        4. Even split of n_outputs across tasks.
    """
    explicit = _parse_class_list(classes_per_task)
    if explicit is None:
        explicit = _parse_class_list(nc_per_task)

    if explicit is not None:
        if n_tasks is not None and len(explicit) not in (1, n_tasks):
            raise ValueError(
                f"Expected 1 or {n_tasks} class counts, got {len(explicit)}: {explicit}"
            )
        if n_tasks is not None and len(explicit) == 1:
            explicit = explicit * n_tasks
        return explicit

    if isinstance(nc_per_task, (int, float)) and n_tasks is not None:
        return [int(nc_per_task) for _ in range(n_tasks)]

    if n_tasks is not None and n_outputs is not None and n_tasks > 0:
        base = n_outputs // n_tasks
        remainder = n_outputs - base * n_tasks
        counts = [base for _ in range(n_tasks)]
        for i in range(remainder):
            counts[i] += 1
        return counts

    raise ValueError("Unable to infer per-task class counts.")


def task_class_count(nc_per_task, task):
    if isinstance(nc_per_task, (list, tuple, np.ndarray)):
        return int(nc_per_task[task])
    return int(nc_per_task)


def max_task_class_count(nc_per_task):
    if isinstance(nc_per_task, (list, tuple, np.ndarray)):
        return int(max(nc_per_task))
    return int(nc_per_task)


def compute_offsets(task, nc_per_task):
    if isinstance(nc_per_task, (list, tuple, np.ndarray)):
        if task >= len(nc_per_task):
            raise ValueError(
                f"Task index {task} out of range for nc_per_task={nc_per_task}"
            )
        offset1 = int(sum(nc_per_task[:task]))
        offset2 = int(offset1 + nc_per_task[task])
    else:
        offset1 = task * nc_per_task
        offset2 = (task + 1) * nc_per_task

    return int(offset1), int(offset2)


def current_task_class_indices(
    task,
    nc_per_task,
    n_outputs: int,
    global_noise_label: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Output columns owned by ``task`` itself, as an index tensor.

    The column span ``[offset1, offset2)`` of a single task, plus the global
    noise label when one is configured -- matching the classes
    :func:`apply_task_incremental_logit_mask` leaves unmasked under TIL, so any
    penalty built on these columns charges exactly what the head predicts over.
    Unlike the cumulative "seen so far" span, this excludes earlier tasks'
    classes, which is what a penalty on the *current* task's outputs needs in
    CIL as well as TIL.

    Args:
        task: Task index.
        nc_per_task: Per-task class counts (list) or a scalar count.
        n_outputs: Total width of the head; ``offset2`` is clamped to it.
        global_noise_label: Always-active noise column, if the dataset has one.
        device: Device for the returned tensor.

    Returns:
        Sorted ``torch.long`` index tensor of the task's output columns.

    Usage:
        >>> current_task_class_indices(1, [3, 3], 6).tolist()
        [3, 4, 5]
    """
    offset1, offset2 = compute_offsets(task, nc_per_task)
    offset2 = min(int(n_outputs), offset2)
    indices = list(range(offset1, offset2))
    if global_noise_label is not None:
        noise = int(global_noise_label)
        if 0 <= noise < int(n_outputs) and noise not in indices:
            indices.append(noise)
    return torch.tensor(sorted(set(indices)), dtype=torch.long, device=device)


def _effective_cil_upto_for_loader(
    *,
    loader: str | None,
    cil_all_seen_upto_task: int | None,
) -> int | None:
    """Return the cumulative CIL task bound to honour, or None for TIL masking.

    Aligns with ``utils.training_forward.model_forward_for_metric_loop``: only
    ``class_incremental_loader`` uses cumulative (seen-so-far) masking. When
    ``loader`` is ``None``, ``cil_all_seen_upto_task`` is used as given (backward
    compatible with call sites that do not pass ``loader``).

    Args:
        loader: Value of ``args.loader`` from the incremental dataloader, if any.
        cil_all_seen_upto_task: Requested CIL upper task index from the caller.

    Returns:
        Task index for the CIL branch, or ``None`` to use the TIL branch.

    Usage:
        >>> _effective_cil_upto_for_loader(
        ...     loader="task_incremental_loader", cil_all_seen_upto_task=3
        ... )
        None
    """
    if loader is not None and loader != "class_incremental_loader":
        return None
    return cil_all_seen_upto_task


def apply_task_incremental_logit_mask(
    logits: torch.Tensor,
    task_index: int,
    nc_per_task,
    n_outputs: int,
    *,
    cil_all_seen_upto_task: int | None = None,
    fill_value: float = -1e9,
    loader: str | None = None,
) -> torch.Tensor:
    """Apply task-wise or class-incremental (CIL) evaluation logit masking.

    **Task-incremental (TIL) inference:** only the logit block for ``task_index``
    is left active; past and future classes are masked.

    **CIL evaluation (``cil_all_seen_upto_task`` set):** all classes introduced
    in tasks ``0..cil_all_seen_upto_task`` (inclusive) stay active; only future
    logits are masked.

    If ``loader`` is ``"task_incremental_loader"`` (or any value other than
    ``"class_incremental_loader"``), ``cil_all_seen_upto_task`` is ignored and
    the TIL branch is used — same rule as the metric forward path in ``main.py``.

    Args:
        logits: Unmasked classifier output ``(batch, n_classes)``.
        task_index: Task index used only for the TIL branch (ignored when
            the effective CIL bound is not ``None``).
        nc_per_task: Per-task class counts or scalar (same convention as
            :func:`compute_offsets`).
        n_outputs: Logit width (truncate mask at this index).
        cil_all_seen_upto_task: If not ``None`` (after ``loader`` resolution),
            cumulative CIL mask through this task index (inclusive).
        fill_value: Mask fill value (large negative logit).
        loader: Optional ``args.loader`` string; when set and not the CIL loader,
            forces TIL masking regardless of ``cil_all_seen_upto_task``.

    Returns:
        Masked logits (clone); input tensor is not modified.

    Usage:
        >>> # TIL: only task 1 classes active
        >>> y = apply_task_incremental_logit_mask(logits, 1, [5, 5, 5], 15)
        >>> # CIL after task 1: classes from tasks 0 and 1 active
        >>> y = apply_task_incremental_logit_mask(
        ...     logits, 1, [5, 5, 5], 15, cil_all_seen_upto_task=1
        ... )
    """
    masked = logits.clone()
    effective_cil = _effective_cil_upto_for_loader(
        loader=loader, cil_all_seen_upto_task=cil_all_seen_upto_task
    )
    if effective_cil is not None:
        _, offset2 = compute_offsets(effective_cil, nc_per_task)
        if offset2 < n_outputs:
            masked[:, offset2:].fill_(fill_value)
        return masked
    offset1, offset2 = compute_offsets(task_index, nc_per_task)
    if offset1 > 0:
        masked[:, :offset1].fill_(fill_value)
    if offset2 < n_outputs:
        masked[:, offset2:].fill_(fill_value)
    return masked


def mask_replay_logits(
    logits: torch.Tensor,
    sample_tasks: torch.Tensor,
    current_task: int,
    nc_per_task,
    n_outputs: int,
    *,
    loader: str | None,
    fill_value: float = -1e9,
) -> torch.Tensor:
    """Training-time logit mask for a batch whose rows may come from past tasks.

    **CIL:** every row, replayed or not, sees all classes of tasks
    ``0..current_task``. Bounding a replayed row by its *own* task instead
    never pushes it away from classes introduced later, while current-task
    rows are pushed away from every old class; the model then learns to
    predict only the newest task (old-task CIL recall ~0).

    **TIL:** each row sees only its own task's class block.

    Args:
        logits: Unmasked logits ``(batch, n_classes)``.
        sample_tasks: Task id per row, shape ``(batch,)``.
        current_task: Task currently being trained (CIL bound).
        nc_per_task: Per-task class counts or scalar (see :func:`compute_offsets`).
        n_outputs: Logit width.
        loader: ``args.loader``; only ``"class_incremental_loader"`` selects CIL.
        fill_value: Value written into masked logits.

    Returns:
        Masked logits (clone); targets stay global class indices.

    Usage:
        logits = mask_replay_logits(raw, bt, t, [5, 6], 11, loader=args.loader)
    """
    if loader == "class_incremental_loader":
        return apply_task_incremental_logit_mask(
            logits,
            int(current_task),
            nc_per_task,
            n_outputs,
            cil_all_seen_upto_task=int(current_task),
            fill_value=fill_value,
            loader=loader,
        )
    masked = logits.clone()
    for task_id in torch.unique(sample_tasks).tolist():
        rows = sample_tasks == int(task_id)
        masked[rows] = apply_task_incremental_logit_mask(
            logits[rows],
            int(task_id),
            nc_per_task,
            n_outputs,
            fill_value=fill_value,
            loader=loader,
        )
    return masked


def to_onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def _check_loss(loss):
    return not bool(torch.isnan(loss).item()) and bool((loss >= 0.0).item())


def compute_accuracy(ypred, ytrue, task_size=10):
    all_acc = {}

    all_acc["total"] = round((ypred == ytrue).sum() / len(ytrue), 3)

    for class_id in range(0, np.max(ytrue), task_size):
        idxes = np.where(
            np.logical_and(ytrue >= class_id, ytrue < class_id + task_size)
        )[0]

        label = "{}-{}".format(
            str(class_id).rjust(2, "0"), str(class_id + task_size - 1).rjust(2, "0")
        )
        all_acc[label] = round((ypred[idxes] == ytrue[idxes]).sum() / len(idxes), 3)

    return all_acc


def get_date():
    return datetime.datetime.now().strftime("%Y%m%d")


def get_date_time():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def log_dir(opt, timestamp=None, config_name=None):
    if timestamp is None:
        timestamp = get_date_time()

    rand_num = str(random.randint(1, 1001))
    dir_name = config_name if config_name else opt.model
    logdir = opt.log_dir + "/%s/%s_%s/%s" % (
        dir_name,
        timestamp,
        opt.expt_name,
        opt.seed,
    )
    tfdir = opt.log_dir + "/%s/%s_%s/%s/%s" % (
        dir_name,
        timestamp,
        opt.expt_name,
        opt.seed,
        "tfdir",
    )

    mkdir(logdir)
    mkdir(tfdir)

    with open(logdir + "/training_parameters.json", "w") as f:
        params = {k: v for k, v in vars(opt).items() if not callable(v)}
        json.dump(params, f, indent=4)

    return logdir, tfdir


def save_list_to_file(path, thelist):
    with open(path, "w") as f:
        for item in thelist:
            f.write("%s\n" % item)


def find_latest_checkpoint(folder_path):
    print("searching for checkpoint in : " + folder_path)
    files = sorted(
        glob.iglob(folder_path + "/*.pth"), key=os.path.getmtime, reverse=True
    )
    print("latest checkpoint is:")
    print(files[0])
    return files[0]


def resolve_task_order_seed(args) -> int:
    """Resolve the effective task-order seed and record where it came from.

    Task presentation order is permuted with its own RNG stream, seeded either by
    the training seed (the default, so that varying ``--seed`` varies task order
    too) or by an explicit ``--task-order-seed``. Passing an explicit value keeps
    the order fixed while ``--seed`` varies, which is how the two effects are
    isolated from one another.

    Must be called before :func:`log_dir`, which snapshots ``vars(args)`` into
    ``training_parameters.json``.

    Args:
        args: Parsed argument namespace. ``args.task_order_seed`` is replaced by
            the resolved integer and ``args.task_order_seed_source`` is set to
            ``"seed"`` or ``"explicit"``.

    Returns:
        The resolved task-order seed.

    Usage:
        >>> import argparse
        >>> args = argparse.Namespace(seed=39, task_order_seed=None)
        >>> resolve_task_order_seed(args)
        39
        >>> args.task_order_seed_source
        'seed'
    """
    raw = getattr(args, "task_order_seed", None)
    if raw is None or (isinstance(raw, str) and len(raw.strip()) == 0):
        args.task_order_seed = int(args.seed)
        args.task_order_seed_source = "seed"
    else:
        args.task_order_seed = int(raw)
        args.task_order_seed_source = "explicit"
    return args.task_order_seed


def init_seed(seed):
    """
    Disable cudnn to maximize reproducibility
    """
    print("Set seed", seed)
    random.seed(seed)
    torch.cuda.cudnn_enabled = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_latest_checkpoint_name(folder_path):
    print("searching for checkpoint in : " + folder_path)
    files = glob.glob(folder_path + "/*.pth")
    min_num = 0
    filename = ""
    for i, filei in enumerate(files):
        ckpt_name = os.path.splitext(filei)
        ckpt_num = int(ckpt_name.split("_")[-1])
        if ckpt_num > min_num:
            min_num = ckpt_num
            filename = filei
    print("latest checkpoint is:")
    print(filename)
    return filename


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError(
            "Unknown type of input, expected torch.Tensor or "
            "np.ndarray, but got {}".format(type(input))
        )


def log_sum_exp(input, dim=None, keepdim=False):
    """Numerically stable LogSumExp.

    Args:
        input (Tensor)
        dim (int): Dimension along with the sum is performed
        keepdim (bool): Whether to retain the last dimension on summing

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        input = input.view(-1)
        dim = 0
    max_val = input.max(dim=dim, keepdim=True)[0]
    output = max_val + (input - max_val).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        output = output.squeeze(dim)
    return output


_REFERENCE_BATCH_SIZE: Final[int] = 256


def scale_learning_rate_for_batch_size(
    base_lr: float,
    batch_size: int,
    reference_batch_size: int = _REFERENCE_BATCH_SIZE,
) -> float:
    """Scale a learning rate linearly with batch size.

    This assumes that ``base_lr`` was tuned for ``reference_batch_size``.
    For example, doubling the batch size from 128 to 256 will double the
    learning rate; halving the batch size will halve the learning rate.

    Args:
        base_lr: Learning rate tuned for ``reference_batch_size``.
        batch_size: Actual training batch size.
        reference_batch_size: Batch size ``base_lr`` corresponds to.

    Returns:
        A scaled learning rate that is proportional to ``batch_size``.

    Usage:
        scaled_lr = scale_learning_rate_for_batch_size(args.lr, args.batch_size)
    """
    if batch_size <= 0 or reference_batch_size <= 0:
        return float(base_lr)
    scale = float(batch_size) / float(reference_batch_size)
    return float(base_lr) * scale


@torch.no_grad()
def proximal_anchor_(
    param: torch.Tensor,
    anchor: torch.Tensor,
    stiffness: torch.Tensor,
    step_size: float,
) -> torch.Tensor:
    """Apply the closed-form proximal step for a quadratic anchor, in place.

    Solves ``argmin_p ||p - param||^2 / (2 * step_size) + sum(stiffness / 2 * (p - anchor)^2)``,
    i.e. ``p = (param + step_size * stiffness * anchor) / (1 + step_size * stiffness)``.
    Unlike an explicit gradient step on the penalty, this never overshoots the
    anchor, so it stays stable for any stiffness. Negative stiffness is clamped
    to zero because the proximal operator is only defined for a convex penalty.

    Args:
        param: Parameter tensor already updated by the task-loss step.
        anchor: Consolidated parameter values to pull towards.
        stiffness: Per-element curvature ``k`` of the penalty ``k / 2 * (p - anchor)^2``.
        step_size: Learning rate of the task-loss step.

    Returns:
        The penalty ``sum(k / 2 * (param - anchor)^2)`` evaluated before anchoring.

    Usage:
        penalty = proximal_anchor_(p, p_star, lamb * fisher, lr)
    """
    stiffness = stiffness.clamp(min=0)
    penalty = 0.5 * (stiffness * (param - anchor).pow(2)).sum()
    rate = step_size * stiffness
    param.add_(rate * anchor).div_(1.0 + rate)
    return penalty
