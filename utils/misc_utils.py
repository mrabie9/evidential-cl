import datetime
import glob
import json
import os
import random
from typing import Final

import numpy as np
import torch


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
    global_noise_label: int | None = None,
    fill_value: float = -1e9,
    loader: str | None = None,
) -> torch.Tensor:
    """Apply task-wise or class-incremental (CIL) evaluation logit masking.

    **Task-incremental (TIL) inference:** only the logit block for ``task_index``
    is left active; past and future classes are masked.

    **CIL evaluation (``cil_all_seen_upto_task`` set):** all **signal** classes
    introduced in tasks ``0..cil_all_seen_upto_task`` (inclusive) stay active;
    only *future* signal logits are masked. If ``global_noise_label`` is set
    (shared IQ noise class), that index stays **unmasked** so the head can
    predict noise jointly with seen classes; otherwise a naive mask
    ``[:, offset2:]`` would zero the noise logit and break detection metrics.

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
        global_noise_label: Optional global noise class index (not counted in
            ``nc_per_task`` / ``compute_offsets``). When set, future-signal mask
            is ``[offset2:noise)`` and ``(noise:]`` instead of ``[offset2:]``.
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
        if global_noise_label is not None and 0 <= int(global_noise_label) < n_outputs:
            gnoise = int(global_noise_label)
            if offset2 < gnoise:
                masked[:, offset2:gnoise].fill_(fill_value)
            tail = gnoise + 1
            if tail < n_outputs:
                masked[:, tail:].fill_(fill_value)
        elif offset2 < n_outputs:
            masked[:, offset2:].fill_(fill_value)
        return masked
    offset1, offset2 = compute_offsets(task_index, nc_per_task)
    if offset1 > 0:
        masked[:, :offset1].fill_(fill_value)
    if offset2 < n_outputs:
        masked[:, offset2:].fill_(fill_value)
    if global_noise_label is not None:
        gnoise = int(global_noise_label)
        if 0 <= gnoise < n_outputs and (gnoise < offset1 or gnoise >= offset2):
            masked[:, gnoise] = logits[:, gnoise]
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
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-2]


def log_dir(opt, timestamp=None, config_name=None):
    if timestamp is None:
        timestamp = get_date_time()

    rand_num = str(random.randint(1, 1001))
    dir_name = config_name if config_name else opt.model
    logdir = opt.log_dir + "/%s/%s-%s/%s" % (
        dir_name,
        opt.expt_name,
        timestamp,
        opt.seed,
    )
    tfdir = opt.log_dir + "/%s/%s-%s/%s/%s" % (
        dir_name,
        opt.expt_name,
        timestamp,
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
