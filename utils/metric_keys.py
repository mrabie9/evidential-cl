"""Canonical metric key names, with fallbacks to the pre-removal names.

Detection metrics (``det_rec`` / ``det_fa``) and the noise class were removed
repo-wide; the surviving classification metrics were renamed ``cls_* ->
macro_*`` so that result files written before and after the change cannot be
silently mixed. Writers emit only the canonical names below. Readers should go
through :func:`metric_aliases` (or :func:`extract_metric`) so that runs
produced before the rename still load.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

# Canonical key -> legacy names it replaced, newest first.
METRIC_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "tr_macro_rec": ("cls_tr_rec",),
    "val_macro_rec": ("val_acc",),
    "val_macro_f1": ("val_f1", "val_f1_c"),
    "train_macro_rec": ("train_cls_rec", "train_rec", "cls_tr_rec"),
    "train_macro_prec": ("train_cls_prec", "train_prec"),
    "train_macro_f1": ("train_f1", "train_f1_c"),
    "val_macro_rec_per_epoch": ("val_cls_rec", "val_rec", "val_acc"),
    "val_macro_prec_per_epoch": ("val_cls_prec", "val_prec"),
    "val_macro_f1_per_epoch": ("val_f1_per_epoch",),
    "zero_shot_macro_rec": ("zero_shot_rec_cls",),
    "zero_shot_macro_prec": ("zero_shot_prec_cls",),
    "zero_shot_macro_f1": ("zero_shot_f1_cls",),
    "zero_shot_total_macro_f1": ("zero_shot_total_f1",),
    "zero_shot_per_task_macro_rec": ("zero_shot_per_task_rec_cls",),
    "zero_shot_per_task_macro_prec": ("zero_shot_per_task_prec_cls",),
    "zero_shot_per_task_macro_f1": ("zero_shot_per_task_f1_cls",),
    "forward_transfer_total_macro_f1_zs": ("forward_transfer_total_f1_zs",),
    "zero_shot_total_macro_f1_zs": ("zero_shot_total_f1_zs",),
    "baseline_total_macro_f1_zs": ("baseline_total_f1_zs",),
    # seed_metrics.json
    "val_macro_f1_seed": ("val_cls_f1",),
    "tr_macro_f1_seed": ("tr_cls_f1",),
}

# Keys that no longer exist anywhere. Readers should ignore them; writers must
# never emit them again.
REMOVED_METRIC_KEYS: frozenset[str] = frozenset(
    {
        "zero_shot_det",
        "zero_shot_pfa",
        "zero_shot_per_task_det",
        "zero_shot_per_task_pfa",
        "train_det_rec",
        "train_det_pfa",
        "train_det_fa",
        "train_det",
        "train_fa",
        "val_det_rec",
        "val_det_pfa",
        "val_det_acc",
        "val_det_fa",
        "val_det",
        "val_fa",
    }
)


def metric_aliases(canonical_key: str) -> list[str]:
    """Return the lookup order for one canonical metric key.

    Args:
        canonical_key: Current metric name, e.g. ``"val_macro_f1"``.

    Returns:
        The canonical key followed by any legacy names it replaced.

    Usage:
        for key in metric_aliases("val_macro_f1"):
            if key in npz:
                break
    """
    return [canonical_key, *METRIC_KEY_ALIASES.get(canonical_key, ())]


def extract_metric(
    data: Mapping[str, Any] | Any,
    canonical_key: str,
    default: Any = None,
) -> Any:
    """Read one metric from a mapping or NPZ, trying legacy names in order.

    Args:
        data: Any object supporting ``in`` and ``__getitem__`` (``dict``,
            ``np.lib.npyio.NpzFile``, ...).
        canonical_key: Current metric name to look up.
        default: Returned when no alias is present.

    Returns:
        The stored value for the first matching alias, else ``default``.

    Usage:
        val_f1 = extract_metric(np.load(path), "val_macro_f1")
    """
    for key in metric_aliases(canonical_key):
        try:
            if key in data:
                return data[key]
        except TypeError:
            continue
    return default


def first_present_key(
    data: Mapping[str, Any] | Any,
    canonical_keys: Sequence[str],
) -> str | None:
    """Return the first canonical key (or alias) actually present in ``data``.

    Args:
        data: Mapping or NPZ to inspect.
        canonical_keys: Canonical keys to try, in preference order.

    Returns:
        The concrete key name found in ``data``, or ``None``.

    Usage:
        key = first_present_key(npz, ["val_macro_f1", "val_macro_rec"])
    """
    for canonical_key in canonical_keys:
        for key in metric_aliases(canonical_key):
            try:
                if key in data:
                    return key
            except TypeError:
                continue
    return None
