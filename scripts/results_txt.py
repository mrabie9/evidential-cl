"""Read F1 statistics out of a seed's ``results.txt``, in either layout.

Two layouts exist on disk:

* **Legacy** (runs before the noise-class removal merge): the first task matrix
  *is* the F1 matrix. Its stats are ``Diagonal F1`` / ``Final F1`` followed by
  bare ``Backward:`` / ``Forward:`` lines, which therefore belong to F1.
* **Current** (``metrics.metrics.append_metric_block``): the first matrix is
  recall (``Diagonal Accuracy`` / ``Final Accuracy`` / ``Backward:`` /
  ``Forward:``), followed by ``Precision (per-task macro ...)`` and
  ``F1 (per-task macro ...)`` blocks whose stats are ``Backward F1:`` etc.

A plain ``re.search(r"Backward:")`` silently returns *recall* BWT on the current
layout, which is why every analyser should read through this module.

Usage:
    from results_txt import f1_stats, f1_matrix
    stats = f1_stats(open(path).read())   # {"Final F1": 0.46, "Backward": 0.06, ...}
    baseline, R = f1_matrix(open(path).read())
"""

from __future__ import annotations

import re

import numpy as np

_STAT_RE = re.compile(
    r"^(Diagonal F1|Final F1|Backward F1|Forward F1|Backward|Forward):\s+"
    r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|nan)",
    re.MULTILINE,
)
_F1_BLOCK_HEADER = "F1 (per-task macro"


def is_current_layout(text: str) -> bool:
    """True when ``text`` carries a separate F1 block (post-merge layout)."""
    return _F1_BLOCK_HEADER in text


def f1_stats(text: str) -> dict[str, float]:
    """F1 ``Diagonal F1`` / ``Final F1`` / ``Backward`` / ``Forward``, as fractions.

    The BWT and FWT keys keep the legacy names ``Backward`` / ``Forward`` so
    existing callers need no other change. Keys whose value is absent from the
    file are omitted.

    Args:
        text: Full contents of a ``results.txt``.

    Returns:
        Mapping of stat name to value.
    """
    found: dict[str, float] = {}
    for name, value in _STAT_RE.findall(text):
        found.setdefault(name, float(value))
    out = {k: found[k] for k in ("Diagonal F1", "Final F1") if k in found}
    if is_current_layout(text):
        for legacy, current in (("Backward", "Backward F1"), ("Forward", "Forward F1")):
            if current in found:
                out[legacy] = found[current]
    elif "Diagonal F1" in found:
        # Legacy: the first (and only) block is F1, so its bare lines are F1's.
        for legacy in ("Backward", "Forward"):
            if legacy in found:
                out[legacy] = found[legacy]
    return out


def f1_matrix(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(baseline, R)`` for the F1 task matrix.

    ``baseline`` is the pre-training zero-shot row printed above the ``|``
    separator; ``R`` is the ``(T, T)`` matrix printed below it (row ``r`` =
    after training task ``r``).

    Raises:
        ValueError: If the file holds no F1 matrix (e.g. an accuracy-only legacy
            file) or the matrix is not square.
    """
    lines = [ln.strip() for ln in text.splitlines()]
    if is_current_layout(text):
        start = next(i for i, ln in enumerate(lines) if ln.startswith(_F1_BLOCK_HEADER))
        lines = lines[start + 1 :]
    elif not re.search(r"^Diagonal F1:", text, re.MULTILINE):
        raise ValueError("results.txt holds no F1 matrix")
    sep = lines.index("|")
    baseline = np.array([float(v) for v in lines[sep - 1].split()])
    rows = []
    for ln in lines[sep + 1 :]:
        if not ln or not re.match(r"^-?\d", ln):
            break
        rows.append([float(v) for v in ln.split()])
    R = np.array(rows)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"non-square F1 matrix {R.shape}")
    return baseline, R
