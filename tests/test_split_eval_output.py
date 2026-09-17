"""Guard ``_split_eval_output`` against ambiguous per-task metric lists."""

# ruff: noqa: E402

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main import _split_eval_output


def test_three_tuple_from_eval_tasks_is_unpacked_in_order() -> None:
    """``eval_tasks`` / ``eval_class_tasks`` both return a fixed 3-tuple."""
    macro_rec, macro_prec, macro_f1 = _split_eval_output(
        ([0.18, 0.2045], [0.31, 0.33], [0.22, 0.26])
    )
    assert macro_rec == [0.18, 0.2045]
    assert macro_prec == [0.31, 0.33]
    assert macro_f1 == [0.22, 0.26]


def test_three_tuple_tolerates_missing_precision_and_f1() -> None:
    """Evaluators may report recall only; the other two come back as ``None``."""
    macro_rec, macro_prec, macro_f1 = _split_eval_output(([0.18, 0.2045], None, None))
    assert macro_rec == [0.18, 0.2045]
    assert macro_prec is None and macro_f1 is None


def test_bare_per_task_list_is_treated_as_recall_only() -> None:
    """A bare list is per-task recall, never a packed metric tuple."""
    macro_rec, macro_prec, macro_f1 = _split_eval_output([0.18, 0.20])
    assert macro_rec == [0.18, 0.20]
    assert macro_prec is None and macro_f1 is None
