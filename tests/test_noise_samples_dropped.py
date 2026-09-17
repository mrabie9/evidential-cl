"""Regression: noise-labelled samples never reach the model or the label space.

Raw IQ ``.npz`` files encode noise as ``y == -1``. Those rows are dropped at
load time so no logit slot is reserved for noise and every per-task class count
covers signal classes only.
"""

# ruff: noqa: E402

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataloaders.task_incremental_loader import drop_noise_samples


def test_drop_noise_samples_removes_negative_labels() -> None:
    """Rows labelled ``-1`` are discarded and inputs stay aligned with labels."""
    samples = np.arange(12, dtype=float).reshape(6, 2)
    labels = np.array([0, -1, 1, -1, 2, 0])

    kept_samples, kept_labels = drop_noise_samples(samples, labels, "unit train")

    assert kept_labels.tolist() == [0, 1, 2, 0]
    assert kept_samples.tolist() == [[0.0, 1.0], [4.0, 5.0], [8.0, 9.0], [10.0, 11.0]]
    assert (kept_labels >= 0).all()


def test_drop_noise_samples_collapses_two_column_labels() -> None:
    """Legacy ``[N, 2]`` (class, detection) labels collapse to 1D class ids."""
    samples = np.arange(8, dtype=float).reshape(4, 2)
    labels = np.array([[3, 1], [-1, 0], [5, 1], [-1, 0]])

    kept_samples, kept_labels = drop_noise_samples(samples, labels, "unit test")

    assert kept_labels.ndim == 1
    assert kept_labels.tolist() == [3, 5]
    assert kept_samples.tolist() == [[0.0, 1.0], [4.0, 5.0]]


def test_drop_noise_samples_is_a_no_op_without_noise() -> None:
    """A split with no negative labels is returned unchanged."""
    samples = np.arange(6, dtype=float).reshape(3, 2)
    labels = np.array([0, 1, 2])

    kept_samples, kept_labels = drop_noise_samples(samples, labels, "unit clean")

    assert kept_labels.tolist() == labels.tolist()
    assert kept_samples.tolist() == samples.tolist()


def test_assert_no_noise_slot_rejects_a_reserved_extra_class() -> None:
    """``n_outputs`` must equal the summed per-task signal-class counts."""
    from dataloaders.task_incremental_loader import IncrementalLoader

    loader = IncrementalLoader.__new__(IncrementalLoader)
    loader.classes_per_task = [5, 5, 4]

    loader._assert_no_noise_slot(14)  # exact fit: no reserved noise slot

    try:
        loader._assert_no_noise_slot(15)
    except ValueError as error:
        assert "non-signal slot" in str(error)
    else:
        raise AssertionError("expected ValueError for a reserved noise slot")
