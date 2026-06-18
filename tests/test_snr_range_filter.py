"""Tests for SNR-range sample filtering (``--snr_range``).

Covers the ``_filter_samples_by_snr`` helper used by the IQ task-incremental
loader to keep only samples whose per-sample SNR (deeprad ``lbl_tr``/``lbl_te``
column 0) lies inside an inclusive dB window: the inclusive bounds, the
column-0 convention for 2-D label arrays, the no-op paths (no window / no SNR
label), length-mismatch skipping, and the empty-window guard.
"""

# ruff: noqa: E402

import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataloaders.task_incremental_loader import _filter_samples_by_snr


def _make_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three samples at -5, 3 and 12 dB with 2-D (lbl-style) SNR labels."""

    samples = np.arange(6, dtype=np.float32).reshape(3, 2)
    labels = np.array([10, 11, 12], dtype=np.int64)
    snr_labels = np.array([[-5.0, 0.0], [3.0, 0.0], [12.0, 0.0]])
    return samples, labels, snr_labels


def test_inclusive_window_keeps_only_in_range() -> None:
    """Bounds are inclusive and column 0 is read as the SNR."""

    samples, labels, snr_labels = _make_data()
    kept_x, kept_y = _filter_samples_by_snr(
        samples, labels, snr_labels, (-5.0, 3.0), "demo"
    )
    assert kept_x.shape[0] == 2
    assert kept_y.tolist() == [10, 11]


def test_none_window_returns_inputs_unchanged() -> None:
    """A ``None`` window disables filtering."""

    samples, labels, snr_labels = _make_data()
    kept_x, kept_y = _filter_samples_by_snr(samples, labels, snr_labels, None, "demo")
    assert kept_x.shape[0] == 3
    assert kept_y.tolist() == labels.tolist()


def test_missing_snr_label_returns_inputs_unchanged() -> None:
    """Files without an SNR label (e.g. rcn) pass through untouched."""

    samples, labels, _ = _make_data()
    kept_x, kept_y = _filter_samples_by_snr(samples, labels, None, (0.0, 20.0), "rcn")
    assert kept_x.shape[0] == 3
    assert kept_y.tolist() == labels.tolist()


def test_length_mismatch_is_skipped() -> None:
    """A misaligned SNR vector skips filtering rather than corrupting labels."""

    samples, labels, _ = _make_data()
    snr_wrong_length = np.array([0.0, 1.0])
    kept_x, kept_y = _filter_samples_by_snr(
        samples, labels, snr_wrong_length, (0.0, 20.0), "mismatch"
    )
    assert kept_x.shape[0] == 3
    assert kept_y.tolist() == labels.tolist()


def test_one_dimensional_snr_label_supported() -> None:
    """A 1-D SNR vector is treated directly as the SNR in dB."""

    samples, labels, _ = _make_data()
    snr_1d = np.array([-5.0, 3.0, 12.0])
    kept_x, kept_y = _filter_samples_by_snr(samples, labels, snr_1d, (10.0, 20.0), "1d")
    assert kept_x.shape[0] == 1
    assert kept_y.tolist() == [12]


def test_empty_window_raises() -> None:
    """A window that removes every sample fails fast."""

    samples, labels, snr_labels = _make_data()
    with pytest.raises(ValueError, match="removed all"):
        _filter_samples_by_snr(samples, labels, snr_labels, (100.0, 200.0), "empty")
