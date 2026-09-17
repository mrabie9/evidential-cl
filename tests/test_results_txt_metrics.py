"""Tests for the per-seed and cross-seed ``results.txt`` metric sections."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from main import _write_seed_metrics, _write_sweep_summary, save_results
from metrics.metrics import append_metric_block

# Two rounds for task 0, one each for tasks 1 and 2; row 0 is the zero-shot row.
RESULT_T = torch.tensor([0.0, 0.0, 1.0, 2.0])
RECALL = torch.tensor(
    [
        [0.1, 0.2, 0.3, 0.0],
        [0.8, 0.2, 0.3, 0.0],
        [0.7, 0.9, 0.4, 0.0],
        [0.6, 0.8, 0.9, 0.0],
    ]
)[:, :3]


def test_append_metric_block_stats_and_lines(tmp_path: Path) -> None:
    """Stats come from the per-task rows and the block is written after existing text."""
    path = tmp_path / "results.txt"
    path.write_text("existing\n")

    stats = append_metric_block(path, "F1", RESULT_T, RECALL)

    assert stats["diag"] == pytest.approx((0.8 + 0.9 + 0.9) / 3)
    assert stats["final"] == pytest.approx((0.6 + 0.8 + 0.9) / 3)
    assert stats["bwt"] == pytest.approx((-0.2 - 0.1 + 0.0) / 3)
    assert stats["fwt"] == pytest.approx((0.0 + 0.0 + 0.1) / 3)
    lines = path.read_text().splitlines()
    assert lines[0] == "existing"
    assert lines[2].startswith("F1 (")
    assert lines[3] == "0.1000 0.2000 0.3000"
    assert lines[4] == "|"
    assert lines[5:8] == [
        "0.8000 0.2000 0.3000",
        "0.7000 0.9000 0.4000",
        "0.6000 0.8000 0.9000",
    ]
    assert lines[10] == "Backward F1: -0.1000"


def test_append_metric_block_skips_empty_matrix(tmp_path: Path) -> None:
    """Empty or misaligned matrices write nothing and return None."""
    path = tmp_path / "results.txt"
    assert append_metric_block(path, "F1", RESULT_T, torch.empty((0, 0))) is None
    assert append_metric_block(path, "F1", RESULT_T, RECALL[:2]) is None
    assert not path.exists()


PRECISION = RECALL * 0.5
F1_MATRIX = RECALL * 0.25


def _save_synthetic_results(log_dir: Path) -> dict:
    """Run ``save_results`` on the synthetic matrices and return its BWT dict."""
    args = SimpleNamespace(
        log_dir=str(log_dir), state_logging=False, calc_test_accuracy=False
    )
    _, _, val_bwt = save_results(
        args,
        RESULT_T,
        RECALL,
        PRECISION,
        F1_MATRIX,
        torch.empty((0,)),
        torch.empty((0, 0)),
        torch.nn.Linear(2, 2),
        1.0,
    )
    return val_bwt


def test_save_results_writes_all_metrics(tmp_path: Path) -> None:
    """results.txt keeps recall first, adds precision/F1 and returns per-metric BWT."""
    val_bwt = _save_synthetic_results(tmp_path)

    text = (tmp_path / "results.txt").read_text()
    assert "Signal-class" not in text
    assert "Precision (" in text and "F1 (" in text and "Summary (validation):" in text
    first_backward = next(
        line for line in text.splitlines() if line.startswith("Backward")
    )
    assert first_backward == "Backward: -0.1000"
    assert val_bwt["rec"] == pytest.approx(-0.1)
    assert val_bwt["prec"] == pytest.approx(-0.05)
    assert val_bwt["f1"] == pytest.approx(-0.025)
    assert "Backward F1: -0.0250" in text


def test_task_confusion_matrix_reads_metric_blocks(tmp_path: Path) -> None:
    """The perf-matrix script recovers the precision and F1 task matrices."""
    script = pytest.importorskip("scripts.task_confusion_matrix")
    _save_synthetic_results(tmp_path)

    matrices = script.perf_matrices_for_seed(tmp_path)
    assert matrices["cls_prec"] == pytest.approx(PRECISION[[1, 2, 3]].numpy())
    assert matrices["macro_f1"] == pytest.approx(F1_MATRIX[[1, 2, 3]].numpy())


def test_seed_metrics_feed_sweep_summary(tmp_path: Path) -> None:
    """Per-seed recall/precision/F1 and BWT are averaged into the sweep summary."""
    for seed, offset in ((0, 0.0), (1, 0.2)):
        seed_dir = tmp_path / str(seed)
        seed_dir.mkdir()
        headline = {
            "val_macro_rec": 0.5 + offset,
            "val_macro_prec": 0.4 + offset,
            "val_macro_f1": 0.3 + offset,
            "tr_macro_rec": 0.9,
            "tr_macro_prec": 0.8,
            "tr_macro_f1": None,
        }
        val_bwt = {"rec": -0.1 - offset, "prec": -0.2, "f1": None}
        _write_seed_metrics(
            SimpleNamespace(seed=seed, log_dir=str(seed_dir)), 3600.0, headline, val_bwt
        )

    payload = json.loads((tmp_path / "0" / "seed_metrics.json").read_text())
    assert payload["val_macro_prec"] == 0.4
    assert payload["val_bwt_rec"] == -0.1

    _write_sweep_summary(str(tmp_path), [0, 1])
    summary = (tmp_path / "results.txt").read_text()

    def stat(label: str) -> str:
        line = next(x for x in summary.splitlines() if x.strip().startswith(label))
        return line.split(":", 1)[1].split("[")[0].strip()

    assert stat("Validation macro_rec ") == "0.6000 +/- 0.1414"
    assert stat("Validation macro_prec ") == "0.5000 +/- 0.1414"
    assert stat("Validation macro_f1 ") == "0.4000 +/- 0.1414"
    assert stat("Training macro_prec ") == "0.8000 +/- 0.0000"
    assert stat("Validation BWT rec ") == "-0.2000 +/- 0.1414"
    assert stat("Validation BWT prec ") == "-0.2000 +/- 0.0000"
    assert "Training macro_f1" not in summary
    assert "Validation BWT f1" not in summary
    assert "Backward transfer (BWT) mean +/- std:" in summary


HEADLINE = {"val_macro_rec": 0.55, "val_macro_prec": 0.33, "val_macro_f1": 0.22}


def _save_with_headline(log_dir: Path, loader: str) -> str:
    """Run ``save_results`` with a headline dict and return results.txt."""
    args = SimpleNamespace(
        log_dir=str(log_dir),
        state_logging=False,
        calc_test_accuracy=False,
        loader=loader,
    )
    save_results(
        args,
        RESULT_T,
        RECALL,
        PRECISION,
        F1_MATRIX,
        torch.empty((0,)),
        torch.empty((0, 0)),
        torch.nn.Linear(2, 2),
        1.0,
        headline=HEADLINE,
    )
    return (log_dir / "results.txt").read_text()


def test_summary_keeps_row_mean_and_adds_headline(tmp_path: Path) -> None:
    """final stays the row mean (bwt = final - diag); the headline sits beside it."""
    text = _save_with_headline(tmp_path, "class_incremental_loader")
    row_mean = (0.6 + 0.8 + 0.9) / 3
    assert "Final Accuracy: {:.4f}".format(row_mean) in text
    assert "Backward: -0.1000" in text
    summary = text.split("Summary (validation):", 1)[1].splitlines()
    assert summary[1].split() == [
        "metric",
        "diagonal",
        "final",
        "bwt",
        "fwt",
        "headline",
    ]
    rows = {line.split()[0]: line.split()[1:] for line in summary[2:5]}
    diag, final, bwt, _, headline = rows["recall"]
    assert float(final) == pytest.approx(row_mean, abs=1e-4)
    assert float(bwt) == pytest.approx(float(final) - float(diag), abs=1e-4)
    assert headline == "0.5500"
    assert rows["precision"][4] == "0.3300"
    assert rows["f1"][4] == "0.2200"

    bundle = torch.load(tmp_path / "results.pt", weights_only=False)
    assert float(bundle[3][1]) == pytest.approx(row_mean)
    assert "# val: 0.867 0.767 -0.100" in bundle[4]
    assert "# headline: rec=0.5500 prec=0.3300 f1=0.2200" in bundle[4]


def test_summary_without_headline_marks_it_missing(tmp_path: Path) -> None:
    """Callers that pass no headline (main_single_round) get n/a, no one-liner field."""
    _save_synthetic_results(tmp_path)
    text = (tmp_path / "results.txt").read_text()
    recall_row = next(x for x in text.splitlines() if x.startswith("recall "))
    assert recall_row.split()[-1] == "n/a"
    bundle = torch.load(tmp_path / "results.pt", weights_only=False)
    assert "# headline:" not in bundle[4]
