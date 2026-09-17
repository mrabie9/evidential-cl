"""Unit tests for hyperparameter tuner path/writeback safeguards."""

from __future__ import annotations

# ruff: noqa: E402

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tuning import hyperparam_tuner


def test_extract_macro_f1_mean_cil_uses_union_headline_not_column_mean(
    tmp_path: Path,
) -> None:
    """CIL scoring must read the pooled headline, not average per-task columns.

    ``val_macro_f1`` under CIL holds per-task columns sliced from one pooled
    pass (see ``main.eval_cil_pooled``); averaging its tail mixes in tasks the
    model has since forgotten instead of scoring the class-union headline
    stored separately as ``cil_union_macro_f1``.
    """
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    np.savez(
        metrics_dir / "task1.npz",
        val_macro_f1=np.array([0.95, 0.36, 0.06, 0.02]),
        cil_union_macro_f1=np.array([0.95, 0.06]),
    )

    til_score = hyperparam_tuner.extract_macro_f1_mean_from_trial_logs(
        tmp_path, num_tasks=2, cil_mode=False
    )
    cil_score = hyperparam_tuner.extract_macro_f1_mean_from_trial_logs(
        tmp_path, num_tasks=2, cil_mode=True
    )

    assert til_score == (0.06 + 0.02) / 2
    assert cil_score == 0.06


def test_resolve_cli_config_path_uses_caller_cwd(tmp_path: Path) -> None:
    """Relative config paths should resolve from caller working directory."""
    caller_cwd = tmp_path / "scripts"
    caller_cwd.mkdir(parents=True, exist_ok=True)
    target_config = tmp_path / "configs" / "models" / "til" / "hat.yaml"
    target_config.parent.mkdir(parents=True, exist_ok=True)
    target_config.write_text("model: hat\n", encoding="utf-8")

    previous_cwd = Path.cwd()
    try:
        os.chdir(caller_cwd)
        resolved = hyperparam_tuner.resolve_cli_config_path(
            "../configs/models/til/hat.yaml"
        )
    finally:
        os.chdir(previous_cwd)

    assert resolved == target_config.resolve()


def test_run_tuning_records_yaml_error_without_failing(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """YAML writeback failures should not fail a successful tuning run."""
    captured_summaries: list[dict[str, Any]] = []

    cli = SimpleNamespace(
        config=["../configs/models/til/hat.yaml"],
        config_dir=[],
        override=[],
        grid=[],
        num_samples=None,
        search_seed=0,
        max_trials=None,
        shuffle=False,
        tune_only=[],
        hierarchical=False,
        lr_first=False,
        lr_key="lr",
        dry_run=False,
        output_root=str(tmp_path / "out"),
        seed_offset=0,
        vary_seed=False,
        keep_expt_name=False,
        stage2_top_k=0,
        stage2_seeds="",
    )

    class _FakeCliParser:
        def parse_args(self) -> SimpleNamespace:
            return cli

    monkeypatch.setattr(hyperparam_tuner, "build_cli", lambda _preset: _FakeCliParser())
    monkeypatch.setattr(
        hyperparam_tuner.file_parser,
        "parse_args_from_yaml",
        lambda _sources: SimpleNamespace(model="hat", expt_name="hat", seed=0),
    )
    monkeypatch.setattr(hyperparam_tuner, "parse_override_specs", lambda *_: {})
    monkeypatch.setattr(hyperparam_tuner, "expand_trials", lambda *_: [{}])
    monkeypatch.setattr(
        hyperparam_tuner.misc_utils, "get_date_time", lambda: "2026-01-01_00-00-00"
    )
    monkeypatch.setattr(
        hyperparam_tuner,
        "run_single_trial",
        lambda *args, **kwargs: {
            "status": "ok",
            "trial": 0,
            "params": {"lr": 0.001},
            "trial_params": {"lr": 0.001},
            "score": 0.9,
            "duration_sec": 0.1,
            "log_dir": str(tmp_path / "run"),
        },
    )
    monkeypatch.setattr(
        hyperparam_tuner,
        "dump_summary",
        lambda _session_dir, summary, _successes: captured_summaries.append(
            dict(summary)
        ),
    )
    monkeypatch.setattr(
        hyperparam_tuner,
        "write_best_params_to_yaml",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("write failed")),
    )

    preset = hyperparam_tuner.TuningPreset(
        model_name="hat", default_grid={"lr": [0.001]}
    )
    hyperparam_tuner.run_tuning(preset)

    assert captured_summaries, "Expected summary persistence calls."
    assert any(
        summary.get("updated_yaml_error") == "write failed"
        for summary in captured_summaries
    )


def test_select_best_trial_prefers_final_hierarchical_stage() -> None:
    """Hierarchical sweeps should not report an early-stage trial as best."""
    search_space = {
        "lr": [0.01, 0.001],
        "reg_lambda": [10, 100],
    }
    successes = [
        {
            "trial": 1,
            "stage": "lr",
            "score": 0.2,
            "params": {"lr": 0.001},
            "trial_params": {"lr": 0.001},
        },
        {
            "trial": 3,
            "stage": "reg_lambda",
            "score": 0.2,
            "params": {"lr": 0.001, "reg_lambda": 100},
            "trial_params": {"reg_lambda": 100},
            "fixed_params": {"lr": 0.001},
        },
    ]
    best = hyperparam_tuner.select_best_trial(
        successes, search_space, hierarchical=True
    )
    assert best is not None
    assert best["trial"] == 3
    assert best["params"]["reg_lambda"] == 100


def test_dedupe_config_sources_preserves_first_occurrence() -> None:
    """Duplicate --config paths should not be applied twice."""
    first = str(ROOT / "configs/tuning_defaults.yaml")
    second = str((ROOT / "configs/tuning_defaults.yaml").resolve())
    deduped = hyperparam_tuner._dedupe_config_sources([first, second])
    assert len(deduped) == 1


def test_default_config_chain_omits_base_yaml() -> None:
    """Tuning defaults must not pull in full-experiment base.yaml."""
    chain = hyperparam_tuner._default_config_chain("eucr", None)
    names = [Path(path).name for path in chain]
    assert "base.yaml" not in names
    assert "tuning_defaults.yaml" in names
    assert "eucr.yaml" in names


def test_select_best_trial_breaks_score_ties_by_param_completeness() -> None:
    """Equal scores should prefer trials with the full searched parameter set."""
    search_space = {"lr": [0.001], "reg_lambda": [100]}
    successes = [
        {"trial": 3, "stage": "lr", "score": 0.1, "params": {"lr": 0.001}},
        {
            "trial": 16,
            "stage": "reg_lambda",
            "score": 0.1,
            "params": {"lr": 0.001, "reg_lambda": 100, "probe_loss_weight": 1.0},
        },
    ]
    best = hyperparam_tuner.select_best_trial(
        successes, search_space, hierarchical=True
    )
    assert best is not None
    assert best["trial"] == 16
def test_hierarchical_writeback_keeps_earlier_stage_winners(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """Every searched key of the best trial is written, not just its own stage.

    Constant ``--override`` values are not searched, so they must stay out of
    the model YAML.
    """
    written: list[dict[str, Any]] = []

    cli = SimpleNamespace(
        config=[str(tmp_path / "model.yaml")],
        config_dir=[],
        override=["lr=0.01"],
        grid=[],
        num_samples=None,
        search_seed=0,
        max_trials=None,
        shuffle=False,
        tune_only=[],
        hierarchical=True,
        lr_first=False,
        lr_key="lr",
        dry_run=False,
        output_root=str(tmp_path / "out"),
        seed_offset=0,
        vary_seed=False,
        keep_expt_name=False,
        stage2_top_k=0,
        stage2_seeds="",
    )

    class _FakeCliParser:
        def parse_args(self) -> SimpleNamespace:
            return cli

    def _fake_trial(_base_args, overrides, trial_params, trial_idx, *_a, **_k):
        params = dict(overrides, **trial_params)
        score = params.get("a", 0) * 10 + params.get("b", 0)
        return {
            "status": "ok",
            "trial": trial_idx,
            "params": params,
            "trial_params": dict(trial_params),
            "score": float(score),
            "duration_sec": 0.0,
            "log_dir": str(tmp_path / "run"),
        }

    monkeypatch.setattr(hyperparam_tuner, "build_cli", lambda _preset: _FakeCliParser())
    monkeypatch.setattr(
        hyperparam_tuner.file_parser,
        "parse_args_from_yaml",
        lambda _sources: SimpleNamespace(model="m", expt_name="m", seed=0),
    )
    monkeypatch.setattr(
        hyperparam_tuner, "parse_override_specs", lambda *_: {"lr": 0.01}
    )
    monkeypatch.setattr(
        hyperparam_tuner.misc_utils, "get_date_time", lambda: "2026-01-01_00-00-00"
    )
    monkeypatch.setattr(hyperparam_tuner, "run_single_trial", _fake_trial)
    monkeypatch.setattr(hyperparam_tuner, "dump_summary", lambda *_: None)
    monkeypatch.setattr(
        hyperparam_tuner,
        "write_best_params_to_yaml",
        lambda _path, values: written.append(dict(values)) or values,
    )

    preset = hyperparam_tuner.TuningPreset(
        model_name="m", default_grid={"a": [1, 2], "b": [3, 4]}
    )
    hyperparam_tuner.run_tuning(preset)

    assert written == [{"a": 2, "b": 4}]
