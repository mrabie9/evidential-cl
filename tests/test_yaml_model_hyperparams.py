"""YAML model hyper-parameters must reach each model's config dataclass.

``parser._apply_config_overrides`` used to drop every YAML key without a
``get_parser`` argument. Model-specific hyper-parameters have no CLI flag, so
``lamb`` (EWC/RWalk), ``si_c`` (SI), ``distill_lambda`` (LwF), ``gamma``/``smax``
(HAT) and the UCL coefficients never left the YAML: every production run used
the dataclass defaults while the tuner, which sets attributes directly, tuned
values that were then ignored.
"""

from __future__ import annotations

# ruff: noqa: E402

import dataclasses
import importlib
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import parser as file_parser

MODEL_CONFIGS = sorted((ROOT / "configs" / "models").glob("*/*.yaml"))


def _config_dataclass(model_name: str):
    """Return the model module's ``*Config`` dataclass exposing ``from_args``."""
    try:
        module = importlib.import_module(f"model.{model_name}")
    except Exception as exc:  # pragma: no cover - optional heavy deps
        pytest.skip(f"model.{model_name} not importable: {exc}")
    for value in vars(module).values():
        if (
            dataclasses.is_dataclass(value)
            and isinstance(value, type)
            and value.__module__ == module.__name__
            and hasattr(value, "from_args")
        ):
            return value
    return None


def test_unknown_yaml_key_is_applied(tmp_path: Path) -> None:
    cfg = tmp_path / "m.yaml"
    cfg.write_text("distill_lambda: 7.5\nlr: 0.5\n", encoding="utf-8")
    args = file_parser.parse_args_from_yaml([str(cfg)])
    assert args.distill_lambda == 7.5
    assert args.lr == 0.5


def test_cli_still_overrides_yaml_for_parser_arguments(tmp_path: Path) -> None:
    cfg = tmp_path / "m.yaml"
    cfg.write_text("lr: 0.5\nsi_c: 0.4\n", encoding="utf-8")
    base = file_parser.parse_args_from_yaml([str(cfg)])
    args = file_parser.get_parser().parse_args(["--lr", "0.1"], namespace=base)
    assert args.lr == 0.1
    assert args.si_c == 0.4


@pytest.mark.parametrize(
    "config_path", MODEL_CONFIGS, ids=lambda p: f"{p.parent.name}/{p.stem}"
)
def test_model_yaml_values_reach_config_dataclass(config_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    model_name = data.get("model")
    if not model_name:
        pytest.skip("config names no model")
    config_cls = _config_dataclass(model_name)
    if config_cls is None:
        pytest.skip(f"model.{model_name} has no config dataclass")

    args = file_parser.parse_args_from_yaml([str(config_path)])
    cfg = config_cls.from_args(args)
    fields = {f.name for f in dataclasses.fields(config_cls)}
    # Deliberately inert keys (see parser.INTENTIONALLY_UNUSED_CONFIG_KEYS) are
    # decided elsewhere, so the dataclass is not expected to echo them.
    checked = {
        key: value
        for key, value in data.items()
        if key in fields and key not in file_parser.INTENTIONALLY_UNUSED_CONFIG_KEYS
    }
    mismatched = {
        key: (value, getattr(cfg, key))
        for key, value in checked.items()
        if getattr(cfg, key) != value
    }
    assert not mismatched, f"YAML value lost (yaml, config): {mismatched}"
