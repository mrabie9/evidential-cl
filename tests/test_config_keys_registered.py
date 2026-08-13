"""Every key in every YAML under ``configs/`` must reach the run.

``parser._apply_config_overrides`` builds its namespace with
``parser.parse_args([])``, so the namespace holds exactly the registered
argparse dests and nothing else. A YAML key that no ``add_argument`` declares
is therefore inapplicable -- and before this guard existed it was skipped in
silence, which is how ``configs/models/til/si.yaml``'s ``si_c: 0.4`` came to
have no effect on any run while the file read as if it did. Seven models were
affected the same way (SI, UCL, RWalk, HAT, EWC, LwF, BCL-dual).

The runtime check in ``_apply_config_overrides`` catches ad-hoc configs at
launch. This catches the committed ones without running anything: it globs the
tree, so a config added later is covered with no list to maintain.
"""

# ruff: noqa: E402

from __future__ import annotations

import glob
import os
import sys
from typing import List, Tuple

import pytest
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from parser import (
    CONFIG_KEY_ALIASES,
    INTENTIONALLY_UNUSED_CONFIG_KEYS,
    _apply_config_overrides,
    get_parser,
)


def _config_paths() -> List[str]:
    """Every YAML under ``configs/``, sorted for a stable failure message."""
    return sorted(
        glob.glob(os.path.join(ROOT, "configs", "**", "*.yaml"), recursive=True)
    )


def _registered_dests() -> set[str]:
    """The dests the parser declares -- exactly what a config key can target."""
    return set(vars(get_parser().parse_args([])))


def test_configs_directory_is_not_empty() -> None:
    """Guard the guard: a broken glob would make every other check vacuous."""
    assert len(_config_paths()) > 10


def test_every_config_key_is_registered_or_declared_inert() -> None:
    registered = _registered_dests()
    offenders: List[Tuple[str, str]] = []
    for path in _config_paths():
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        for key in data:
            if key in registered or key in CONFIG_KEY_ALIASES:
                continue
            if key in INTENTIONALLY_UNUSED_CONFIG_KEYS:
                continue
            offenders.append((os.path.relpath(path, ROOT), key))

    assert not offenders, (
        "Config key(s) that no argparse argument declares, so they would have "
        "no effect on the run:\n"
        + "\n".join(f"    {path}: {key}" for path, key in offenders)
        + "\nRegister the argument in parser.get_parser(), remove the key, or "
        "add it to parser.INTENTIONALLY_UNUSED_CONFIG_KEYS with a comment "
        "explaining why it is inert."
    )


def test_aliases_and_inert_keys_do_not_overlap_registered_dests() -> None:
    """An alias or inert key that is also a dest would never be reached."""
    registered = _registered_dests()
    assert not (set(CONFIG_KEY_ALIASES) & registered)
    assert not (INTENTIONALLY_UNUSED_CONFIG_KEYS & registered)


def test_alias_targets_are_themselves_registered() -> None:
    """An alias pointing at an unregistered dest would silently do nothing."""
    registered = _registered_dests()
    for source, target in CONFIG_KEY_ALIASES.items():
        assert target in registered, f"alias {source!r} targets unknown dest {target!r}"


def test_unrecognised_key_raises_rather_than_being_skipped(tmp_path) -> None:
    """The failure mode this whole module exists to prevent."""
    config = tmp_path / "bad.yaml"
    config.write_text("lr: 0.01\ndefinitely_not_an_argument: 7\n", encoding="utf-8")

    args = get_parser().parse_args([])
    with pytest.raises(ValueError, match="definitely_not_an_argument"):
        _apply_config_overrides(args, [config])


def test_recognised_keys_are_applied(tmp_path) -> None:
    """The positive case, so the guard cannot pass by rejecting everything."""
    config = tmp_path / "good.yaml"
    config.write_text("lr: 0.0123\nglances: 4\n", encoding="utf-8")

    args = _apply_config_overrides(get_parser().parse_args([]), [config])
    assert args.lr == pytest.approx(0.0123)
    assert args.inner_steps == 4  # applied through CONFIG_KEY_ALIASES


@pytest.mark.parametrize(
    "key, config_fragment",
    [
        ("si_c", "configs/models/til/si.yaml"),
        ("lamb", "configs/models/til/ewc.yaml"),
        ("alpha", "configs/models/til/rwalk.yaml"),
        ("gamma", "configs/models/til/hat.yaml"),
        ("smax", "configs/models/til/hat.yaml"),
        ("beta", "configs/models/til/ucl.yaml"),
        ("ratio", "configs/models/til/ucl.yaml"),
        ("lr_rho", "configs/models/til/ucl.yaml"),
    ],
)
def test_previously_dropped_keys_now_reach_the_namespace(
    key: str, config_fragment: str
) -> None:
    """Regression pins for the specific keys that were silently discarded."""
    import parser as file_parser

    args = file_parser.parse_args_from_yaml(["configs/base.yaml", config_fragment])
    with open(os.path.join(ROOT, config_fragment), "r", encoding="utf-8") as handle:
        expected = (yaml.safe_load(handle) or {})[key]
    assert getattr(args, key) == pytest.approx(float(expected))
