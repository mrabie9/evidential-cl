"""tune_adab1n.py host selection and config-chain ordering."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tuning.presets import TUNING_PRESETS  # noqa: E402
from tuning.tune_adab1n import (  # noqa: E402
    DEFAULT_HOST,
    HOSTS,
    default_config_args,
    split_host,
)


def test_every_host_maps_to_a_real_preset() -> None:
    for preset_key, _, _ in HOSTS.values():
        assert preset_key in TUNING_PRESETS


def test_every_host_config_fragment_exists() -> None:
    for _, host_cfg, adab1n_cfg in HOSTS.values():
        assert (ROOT / host_cfg).is_file(), host_cfg
        assert (ROOT / adab1n_cfg).is_file(), adab1n_cfg


@pytest.mark.parametrize("host", sorted(HOSTS))
def test_adab1n_fragment_is_written_not_the_host_baseline(host: str) -> None:
    """The harness writes best params into the LAST --config file.

    If the host's own fragment came last, a sweep would silently persist
    norm_type=adab1n into that model's baseline config and change every later run
    of it. The AdaB1N fragment must therefore be last, and must not be the host's.
    """
    _, host_cfg, adab1n_cfg = HOSTS[host]
    args = default_config_args(host)

    assert args[-1] == adab1n_cfg
    assert adab1n_cfg != host_cfg
    assert args.index(host_cfg) < args.index(adab1n_cfg)


@pytest.mark.parametrize("host", sorted(HOSTS))
def test_host_selection_round_trips(host: str) -> None:
    assert split_host(["--host", host]) == (host, [])
    assert split_host([f"--host={host}"]) == (host, [])


def test_default_host_when_unspecified() -> None:
    assert split_host(["--seeds", "0"]) == (DEFAULT_HOST, ["--seeds", "0"])


def test_unknown_host_is_rejected() -> None:
    with pytest.raises(SystemExit):
        split_host(["--host", "nope"])
    with pytest.raises(SystemExit):
        split_host(["--host"])
