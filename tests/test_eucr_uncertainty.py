"""Unit tests for the EUCR Dempster combination fix and uncertainty readout.

Fast, CPU-only, no data loading: exercises the corrected ignorance combination
in :class:`Dempster_layer` and the nonspecificity/discord/both readouts in
:mod:`model.eucr_consolidation`.
"""

from __future__ import annotations

# ruff: noqa: E402

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.eucr_consolidation import _stage_uncertainty
from model.evidential_modules import Dempster_layer


def _make_masses(mass_list: list[list[float]]) -> torch.Tensor:
    """Stack per-prototype mass vectors into a ``[1, P, C+1]`` batch tensor."""
    return torch.tensor(mass_list, dtype=torch.float64).unsqueeze(0)


def test_dempster_ignorance_is_not_triple_counted() -> None:
    """The combined ignorance must be omega1*omega2, not 3*omega1*omega2."""
    # Two prototypes, two classes (+ omega). m = [c0, c1, omega].
    m1 = [0.5, 0.2, 0.3]
    m2 = [0.1, 0.6, 0.3]
    layer = Dempster_layer(n_prototypes=2, num_class=2)
    combined = layer(_make_masses([m1, m2]))[0]

    # Correct (unnormalised) Dempster terms:
    omega = m1[2] * m2[2]  # 0.09
    c0 = m1[0] * m2[0] + m1[0] * m2[2] + m1[2] * m2[0]
    c1 = m1[1] * m2[1] + m1[1] * m2[2] + m1[2] * m2[1]
    total = c0 + c1 + omega
    expected = torch.tensor(
        [c0 / total, c1 / total, omega / total], dtype=torch.float64
    )

    assert torch.allclose(combined, expected, atol=1e-9)
    # The old triple-count would have put 3*omega/(...) here; ensure we are well
    # below that inflated value.
    assert combined[-1] < (3 * omega) / total


def test_combination_preserves_unit_mass() -> None:
    layer = Dempster_layer(n_prototypes=3, num_class=3)
    masses = _make_masses(
        [[0.3, 0.1, 0.2, 0.4], [0.2, 0.2, 0.1, 0.5], [0.05, 0.05, 0.7, 0.2]]
    )
    combined = layer(masses)[0]
    assert torch.allclose(
        combined.sum(), torch.tensor(1.0, dtype=torch.float64), atol=1e-9
    )
    assert (combined >= 0).all()


def test_discord_uniform_is_one_and_peaked_is_zero() -> None:
    classes = 4
    uniform = torch.full((1, classes), 1.0 / classes)
    omega0 = torch.zeros(1)
    discord_uniform = _stage_uncertainty(uniform, omega0, "discord")
    assert math.isclose(discord_uniform.item(), 1.0, abs_tol=1e-4)

    peaked = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    discord_peaked = _stage_uncertainty(peaked, omega0, "discord")
    assert discord_peaked.item() < 1e-3


def test_nonspecificity_returns_omega_and_both_sums() -> None:
    beliefs = torch.tensor([[0.4, 0.3, 0.1, 0.1]])  # sums to 1 - omega = 0.9
    omega = torch.tensor([0.1])
    nonspec = _stage_uncertainty(beliefs, omega, "nonspecificity")
    discord = _stage_uncertainty(beliefs, omega, "discord")
    both = _stage_uncertainty(beliefs, omega, "both")
    assert torch.allclose(nonspec, omega)
    assert torch.allclose(both, nonspec + discord)


def test_discord_survives_when_omega_is_zero() -> None:
    """The whole point: discord stays informative once the chain kills omega."""
    omega = torch.zeros(1)
    confident = _stage_uncertainty(torch.tensor([[0.9, 0.05, 0.05]]), omega, "discord")
    conflicted = _stage_uncertainty(
        torch.tensor([[0.34, 0.33, 0.33]]), omega, "discord"
    )
    assert conflicted.item() > confident.item()
