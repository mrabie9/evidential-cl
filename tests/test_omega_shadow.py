"""Tests for the shadow path integrals (``WOE_OMEGA_SHADOW``).

The B6 comparison ran one trajectory per tracked scalar, so a disagreement
between two ``Omega`` fields mixes "the scalars select different parameters"
with "the arms took different paths at different anchor strengths". A shadow
path integral removes the second half: extra scalars are differentiated at the
same importance windows, against the same displacements, on the trajectory the
*live* scalar is driving. Nothing shadowed reaches the loss.

Covered here: exact agreement with the live buffer when the shadow is the
tracked scalar itself, the additivity that makes ``logit``/``conflict`` a
decomposition rather than two loosely related probes, isolation from the live
trajectory, and inertness when the env var is unset.
"""

# ruff: noqa: E402

from __future__ import annotations

import os
import sys

import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.woe_si import Net as WoeSiNet
from tests.test_least_commitment import _iq_batch, _woe_args


def _run(model: WoeSiNet, steps: int = 3, rows: int = 4) -> None:
    torch.manual_seed(3)
    x, y = _iq_batch(rows=rows), torch.randint(0, 3, (rows,), dtype=torch.long)
    for _ in range(steps):
        model.observe(x, y, 0)


def test_inert_without_the_env_var() -> None:
    """No buffers, hence no extra forward/backward, when unset."""
    torch.manual_seed(0)
    model = WoeSiNet(1024, 6, 2, _woe_args())
    _run(model)
    assert model._shadow_scalars == ()
    assert model._shadow_w == {} and model._shadow_omega == {}


def test_rejects_an_unknown_scalar(monkeypatch) -> None:
    monkeypatch.setenv("WOE_OMEGA_SHADOW", "i2,nonsense")
    with pytest.raises(ValueError, match="WOE_OMEGA_SHADOW"):
        WoeSiNet(1024, 6, 2, _woe_args())


def test_shadowing_the_tracked_scalar_reproduces_the_live_omega(monkeypatch) -> None:
    """``i2`` shadowed must equal ``i2`` tracked, elementwise.

    This is what licenses reading any *other* shadow field as a counterfactual
    ``Omega``: the machinery is validated against the one case whose answer is
    known. It holds exactly rather than approximately because the shadow reuses
    the live centring reference -- the feature mean is EMA-updated by the
    importance path before the scalar is formed, and the shadow deliberately
    does not update it again -- and runs at ``bn_training=False``, which also
    switches dropout off, so the second forward is deterministic.
    """
    monkeypatch.setenv("WOE_OMEGA_SHADOW", "i2")
    torch.manual_seed(1)
    model = WoeSiNet(1024, 6, 2, _woe_args())
    _run(model)
    model.on_task_end()

    for name in model._tracked_names:
        live = getattr(model, f"{model._param_to_key[name]}_woe_omega")
        shadow = model._shadow_omega["i2"][name]
        assert torch.allclose(live, shadow, atol=0.0, rtol=0.0)


def test_the_halves_sum_to_the_whole_path_integral(monkeypatch) -> None:
    """``omega_logit + omega_conflict == omega_i2`` before the projection.

    ``I_2 = ||z'||^2 + 2 sum_k w+_k w-_k`` is an identity, so the gradient
    fields add and the path integral -- linear in the gradient -- adds with
    them. The identity is asserted on ``w`` rather than on ``Omega`` because
    consolidation applies ``abs``/``relu``, which is not additive: two halves
    that partly cancel produce ``Omega`` mass the sum never sees. That
    non-additivity is precisely what makes the near-collinearity question
    empirical rather than algebraic.
    """
    monkeypatch.setenv("WOE_OMEGA_SHADOW", "i2,logit,conflict")
    torch.manual_seed(2)
    model = WoeSiNet(1024, 6, 2, _woe_args())
    _run(model)

    for name in model._tracked_names:
        halves = model._shadow_w["logit"][name] + model._shadow_w["conflict"][name]
        whole = model._shadow_w["i2"][name]
        assert torch.allclose(halves, whole, atol=1e-6, rtol=1e-4)


def test_shadowing_does_not_move_the_trajectory(monkeypatch) -> None:
    """A diagnostic that perturbed the run it observes would be worthless.

    The risks are concrete: an extra forward could advance the dropout stream or
    the BatchNorm running statistics, and an extra EMA update could move the
    centring reference. All three are ruled out by comparing parameters against
    an identically seeded run with shadowing off.
    """

    def _trained() -> WoeSiNet:
        torch.manual_seed(4)
        model = WoeSiNet(1024, 6, 2, _woe_args(woe_lambda=1.0))
        _run(model)
        model.on_task_end()
        _run(model)
        return model

    monkeypatch.delenv("WOE_OMEGA_SHADOW", raising=False)
    plain = _trained()
    monkeypatch.setenv("WOE_OMEGA_SHADOW", "logit,conflict,ce,z2,phi2")
    shadowed = _trained()

    assert shadowed._shadow_scalars == ("logit", "conflict", "ce", "z2", "phi2")
    for (name, a), (_, b) in zip(
        plain.net.named_parameters(), shadowed.net.named_parameters()
    ):
        assert torch.equal(a, b), f"shadowing moved {name}"
    assert torch.equal(plain.woe_feature_mean, shadowed.woe_feature_mean)


def test_gradient_geometry_records_norms_and_cosines(monkeypatch) -> None:
    """The gradient-side companion to the ``[LC] split`` value shares.

    ``cos_live_i2`` is the self-check: the live gradient and a shadow of the
    same scalar are the same vector, so their mean cosine must be exactly 1 per
    window. If it ever is not, the shadow forward has drifted from the live one
    and every other number in the block is suspect.
    """
    monkeypatch.setenv("WOE_LC_DEBUG", "1")
    monkeypatch.setenv("WOE_OMEGA_SHADOW", "i2,logit,conflict")
    torch.manual_seed(5)
    model = WoeSiNet(1024, 6, 2, _woe_args())
    _run(model, steps=2)

    assert model._grad_stat_steps == 2
    assert model._grad_stat_sums["cos_live_i2"] == pytest.approx(2.0, abs=1e-5)
    for half in ("logit", "conflict"):
        assert model._grad_stat_sums[f"norm_{half}"] > 0.0
    model.on_task_end()
    assert model._grad_stat_steps == 0 and model._grad_stat_sums == {}


def test_gradient_geometry_is_inert_without_debug(monkeypatch) -> None:
    monkeypatch.delenv("WOE_LC_DEBUG", raising=False)
    monkeypatch.setenv("WOE_OMEGA_SHADOW", "logit,conflict")
    torch.manual_seed(6)
    model = WoeSiNet(1024, 6, 2, _woe_args())
    _run(model, steps=2)
    assert model._grad_stat_steps == 0 and model._grad_stat_sums == {}
