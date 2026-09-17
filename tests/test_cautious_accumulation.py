"""Tests for cautious (max) accumulation of importance (``--woe_omega_accum``).

Dempster's rule of combination adds weights of evidence, and is licensed only
for *distinct* bodies of evidence. Sequential tasks are not distinct -- they
share a backbone and each is initialised from the previous solution -- so the
applicable rule is Denoeux's **cautious** rule, whose canonical weight functions
combine by minimum. A weight of evidence is minus the log of that weight
function, so on this scale the cautious rule is an elementwise **maximum**::

    Dempster  (distinct):     Omega_i = sum_t Omega_i^t
    cautious  (non-distinct): Omega_i = max_t Omega_i^t

Covered here: that ``"max"`` really is the elementwise maximum of the per-task
contributions, the idempotence that distinguishes it from summation (relearning
the same task adds nothing), that it leaves the *support* of ``Omega`` unchanged
so the count of anchored parameters is identical to ``"sum"``, and that it never
exceeds ``"sum"`` on real training data. The last three are the properties the
claim in ``docs/woe-cl/README.md`` rests on -- saturation is what online-EWC and
SI patch with hand-chosen decay factors, and a max removes it by construction.
"""

# ruff: noqa: E402

from __future__ import annotations

import math
import os
import sys
from types import SimpleNamespace
from typing import Dict, List

import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.woe_si import Net as WoeSiNet


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _woe_args(**overrides) -> SimpleNamespace:
    """Minimal namespace with the fields ``ResNet1D`` / WoE-SI expect."""
    return SimpleNamespace(
        arch="resnet1d",
        cuda=False,
        use_detector_arch=False,
        class_weighted_ce=False,
        data_scaling="none",
        use_iq_aug_features=False,
        iq_aug_feature_type="power",
        classes_per_task=overrides.get("classes_per_task", [3, 3]),
        nc_per_task_list="",
        nc_per_task=None,
        noise_label=None,
        loader="task_incremental_loader",
        alpha_init=1e-3,
        optimizer="sgd",
        clipgrad=100.0,
        cls_lambda=1.0,
        det_memories=0,
        det_replay_batch=64,
        inner_steps=1,
        lr=overrides.get("lr", 0.01),
        woe_lambda=overrides.get("woe_lambda", 0.0),
        woe_xi=overrides.get("woe_xi", 1e-3),
        woe_centering_mode="centered_uniform",
        woe_mu_momentum=0.9,
        woe_importance_stride=1,
        woe_conflict_weighting=False,
        woe_reg_level=overrides.get("woe_reg_level", "parameter"),
        woe_anchor_mode="proximal",
        woe_omega_winsorise=0.0,
        woe_omega_transform=overrides.get("woe_omega_transform", "abs"),
        woe_omega_accum=overrides.get("woe_omega_accum", "sum"),
        woe_importance_scalar="i2",
        woe_evidence_scale="weight",
        woe_evidence_belief_tau=1.0,
        woe_evidence_asymmetric=False,
        woe_evidence_distill_lambda=0.0,
        woe_lwf_lambda=0.0,
        woe_lwf_temperature=5.0,
        woe_lc_lambda=0.0,
        woe_lc_readout_only=False,
        woe_lc_term="i2",
    )


def _omega(model: WoeSiNet) -> Dict[str, torch.Tensor]:
    """Snapshot of every cumulative ``Omega`` buffer, keyed by parameter name."""
    return {
        name: getattr(model, f"{model._param_to_key[name]}_woe_omega").clone()
        for name in model._tracked_names
    }


def _total_omega(model: WoeSiNet) -> float:
    return sum(float(buf.sum().item()) for buf in _omega(model).values())


def _nonzero_count(model: WoeSiNet) -> int:
    return sum(int((buf > 0).sum().item()) for buf in _omega(model).values())


def _write_path_integral(model: WoeSiNet, value: float) -> None:
    """Fill every per-task accumulator ``omega^t`` with a constant.

    Consolidation divides by ``delta_total^2 + xi``, and on a model whose
    parameters have not moved since the last consolidation ``delta_total`` is
    exactly zero -- so the per-task contribution is exactly ``value / xi`` in
    every element, which makes the combination rule testable in closed form.
    """
    for name in model._tracked_names:
        key = model._param_to_key[name]
        getattr(model, f"{key}_woe_w").fill_(value)


def _consolidate(model: WoeSiNet, task: int) -> None:
    """Run one end-of-task consolidation for ``task``."""
    model.current_task = task
    model._consolidate_current_task()


# ----------------------------------------------------------------------
# The combination rule, in closed form
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "accum,expected_factor",
    [("sum", 0.3 + 0.1 + 0.2), ("max", 0.3)],
)
def test_accumulation_rule_matches_closed_form(
    accum: str, expected_factor: float
) -> None:
    """Three consolidations of known mass combine by sum / by max exactly."""
    torch.manual_seed(0)
    xi = 1e-3
    model = WoeSiNet(2, 6, 2, _woe_args(woe_omega_accum=accum, woe_xi=xi))
    for task, mass in enumerate((0.3, 0.1, 0.2)):
        _write_path_integral(model, mass)
        _consolidate(model, task)
    expected = expected_factor / xi
    for name, buf in _omega(model).items():
        assert torch.allclose(
            buf, torch.full_like(buf, expected), rtol=1e-5
        ), f"{name}: {accum} accumulation did not match the closed form"


def test_cautious_accumulation_is_idempotent() -> None:
    """Re-consolidating identical evidence changes nothing under the max rule.

    This is the property that separates the two rules, and the one the claim
    rests on: Dempster's rule treats a repeated task as fresh evidence and lets
    importance grow without bound (which is why online-EWC and SI need a decay
    factor), while the cautious rule recognises it as the same body of evidence
    and is unmoved.
    """
    torch.manual_seed(1)
    cautious = WoeSiNet(2, 6, 2, _woe_args(woe_omega_accum="max"))
    dempster = WoeSiNet(2, 6, 2, _woe_args(woe_omega_accum="sum"))

    for model in (cautious, dempster):
        _write_path_integral(model, 0.25)
        _consolidate(model, 0)
    after_first = {"max": _total_omega(cautious), "sum": _total_omega(dempster)}
    assert after_first["max"] == pytest.approx(after_first["sum"])

    for model in (cautious, dempster):
        _write_path_integral(model, 0.25)
        _consolidate(model, 1)

    assert _total_omega(cautious) == pytest.approx(after_first["max"])
    assert _total_omega(dempster) == pytest.approx(2.0 * after_first["sum"])


def test_cautious_accumulation_preserves_the_anchored_support() -> None:
    """``max`` changes magnitudes but not *which* parameters are anchored.

    A6 established that ``woe_lambda`` tracks the count of anchored parameters
    rather than their total mass. A maximum of non-negatives is non-zero exactly
    where some term is, so the count is invariant by construction -- meaning the
    two rules differ only in scale, and lambda has to be re-centred by the
    measured mass ratio rather than by a change of support.
    """
    torch.manual_seed(2)
    masses = (0.4, 0.0, 0.15)
    counts: Dict[str, int] = {}
    totals: Dict[str, float] = {}
    for accum in ("sum", "max"):
        model = WoeSiNet(2, 6, 2, _woe_args(woe_omega_accum=accum))
        # Alternate a live and a dead half so the support is a strict subset.
        for task, mass in enumerate(masses):
            for name in model._tracked_names:
                key = model._param_to_key[name]
                buffer = getattr(model, f"{key}_woe_w")
                flat = buffer.reshape(-1)
                flat[: flat.numel() // 2] = mass
                flat[flat.numel() // 2 :] = 0.0
            _consolidate(model, task)
        counts[accum] = _nonzero_count(model)
        totals[accum] = _total_omega(model)

    assert counts["max"] == counts["sum"] > 0
    # 0.4 vs 0.4 + 0.15: strictly less mass over the identical support.
    assert totals["max"] < totals["sum"]


# ----------------------------------------------------------------------
# On the real training path
# ----------------------------------------------------------------------
def test_cautious_omega_never_exceeds_dempster_on_real_batches() -> None:
    """End to end over three tasks: max <= sum elementwise, both finite."""
    torch.manual_seed(3)
    x = torch.randn(8, 2, 128)
    labels = [torch.randint(0, 3, (8,)), torch.randint(3, 6, (8,))]

    snapshots: Dict[str, Dict[str, torch.Tensor]] = {}
    for accum in ("sum", "max"):
        torch.manual_seed(3)
        model = WoeSiNet(2, 6, 2, _woe_args(woe_omega_accum=accum))
        for task, y in enumerate(labels):
            for _ in range(3):
                model.observe(x, y, task)
        model.on_task_end()
        snapshots[accum] = _omega(model)

    totals: List[float] = []
    for name, cautious in snapshots["max"].items():
        dempster = snapshots["sum"][name]
        assert torch.isfinite(cautious).all()
        assert (cautious <= dempster + 1e-6).all(), f"{name}: max exceeded sum"
        totals.append(float(cautious.sum().item()))
    assert math.isfinite(sum(totals))
    assert sum(totals) > 0.0


def test_invalid_accumulation_rule_raises() -> None:
    with pytest.raises(ValueError, match="woe_omega_accum"):
        WoeSiNet(2, 6, 2, _woe_args(woe_omega_accum="bogus"))
