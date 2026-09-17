"""Tests for the shared proximal quadratic anchor (``utils.proximal_anchor``).

Covers the closed-form update's algebra, its stability where explicit descent on
the same penalty diverges, and the wiring into EWC / SI / RWalk / UCL: that the
anchor leaves the backward pass, that it is a no-op before the first
consolidation, and that a large penalty strength pins parameters instead of
blowing them up.
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

from utils.proximal_anchor import (
    ANCHOR_MODES,
    anchor_curvature,
    apply_proximal_anchor,
    optimizer_learning_rate,
    proximal_anchor_coefficient,
    resolve_anchor_mode,
    validate_anchor_mode,
)


# ======================================================================
# Functional core
# ======================================================================
def test_update_matches_the_closed_form_convex_combination() -> None:
    param = torch.tensor([2.0, -1.0])
    importance = torch.tensor([3.0, 7.0])
    anchor = torch.tensor([0.5, 0.5])
    coefficient = 0.25

    expected = (param + importance * coefficient * anchor) / (
        1.0 + importance * coefficient
    )
    apply_proximal_anchor(param, importance, anchor, coefficient)
    assert torch.allclose(param, expected)


def test_zero_importance_leaves_the_parameter_untouched() -> None:
    param = torch.tensor([2.0, -1.0])
    original = param.clone()
    apply_proximal_anchor(param, torch.zeros(2), torch.zeros(2), 0.5)
    assert torch.equal(param, original)


def test_huge_importance_pins_the_parameter_to_its_anchor() -> None:
    param = torch.tensor([100.0])
    anchor = torch.tensor([1.0])
    apply_proximal_anchor(param, torch.tensor([1e12]), anchor, 1.0)
    assert torch.allclose(param, anchor, atol=1e-6)


def test_update_never_overshoots_the_anchor() -> None:
    """The result is a convex combination, so it stays inside [theta, theta*]."""
    param = torch.tensor([5.0])
    anchor = torch.tensor([-3.0])
    for importance in (1e-3, 1.0, 1e3, 1e9):
        moving = param.clone()
        apply_proximal_anchor(moving, torch.tensor([importance]), anchor, 10.0)
        assert anchor.item() <= moving.item() <= param.item()


def test_negative_importance_is_clamped_rather_than_inverting_the_pull() -> None:
    param = torch.tensor([2.0])
    original = param.clone()
    apply_proximal_anchor(param, torch.tensor([-50.0]), torch.tensor([0.0]), 1.0)
    assert torch.equal(param, original)


def test_stable_where_explicit_descent_on_the_same_penalty_diverges() -> None:
    """lr * k * Omega = 4 is outside the explicit-descent window (< 2)."""
    learning_rate, curvature, importance = 0.1, 2.0, 20.0
    anchor = torch.zeros(1)

    explicit = torch.tensor([1.0])
    for _ in range(50):
        explicit = explicit - learning_rate * curvature * importance * (
            explicit - anchor
        )
    assert explicit.abs().item() > 1e6

    proximal = torch.tensor([1.0])
    coefficient = proximal_anchor_coefficient(learning_rate, curvature)
    for _ in range(50):
        apply_proximal_anchor(proximal, torch.tensor([importance]), anchor, coefficient)
    assert proximal.abs().item() < 1.0


def test_curvature_reflects_how_each_method_writes_its_penalty() -> None:
    # si/woe_si fold no 1/2 into lambda; ewc writes an explicit 0.5.
    assert anchor_curvature("si", 0.4) == pytest.approx(0.8)
    assert anchor_curvature("woe_si", 3000.0) == pytest.approx(6000.0)
    assert anchor_curvature("ewc", 100.0) == pytest.approx(100.0)
    assert anchor_curvature("rwalk", 1.0) == pytest.approx(2.0)


def test_curvature_rejects_unknown_methods() -> None:
    with pytest.raises(ValueError):
        anchor_curvature("ucl", 1.0)


def test_anchor_mode_validation() -> None:
    assert set(ANCHOR_MODES) == {"loss", "proximal"}
    assert validate_anchor_mode("proximal") == "proximal"
    with pytest.raises(ValueError):
        validate_anchor_mode("prox")


def test_resolve_anchor_mode_falls_back_when_absent_or_none() -> None:
    class Args:
        pass

    args = Args()
    assert resolve_anchor_mode(args) == "proximal"
    args.anchor_mode = None
    assert resolve_anchor_mode(args) == "proximal"
    args.anchor_mode = "loss"
    assert resolve_anchor_mode(args) == "loss"


def test_optimizer_learning_rate_reads_the_live_group_value() -> None:
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=0.01)
    assert optimizer_learning_rate(optimizer) == pytest.approx(0.01)
    optimizer.param_groups[0]["lr"] = 0.002
    assert optimizer_learning_rate(optimizer) == pytest.approx(0.002)


# ======================================================================
# Learner wiring
# ======================================================================
def _minimal_args(anchor_mode: str, **overrides: object) -> object:
    """Namespace with the fields ``ResNet1D`` and the regularisers expect."""
    args = type("Args", (), {})()
    args.class_incremental = True
    args.classes_per_task = [2, 2]
    args.nc_per_task_list = ""
    args.nc_per_task = None
    args.class_weighted_ce = False
    args.use_iq_aug_features = False
    args.data_scaling = "none"
    args.iq_aug_feature_type = "power"
    args.loader = "task_incremental_loader"
    args.lr = 0.01
    args.optimizer = "sgd"
    args.inner_steps = 1
    args.anchor_mode = anchor_mode
    args.clipgrad = 100.0
    args.cls_lambda = 1.0
    args.norm_track_stats = True
    args.alpha_init = 1e-3
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _batch(n: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    return torch.randn(n, 2, 1024), torch.randint(0, 2, (n,))


def _build(module_name: str, args: object) -> torch.nn.Module:
    import importlib

    return importlib.import_module(f"model.{module_name}").Net(2, 4, 2, args)


LEARNERS = ("ewc", "si", "rwalk")


@pytest.mark.parametrize("module_name", LEARNERS)
def test_proximal_mode_keeps_the_anchor_out_of_the_backward_pass(
    module_name: str,
) -> None:
    """The loss form carries the penalty on task 1; the proximal form does not."""
    penalty_getter = {
        "ewc": "_ewc_penalty",
        "si": "_surrogate_loss",
        "rwalk": "_regulariser",
    }[module_name]
    strength = 1e6
    reported = {}
    for mode in ("loss", "proximal"):
        torch.manual_seed(4321)
        net = _build(module_name, _minimal_args(mode, lamb=strength, si_c=strength))
        x, y = _batch()
        net.observe(x, y, 0)
        net.on_task_end()
        # Task 1 owns the second class slice, so its targets are offset. The
        # first step of a task always reports an anchor of exactly 0 (theta is
        # still theta*), so the comparison needs a step's worth of drift first.
        net.observe(x, y + 2, 1)
        reported[mode] = net.observe(x, y + 2, 1)[0]
        # The penalty itself is still computable in both modes -- proximal only
        # changes where it is applied, not whether the importance exists.
        assert float(getattr(net, penalty_getter)().sum().detach()) >= 0.0

    # A cross-entropy over 4 classes is O(1); the loss form adds a huge anchor.
    assert reported["proximal"] < 100.0
    assert reported["loss"] > reported["proximal"]


@pytest.mark.parametrize("module_name", LEARNERS)
def test_first_task_is_identical_under_both_modes(module_name: str) -> None:
    """No importance exists yet, so the anchor cannot change task 0."""
    outputs = []
    for mode in ("loss", "proximal"):
        torch.manual_seed(1234)
        net = _build(module_name, _minimal_args(mode, lamb=10.0, si_c=10.0))
        x, y = _batch()
        net.observe(x, y, 0)
        outputs.append(
            torch.cat([p.detach().reshape(-1) for p in net.net.parameters()])
        )
    assert torch.allclose(outputs[0], outputs[1], atol=1e-6)


@pytest.mark.parametrize("module_name", LEARNERS)
def test_large_penalty_strength_pins_parameters_instead_of_diverging(
    module_name: str,
) -> None:
    """A strength that makes the loss form diverge must stay finite and bounded."""
    strength = 1e9
    net = _build(
        module_name, _minimal_args("proximal", lamb=strength, si_c=strength, eps=0.01)
    )
    x, y = _batch()
    net.observe(x, y, 0)
    net.on_task_end()
    anchored = {
        name: param.detach().clone() for name, param in net.net.named_parameters()
    }

    for _ in range(3):
        net.observe(x, y + 2, 1)

    for name, param in net.net.named_parameters():
        assert torch.isfinite(param).all(), f"{module_name}: {name} went non-finite"
        drift = (param.detach() - anchored[name]).abs().max().item()
        assert drift < 1.0, f"{module_name}: {name} drifted {drift} despite the anchor"


def test_ucl_proximal_mu_anchor_is_finite_and_bounded() -> None:
    """UCL's mu strength is an inverse sigma, so it is the worst-conditioned."""
    ucl = _build(
        "ucl_bresnet",
        _minimal_args(
            "proximal",
            alpha=1e6,
            beta=0.02,
            ratio=0.1,
            lr_rho=0.005,
            eval_samples=1,
            split=True,
        ),
    )
    x, y = _batch()
    ucl.observe(x, y, 0)
    # UCL uses per-task heads, so task 1's labels live in the second class slice.
    y_next = y + 2
    ucl.observe(x, y_next, 1)
    before = {
        name: param.detach().clone()
        for name, param in ucl.model.named_parameters()
        if name.endswith("weight_mu")
    }
    for _ in range(3):
        ucl.observe(x, y_next, 1)

    for name, param in ucl.model.named_parameters():
        assert torch.isfinite(param).all(), f"ucl: {name} went non-finite"
    for name, snapshot in before.items():
        param = dict(ucl.model.named_parameters())[name]
        assert (param.detach() - snapshot).abs().max().item() < 1.0
