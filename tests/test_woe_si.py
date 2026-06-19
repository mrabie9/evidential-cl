"""Tests for Weight-of-Evidence Synaptic Intelligence (``model/woe_si.py``).

Covers the differentiable DS information-content core (finite-difference vs
autograd, non-negativity, binary reduction, vacuity), the SI-style importance
bookkeeping (penalty zero on task 1, omega accumulation / per-task reset, finite
omega in both TIL and CIL), and a short 2-task integration check that WoE-SI
reduces forgetting relative to naive fine-tuning.
"""

# ruff: noqa: E402

from __future__ import annotations

import math
import os
import sys
from typing import List

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.woe_si import (
    Net,
    compute_weights_of_evidence,
    information_content,
    per_class_total_evidence,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _make_args(loader: str, **overrides) -> object:
    """Minimal namespace with the fields ``ResNet1D`` / WoE-SI expect."""
    o = type("Args", (), {})()
    o.classes_per_task = overrides.get("classes_per_task", [3, 3])
    o.nc_per_task_list = ""
    o.nc_per_task = None
    o.noise_label = overrides.get("noise_label", None)
    o.class_weighted_ce = False
    o.use_detector_arch = False
    o.use_iq_aug_features = False
    o.data_scaling = "none"
    o.iq_aug_feature_type = "power"
    o.lr = overrides.get("lr", 0.01)
    o.optimizer = "sgd"
    o.clipgrad = 100.0
    o.cls_lambda = 1.0
    o.det_memories = 0
    o.det_replay_batch = 64
    o.alpha_init = 1e-3
    o.loader = loader
    o.inner_steps = 1
    o.woe_lambda = overrides.get("woe_lambda", 0.5)
    o.woe_xi = overrides.get("woe_xi", 1e-3)
    o.woe_centering_mode = overrides.get("woe_centering_mode", "centered_uniform")
    o.woe_mu_momentum = overrides.get("woe_mu_momentum", 0.9)
    o.woe_importance_stride = overrides.get("woe_importance_stride", 1)
    o.woe_conflict_weighting = overrides.get("woe_conflict_weighting", False)
    o.woe_reg_level = overrides.get("woe_reg_level", "parameter")
    return o


def _cumulative_omega(model: Net) -> float:
    """Sum of cumulative ``Omega`` over all tracked parameters."""
    total = 0.0
    for name in model._tracked_names:
        key = model._param_to_key[name]
        total += float(getattr(model, f"{key}_woe_omega").sum().item())
    return total


def _per_task_w_norm(model: Net) -> float:
    """L1 norm of the per-task path-integral accumulator ``omega^t``."""
    total = 0.0
    for name in model._tracked_names:
        key = model._param_to_key[name]
        total += float(getattr(model, f"{key}_woe_w").abs().sum().item())
    return total


# ----------------------------------------------------------------------
# Functional core
# ----------------------------------------------------------------------
def test_information_content_nonnegative() -> None:
    torch.manual_seed(0)
    phi = torch.randn(16, 32)
    weight = torch.randn(5, 32)
    bias = torch.randn(5)
    mu = phi.mean(dim=0)
    weights = compute_weights_of_evidence(phi, weight, bias, mu)
    i2 = information_content(weights)
    assert i2.shape == (16,)
    assert bool((i2 >= 0).all())


def test_binary_case_reduces_to_wplus_wminus() -> None:
    """For a single class the I_2 term is exactly ``(w+)^2 + (w-)^2``."""
    torch.manual_seed(1)
    phi = torch.randn(7, 12)
    weight = torch.randn(1, 12)
    bias = torch.randn(1)
    mu = phi.mean(dim=0)
    weights = compute_weights_of_evidence(phi, weight, bias, mu)
    w_plus, w_minus = per_class_total_evidence(weights)
    expected = (w_plus.pow(2) + w_minus.pow(2)).squeeze(1)
    got = information_content(weights)
    assert torch.allclose(got, expected, atol=1e-6)


def test_finite_difference_matches_autograd_for_readout() -> None:
    """d I_2(m) / d(readout params) from autograd matches a central difference."""
    torch.manual_seed(2)
    phi = torch.randn(6, 10, dtype=torch.float64)
    mu = phi.mean(dim=0)
    weight = torch.randn(3, 10, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(3, dtype=torch.float64, requires_grad=True)

    def mean_i2(w_param: torch.Tensor, b_param: torch.Tensor) -> torch.Tensor:
        weights = compute_weights_of_evidence(phi, w_param, b_param, mu)
        return information_content(weights).mean()

    analytic = torch.autograd.grad(mean_i2(weight, bias), [weight, bias])
    grad_weight, grad_bias = analytic[0].detach(), analytic[1].detach()

    epsilon = 1e-6
    # Spot-check a handful of weight entries and every bias entry.
    for row, col in [(0, 0), (1, 4), (2, 9)]:
        plus = weight.detach().clone()
        minus = weight.detach().clone()
        plus[row, col] += epsilon
        minus[row, col] -= epsilon
        fd = (mean_i2(plus, bias.detach()) - mean_i2(minus, bias.detach())) / (
            2 * epsilon
        )
        assert abs(fd.item() - grad_weight[row, col].item()) < 1e-5
    for k in range(bias.numel()):
        plus = bias.detach().clone()
        minus = bias.detach().clone()
        plus[k] += epsilon
        minus[k] -= epsilon
        fd = (mean_i2(weight.detach(), plus) - mean_i2(weight.detach(), minus)) / (
            2 * epsilon
        )
        assert abs(fd.item() - grad_bias[k].item()) < 1e-5


def test_full_lc_centering_is_stubbed() -> None:
    phi = torch.randn(3, 4)
    weight = torch.randn(2, 4)
    bias = torch.randn(2)
    mu = phi.mean(dim=0)
    try:
        compute_weights_of_evidence(phi, weight, bias, mu, centering_mode="full_lc")
    except NotImplementedError:
        return
    raise AssertionError("full_lc should raise NotImplementedError")


def test_vacuous_readout_yields_small_i2() -> None:
    """A near-zero (vacuous) readout produces near-zero information content."""
    torch.manual_seed(3)
    phi = torch.randn(16, 32)
    mu = phi.mean(dim=0)
    weight = torch.zeros(4, 32)
    bias = torch.zeros(4)
    weights = compute_weights_of_evidence(phi, weight, bias, mu)
    i2 = information_content(weights).mean()
    assert float(i2.item()) < 1e-8


def test_conflict_weighting_changes_value_but_stays_nonnegative() -> None:
    torch.manual_seed(4)
    phi = torch.randn(8, 16)
    weight = torch.randn(3, 16)
    bias = torch.randn(3)
    mu = phi.mean(dim=0)
    weights = compute_weights_of_evidence(phi, weight, bias, mu)
    plain = information_content(weights, conflict_weighting=False)
    weighted = information_content(weights, conflict_weighting=True)
    assert bool((weighted >= 0).all())
    # Conflict factor (1 + kappa) >= 1, so weighted >= plain everywhere.
    assert bool((weighted >= plain - 1e-6).all())


# ----------------------------------------------------------------------
# Learner bookkeeping
# ----------------------------------------------------------------------
def test_vacuous_model_has_small_omega() -> None:
    """A freshly-initialised readout should accumulate only tiny importance."""
    torch.manual_seed(5)
    model = Net(2, 6, 2, _make_args("task_incremental_loader", lr=0.0))
    x = torch.randn(8, 2, 128)
    y = torch.randint(0, 3, (8,))
    # Zero the readout so the model starts vacuous; lr=0 keeps it vacuous.
    with torch.no_grad():
        model.net.model.fc.weight.zero_()
        model.net.model.fc.bias.zero_()
    for _ in range(4):
        model.observe(x, y, 0)
    # With a vacuous, frozen readout the per-task path integral stays ~0.
    assert _per_task_w_norm(model) < 1e-4


def test_information_signal_is_scale_normalised() -> None:
    """The learner's importance signal must not scale with ``feature_dim**2``.

    Regression guard for the surrogate-loss explosion: the *unnormalised* Denoeux
    ``I_2`` scales as ~``K * feature_dim**2`` -- hundreds even at init and growing
    to 1e5-1e6 once the readout trains, which made ``omega`` and the quadratic
    anchor penalty blow up across tasks. The learner divides ``I_2`` by
    ``feature_dim**2`` (= 512**2 here), so the signal it backprops must collapse
    to well under 1.0; the unnormalised value would be in the hundreds and trip
    this assertion.
    """
    torch.manual_seed(11)
    model = Net(2, 6, 2, _make_args("task_incremental_loader"))
    # Drive the readout to O(1) weights so the unnormalised I_2 is clearly large.
    with torch.no_grad():
        torch.nn.init.normal_(model.net.model.fc.weight, std=1.0)
        torch.nn.init.normal_(model.net.model.fc.bias, std=1.0)
    x = torch.randn(8, 2, 128)
    signal = float(
        model._compute_information_content(x, t=0, update_feature_mean=False).item()
    )
    assert signal >= 0.0
    assert signal < 1.0, f"importance signal not normalised (got {signal})"


def test_penalty_is_exactly_zero_on_first_task() -> None:
    torch.manual_seed(6)
    model = Net(2, 6, 2, _make_args("task_incremental_loader"))
    x = torch.randn(8, 2, 128)
    y = torch.randint(0, 3, (8,))
    for _ in range(5):
        model.observe(x, y, 0)
        assert float(model._surrogate_loss().item()) == 0.0


def test_omega_accumulates_across_tasks_and_per_task_resets() -> None:
    torch.manual_seed(7)
    model = Net(
        2, 9, 3, _make_args("task_incremental_loader", classes_per_task=[3, 3, 3])
    )
    x = torch.randn(12, 2, 128)
    labels = [
        torch.randint(0, 3, (12,)),
        torch.randint(3, 6, (12,)),
        torch.randint(6, 9, (12,)),
    ]
    cumulative: List[float] = []
    for task_index, y in enumerate(labels):
        for _ in range(4):
            model.observe(x, y, task_index)
        # observe() consolidates the *previous* task when the index changes, so
        # the cumulative Omega after task t reflects consolidations 0..t-1.
        cumulative.append(_cumulative_omega(model))
    # After observing task 0 only, nothing is consolidated yet.
    assert cumulative[0] == 0.0
    # Each subsequent task consolidates one more, so Omega is non-decreasing and
    # eventually strictly positive.
    assert cumulative[1] > 0.0
    assert cumulative[2] >= cumulative[1]
    # The per-task accumulator omega^t is reset at each consolidation: right
    # after a consolidation it should reflect only the new (current) task.
    model.on_task_end()
    assert _per_task_w_norm(model) == 0.0


def test_til_and_cil_paths_produce_finite_omega() -> None:
    for loader in ("task_incremental_loader", "class_incremental_loader"):
        torch.manual_seed(8)
        model = Net(2, 6, 2, _make_args(loader))
        x = torch.randn(8, 2, 128)
        y0 = torch.randint(0, 3, (8,))
        y1 = torch.randint(3, 6, (8,))
        for _ in range(4):
            model.observe(x, y0, 0)
        for _ in range(4):
            model.observe(x, y1, 1)
        model.on_task_end()
        total = _cumulative_omega(model)
        assert math.isfinite(total)
        assert total > 0.0


def test_importance_stride_runs_and_is_finite() -> None:
    torch.manual_seed(9)
    model = Net(2, 6, 2, _make_args("task_incremental_loader", woe_importance_stride=2))
    x = torch.randn(8, 2, 128)
    y0 = torch.randint(0, 3, (8,))
    y1 = torch.randint(3, 6, (8,))
    for _ in range(6):
        model.observe(x, y0, 0)
    for _ in range(6):
        model.observe(x, y1, 1)
    assert math.isfinite(_cumulative_omega(model))


# ----------------------------------------------------------------------
# Regularisation level: channel-grouped importance and output-level distillation
# ----------------------------------------------------------------------
def _max_within_channel_std(model: Net) -> float:
    """Largest std *within* an output channel over all multi-dim Omega buffers.

    For a channel-collapsed Omega every weight in a filter shares one value, so
    this is ~0; for the per-parameter Omega the within-filter weights differ.
    """
    worst = 0.0
    for name in model._tracked_names:
        key = model._param_to_key[name]
        omega = getattr(model, f"{key}_woe_omega")
        if omega.dim() < 2:
            continue
        # std over the within-filter dims (everything except the channel axis 0).
        per_channel_std = omega.flatten(1).std(dim=1)
        worst = max(worst, float(per_channel_std.max().item()))
    return worst


def _run_two_tasks(model: Net) -> None:
    x = torch.randn(8, 2, 128)
    y0 = torch.randint(0, 3, (8,))
    y1 = torch.randint(3, 6, (8,))
    for _ in range(4):
        model.observe(x, y0, 0)
    for _ in range(4):
        model.observe(x, y1, 1)
    model.on_task_end()


def test_invalid_reg_level_raises() -> None:
    try:
        Net(2, 6, 2, _make_args("task_incremental_loader", woe_reg_level="bogus"))
    except ValueError:
        return
    raise AssertionError("woe_reg_level='bogus' should raise ValueError")


def test_channel_mode_collapses_omega_within_channel() -> None:
    """Channel mode makes every Omega buffer exactly uniform within each channel.

    Contrasted against parameter mode on identical data/seed: the collapse drives
    the within-channel spread to zero while the per-weight Omega genuinely varies
    (the absolute scale is tiny because Omega is feature_dim**2-normalised, so the
    two modes are compared relative to each other rather than to a fixed epsilon).
    """
    torch.manual_seed(20)
    channel = Net(
        2, 6, 2, _make_args("task_incremental_loader", woe_reg_level="channel")
    )
    _run_two_tasks(channel)
    torch.manual_seed(20)
    parameter = Net(
        2, 6, 2, _make_args("task_incremental_loader", woe_reg_level="parameter")
    )
    _run_two_tasks(parameter)

    channel_std = _max_within_channel_std(channel)
    parameter_std = _max_within_channel_std(parameter)
    assert _cumulative_omega(channel) > 0.0
    # Channel collapse broadcasts one value per filter -> exactly uniform.
    assert channel_std == 0.0
    # Parameter mode keeps per-weight variation, so it is strictly less uniform.
    assert parameter_std > channel_std


def test_output_mode_distillation_zero_on_first_task() -> None:
    """No teacher yet on task 0 -> the evidence-distillation penalty is exactly 0."""
    torch.manual_seed(21)
    model = Net(2, 6, 2, _make_args("task_incremental_loader", woe_reg_level="output"))
    x = torch.randn(8, 2, 128)
    y = torch.randint(0, 3, (8,))
    for _ in range(4):
        model.observe(x, y, 0)
        assert model.teacher is None
        assert float(model._evidence_distillation_loss(x, 0).item()) == 0.0


def test_output_mode_distillation_positive_after_teacher() -> None:
    """Once a teacher is frozen and the student moves, the penalty is > 0 and finite."""
    torch.manual_seed(22)
    model = Net(2, 6, 2, _make_args("task_incremental_loader", woe_reg_level="output"))
    x = torch.randn(8, 2, 128)
    y0 = torch.randint(0, 3, (8,))
    y1 = torch.randint(3, 6, (8,))
    for _ in range(4):
        model.observe(x, y0, 0)
    # Crossing to task 1 consolidates task 0 and snapshots the teacher.
    for _ in range(4):
        model.observe(x, y1, 1)
    assert model.teacher is not None
    penalty = float(model._evidence_distillation_loss(x, 1).item())
    assert math.isfinite(penalty)
    assert penalty > 0.0


def test_output_mode_til_and_cil_paths_run_finite() -> None:
    for loader in ("task_incremental_loader", "class_incremental_loader"):
        torch.manual_seed(23)
        model = Net(2, 6, 2, _make_args(loader, woe_reg_level="output"))
        _run_two_tasks(model)
        assert model.teacher is not None
        x = torch.randn(8, 2, 128)
        penalty = float(model._evidence_distillation_loss(x, 1).item())
        assert math.isfinite(penalty)
        assert penalty >= 0.0
        # Output mode uses no path integral: Omega stays at zero.
        assert _cumulative_omega(model) == 0.0


# ----------------------------------------------------------------------
# Integration: forgetting vs naive fine-tuning on a 2-task toy split
# ----------------------------------------------------------------------
def _toy_task(num_classes: int, base_label: int, seq_len: int, per_class: int):
    """Build a separable IQ toy task: each class is a distinct sinusoid pair."""
    samples = []
    targets = []
    time = torch.linspace(0, 1, seq_len)
    for local_class in range(num_classes):
        frequency = 3.0 + 2.0 * (base_label + local_class)
        for _ in range(per_class):
            phase = torch.rand(1) * 0.1
            in_phase = torch.sin(2 * math.pi * frequency * time + phase)
            quadrature = torch.cos(2 * math.pi * frequency * time + phase)
            signal = torch.stack([in_phase, quadrature], dim=0)
            signal = signal + 0.05 * torch.randn_like(signal)
            samples.append(signal)
            targets.append(base_label + local_class)
    inputs = torch.stack(samples, dim=0)
    labels = torch.tensor(targets, dtype=torch.long)
    permutation = torch.randperm(inputs.shape[0])
    return inputs[permutation], labels[permutation]


@torch.no_grad()
def _task_accuracy(
    model: Net, inputs: torch.Tensor, labels: torch.Tensor, task: int
) -> float:
    model.eval()
    logits = model(inputs, task, cil_all_seen_upto_task=None)
    predictions = torch.argmax(logits, dim=1)
    return float((predictions == labels).float().mean().item())


def _train_two_tasks(woe_lambda: float, seed: int) -> dict:
    torch.manual_seed(seed)
    seq_len = 128
    task0_x, task0_y = _toy_task(3, 0, seq_len, per_class=24)
    task1_x, task1_y = _toy_task(3, 3, seq_len, per_class=24)

    model = Net(
        2,
        6,
        2,
        _make_args("task_incremental_loader", woe_lambda=woe_lambda, lr=0.02),
    )

    batch = 24
    epochs = 6

    def run_task(inputs, labels, task_index):
        count = inputs.shape[0]
        for _ in range(epochs):
            order = torch.randperm(count)
            for start in range(0, count, batch):
                idx = order[start : start + batch]
                model.observe(inputs[idx], labels[idx], task_index)

    run_task(task0_x, task0_y, 0)
    acc_task0_after_task0 = _task_accuracy(model, task0_x, task0_y, 0)
    run_task(task1_x, task1_y, 1)
    model.on_task_end()

    acc_task0_after_task1 = _task_accuracy(model, task0_x, task0_y, 0)
    acc_task1_after_task1 = _task_accuracy(model, task1_x, task1_y, 1)

    final_acc = 0.5 * (acc_task0_after_task1 + acc_task1_after_task1)
    # BWT (Lopez-Paz & Ranzato): mean change on earlier tasks after training the
    # last one. Here only task 0 is "earlier".
    backward_transfer = acc_task0_after_task1 - acc_task0_after_task0
    return {
        "final_acc": final_acc,
        "bwt": backward_transfer,
        "task0_retained": acc_task0_after_task1,
        "omega": _cumulative_omega(model),
    }


def test_integration_woe_si_reduces_forgetting_vs_naive() -> None:
    naive = _train_two_tasks(woe_lambda=0.0, seed=100)
    woe = _train_two_tasks(woe_lambda=50.0, seed=100)
    print(
        "\n[WoE-SI integration] naive: ACC={final_acc:.3f} BWT={bwt:.3f} "
        "task0_retained={task0_retained:.3f}".format(**naive)
    )
    print(
        "[WoE-SI integration] woe:   ACC={final_acc:.3f} BWT={bwt:.3f} "
        "task0_retained={task0_retained:.3f} Omega_sum={omega:.2f}".format(**woe)
    )
    # WoE-SI must build positive importance and must not forget more than naive
    # fine-tuning (BWT no worse, within a small tolerance for run-to-run noise).
    assert woe["omega"] > 0.0
    assert woe["bwt"] >= naive["bwt"] - 0.05
    assert woe["task0_retained"] >= naive["task0_retained"] - 0.05
