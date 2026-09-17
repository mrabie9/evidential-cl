"""AdcIqAdapter: the zero-ADC short-circuit is per row, not per batch."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.resnet1d import AdcIqAdapter  # noqa: E402


def _adapter() -> AdcIqAdapter:
    adapter = AdcIqAdapter()
    with torch.no_grad():
        # Weights whose ADC0 entry normalises well away from 1, so the mixing
        # path is clearly distinguishable from the identity path. Shared
        # across the I and Q channels.
        adapter.weight.copy_(torch.tensor([1.7, 0.8, 0.8]))
    return adapter


def _padded_rows(n: int) -> torch.Tensor:
    """(n, 3, 2, L) rows carrying ADC0 only, with ADC1/ADC2 exactly zero."""
    x = torch.zeros(n, 3, 2, 16)
    x[:, 0] = torch.randn(n, 2, 16)
    return x


def _three_adc_row() -> torch.Tensor:
    return torch.randn(1, 3, 2, 16)


def test_padded_rows_are_unchanged_by_batch_composition() -> None:
    """The regression: one genuine 3-ADC row must not rescale its batch mates."""
    torch.manual_seed(0)
    adapter = _adapter()
    padded = _padded_rows(8)

    alone = adapter(padded)
    with_real = adapter(torch.cat([padded, _three_adc_row()]))[:8]

    assert torch.allclose(alone, with_real, atol=1e-6)


def test_padded_rows_keep_their_adc0_channels() -> None:
    torch.manual_seed(0)
    adapter = _adapter()
    padded = _padded_rows(4)

    out = adapter(torch.cat([padded, _three_adc_row()]))[:4]

    assert torch.allclose(out, padded[:, 0, :, :], atol=1e-6)


def test_three_adc_rows_still_get_the_learned_mixing() -> None:
    torch.manual_seed(0)
    adapter = _adapter()
    real = _three_adc_row()

    out = adapter(torch.cat([_padded_rows(4), real]))[4:]

    weight = adapter.weight
    normalized = weight / weight.sum()
    expected = torch.einsum("bial,a->bil", real.permute(0, 2, 1, 3), normalized)
    assert torch.allclose(out, expected, atol=1e-6)
    assert not torch.allclose(out, real[:, 0, :, :], atol=1e-3)


def test_all_padded_batch_is_identity() -> None:
    torch.manual_seed(0)
    adapter = _adapter()
    padded = _padded_rows(6)

    assert torch.allclose(adapter(padded), padded[:, 0, :, :], atol=1e-6)


def test_gradients_reach_the_mixing_weights() -> None:
    torch.manual_seed(0)
    adapter = _adapter()
    batch = torch.cat([_padded_rows(4), _three_adc_row()])

    adapter(batch).sum().backward()

    assert adapter.weight.grad is not None
    assert torch.isfinite(adapter.weight.grad).all()
    assert adapter.weight.grad.abs().sum() > 0
