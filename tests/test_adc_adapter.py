import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.resnet1d import AdcIqAdapter, ResNet1D


def test_adapter_shape():
    adapter = AdcIqAdapter()
    x = torch.randn(4, 3, 2, 128)
    y = adapter(x)
    assert y.shape == (4, 2, 128)


def test_prepare_input_3adc_flat():
    model = ResNet1D(num_classes=4)
    x = torch.randn(5, 3, 256)  # (B, 3, 2L)
    out = model._prepare_input(x)
    assert out.shape == (5, 3, 2, 128)


def test_prepare_input_2adc_flat():
    model = ResNet1D(num_classes=4)
    x = torch.randn(6, 256)  # (B, 2L)
    out = model._prepare_input(x)
    assert out.shape == (6, 2, 128)


def test_prepare_input_ambiguous_flat():
    model = ResNet1D(num_classes=4)
    x = torch.randn(3, 12)  # divisible by both 2 and 3
    try:
        model._prepare_input(x)
    except ValueError as exc:
        assert "Ambiguous flat input shape" in str(exc)
    else:
        raise AssertionError("Expected ValueError for ambiguous flat input.")


def test_adapter_known_mix():
    adapter = AdcIqAdapter()
    with torch.no_grad():
        adapter.weight.copy_(torch.tensor([1.0, 0.0, 0.0]))
        adapter.bias.zero_()
    b, l = 2, 16
    i = torch.randn(b, l)  # [b, l]
    q = torch.randn(b, l)  # [b, l]
    x = torch.stack(
        [
            torch.stack([i, q], dim=1),
            torch.randn(b, 2, l),
            torch.randn(b, 2, l),
        ],
        dim=1,
    )  # (B, 3, 2, L)
    y = adapter(x)  # (B, 2, L)
    # Weight selects ADC0 only, using the same coefficients for I and Q.
    assert torch.allclose(y[:, 0], i, atol=1e-6)
    assert torch.allclose(y[:, 1], q, atol=1e-6)


def test_adapter_shares_weights_across_iq():
    """The ADC mixing weights must be identical for the I and Q channels."""
    adapter = AdcIqAdapter()
    with torch.no_grad():
        adapter.weight.copy_(torch.tensor([0.6, 0.3, 0.1]))
        adapter.bias.zero_()
    b, l = 3, 8
    x = torch.randn(b, 3, 2, l)
    y = adapter(x)
    expected = torch.einsum("bal,a->bl", x[:, :, 0, :], adapter.weight)
    assert torch.allclose(y[:, 0], expected, atol=1e-6)
    expected_q = torch.einsum("bal,a->bl", x[:, :, 1, :], adapter.weight)
    assert torch.allclose(y[:, 1], expected_q, atol=1e-6)


def test_adapter_grad_flow():
    adapter = AdcIqAdapter()
    x = torch.randn(3, 3, 2, 8, requires_grad=True)
    y = adapter(x).sum()
    y.backward()
    assert adapter.weight.grad is not None
    assert adapter.weight.grad.abs().sum().item() > 0


def test_resnet1d_integration():
    model = ResNet1D(num_classes=4)
    x3 = torch.randn(2, 3, 2, 32)
    out3 = model(x3)
    assert out3.shape == (2, 4)
    x2 = torch.randn(2, 2, 32)
    out2 = model(x2)
    assert out2.shape == (2, 4)


if __name__ == "__main__":
    test_adapter_shape()
    test_prepare_input_3adc_flat()
    test_prepare_input_2adc_flat()
    test_prepare_input_ambiguous_flat()
    test_adapter_known_mix()
    test_adapter_shares_weights_across_iq()
    test_adapter_grad_flow()
    test_resnet1d_integration()
    print("All adapter tests passed.")
