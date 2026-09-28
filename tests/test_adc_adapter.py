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
        # Logits: softmax([0, -inf, -inf]) selects ADC0 exactly.
        adapter.weight.copy_(torch.tensor([0.0, float("-inf"), float("-inf")]))
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
        adapter.weight.copy_(torch.log(torch.tensor([0.6, 0.3, 0.1])))
        adapter.bias.zero_()
    mix = torch.tensor([0.6, 0.3, 0.1])
    b, l = 3, 8
    x = torch.randn(b, 3, 2, l)
    y = adapter(x)
    expected = torch.einsum("bal,a->bl", x[:, :, 0, :], mix)
    assert torch.allclose(y[:, 0], expected, atol=1e-6)
    expected_q = torch.einsum("bal,a->bl", x[:, :, 1, :], mix)
    assert torch.allclose(y[:, 1], expected_q, atol=1e-6)


def test_adapter_grad_flow():
    adapter = AdcIqAdapter()
    x = torch.randn(3, 3, 2, 8, requires_grad=True)
    y = adapter(x).sum()
    y.backward()
    assert adapter.weight.grad is not None
    assert adapter.weight.grad.abs().sum().item() > 0


def test_default_mix_is_uniform():
    adapter = AdcIqAdapter()
    assert torch.allclose(adapter.mixing_weights(), torch.full((3,), 1.0 / 3))


def test_mix_stays_convex_where_the_old_sum_normalisation_blew_up():
    """Regression for the CIL 1-epoch collapse (bcl_dual seed 9).

    The old ``w / w.sum()`` mix turned these logged weights into
    ``[+6.7, -9.4, +3.6]``: summing to 1 but subtractive and ~21x the noise
    gain of the uniform mix. The softmax mix must stay a convex combination.
    """
    adapter = AdcIqAdapter()
    for logged in ([1.61, -2.24, 0.87], [1.28, -2.34, 1.13], [1.0, -1.0, 0.0]):
        with torch.no_grad():
            adapter.weight.copy_(torch.tensor(logged))
        mix = adapter.mixing_weights()
        assert torch.all(mix >= 0)
        assert torch.isclose(mix.sum(), torch.tensor(1.0))
        # A convex mix is never noisier than using one ADC alone.
        assert mix.norm() <= 1.0 + 1e-6


def test_mix_gradient_is_bounded_near_zero_sum():
    adapter = AdcIqAdapter()
    with torch.no_grad():
        adapter.weight.copy_(torch.tensor([1.0, -1.0 + 1e-6, 0.0]))
    x = torch.randn(4, 3, 2, 8)
    adapter(x).pow(2).mean().backward()
    assert torch.isfinite(adapter.weight.grad).all()
    assert adapter.weight.grad.abs().max() < 10.0


def test_conv_path_rows_are_convex():
    adapter = AdcIqAdapter()
    with torch.no_grad():
        adapter.proj_3ch.weight.copy_(
            torch.tensor([[1.0, -1.0, 0.0], [2.0, -3.0, 1.0]]).view(2, 3, 1)
        )
    mix = adapter.conv_mixing_weights()
    assert torch.all(mix >= 0)
    assert torch.allclose(mix.sum(dim=1), torch.ones(2))
    x = torch.randn(2, 3, 16)
    expected = torch.einsum("bcl,oc->bol", x, mix)
    assert torch.allclose(adapter(x), expected, atol=1e-6)


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
