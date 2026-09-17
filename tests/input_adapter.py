#! /usr/bin/env python3
# ruff: noqa: E402
import os
import sys

# Add repo root to sys.path so `model` can be imported when this file is run directly
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import pytest
from model.resnet1d import AdcIqAdapter


def test_adc_iq_adapter_shape_and_type():
    adapter = AdcIqAdapter()
    x = torch.randn(5, 3, 2, 128, dtype=torch.float32)  # (B, 3, 2, L)

    y = adapter(x)

    assert y.shape == (5, 2, 128)
    assert y.dtype == x.dtype
    assert y.device == x.device


def test_adc_iq_adapter_linear_mixing():
    B, L = 2, 4
    adapter = AdcIqAdapter()

    # Make mixing easy to reason about: both IQ channels use
    # (adc0 + adc1 + adc2) / 3, the same weights for I and Q.
    with torch.no_grad():
        adapter.weight.zero_()
        adapter.bias.zero_()
        adapter.weight.copy_(torch.tensor([1 / 3, 1 / 3, 1 / 3]))  # sum = 1

    # Build an input with simple structure so we can predict the output
    x = torch.zeros(B, 3, 2, L)
    # adc0 = 1, adc1 = 2, adc2 = 3, same for both IQ channels and all L
    x[:, 0, :, :] = 1.0
    x[:, 1, :, :] = 2.0
    x[:, 2, :, :] = 3.0
    # x = x[:,0:2,:,:]
    print(x.shape)
    y = adapter(x)  # (B, 2, L)
    # Both IQ channels and all L: (1 + 2 + 3) / 3 = 2
    assert torch.allclose(y[:, 0, :], torch.full((B, L), 2.0))
    assert torch.allclose(y[:, 1, :], torch.full((B, L), 2.0))


def test_adc_iq_adapter_raises_on_invalid_shape():
    adapter = AdcIqAdapter()

    # Wrong dim
    with pytest.raises(ValueError):
        adapter(torch.randn(3, 2, 128))  # 3D instead of 4D

    # Wrong channels
    with pytest.raises(ValueError):
        adapter(torch.randn(2, 2, 2, 16))  # channel dim = 2

    # Wrong IQ axis
    with pytest.raises(ValueError):
        adapter(torch.randn(2, 3, 3, 16))  # IQ dim != 2


test_adc_iq_adapter_raises_on_invalid_shape()
test_adc_iq_adapter_linear_mixing()
test_adc_iq_adapter_shape_and_type()
test_adc_iq_adapter_raises_on_invalid_shape()
print("All tests passed.")
