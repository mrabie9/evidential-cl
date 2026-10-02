"""Joint vs separate replay/current forwards under BatchNorm."""

import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.task_bn import (  # noqa: E402
    forward_replay_and_current,
    replay_forward_is_joint,
)


def _batchnorm_net() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv1d(2, 4, 3), nn.BatchNorm1d(4), nn.Flatten()).train()


def test_joint_forward_matches_concatenated_batch():
    net = _batchnorm_net()
    replay_x = torch.randn(5, 2, 16) + 3.0
    current_x = torch.randn(7, 2, 16)

    replay_logits, current_logits = forward_replay_and_current(
        net, net, replay_x, current_x, joint=True
    )
    expected = net(torch.cat([replay_x, current_x]))

    assert torch.allclose(replay_logits, expected[:5])
    assert torch.allclose(current_logits, expected[5:])


def test_separate_forward_normalizes_each_block_alone():
    net = _batchnorm_net()
    replay_x = torch.randn(5, 2, 16) + 3.0
    current_x = torch.randn(7, 2, 16)

    _, joint_current = forward_replay_and_current(
        net, net, replay_x, current_x, joint=True
    )
    _, separate_current = forward_replay_and_current(
        net, net, replay_x, current_x, joint=False
    )

    assert torch.allclose(separate_current, net(current_x))
    assert not torch.allclose(separate_current, joint_current)


def test_only_class_incremental_runs_forward_jointly():
    assert replay_forward_is_joint("class_incremental_loader")
    assert not replay_forward_is_joint("task_incremental_loader")
    assert not replay_forward_is_joint(None)
