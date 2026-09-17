"""iCaRL must follow the authors' reference implementation (srebuffi/iCaRL)."""

# ruff: noqa: E402

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.icarl import Net as IcarlNet

LENGTH = 32


def _make_args(
    loader: str = "task_incremental_loader", eval_bn_stats: str = "batch"
) -> object:
    args = type("Args", (), {})()
    args.classes_per_task = [2, 2]
    args.nc_per_task_list = ""
    args.nc_per_task = None
    args.batch_size = 8
    args.n_memories = 8
    args.inner_steps = 1
    args.cuda = False
    args.alpha_init = 1e-3
    args.lr = 1e-2
    args.samples_per_task = 16
    args.n_epochs = 1
    args.arch = "resnet1d"
    args.dataset = "iq"
    args.loader = loader
    args.bn_mode = "shared"
    args.eval_bn_stats = eval_bn_stats
    args.class_weighted_ce = False
    args.grad_clip_norm = 0.0
    args.icarl_feature_chunk_size = 4
    return args


def _task_batches(task: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Two batches of 8 covering a task's 16 samples, both classes in each."""
    generator = torch.Generator().manual_seed(task)
    x = torch.randn(16, 2, LENGTH, generator=generator) + 3.0 * task
    y = torch.arange(16) % 2 + 2 * task
    return [(x[:8], y[:8]), (x[8:], y[8:])]


def _train(model: IcarlNet, tasks: int) -> None:
    for task in range(tasks):
        for x, y in _task_batches(task):
            model.observe(x, y, task)


def _reference_herding(features: np.ndarray, count: int) -> list[int]:
    """``alpha_dr_herding`` loop of ``main_cifar_100_theano.py``, returned as a ranking."""
    D = features.T
    mu = np.mean(D, axis=1)
    alpha = np.zeros(D.shape[1])
    w_t = mu
    iter_herding = 0
    iter_herding_eff = 0
    while np.sum(alpha != 0) != min(count, D.shape[1]) and iter_herding_eff < 1000:
        ind_max = int(np.argmax(np.dot(w_t, D)))
        iter_herding_eff += 1
        if alpha[ind_max] == 0:
            alpha[ind_max] = 1 + iter_herding
            iter_herding += 1
        w_t = w_t + mu - D[:, ind_max]
    picked = np.nonzero(alpha)[0]
    return [int(i) for i in picked[np.argsort(alpha[picked])]]


def test_herding_matches_reference_ranking() -> None:
    torch.manual_seed(0)
    features = F.normalize(torch.randn(40, 6, dtype=torch.float64), dim=1)
    ours = IcarlNet._herding_order(features, 12).tolist()
    assert ours == _reference_herding(features.numpy(), 12)


def test_task_end_herds_per_class_budget_and_freezes_network() -> None:
    torch.manual_seed(0)
    model = IcarlNet(2 * LENGTH, 4, 2, _make_args())
    _train(model, tasks=1)

    assert model.memx is None and model.memy is None
    assert model.exemplar_x.device.type == "cpu"
    # 8 memories over 2 seen classes.
    assert torch.bincount(model.exemplar_y).tolist() == [4, 4]

    assert model.old_net is not None
    assert all(not p.requires_grad for p in model.old_net.parameters())
    assert model.old_net.args is model.net.args, "args must be shared, not deep-copied"
    for frozen, live in zip(model.old_net.parameters(), model.net.parameters()):
        assert torch.equal(frozen, live)


def test_second_task_keeps_head_of_old_rankings() -> None:
    torch.manual_seed(0)
    model = IcarlNet(2 * LENGTH, 4, 2, _make_args())
    _train(model, tasks=1)
    first_x, first_y = model.exemplar_x.clone(), model.exemplar_y.clone()

    _train(model, tasks=2)
    # 8 memories over 4 seen classes.
    assert torch.bincount(model.exemplar_y).tolist() == [2, 2, 2, 2]
    for c in (0, 1):
        expected = first_x[first_y == c][:2]
        assert torch.equal(model.exemplar_x[model.exemplar_y == c], expected)


def test_distillation_targets_replace_old_units_on_every_row() -> None:
    torch.manual_seed(0)
    # Running statistics keep the expectation a plain forward: this asserts the
    # target math, not the BatchNorm policy (iCaRL follows evaluation, which is
    # batch statistics for both loaders since 2026-09-16, and reads its features
    # in randomly ordered chunks there).
    args = _make_args(loader="class_incremental_loader", eval_bn_stats="running")
    model = IcarlNet(2 * LENGTH, 4, 2, args)
    _train(model, tasks=1)

    x, y = _task_batches(1)[0]
    replay_x, replay_y = model._sample_exemplars(8, 16)
    labels = torch.cat((y, replay_y))
    targets = model._targets(x, replay_x, labels, t=1)

    with torch.no_grad():
        expected_old = torch.sigmoid(model.old_net(torch.cat((x, replay_x))))[:, :2]
    assert torch.allclose(targets[:, :2], expected_old)
    assert torch.equal(targets[:, 2:], F.one_hot(labels, 4).float()[:, 2:])


def test_exemplar_share_follows_augmented_training_set() -> None:
    torch.manual_seed(0)
    model = IcarlNet(2 * LENGTH, 4, 2, _make_args())
    _train(model, tasks=1)
    replay_x, _ = model._sample_exemplars(batch_count=8, samples_per_task=16)
    # 8 new rows * 8 exemplars / 16 task samples.
    assert replay_x.size(0) == 4


def test_til_forward_is_nearest_mean_within_task() -> None:
    torch.manual_seed(0)
    model = IcarlNet(2 * LENGTH, 4, 2, _make_args())
    _train(model, tasks=1)
    model.eval()
    x, _ = _task_batches(0)[0]

    scores = model(x, 0)
    assert torch.all(scores[:, 2:] == -1e9)

    feats = F.normalize(model.net.forward_features(x), dim=1)
    means = model._class_means([0])[0]
    assert torch.allclose(scores[:, :2], -torch.cdist(feats, means).pow(2), atol=1e-5)


def test_class_means_are_normalised_means_of_normalised_features() -> None:
    torch.manual_seed(0)
    # Running statistics, as above: the assertion is about the mean math.
    args = _make_args(loader="class_incremental_loader", eval_bn_stats="running")
    model = IcarlNet(2 * LENGTH, 4, 2, args)
    _train(model, tasks=1)
    model.eval()

    means, class_ids = model._class_means([0])
    with torch.no_grad():
        feats = F.normalize(model.net.forward_features(model.exemplar_x), dim=1)
    for row, c in enumerate(class_ids):
        expected = F.normalize(feats[model.exemplar_y == c].mean(0), dim=0)
        assert torch.allclose(means[row], expected, atol=1e-5)


@pytest.mark.parametrize(
    "loader, expected_scored",
    [("task_incremental_loader", [2, 3]), ("class_incremental_loader", [0, 1, 2, 3])],
)
def test_forward_candidate_classes(loader: str, expected_scored: list[int]) -> None:
    torch.manual_seed(0)
    model = IcarlNet(2 * LENGTH, 4, 2, _make_args(loader=loader))
    _train(model, tasks=2)
    model.eval()
    x, _ = _task_batches(1)[0]

    scores = model(x, 1, cil_all_seen_upto_task=1)
    scored = [c for c in range(4) if bool(torch.all(scores[:, c] > -1e9))]
    assert scored == expected_scored


def test_forward_falls_back_to_masked_logits_before_task_has_exemplars() -> None:
    torch.manual_seed(0)
    model = IcarlNet(2 * LENGTH, 4, 2, _make_args())
    _train(model, tasks=1)
    model.eval()
    x, _ = _task_batches(1)[0]

    scores = model(x, 1)
    with torch.no_grad():
        logits = model.netforward(x)
    assert torch.all(scores[:, :2] == -1e9)
    assert torch.allclose(scores[:, 2:], logits[:, 2:])


def test_feature_extraction_is_chunked_and_order_preserving() -> None:
    torch.manual_seed(0)
    model = IcarlNet(2 * LENGTH, 4, 2, _make_args())
    seen_batch_sizes: list[int] = []

    def _fake_features(batch: torch.Tensor) -> torch.Tensor:
        seen_batch_sizes.append(int(batch.size(0)))
        return batch.flatten(1)[:, : model.n_feat].clone()

    model.net.forward_features = _fake_features  # type: ignore[method-assign]
    inputs = torch.randn(9, 2, model.n_feat)
    feats = model._eval_features(inputs)

    assert seen_batch_sizes == [4, 4, 1]
    expected = F.normalize(inputs.flatten(1)[:, : model.n_feat], dim=1)
    assert torch.allclose(feats, expected)
