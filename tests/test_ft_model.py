"""ft: naive fine-tuning lower bound, trained on each new task's data only."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import parser as file_parser  # noqa: E402
from model import ft, iid2  # noqa: E402
from tuning.presets import TUNING_PRESETS  # noqa: E402


def _mode_args(mode: str, model: str) -> object:
    return file_parser.parse_args_from_yaml(
        [
            str(ROOT / "configs" / "base.yaml"),
            str(ROOT / "configs" / "models" / mode / f"{model}.yaml"),
        ]
    )


def _ft_args() -> object:
    args = _mode_args("til", "ft")
    args.cuda = False
    args.arch = "resnet1d"
    args.dataset = "iq"
    args.data_scaling = "none"
    args.classes_per_task = [2, 3]
    args.nc_per_task_list = ""
    args.nc_per_task = None
    args.class_weighted_ce = False
    args.loader = "task_incremental_loader"
    args.norm_type = "batchnorm"
    return args


@pytest.mark.parametrize("model", ["ft", "iid2"])
@pytest.mark.parametrize(
    "mode, loader",
    [("til", "task_incremental_loader"), ("cil", "class_incremental_loader")],
)
def test_bound_configs_pin_their_loader(model: str, mode: str, loader: str) -> None:
    """Launchers pick configs/models/<mode>/; the loader must not fall back to base."""
    args = _mode_args(mode, model)
    assert args.model == model
    assert args.loader == loader
    assert not (ROOT / "configs" / "models" / f"{model}.yaml").exists()


def test_ft_does_not_replay() -> None:
    assert ft.Net.cumulative_replay is False
    assert iid2.Net.cumulative_replay is True


def test_ft_trains_on_a_single_task_batch() -> None:
    torch.manual_seed(0)
    model = ft.Net(2 * 32, 5, 2, _ft_args())
    x = torch.randn(8, 2, 32)
    y = torch.tensor([2, 3, 4, 2, 3, 4, 2, 3])

    loss, _, metric_logits = model.observe(x, y, 1)

    assert torch.isfinite(torch.tensor(loss))
    predictions = metric_logits.argmax(dim=1)
    assert ((predictions >= 2) & (predictions < 5)).all()


def test_tuning_preset_targets_ft() -> None:
    assert TUNING_PRESETS["ft"].model_name == "ft"
