"""Functional check: configured CL algorithms learn the global noise class.

Each model (as declared in ``configs/models/*.yaml``) is instantiated with a tiny
synthetic IQ setup: three signal classes (0–2) and a shared noise index (3).
After a short ``observe`` training loop on random batches that mix signal and
noise labels, we require a majority of noise-only probe samples to be classified
as the noise class under the same logits masking used for CIL-style evaluation.

**Excluded** from this smoke suite (meta-gradient / BGD interact badly with the
minimal stub here): ``anml``, ``meralg1``, ``meta-bgd``.
"""

# ruff: noqa: E402

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import parser as file_parser  # noqa: E402
from utils import misc_utils  # noqa: E402

CONFIG_MODEL_DIR = ROOT / "configs" / "models"
NOISE_LABEL = 3
N_SIGNAL = 3
N_OUTPUTS = N_SIGNAL + 1
N_TASKS = 1
SEQ_LEN = 96
N_INPUTS_IQ = 2 * SEQ_LEN
TRAIN_STEPS_DEFAULT = 180
TRAIN_STEPS_FAST = 120
# AGEM / GEM converge slowly under the fully deterministic per-op Generator stream
# used in this test (same seed as pytest); extra steps keep the probe well above chance.
TRAIN_STEPS_EPISODIC_MEMORY = 400
NOISE_PROBE_BATCH = 72
MIN_NOISE_ACCURACY = 0.56

# Meta-learning / special baselines excluded from this smoke suite (fragile in
# minimal synthetic IQ setup or unrelated API).
#
# ``eucr`` is also excluded: its Dempster-Shafer evidential head requires Adam
# (adaptive per-parameter steps) plus the KL warm-up to escape the flat
# maximum-ignorance region. Under this suite's CE-oriented regime (SGD, lr 0.08,
# 180 steps on prior-only random inputs) it stays at ignorance, but it trains to
# confident, calibrated predictions under its own ``configs/models/*/eucr.yaml``
# (Adam) setup.
MODEL_MODULES_EXCLUDED_FROM_NOISE_SMOKE: frozenset[str] = frozenset(
    ("anml", "meralg1", "meta-bgd", "eucr")
)


def _ordered_model_config_paths() -> list[Path]:
    """Return deterministically ordered model config files.

    Preference order (per model) is handled by `_yaml_chain_for_module`:
    legacy top-level path first, then nested mode-specific paths.
    """

    return sorted(CONFIG_MODEL_DIR.rglob("*.yaml"), key=lambda p: p.as_posix())


def _ordered_model_modules_from_configs() -> list[str]:
    """Return unique ``model:`` ids from ``configs/models/**/*.yaml``."""
    discovered: list[str] = []
    seen: set[str] = set()
    for path in _ordered_model_config_paths():
        with path.open(encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        module_name = payload.get("model")
        if not module_name or not isinstance(module_name, str):
            continue
        if module_name in seen:
            continue
        seen.add(module_name)
        discovered.append(module_name)
    return discovered


def _yaml_chain_for_module(model_module: str) -> list[str]:
    """``base.yaml`` plus the YAML that declares this ``model`` (if any)."""
    chain = [str(ROOT / "configs" / "base.yaml")]
    matching_paths: list[Path] = []
    for path in _ordered_model_config_paths():
        with path.open(encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if payload.get("model") == model_module:
            matching_paths.append(path)

    if matching_paths:
        legacy_matches = [
            path for path in matching_paths if path.parent == CONFIG_MODEL_DIR
        ]
        if legacy_matches:
            chain.append(str(legacy_matches[0]))
        else:
            chain.append(str(matching_paths[0]))
    return chain


def _training_steps(model_module: str) -> int:
    if model_module in ("agem", "gem"):
        return TRAIN_STEPS_EPISODIC_MEMORY
    if model_module in (
        "bcl_dual",
        "ctn",
        "eralg4",
        "lamaml_cifar",
    ):
        return TRAIN_STEPS_FAST
    return TRAIN_STEPS_DEFAULT


def _build_args(model_module: str) -> object:
    args = file_parser.parse_args_from_yaml(_yaml_chain_for_module(model_module))
    args.model = model_module
    args.cuda = False
    args.dataset = "iq"
    args.arch = "resnet1d"
    args.data_scaling = "none"
    args.class_weighted_ce = False
    # Per-task heads must span global labels 0..NOISE_LABEL inclusive.  The IQ
    # loader counts *signal* classes per task and adds noise separately; UCL's
    # split heads otherwise emit only K logits for K ``signal'' ids.
    if model_module == "ucl_bresnet":
        args.classes_per_task = [N_OUTPUTS]
    else:
        args.classes_per_task = [N_SIGNAL] * N_TASKS
    args.nc_per_task_list = ""
    args.nc_per_task = None
    args.noise_label = NOISE_LABEL
    args.batch_size = 16
    args.memories = 128
    args.n_memories = 128
    args.replay_batch_size = 8
    args.n_meta = 1
    args.lr = 0.08
    args.optimizer = "sgd"
    args.gamma = 0.5
    args.memory_strength = 0.3
    args.smax = 400.0
    args.second_order = False
    args.learn_lr = False
    args.sync_update = False
    args.meta_batches = 1
    args.cifar_batches = 1
    args.bgd_optimizer = "sgd"
    args.train_mc_iters = 1
    args.inner_steps = 1
    args.split = False
    args.class_incremental = True
    args.get_samples_per_task = lambda _task_id: 128
    args.validation = 0.0
    args.det_memories = 0
    args.use_detector_arch = False
    args.eval_samples = 1
    return args


def _rand_batch(
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(
        batch_size,
        2,
        SEQ_LEN,
        device=device,
        generator=generator,
    )
    y = torch.empty(batch_size, dtype=torch.long, device=device)
    frac_noise = 0.42
    noise_mask = torch.rand(batch_size, generator=generator, device=device) < frac_noise
    n_signal = int((~noise_mask).sum().item())
    if n_signal:
        y[~noise_mask] = torch.randint(
            0,
            N_SIGNAL,
            (n_signal,),
            device=device,
            generator=generator,
        )
    if noise_mask.any():
        y[noise_mask] = NOISE_LABEL
    return x, y


def _forward_logits(
    model: object,
    *,
    model_module: str,
    x: torch.Tensor,
    task_index: int = 0,
) -> torch.Tensor:
    """Return global logits suitable for ``argmax`` over noise + seen classes."""
    if model_module == "anml":
        return model(x, fast_weights=None)  # type: ignore[operator]

    if model_module == "icarl":
        raw = model.netforward(x)  # type: ignore[attr-defined]
        return misc_utils.apply_task_incremental_logit_mask(
            raw,
            task_index,
            model.classes_per_task,  # type: ignore[attr-defined]
            model.n_outputs,  # type: ignore[attr-defined]
            cil_all_seen_upto_task=task_index,
            global_noise_label=model.noise_label,  # type: ignore[attr-defined]
        )

    if model_module == "hat":
        bridge_device = x.device if x.is_cuda else model._device()  # type: ignore[attr-defined]
        logits, _ = model.bridge.forward(  # type: ignore[attr-defined]
            model._task_tensor(task_index, bridge_device),  # type: ignore[attr-defined]
            x,
            model.smax,  # type: ignore[attr-defined]
            return_masks=True,
        )
        return misc_utils.apply_task_incremental_logit_mask(
            logits,
            task_index,
            model.classes_per_task,  # type: ignore[attr-defined]
            model.n_outputs,  # type: ignore[attr-defined]
            cil_all_seen_upto_task=task_index,
            global_noise_label=model.noise_label,  # type: ignore[attr-defined]
        )

    if model_module == "meta-bgd":
        raw = model.net.forward(x)  # type: ignore[attr-defined]
        return misc_utils.apply_task_incremental_logit_mask(
            raw,
            task_index,
            model.classes_per_task,  # type: ignore[attr-defined]
            model.n_outputs,  # type: ignore[attr-defined]
            cil_all_seen_upto_task=task_index,
            global_noise_label=model.noise_label,  # type: ignore[attr-defined]
        )

    if model_module == "ucl_bresnet":
        out = model(x, task_index)  # type: ignore[operator]
        return misc_utils.apply_task_incremental_logit_mask(
            out,
            task_index,
            model.classes_per_task,  # type: ignore[attr-defined]
            model.n_outputs,  # type: ignore[attr-defined]
            cil_all_seen_upto_task=task_index,
            global_noise_label=model.noise_label,  # type: ignore[attr-defined]
        )

    try:
        return model(  # type: ignore[operator]
            x,
            task_index,
            cil_all_seen_upto_task=task_index,
        )
    except TypeError:
        try:
            return model(x, task_index)  # type: ignore[operator]
        except TypeError:
            return model(x)  # type: ignore[operator]


def _noise_accuracy(
    model: object,
    model_module: str,
    device: torch.device,
    generator: torch.Generator,
) -> float:
    model.eval()
    x_probe, _ = _rand_batch(NOISE_PROBE_BATCH, generator, device)
    y_noise = torch.full(
        (NOISE_PROBE_BATCH,), NOISE_LABEL, dtype=torch.long, device=device
    )
    with torch.no_grad():
        logits = _forward_logits(
            model, model_module=model_module, x=x_probe, task_index=0
        )
    preds = torch.argmax(logits, dim=1)
    return float((preds == y_noise).float().mean().item())


def _run_training(
    model: object,
    model_module: str,
    batch_size: int,
    n_steps: int,
    device: torch.device,
    generator: torch.Generator,
) -> None:
    model.train()
    for _ in range(n_steps):
        x, y = _rand_batch(batch_size, generator, device)
        model.observe(x.cpu(), y.cpu(), 0)


MODEL_MODULES = [
    name
    for name in _ordered_model_modules_from_configs()
    if name not in MODEL_MODULES_EXCLUDED_FROM_NOISE_SMOKE
]


@pytest.mark.parametrize("model_module", MODEL_MODULES, ids=lambda m: m)
def test_model_learns_global_noise_class(model_module: str) -> None:
    """Train on mixed signal/noise IQ minibatches; probe noise classification."""
    torch.manual_seed(0)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    device = torch.device("cpu")

    args = _build_args(model_module)
    batch_size = int(args.batch_size)

    Model = importlib.import_module(f"model.{model_module}")
    model = Model.Net(N_INPUTS_IQ, N_OUTPUTS, N_TASKS, args)
    model.to(device)
    if model_module == "icarl":
        model.gpu = False  # type: ignore[attr-defined]

    n_steps = _training_steps(model_module)
    if model_module in ("icarl", "lamaml_cifar", "eralg4"):
        n_steps = max(n_steps, 260)
    _run_training(model, model_module, batch_size, n_steps, device, generator)

    accuracy = _noise_accuracy(model, model_module, device, generator)
    assert accuracy >= MIN_NOISE_ACCURACY, (
        f"{model_module}: noise accuracy {accuracy:.3f} "
        f"(expected >= {MIN_NOISE_ACCURACY})"
    )


# Re-export for optional `pytest tests/test_noise_label_learning_all_models.py -k rwalk`
__all__ = [
    "MODEL_MODULES",
    "MODEL_MODULES_EXCLUDED_FROM_NOISE_SMOKE",
    "test_model_learns_global_noise_class",
]
