"""Shared offline harness for Gate B1 (docs/eucr/preregistration.md).

Loads a finished EUCR run's checkpoints and re-scores task 0 under either
normalisation regime, with no re-training. Everything here is read-only with
respect to the run directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

B0_EXTRA = REPO / "logs/eucr/b0_extra-2026-08-25_14-31-08-7745"
SEEDS = [1, 2, 3, 7, 11]


def pin_determinism(seed: int = 0) -> None:
    """Make every arm bit-reproducible across runs.

    The batch regime evaluates with cuDNN in benchmark mode by default
    (parser sets cudnn_benchmark=True), which re-picks convolution algorithms by
    timing on each run. The resulting last-bit differences are amplified twice
    over: BatchNorm recomputes its statistics from those activations, and at ck3
    under running statistics the model sits near chance, where a tiny logit
    perturbation moves many argmaxes. Measured: Delta_head moved by up to 0.16
    between identical runs before this was pinned.
    """
    import random


    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _seed_dir(seed: int, root: Path = B0_EXTRA) -> Path:
    return root / str(seed)


def load_args(seed: int, root: Path = B0_EXTRA):
    """Parser defaults overlaid with the run's own training_parameters.json."""
    import parser as file_parser

    args = file_parser.get_parser().parse_args([])
    saved = json.load(open(_seed_dir(seed, root) / "training_parameters.json"))
    for key, value in saved.items():
        setattr(args, key, value)
    args.seed = int(seed)
    args.cuda = bool(getattr(args, "cuda", False)) and torch.cuda.is_available()
    # Paths in the saved args are relative to the repo root.
    args.data_path = str(REPO / saved["data_path"])
    return args


def build_loader(args):
    import importlib

    Loader = importlib.import_module("dataloaders." + args.loader)
    loader = Loader.IncrementalLoader(args, seed=args.seed)
    n_inputs, n_outputs, n_tasks = loader.get_dataset_info()
    args.classes_per_task = getattr(loader, "classes_per_task", None)
    return loader, n_inputs, n_outputs, n_tasks


def build_model(args, n_inputs, n_outputs, n_tasks):
    import importlib

    Model = importlib.import_module("model." + args.model)
    model = Model.Net(n_inputs, n_outputs, n_tasks, args)
    if args.cuda:
        model.cuda()
    return model


def load_checkpoint(model, seed: int, task: int, root: Path = B0_EXTRA):
    """Load ``task_{task}.pt`` in place. Returns the checkpoint dict."""
    path = _seed_dir(seed, root) / "checkpoints" / f"task_{task}.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    # `ce_heads` postdates these checkpoints. They are never touched on the eval
    # path (forward only builds ce_logits when explicitly requested), and Arm 1
    # fits its own probe rather than loading one, so their absence is expected.
    # Anything else missing means the architecture moved under the checkpoint.
    unexpected_missing = [k for k in missing if not k.startswith("backbone.ce_heads.")]
    if unexpected_missing or unexpected:
        raise RuntimeError(
            f"state_dict mismatch for seed {seed} task {task}: "
            f"missing={unexpected_missing[:5]} unexpected={list(unexpected)[:5]}"
        )
    return ckpt


def set_regime(model, regime: str) -> None:
    """Select the BatchNorm statistics used at inference.

    ``batch``   -- EUCR.forward pins bn_training=True (the shipped default and
                   the protocol every other ResNet1D learner is scored under).
    ``running`` -- bn_stats_mode falls through, so model.eval() leaves the frozen
                   running buffers in place.
    """
    if regime not in ("batch", "running"):
        raise ValueError(regime)
    model.bn_stats_mode = regime


@torch.no_grad()
def score_task(model, tasks, args, task_index: int = 0) -> float:
    """Task ``task_index`` classification F1 through the real eval path."""
    import main

    # eval_tasks returns (recall, precision, f1, det, det_fa); with
    # specific_task set, each list holds exactly the requested task.
    _, _, f1_results, _, _ = main.eval_tasks(
        model, tasks, args, specific_task=task_index
    )
    return float(f1_results[0])


class Harness:
    """One seed: loader + model, reusable across checkpoints and regimes."""

    def __init__(self, seed: int, root: Path = B0_EXTRA):
        self.seed = int(seed)
        self.root = root
        pin_determinism()
        self.args = load_args(seed, root)
        self.loader, n_in, n_out, n_tasks = build_loader(self.args)
        self.model = build_model(self.args, n_in, n_out, n_tasks)
        self.test_tasks = self.loader.get_tasks("test")
        self._train_tasks = None

    @property
    def train_tasks(self):
        if self._train_tasks is None:
            self._train_tasks = self.loader.get_tasks("train")
        return self._train_tasks

    def at(self, ckpt_task: int, regime: str):
        load_checkpoint(self.model, self.seed, ckpt_task, self.root)
        set_regime(self.model, regime)
        return self.model

    def f1_task0(self, ckpt_task: int, regime: str) -> float:
        self.at(ckpt_task, regime)
        return score_task(self.model, self.test_tasks, self.args, 0)


# ----------------------------------------------------------------------
# Feature access
#
# Both heads read `features = self.feat_norm(self.do(out))` (eucr_backbone.py:384),
# so a forward hook on `feat_norm` captures exactly the tensor Gate B1's H2 is
# about -- the post-LayerNorm pooled feature, not the pre-norm pooled activation.


class _NoDropout:
    """Temporarily replace the backbone's shared Dropout with Identity.

    The batch regime works by putting the whole backbone in train mode, which
    also switches dropout ON at evaluation. That is faithful to the shipped
    protocol but it injects noise into a feature-difference estimate, so Arm 2
    disables it in *both* regimes to leave BatchNorm as the only difference.
    """

    def __init__(self, backbone):
        self.backbone = backbone

    def __enter__(self):
        self._saved = self.backbone.do
        self.backbone.do = torch.nn.Identity()
        return self.backbone

    def __exit__(self, *exc):
        self.backbone.do = self._saved
        return False


@torch.no_grad()
def extract_features(model, loader, task: int, regime: str, dropout: bool = False):
    """Post-LayerNorm pooled features and labels for one task's loader.

    Returns ``(features [N, D], labels [N])`` on CPU, in loader order.
    """
    set_regime(model, regime)
    model.eval()
    device = next(model.parameters()).device
    captured = {}

    def hook(_module, _inputs, output):
        captured["z"] = output.detach()

    # The batch regime works by putting the backbone in TRAIN mode, and a
    # BatchNorm in train mode *writes* its running buffers. Left unguarded, a
    # batch-regime pass silently adapts the running statistics toward the data
    # just scored, so a running-regime pass afterwards is no longer the
    # checkpoint's. Snapshot and restore so extraction order cannot matter.
    bn_backup = [
        (
            m,
            m.running_mean.detach().clone(),
            m.running_var.detach().clone(),
            m.num_batches_tracked.detach().clone(),
        )
        for m in model.backbone.modules()
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
        and m.running_mean is not None
    ]

    handle = model.backbone.feat_norm.register_forward_hook(hook)
    feats, labels = [], []
    ctx = _NoDropout(model.backbone) if not dropout else _nullcontext()
    try:
        with ctx:
            for batch in loader:
                xb, yb = batch[0], batch[1]
                xb = xb.to(device)
                model(xb, task)
                feats.append(captured["z"].float().cpu())
                y = yb[:, 0] if yb.ndim == 2 and yb.shape[1] == 2 else yb
                labels.append(torch.as_tensor(y).long().cpu())
    finally:
        handle.remove()
        for module, mean, var, count in bn_backup:
            module.running_mean.copy_(mean)
            module.running_var.copy_(var)
            module.num_batches_tracked.copy_(count)
    return torch.cat(feats), torch.cat(labels)


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def macro_f1_from_global_preds(preds, targets, model, task: int, args) -> float:
    """Batch-averaged macro F1, matching main.eval_tasks' accounting exactly.

    eval_tasks averages `macro_f1_including_noise` over batches rather than
    pooling, so a pooled score is not comparable to the published numbers.
    """
    from utils.training_metrics import macro_f1_including_noise

    batch_size = int(getattr(args, "test_batch_size", 512))
    scores = []
    for start in range(0, preds.numel(), batch_size):
        stop = start + batch_size
        scores.append(macro_f1_including_noise(preds[start:stop], targets[start:stop]))
    return float(sum(scores) / len(scores)) if scores else 0.0


def pooled_macro_f1(preds, targets) -> float:
    """Macro F1 over the whole set at once, not averaged over batches.

    `main.eval_tasks` averages per batch, which is order-dependent: the deeprad
    files are class-ordered, so a natural-order batch holds few classes and a
    shuffled one holds all of them. Any analysis that reorders or subsets the
    stream (Arm 2's split-half) must use this instead, and its absolute values
    are therefore not comparable to the published per-batch numbers -- only
    ratios computed the same way are.
    """
    from utils.training_metrics import macro_f1_including_noise

    return float(macro_f1_including_noise(preds, targets))
