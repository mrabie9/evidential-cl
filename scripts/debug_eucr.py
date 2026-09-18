"""Bring up ``model.eucr.Net`` outside ``main.py`` so it can be debugged.

``main.py`` buries EUCR's construction behind a YAML chain, a seed-sweep
launcher, log-directory setup and a 2000-line training harness, none of which
you want in a debugger. This script reproduces just the parts EUCR depends on
-- the config chain, the incremental loader, ``classes_per_task`` -- hands back
a live model plus a real batch, and optionally drives ``observe`` /
``finalize_task_after_training`` so a breakpoint anywhere in ``model/eucr.py``,
``model/eucr_backbone.py`` or ``model/eucr_consolidation.py`` gets hit.

Multi-task mode (``--tasks N``) is the one to use for the consolidation
penalty. The penalty ``lambda * sum_i Omega_i (theta_i - theta_star_i)^2`` is
identically zero until at least one task has been consolidated, and stays zero
until the weights drift off the anchor -- so a single-task session has nothing
to inspect. ``--tasks N`` trains and consolidates N tasks, then moves to the
next task and steps *without* consolidating, which is exactly the state the
penalty is live in. ``--report`` then breaks it down per parameter tensor.

Two data modes:

``--fake`` (default)
    No dataset, no disk. Synthesises a ``[B, 2, 512]`` batch matching what the
    real IQ loader hands the backbone, with in-range labels. Starts in ~2
    seconds; use it for shape/NaN/gradient bugs and for the structure of the
    penalty (which tensors carry it, how importance accumulates).

``--real``
    Builds ``dataloaders.task_incremental_loader.IncrementalLoader`` on the
    cheap 4-task reproduction recipe, so batches, class counts and the noise
    label match a real run. Use it when the numbers have to mean something.

Examples::

    # three consolidated tasks, then inspect the live penalty on task 3
    python scripts/debug_eucr.py --tasks 3 --steps 5 --report

    # same on real data, with a breakpoint-friendly pdb prompt at the end
    python scripts/debug_eucr.py --real --tasks 2 --steps 5 --report --pdb

    # single task, construct only, drop into pdb with model/args/x/y bound
    python scripts/debug_eucr.py --pdb

    # override any parser/config field
    python scripts/debug_eucr.py --tasks 3 --set reg_lambda=1000 --report

    # from inside a debugger / `python -i`
    from scripts.debug_eucr import build_session
    s = build_session(fake=True)
    s.run_tasks(3, steps=5)
    s.penalty_report()
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import parser as file_parser  # noqa: E402
from model import eucr_consolidation as cons  # noqa: E402
from utils import misc_utils  # noqa: E402

# The cheap faithful reproduction recipe (4 tasks, small budget).
CONFIG_CHAIN = [
    os.path.join(REPO_ROOT, "configs", "base.yaml"),
    os.path.join(REPO_ROOT, "configs", "models", "til", "eucr.yaml"),
]
# What the real loader reports for the cheap recipe above; only used by --fake,
# where there is no dataset to read the shape or label space from. The IQ
# loader hands the backbone [B, 2, 512] (I/Q split of a 1024-sample record).
FAKE_SHAPE = "2x512"
FAKE_CLASSES_PER_TASK = "6,7,7,7"
CHEAP_RECIPE = {
    "task_order_files": "t2-rcn,t0-rcn,t0-deeprad,t1-deeprad",
    "samples_per_task": 10000,
    "n_epochs": 10,
    "expt_name": "debug-eucr",
}


def _coerce(value: str) -> Any:
    """Turn a ``--set key=value`` string into an int/float/bool/None/str."""
    low = value.strip().lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def build_args(overrides: Optional[dict] = None, *, cuda: bool = True):
    """Parse the EUCR config chain into an ``args`` namespace, then override.

    Mirrors ``main.py``: same YAML chain, same batch-size learning-rate scaling.
    Everything to do with logging, seed sweeps and resume is left out.
    """
    args = file_parser.parse_args_from_yaml(CONFIG_CHAIN)

    for key, value in CHEAP_RECIPE.items():
        setattr(args, key, value)
    for key, value in (overrides or {}).items():
        setattr(args, key, value)

    args.cuda = bool(cuda) and torch.cuda.is_available()
    args.state_logging = False
    # main.py scales lr with batch size before the model is built; EucrConfig
    # reads args.lr in __init__, so this has to happen here too.
    args.lr = misc_utils.scale_learning_rate_for_batch_size(args.lr, args.batch_size)

    # data_path is relative to the repo root in the configs, but a debugger can
    # be launched from anywhere.
    if not os.path.isabs(args.data_path):
        candidate = os.path.join(REPO_ROOT, args.data_path)
        if os.path.exists(candidate):
            args.data_path = candidate
    return args


@dataclass
class Session:
    """Everything a debugger needs: the model, its inputs, and the harness API."""

    args: Any
    model: torch.nn.Module
    n_inputs: int
    n_outputs: int
    n_tasks: int
    loader: Any = None
    train_loader: Any = None
    x: Optional[torch.Tensor] = None
    y: Optional[torch.Tensor] = None
    task: int = 0
    history: list = field(default_factory=list)
    tasks: list = field(default_factory=list)
    _iter: Any = None

    # -- data ---------------------------------------------------------------
    def batch(self, task: Optional[int] = None):
        """Return the currently held ``(x, y)``, switching task if asked."""
        if task is not None and task != self.task:
            self.load_task(task)
        return self.x, self.y

    def next_batch(self):
        """Advance to a fresh batch (real loader cycles; fake resamples)."""
        if self.loader is None:
            self.x, self.y = _fake_batch(
                self.args, self.n_inputs, self.task, self.model
            )
            return self.x, self.y
        device = _device_of(self.model)
        for _attempt in range(2):
            if self._iter is None:
                self._iter = iter(self.train_loader)
            try:
                batch = next(self._iter)
            except StopIteration:
                self._iter = None
                continue
            self.x, self.y = batch[0].to(device), batch[1].to(device)
            return self.x, self.y
        raise RuntimeError("train_loader yielded no batches")

    def load_task(self, task: int) -> None:
        """Move to ``task``. A real loader is one-way, so this only goes up."""
        if self.loader is not None:
            if task < self.task and self.train_loader is not None:
                raise ValueError(
                    "IncrementalLoader.new_task() is one-way; cannot go back to "
                    "task {} from {}".format(task, self.task)
                )
            while getattr(self.loader, "_current_task", 0) <= task:
                _info, train_loader, _val, _test = self.loader.new_task()
                self.train_loader = train_loader
            self._iter = None
        self.task = task
        self.next_batch()

    # -- training -----------------------------------------------------------
    def step(self, n: int = 1, epoch: int = 0, fresh: bool = True, quiet: bool = False):
        """Run ``model.observe`` ``n`` times, by default on a fresh batch each time.

        ``observe`` is the single call the training harness makes per batch, so
        a breakpoint in ``Net.observe`` -- or anywhere below it -- fires here.
        Pass ``fresh=False`` to hammer the same batch (useful when you want a
        deterministic repeat of one forward/backward).
        """
        self.model.real_epoch = epoch
        out = None
        for i in range(n):
            if fresh and i > 0:
                self.next_batch()
            loss, recall, logits = self.model.observe(self.x, self.y, self.task)
            pen = self.penalty()
            out = (loss, recall, logits)
            self.history.append(
                {
                    "task": self.task,
                    "step": i,
                    "loss": loss,
                    "recall": recall,
                    "penalty": pen,
                }
            )
            if not quiet:
                print(
                    "  step {:>3d}  loss={:<11.5f} recall={:.4f}  "
                    "penalty={:.6g}  lambda*penalty={:.6g}".format(
                        i, loss, recall, pen, self.model.reg_lambda * pen
                    )
                )
        return out

    def finalize(self, quiet: bool = False):
        """Run end-of-task consolidation (importance + anchor snapshot).

        Records what this task added to the running importance sum, so
        :meth:`penalty_report` can attribute the anchor per task.
        """
        before = _tensor_totals(self.model.importance)
        self.model.finalize_task_after_training(self._importance_loader())
        after = _tensor_totals(self.model.importance)
        delta = {k: after[k] - before.get(k, 0.0) for k in after}
        row = {
            "task": self.task,
            "omega_total": sum(after.values()),
            "omega_added": sum(delta.values()),
            "per_tensor_added": delta,
            "n_tensors": len(after),
        }
        self.tasks.append(row)
        if not quiet:
            print(
                "  consolidated task {}: Omega mass {:.4g} (+{:.4g}) "
                "over {} tensors".format(
                    row["task"],
                    row["omega_total"],
                    row["omega_added"],
                    row["n_tensors"],
                )
            )
        return self.model.importance

    def _importance_loader(self):
        """The loader ``compute_importance`` iterates (fake mode fabricates one)."""
        if self.train_loader is not None:
            return self.train_loader
        n = min(int(self.model.importance_batches or 4), 4)
        return [
            _fake_batch(self.args, self.n_inputs, self.task, self.model)
            for _ in range(n)
        ]

    def run_tasks(self, n: int, steps: int = 5, epoch: int = 0, quiet: bool = False):
        """Train and consolidate ``n`` tasks, then step on the next one live.

        After the loop the model holds importance accumulated over ``n`` tasks
        and ``theta_star`` from task ``n-1``. It then advances to task ``n``
        (or stays on the last task if the run is out of tasks) and takes
        ``steps`` steps *without* consolidating: the weights drift off the
        anchor, so the penalty is non-zero and there is something to inspect.
        """
        if n > self.n_tasks:
            raise ValueError(
                "asked for {} tasks but the run only has {}".format(n, self.n_tasks)
            )
        for t in range(n):
            print("--- task {} ({} steps, then consolidate) ---".format(t, steps))
            self.load_task(t)
            self.step(steps, epoch=epoch, quiet=quiet)
            self.finalize(quiet=quiet)

        live = min(n, self.n_tasks - 1)
        print("--- live task {} ({} steps, NOT consolidated) ---".format(live, steps))
        if live != self.task:
            self.load_task(live)
        self.step(steps, epoch=epoch, quiet=quiet)
        return self.penalty()

    # -- the penalty --------------------------------------------------------
    def penalty(self) -> float:
        """Scalar ``sum_i Omega_i (theta_i - theta_star_i)^2``, un-scaled by lambda."""
        with torch.no_grad():
            value = cons.penalty(
                self.model.backbone, self.model.importance, self.model.theta_star
            )
        return float(value)

    def penalty_terms(self) -> list:
        """Per-tensor breakdown of the consolidation penalty.

        Each row carries the importance ``Omega``, the displacement from the
        anchor, and the product that actually enters the loss -- which is the
        only one of the three that says where the penalty is coming from.
        """
        imp = self.model.importance or {}
        anchor = self.model.theta_star or {}
        rows = []
        with torch.no_grad():
            for name, param in self.model.backbone.named_parameters():
                if name not in imp or name not in anchor:
                    continue
                omega = imp[name].to(param.device)
                delta = param.detach() - anchor[name].to(param.device)
                term = float((omega * delta**2).sum())
                rows.append(
                    {
                        "name": name,
                        "numel": int(param.numel()),
                        "term": term,
                        "omega_mean": float(omega.mean()),
                        "omega_max": float(omega.max()),
                        "delta_rms": float(delta.pow(2).mean().sqrt()),
                        "delta_max": float(delta.abs().max()),
                    }
                )
        total = sum(r["term"] for r in rows) or 1.0
        for r in rows:
            r["share"] = r["term"] / total
        rows.sort(key=lambda r: r["term"], reverse=True)
        return rows

    def penalty_report(self, top: int = 15) -> list:
        """Print where the consolidation penalty lives, per tensor and per module."""
        rows = self.penalty_terms()
        lam = float(self.model.reg_lambda)
        total = sum(r["term"] for r in rows)
        last_loss = self.history[-1]["loss"] if self.history else float("nan")

        print("--- EUCR consolidation penalty ---")
        print(
            "  tasks consolidated : {}  (anchor theta_star from task {})".format(
                len(self.tasks),
                self.tasks[-1]["task"] if self.tasks else "none",
            )
        )
        print(
            "  granularity={}  uncertainty={}  importance_batches={}".format(
                self.model.reg_granularity,
                self.model.uncertainty_mode,
                self.model.importance_batches,
            )
        )
        if not rows:
            print("  penalty is structurally zero: no importance/anchor yet.")
            return rows
        print(
            "  P = {:.6g}   lambda = {:g}   lambda*P = {:.6g}".format(
                total, lam, lam * total
            )
        )
        print(
            "  last observe() loss = {:.6g}  ->  the anchor is {:.3%} of it".format(
                last_loss, (lam * total / last_loss) if last_loss else float("nan")
            )
        )

        print(
            "\n  {:<34} {:>10} {:>11} {:>11} {:>10} {:>7}".format(
                "tensor", "numel", "term", "Omega mean", "|dtheta|rms", "share"
            )
        )
        for r in rows[:top]:
            print(
                "  {:<34} {:>10,} {:>11.4g} {:>11.4g} {:>10.4g} {:>6.1%}".format(
                    r["name"][-34:],
                    r["numel"],
                    r["term"],
                    r["omega_mean"],
                    r["delta_rms"],
                    r["share"],
                )
            )
        if len(rows) > top:
            rest = rows[top:]
            print(
                "  {:<34} {:>10,} {:>11.4g} {:>11} {:>10} {:>6.1%}".format(
                    "... {} more tensors".format(len(rest)),
                    sum(r["numel"] for r in rest),
                    sum(r["term"] for r in rest),
                    "",
                    "",
                    sum(r["share"] for r in rest),
                )
            )

        groups: dict = {}
        for r in rows:
            key = r["name"].split(".")[0]
            g = groups.setdefault(key, {"term": 0.0, "numel": 0})
            g["term"] += r["term"]
            g["numel"] += r["numel"]
        print("\n  by module:")
        for key, g in sorted(groups.items(), key=lambda kv: -kv[1]["term"]):
            print(
                "  {:<34} {:>10,} {:>11.4g} {:>29.1%}".format(
                    key, g["numel"], g["term"], g["term"] / (total or 1.0)
                )
            )

        if len(self.tasks) > 1:
            print("\n  Omega mass added per consolidated task:")
            for row in self.tasks:
                print(
                    "    task {}: +{:.4g}  (running total {:.4g})".format(
                        row["task"], row["omega_added"], row["omega_total"]
                    )
                )
        return rows

    # -- the Dempster-Shafer heads -----------------------------------------
    def ds_modules(self) -> dict:
        """Every ``Dempster_Shafer_module`` in the model, keyed by label.

        ``head[t]`` is task ``t``'s own DS head; ``probe[s]`` is the DS readout
        hanging off backbone stage ``s``. Both are built from the same five
        layers -- ``ds1`` (:class:`Distance_layer`) -> ``ds1_activate`` ->
        ``ds2`` (belief) -> ``ds2_omega`` -> ``ds3_dempster``.

        None of these ever appear in :meth:`penalty_report`: consolidation
        excludes every name containing ``ds_head`` / ``dm_head`` / ``probes``
        because the heads are per-task, not shared. They are trained, just not
        anchored.
        """
        bb = self.model.backbone
        mods = {}
        for task_index in sorted(bb.ds_heads.keys(), key=int):
            mods["head[{}]".format(task_index)] = bb.ds_heads[task_index]
        for stage in sorted(bb.probes.keys(), key=int):
            mods["probe[{}]".format(stage)] = bb.probes[stage].ds_module
        return mods

    def ds_trace(self, task: Optional[int] = None) -> dict:
        """One forward pass with hooks on every DS layer.

        Returns ``{(label, layer): (inputs, output)}``. Only the current task's
        head runs under TIL, so other ``head[t]`` entries are simply absent.
        """
        layers = ("ds1", "ds1_activate", "ds2", "ds2_omega", "ds3_dempster")
        captured: dict = {}
        handles = []

        def make_hook(key):
            def hook(_module, inputs, output):
                captured[key] = (inputs, output)

            return hook

        for label, ds in self.ds_modules().items():
            for layer in layers:
                handles.append(
                    getattr(ds, layer).register_forward_hook(make_hook((label, layer)))
                )
        try:
            with torch.no_grad():
                self.model.backbone(
                    self.x,
                    self.task if task is None else task,
                    return_probes=True,
                )
        finally:
            for handle in handles:
                handle.remove()
        return captured

    def _lr_of(self, param) -> Optional[float]:
        for group in self.model.opt.param_groups:
            if any(p is param for p in group["params"]):
                return float(group["lr"])
        return None

    def ds_report(self, only: Optional[str] = None, task: Optional[int] = None) -> None:
        """Print the internal state of each DS head: parameters and live tensors.

        The parameter block is the head's own state (prototypes, the activation
        gamma/alpha, the belief split). The trace block is what those parameters
        did to *this* batch, layer by layer, which is the only way to see that
        ``Distance_layer`` is in the path at all -- it has no bias, no
        activation, and its output is consumed immediately by the next layer.
        """
        trace = self.ds_trace(task=task)
        for label, ds in self.ds_modules().items():
            if only and only not in label:
                continue
            ran = (label, "ds1") in trace
            print("\n--- DS module {} ---".format(label))
            print(
                "  prototypes={}  classes={}  features={}  metric={}  "
                "activation_norm={}".format(
                    ds.n_prototypes,
                    ds.n_classes,
                    ds.n_feature_maps,
                    ds.metric,
                    ds.ds1_activate.activation_norm,
                )
            )
            print(
                "  temper={:g}  dempster divisor={:g}{}".format(
                    ds.ds3_dempster.temper,
                    ds.ds3_dempster._divisor,
                    "" if ran else "   [not run this pass]",
                )
            )

            print(
                "  {:<26} {:>16} {:>10} {:>10} {:>10} {:>9} {:>8}".format(
                    "parameter", "shape", "min", "mean", "max", "|grad|", "lr"
                )
            )
            for name, raw in ds.named_parameters():
                param = raw.detach()
                grad = (
                    "-" if raw.grad is None else "{:.3g}".format(float(raw.grad.norm()))
                )
                lr = self._lr_of(raw)
                print(
                    "  {:<26} {:>16} {:>10.4g} {:>10.4g} {:>10.4g} {:>9} {:>8}".format(
                        name,
                        str(tuple(param.shape)),
                        float(param.min()),
                        float(param.mean()),
                        float(param.max()),
                        grad,
                        "-" if lr is None else "{:.3g}".format(lr),
                    )
                )

            with torch.no_grad():
                gamma = ds.ds1_activate.eta.weight.square()
                alpha = torch.sigmoid(ds.ds1_activate.xi.weight)
                beta = ds.ds2.beta.square()
                u = beta / beta.sum(dim=0, keepdim=True).clamp_min(1e-12)
                protos = torch.nn.functional.normalize(ds.ds1.w, dim=-1)
                cos = protos @ protos.t()
                off = cos[~torch.eye(cos.size(0), dtype=torch.bool, device=cos.device)]
            print("  derived:")
            print(
                "    gamma=eta^2      {:.4g} .. {:.4g}  (activation exp(-gamma*d))".format(
                    float(gamma.min()), float(gamma.max())
                )
            )
            print(
                "    alpha=sigmoid(xi) {:.4g} .. {:.4g}".format(
                    float(alpha.min()), float(alpha.max())
                )
            )
            print(
                "    belief split u    max share per prototype {:.4g} "
                "(uniform = {:.4g})".format(
                    float(u.max(dim=0).values.mean()), 1.0 / max(ds.n_classes, 1)
                )
            )
            print(
                "    prototype cosine  mean {:.4g}  max {:.4g}  (1.0 = collapsed)".format(
                    float(off.mean()), float(off.max())
                )
            )

            if not ran:
                continue
            dist = trace[(label, "ds1")][1]
            act = trace[(label, "ds1_activate")][1]
            raw_in = trace[(label, "ds1")][0][0]
            mass_p = trace[(label, "ds2_omega")][1]
            fused = trace[(label, "ds3_dempster")][1]
            with torch.no_grad():
                nearest = dist.argmin(dim=-1)
                used = int(torch.unique(nearest).numel())
                at_max = float((act > 0.999).float().sum(dim=-1).mean())
                omega = fused[:, -1]
                beliefs = fused[:, :-1]
            print("  live trace on this batch:")
            _print_tensor("    ds1 input (features)", raw_in)
            _print_tensor("    ds1 out   (distance)", dist)
            _print_tensor("    ds1_activate out (s)", act)
            _print_tensor("    ds2_omega out (m_p)", mass_p)
            _print_tensor("    ds3 fused mass", fused)
            print(
                "    omega (ignorance): mean {:.4g}  max {:.4g}".format(
                    float(omega.mean()), float(omega.max())
                )
            )
            print(
                "    belief argmax agrees with nearest prototype's class: "
                "{:.1%}".format(
                    float(
                        (beliefs.argmax(dim=-1) == (nearest % max(ds.n_classes, 1)))
                        .float()
                        .mean()
                    )
                )
            )
            print(
                "    prototypes used (argmin over batch): {}/{}   "
                "activations pinned at ~1: {:.2f}/sample".format(
                    used, ds.n_prototypes, at_max
                )
            )

    # -- misc ---------------------------------------------------------------
    def summary(self) -> None:
        n_params = sum(p.numel() for p in self.model.parameters())
        n_cons = sum(1 for _ in cons._named_consolidatable_params(self.model.backbone))
        print("--- EUCR debug session ---")
        print("  data          : {}".format("fake" if self.loader is None else "real"))
        print(
            "  n_inputs={}  n_outputs={}  n_tasks={}".format(
                self.n_inputs, self.n_outputs, self.n_tasks
            )
        )
        print("  classes_per_task: {}".format(self.model.classes_per_task))
        print("  device          : {}".format(next(self.model.parameters()).device))
        print(
            "  parameters      : {:,} ({} consolidatable tensors)".format(
                n_params, n_cons
            )
        )
        print(
            "  optimizer       : {} (lr groups {})".format(
                type(self.model.opt).__name__,
                [round(g["lr"], 6) for g in self.model.opt.param_groups],
            )
        )
        ds = self.ds_modules()
        if ds:
            head = next(iter(ds.values()))
            print(
                "  DS modules      : {} ({} heads + {} probes); head[0] has {} "
                "prototypes over {} classes, metric={}".format(
                    len(ds),
                    len(self.model.backbone.ds_heads),
                    len(self.model.backbone.probes),
                    head.n_prototypes,
                    head.n_classes,
                    head.metric,
                )
            )
        print("  cfg             : {}".format(self.model.cfg))
        if self.x is not None:
            print(
                "  batch           : x{} {} / y{} labels {}..{}".format(
                    tuple(self.x.shape),
                    self.x.dtype,
                    tuple(self.y.shape),
                    int(self.y.min()),
                    int(self.y.max()),
                )
            )


def _print_tensor(label: str, tensor: torch.Tensor) -> None:
    t = tensor.detach().float()
    print(
        "  {:<24} {:>16}  min {:>10.4g}  mean {:>10.4g}  max {:>10.4g}".format(
            label.strip(),
            str(tuple(t.shape)),
            float(t.min()),
            float(t.mean()),
            float(t.max()),
        )
    )


def _tensor_totals(importance) -> dict:
    if not importance:
        return {}
    return {k: float(v.sum()) for k, v in importance.items()}


def _device_of(model) -> torch.device:
    return next(model.parameters()).device


def _fake_batch(args, n_inputs, task, model):
    """Synthesise a batch matching the real loader's shape and label range."""
    device = _device_of(model)
    channels = int(getattr(args, "_fake_channels", 2))
    batch = int(args.batch_size)
    x = torch.randn(batch, channels, n_inputs, device=device)
    offset1, offset2 = model.compute_offsets(task)
    y = torch.randint(offset1, max(offset2, offset1 + 1), (batch,), device=device)
    return x, y.long()


def build_session(
    *,
    fake: bool = True,
    task: int = 0,
    cuda: bool = True,
    seed: int = 0,
    fake_shape: str = FAKE_SHAPE,
    fake_classes: str = FAKE_CLASSES_PER_TASK,
    overrides: Optional[dict] = None,
) -> Session:
    """Construct args, (optionally) the loader, and the EUCR model.

    Returns a :class:`Session` holding the live model and one batch, ready to
    be stepped. This is the function to call from a debugger or ``python -i``.
    """
    args = build_args(overrides, cuda=cuda)
    args.seed = seed
    misc_utils.init_seed(seed)

    loader = None
    if fake:
        channels, n_inputs = (int(v) for v in fake_shape.lower().split("x"))
        args._fake_channels = channels
        n_tasks = len(str(args.task_order_files).split(","))
        # Without a dataset there is nothing to infer the label space from,
        # so use whatever the config/CLI specifies and otherwise mirror what
        # the real loader reports for the cheap recipe.
        spec = (
            getattr(args, "nc_per_task_list", "")
            or getattr(args, "nc_per_task", None)
            or fake_classes
        )
        args.classes_per_task = misc_utils.build_task_class_list(
            n_tasks, None, nc_per_task=spec
        )
        n_outputs = int(sum(args.classes_per_task))
    else:
        Loader = importlib.import_module("dataloaders." + args.loader)
        loader = Loader.IncrementalLoader(args, seed=seed)
        n_inputs, n_outputs, n_tasks = loader.get_dataset_info()
        args.get_samples_per_task = getattr(loader, "get_samples_per_task", None)
        args.get_task_train_loader = getattr(loader, "get_tasks", None)
        args.classes_per_task = getattr(loader, "classes_per_task", None) or None
        if not args.classes_per_task:
            args.classes_per_task = misc_utils.build_task_class_list(
                n_tasks,
                n_outputs,
                nc_per_task=(args.nc_per_task_list or args.nc_per_task),
            )

    Model = importlib.import_module("model." + args.model)
    model = Model.Net(n_inputs, n_outputs, n_tasks, args)
    if args.cuda:
        model.cuda()

    session = Session(
        args=args,
        model=model,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        n_tasks=n_tasks,
        loader=loader,
    )
    session.load_task(task)
    return session


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    data = ap.add_mutually_exclusive_group()
    data.add_argument(
        "--fake",
        dest="fake",
        action="store_true",
        default=True,
        help="synthetic batch, no dataset on disk (default)",
    )
    data.add_argument(
        "--real",
        dest="fake",
        action="store_false",
        help="build the real incremental loader (cheap 4-task recipe)",
    )
    ap.add_argument(
        "--tasks",
        type=int,
        default=0,
        help="train and consolidate this many tasks, then step on the next one "
        "without consolidating (the state the penalty is live in)",
    )
    ap.add_argument("--task", type=int, default=0, help="single-task index to bring up")
    ap.add_argument("--steps", type=int, default=0, help="observe() calls per task")
    ap.add_argument("--epoch", type=int, default=0, help="value for model.real_epoch")
    ap.add_argument(
        "--finalize",
        action="store_true",
        help="single-task mode: consolidate after the steps",
    )
    ap.add_argument(
        "--report",
        action="store_true",
        help="print the per-tensor consolidation penalty breakdown",
    )
    ap.add_argument(
        "--report-top",
        type=int,
        default=15,
        help="tensors to list in the penalty table (default 15)",
    )
    ap.add_argument(
        "--ds",
        action="store_true",
        default=True,
        help="print the internal state of every Dempster-Shafer head "
        "(prototypes, gamma/alpha, belief split) plus a live layer-by-layer trace",
    )
    ap.add_argument(
        "--ds-module",
        default=None,
        metavar="SUBSTR",
        help="restrict --ds to modules whose label matches, e.g. head or probe[4]",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is up")
    ap.add_argument(
        "--fake-shape",
        default=FAKE_SHAPE,
        help="CxL for the synthetic batch (default {})".format(FAKE_SHAPE),
    )
    ap.add_argument(
        "--fake-classes",
        default=FAKE_CLASSES_PER_TASK,
        help="per-task class counts for --fake (default {})".format(
            FAKE_CLASSES_PER_TASK
        ),
    )
    ap.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override any config/parser field; repeatable",
    )
    ap.add_argument(
        "--pdb",
        action="store_true",
        help="drop into pdb with model/args/x/y bound after setup",
    )
    ns = ap.parse_args(argv)

    overrides = {}
    for item in ns.overrides:
        if "=" not in item:
            ap.error("--set expects KEY=VALUE, got {!r}".format(item))
        key, value = item.split("=", 1)
        overrides[key.strip()] = _coerce(value)

    session = build_session(
        fake=ns.fake,
        task=0 if ns.tasks else ns.task,
        cuda=not ns.cpu,
        seed=ns.seed,
        fake_shape=ns.fake_shape,
        fake_classes=ns.fake_classes,
        overrides=overrides,
    )
    session.summary()

    if ns.tasks:
        session.run_tasks(ns.tasks, steps=max(ns.steps, 1), epoch=ns.epoch)
    else:
        if ns.steps:
            print("--- observe x{} on task {} ---".format(ns.steps, session.task))
            session.step(ns.steps, epoch=ns.epoch)
        if ns.finalize:
            print("--- finalize_task_after_training ---")
            session.finalize()

    if ns.report or ns.tasks:
        session.penalty_report(top=ns.report_top)

    if ns.ds:
        session.ds_report(only=ns.ds_module)

    if ns.pdb:
        import pdb

        model = session.model  # noqa: F841  (bound for the pdb prompt)
        args = session.args  # noqa: F841
        x, y = session.x, session.y  # noqa: F841
        print("\npdb: `session`, `model`, `args`, `x`, `y` are bound.")
        print("     session.penalty_terms()  /  session.penalty_report()")
        print(
            "     session.ds_modules()  /  session.ds_report()  /  session.ds_trace()"
        )
        print("     model.importance, model.theta_star  are the penalty's two halves")
        pdb.set_trace()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
