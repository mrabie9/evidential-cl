"""PR-4 Phase 1: ignorance ``m(Theta)`` vs conflict ``kappa`` as a forgetting readout.

Offline read-out over a trained checkpoint. Touches nothing the model optimises.

WHAT IS COMPUTED
----------------
For a checkpoint at task ``t`` and each earlier task ``s <= t``, over task ``s``'s
test data:

1. ``phi = forward_features(x)``, 512-d, ``bn_training=False`` under ``no_grad``.
   ``phi -> fc`` is the affine cut; nothing sits between them.
2. Restrict to task ``s``'s column block ``C_s = [offset1_s, offset2_s)``.
3. **Block-centre** ``beta`` within ``C_s`` (Denoeux Prop 2 at the granularity the
   architecture uses). Under TIL masking, columns outside ``C_s`` are never
   normalised together with ``C_s`` in any softmax, so the gauge freedom is
   per-block; centring across all 64 would couple blocks the model never compares.
4. ``w_jk = beta*_jk (phi_j - mu_j) + beta'*_0k / J`` via the existing
   ``compute_weights_of_evidence`` (which centres *features*, not ``beta`` -- so the
   block-centring here is complementary, not duplicated).
   Default ``--centering prop2_uniform`` since 2026-09-03: Denoeux Prop 2 Eq 38,
   ``beta'*_0k = beta*_0k + sum_q beta*_qk mu_q``, under which ``sum_j w_jk = z*_k``
   exactly. The project's ``centered_uniform`` drops that sum, which is measured
   **40-191x larger** than the ``beta*_0k`` it keeps -- so the readouts were being
   quoted in a convention that is not Denoeux's. PR-4 Sec 4 was measured under
   ``centered_uniform`` and Sec 7 under ``prop2_uniform``; the orderings differ
   (Spearman +0.70 to +0.77 between conventions), so they are not comparable.
5. ``m(Theta)`` and ``kappa`` from the combined mass function.

THE MASS FUNCTION (derivation, since the paper's numbered results are not
available to transcribe in this session -- see PR-4's derivation flag)
----------------------------------------------------------------------
The construction gives ``2K`` simple support functions: for each class ``k``,

    S+_k :  m({theta_k}) = 1 - a_k,      m(Theta) = a_k,   a_k = exp(-w+_k)
    S-_k :  m(~{theta_k}) = 1 - b_k,     m(Theta) = b_k,   b_k = exp(-w-_k)

Conjunctive (unnormalised Dempster) combination picks one focal set from each and
intersects. Writing ``P_a = prod_k a_k``:

* result ``{theta_k}``  <- choose ``{theta_k}`` from ``S+_k``, ``Theta`` from every
  other ``S+_l``, ``Theta`` from ``S-_k``, anything from ``S-_l (l != k)``:

      m({theta_k}) = (1 - a_k) * b_k * prod_{l != k} a_l
                   = P_a * (exp(w+_k) - 1) * exp(-w-_k)

* result ``Theta \\ T`` for ``T`` a proper subset <- choose ``Theta`` from every
  ``S+``, and ``~{theta_l}`` for ``l`` in ``T``:

      m(Theta \\ T) = P_a * prod_{l in T} (1 - b_l) * prod_{l not in T} b_l

  so in particular ``m(Theta) = P_a * prod_l b_l`` (``T`` empty), and summing over
  all proper ``T`` gives ``P_a * [1 - prod_l (1 - b_l)]``.

* everything else intersects to the empty set, so

      kappa = m(empty) = 1 - P_a * S,
      S = sum_k (exp(w+_k) - 1) exp(-w-_k) + 1 - prod_l (1 - exp(-w-_l))

**``P_a`` cancels from every normalised quantity**, which is what makes the
Dempster-normalised masses numerically stable:

      m~(Theta) = exp(-W-) / S,        W- = sum_k w-_k

**But raw ``kappa`` does not survive.** ``P_a = exp(-sum_k w+_k)`` and with
``K >= 6`` classes ``sum_k w+_k`` is large, so ``kappa = 1 - P_a * S`` saturates at
1 to many decimal places and cannot order anything. The workable comparable is
``log(1 - kappa) / K_s`` -- a per-class log non-conflict -- and that is what the
cross-task ordering uses. Raw ``kappa`` is printed too, so the saturation is
visible rather than asserted.

**That saturation is a bookkeeping choice, not a property of the data** (found
2026-09-03, against Beechey et al., Information Fusion 92 (2023) 115-126). The
combination above is one-stage: all ``2K`` simple support functions at once. Doing
it in two stages -- combine and *normalise* the ``K`` positive-evidence functions,
likewise the ``K`` negative ones, then combine the two -- gives the same masses
(Dempster's rule is associative with normalisation) but a different ``m(empty)``,
because the within-family singleton disagreements (``{th_1}`` vs ``{th_2}``, which
with ``K >= 6`` and ``w+ ~ 5`` are near-certain and carry nothing) are quotiented
out first. That is the paper's Eq 11a/11b/12, and it is reported here as
``kappa_staged``. It has real range (0.43-0.86 across tasks, 0.08-0.95 per sample)
and is the **only** quantity in this script to pass the kill criterion below.
Whether it is a *forgetting* readout is a separate question and the answer so far
is no -- see ``docs/woe-cl/pr4-ignorance-conflict.md``.

GAUGE
-----
Three ``mu`` gauges are computed and the task ordering compared across them; see
PR-4's registered kill criterion. ``E_k = sum_j |w_jk| - |z*_k| >= 0`` is reported
alongside, isolating the gauge-dependent part from the gauge-invariant logit.

Usage:
    python scripts/ds_ignorance_conflict.py [--checkpoint PATH] [--per-task-n N]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.resnet1d import ResNet1D  # noqa: E402
from model.woe_si import compute_weights_of_evidence  # noqa: E402
from utils import misc_utils  # noqa: E402

DEFAULT_CKPT = (
    "logs/woe_si_lc/mufz_ema_lam240000.0_s0-2026-08-21_17-26-19-5905/"
    "0/checkpoints/task_9.pt"
)


# ----------------------------------------------------------------------
def block_centre(weight: torch.Tensor, bias: torch.Tensor):
    """Denoeux Prop 2 gauge fixing, applied **within** one task's column block.

    ``weight`` is ``(K_s, J)`` and ``bias`` is ``(K_s,)``, already sliced to the
    block. Subtracting the within-block mean over classes leaves every pairwise
    logit difference inside the block unchanged -- which is all TIL inference
    depends on -- while fixing the gauge the weights of evidence are read in.
    """
    return weight - weight.mean(dim=0, keepdim=True), bias - bias.mean()


def mass_quantities(
    w_plus: torch.Tensor, w_minus: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """``m~(Theta)``, ``kappa``, ``log(1 - kappa)`` from per-class totals.

    Args:
        w_plus / w_minus: ``(batch, K)`` non-negative per-class total evidence.

    Returns:
        Dict of ``(batch,)`` tensors. See the module docstring for the derivation.
    """
    w_plus = w_plus.double()
    w_minus = w_minus.double()
    # S = sum_k (e^{w+_k} - 1) e^{-w-_k} + 1 - prod_l (1 - e^{-w-_l})
    # expm1 keeps precision when w+ is small; the sum is done in log space when
    # w+ is large enough for exp to overflow.
    term = torch.expm1(w_plus) * torch.exp(-w_minus)
    singleton_sum = term.sum(dim=1)
    prod_one_minus_b = torch.exp(
        torch.log1p(-torch.exp(-w_minus).clamp(max=1 - 1e-16)).sum(dim=1)
    )
    S = singleton_sum + 1.0 - prod_one_minus_b
    W_plus = w_plus.sum(dim=1)
    W_minus = w_minus.sum(dim=1)

    log_nonconflict = -W_plus + torch.log(S.clamp(min=1e-300))
    kappa = 1.0 - torch.exp(log_nonconflict)
    m_theta_norm = torch.exp(-W_minus - torch.log(S.clamp(min=1e-300)))

    # Beechey Eq 11a/11b/12 (Information Fusion 92 (2023) 115-126), the *staged*
    # conflict. The K positive-evidence and K negative-evidence simple support
    # functions are each combined and **normalised** first, and only then combined
    # with each other -- so this m(empty) is the cross-family tension alone.
    # `kappa` above is the one-stage m(empty) over all 2K at once, which also
    # counts every within-family singleton disagreement ({th_1} vs {th_2} -> empty);
    # with K >= 6 and w+ ~ 5 those are near-certain and carry nothing, which is why
    # it saturates at 1. Dempster's rule is associative with normalisation, so
    # `m_theta` is identical either way; only the discarded mass differs.
    #   1 - kappa_staged = eta+ eta- S,  1/eta+ = sum_k(e^{w+_k} - 1) + 1,
    #                                    1/eta- = 1 - prod_l (1 - e^{-w-_l})
    # Verified against a full power-set enumeration for K = 3 (both forms exact,
    # and both give the same m(Theta) to machine precision).
    inv_eta_plus = torch.expm1(w_plus).sum(dim=1) + 1.0
    inv_eta_minus = (1.0 - prod_one_minus_b).clamp(min=1e-300)
    log_nonconflict_staged = (
        torch.log(S.clamp(min=1e-300))
        - torch.log(inv_eta_plus)
        - torch.log(inv_eta_minus)
    )
    kappa_staged = -torch.expm1(log_nonconflict_staged)

    # What the staging discards *before* Eq 12 is reached. `eta+` normalises away
    # the conflict among the K positive simple support functions -- two classes
    # both strongly supported -- and `eta-` the conflict among the negative ones.
    #   1 - kappa_+ = exp(-W+) / eta+ ,     kappa_- = prod_l (1 - e^{-w-_l})
    # The product (1 - kappa_+)(1 - kappa_-)(1 - kappa_staged) = 1 - kappa is
    # invariant, so the one-stage `kappa` is the whole conflict and Eq 12's is one
    # factor of it; which factor gets the name is the grouping choice.
    log_nonconflict_plus = -W_plus + torch.log(inv_eta_plus)
    kappa_plus = -torch.expm1(log_nonconflict_plus)
    kappa_minus = prod_one_minus_b
    return {
        "m_theta": m_theta_norm,
        "kappa": kappa,
        "log_nonconflict": log_nonconflict,
        "kappa_staged": kappa_staged,
        "log_nonconflict_staged": log_nonconflict_staged,
        "kappa_plus": kappa_plus,
        "log_nonconflict_plus": log_nonconflict_plus,
        "kappa_minus": kappa_minus,
        "W_plus": W_plus,
        "W_minus": W_minus,
    }


# ----------------------------------------------------------------------
def build_args(classes_per_task: List[int]):
    o = argparse.Namespace()
    o.use_iq_aug_features = False
    o.data_scaling = "normalize"
    o.iq_aug_feature_type = "power"
    o.alpha_init = 1e-3
    o.use_groupnorm = False
    o.norm_type = "batchnorm"
    o.classes_per_task = classes_per_task
    return o


def subset_loader(loader, dataset, task: int, n: int, mode: str, seed: int):
    """A loader over a **random** n-sample subset of one task's split.

    Taking the first ``n`` rows instead would be wrong here, and measurably so:
    four of the ten task files (the `deeprad` ones, tasks 3/4/6/9) are stored with
    the signal classes first and the noise class last, so a sequential draw of
    2048 from a 24480-row task contains **none** of the noise class -- which is
    half the task. Measured proportion gap between a sequential draw and the full
    split: <= 0.03 on the six shuffled files, **0.50** on the four ordered ones.
    A ``mu`` built that way is a signal-only mean for those tasks and a full-task
    mean for the rest -- exactly the differently-weighted-per-task estimator that
    balanced pooling exists to avoid.

    Seeded, so the draw is reproducible and identical across gauges.
    """
    x, y = dataset[task][1], dataset[task][2]
    total = x.shape[0]
    take = min(int(n), total)
    rng = np.random.default_rng(seed + task)
    idx = np.sort(rng.choice(total, size=take, replace=False))
    return loader._get_loader(x[idx], y[idx], mode=mode), take


@torch.no_grad()
def features_for(net, loader, device) -> torch.Tensor:
    out = []
    for batch in loader:
        xb = batch[0] if isinstance(batch, (list, tuple)) else batch
        xb = xb.to(device, non_blocking=True)
        out.append(net.forward_features(xb, bn_training=False).double().cpu())
    return torch.cat(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join(ROOT, DEFAULT_CKPT))
    parser.add_argument("--per-task-n", type=int, default=2048)
    parser.add_argument("--test-limit", type=int, default=4096)
    parser.add_argument(
        "--centering",
        default="prop2_uniform",
        choices=("prop2_uniform", "centered_uniform", "raw_uniform"),
        help=(
            "alpha_jk convention. 'centered_uniform' (default, and what every "
            "recorded PR-4 number used) offsets by beta*_0k/J; 'prop2_uniform' is "
            "Denoeux Prop 2 / Beechey Eq 5, offsetting by beta'*_0k/J with "
            "beta'*_0k = beta*_0k + sum_q beta*_qk mu_q, under which "
            "sum_j w_jk = z*_k exactly."
        ),
    )
    args_cli = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args_cli.checkpoint, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    print(f"checkpoint: {args_cli.checkpoint}  (task {payload['task']})")

    fc_w = state["net.model.fc.weight"].double()
    fc_b = state["net.model.fc.bias"].double()
    n_outputs, J = fc_w.shape
    print(f"fc: weight {tuple(fc_w.shape)}  bias {tuple(fc_b.shape)}  J={J}")

    # --- data -------------------------------------------------------------
    import dataloaders.task_incremental_loader as Loader  # noqa: E402

    base = argparse.Namespace()
    for key, value in {
        "data_path": "data/rff/radar/full",
        "task_order_files": (
            "t2-rcn,t0-rcn,t2-uclresm-25noise,t0-deeprad,t1-deeprad,"
            "t1-uclresm-25noise,t3-deeprad,t0-uclresm-25noise,t1-rcn,t2-deeprad"
        ),
        "dataset": "iq",
        "loader": "task_incremental_loader",
        "samples_per_task": -1,
        "batch_size": 256,
        "test_batch_size": 512,
        "workers": 0,
        "validation": 0.5,
        "data_scaling": "normalize",
        "snr_range": None,
        "class_order": "random",
        "increment": 5,
        "classes_per_it": 6,
        "seed": 0,
        "model": "woe_si",
        "task_order_seed": None,
        "use_iq_aug_features": False,
        "iq_aug_feature_type": "power",
        "n_memories": 5120,
    }.items():
        setattr(base, key, value)
    loader = Loader.IncrementalLoader(base, seed=0)
    _, _, n_tasks = loader.get_dataset_info()
    classes_per_task = list(loader.classes_per_task)
    print(f"classes_per_task: {classes_per_task}  n_tasks={n_tasks}")

    net = ResNet1D(n_outputs, build_args(classes_per_task)).to(device)
    missing = net.model.load_state_dict(
        {
            k[len("net.model.") :]: v
            for k, v in state.items()
            if k.startswith("net.model.")
        },
        strict=False,
    )
    print(
        f"loaded backbone (missing={len(missing.missing_keys)}, "
        f"unexpected={len(missing.unexpected_keys)})"
    )
    net.eval()

    # --- gauges -----------------------------------------------------------
    per_task_mu: Dict[int, torch.Tensor] = {}
    per_task_se: Dict[int, torch.Tensor] = {}
    for s in range(n_tasks):
        sub, took = subset_loader(
            loader, loader.train_dataset, s, args_cli.per_task_n, "train", 12345
        )
        feats = features_for(net, sub, device)
        assert feats.shape[0] == took
        per_task_mu[s] = feats.mean(dim=0)
        per_task_se[s] = feats.std(dim=0) / (feats.shape[0] ** 0.5)
        print(
            f"  task {s}: mu from n={feats.shape[0]}  ||mu||={per_task_mu[s].norm():.4f}"
            f"  mean per-coord SE={per_task_se[s].mean():.5f}"
        )

    # Balanced pooled: equal n per task, so the pooled mean is the mean of the
    # per-task means (equal counts per task, NOT equal counts per class).
    mu_pooled = torch.stack([per_task_mu[s] for s in range(n_tasks)]).mean(dim=0)
    se_pooled = (
        torch.stack([per_task_se[s] ** 2 for s in range(n_tasks)]).sum(dim=0) ** 0.5
    ) / n_tasks
    mu_final = per_task_mu[n_tasks - 1]
    gauges = {
        "pooled": lambda s: mu_pooled,
        "per_task": lambda s: per_task_mu[s],
        "final_task": lambda s: mu_final,
        # Beechey's own choice: `calculate_masses` computes `mu = mean(layer2)`
        # from whatever batch it was handed, so the gauge is the *test* batch
        # being scored -- transductive, and refitted per experimental condition
        # (it is recomputed on the DeepFool and JSMA sets too). Ours above are
        # all fitted on training data. Included so the difference is measured
        # rather than argued.
        "test_batch": lambda s: None,
    }
    se_norm = float((se_pooled**2).sum() ** 0.5)
    print(
        f"\nbalanced-pooled mu: ||mu||={mu_pooled.norm():.4f}  "
        f"mean per-coord SE={se_pooled.mean():.6f}  "
        f"max SE={se_pooled.max():.6f}  ||SE||={se_norm:.4f}"
    )
    # The spec's check: is Monte Carlo error small next to ||mu_ema - mu_pooled||?
    # Note mu_ema is the checkpoint's `woe_feature_mean`, which is the EMA over
    # task 9 ONLY -- it is reset at every boundary, and task 9 is never
    # consolidated because `on_task_end` is not called from main.py. So it is a
    # task-9 statistic, not a global one.
    if "woe_feature_mean" in state:
        mu_ema = state["woe_feature_mean"].double()
        gap = float((mu_ema - mu_pooled).norm())
        ok = "OK - MC negligible" if se_norm < 0.05 * gap else "MC NOT negligible"
        print(
            f"  ||mu_ema - mu_pooled|| = {gap:.4f}   "
            f"(mu_ema is the task-9 EMA, ||mu_ema||={mu_ema.norm():.4f})"
        )
        print(f"  ||SE|| / gauge gap     = {se_norm / gap:.5f}   {ok}")

    # --- how big is the term `centered_uniform` drops? ---------------------
    print(f"\ncentering mode: {args_cli.centering}")
    print("  per-class offset numerator, block-centred (pooled mu):")
    print(f"  {'task':>4} {'|beta*_0k|':>12} {'|sum_q b*_qk mu_q|':>20} {'ratio':>9}")
    for s_ in range(n_tasks):
        cols_ = misc_utils.current_task_class_indices(s_, classes_per_task, n_outputs)
        w_b, b_b = block_centre(fc_w[cols_], fc_b[cols_])
        drop = w_b @ mu_pooled
        print(
            f"  {s_:>4} {b_b.abs().mean():>12.3e} {drop.abs().mean():>20.3e} "
            f"{float(drop.abs().mean() / b_b.abs().mean()):>9.1f}"
        )

    # --- readout ----------------------------------------------------------
    results: Dict[str, Dict[int, Dict[str, float]]] = {g: {} for g in gauges}
    for s in range(n_tasks):
        sub, _ = subset_loader(
            loader, loader.test_dataset, s, args_cli.test_limit, "test", 999
        )
        feats = features_for(net, sub, device)
        cols = misc_utils.current_task_class_indices(s, classes_per_task, n_outputs)
        K_s = cols.numel()
        w_blk, b_blk = block_centre(fc_w[cols], fc_b[cols])
        for name, mu_of in gauges.items():
            mu_s = feats.mean(dim=0) if name == "test_batch" else mu_of(s)
            w = compute_weights_of_evidence(
                feats, w_blk, b_blk, mu_s, centering_mode=args_cli.centering
            )
            w_plus = torch.relu(w).sum(dim=2)
            w_minus = torch.relu(-w).sum(dim=2)
            q = mass_quantities(w_plus, w_minus)
            z_star = w.sum(dim=2)
            excess = w.abs().sum(dim=2) - z_star.abs()
            results[name][s] = {
                "K": K_s,
                "m_theta": float(q["m_theta"].mean()),
                "kappa": float(q["kappa"].mean()),
                "log_nc_per_class": float((q["log_nonconflict"] / K_s).mean()),
                "log_nc_se": float(
                    (q["log_nonconflict"] / K_s).std() / (feats.shape[0] ** 0.5)
                ),
                "kappa_staged": float(q["kappa_staged"].mean()),
                "kappa_staged_se": float(
                    q["kappa_staged"].std() / (feats.shape[0] ** 0.5)
                ),
                "kappa_staged_min": float(q["kappa_staged"].min()),
                "kappa_staged_max": float(q["kappa_staged"].max()),
                "log_nc_staged": float((q["log_nonconflict_staged"] / K_s).mean()),
                "kappa_plus": float(q["kappa_plus"].mean()),
                "log_nc_plus": float((q["log_nonconflict_plus"] / K_s).mean()),
                "kappa_minus": float(q["kappa_minus"].mean()),
                "invariance_err": float(
                    (
                        q["log_nonconflict_plus"]
                        + torch.log1p(-q["kappa_minus"].clamp(max=1 - 1e-16))
                        + q["log_nonconflict_staged"]
                        - q["log_nonconflict"]
                    )
                    .abs()
                    .max()
                ),
                "m_theta_se": float(q["m_theta"].std() / (feats.shape[0] ** 0.5)),
                "excess": float(excess.mean()),
                "w_plus": float(w_plus.mean()),
                "w_plus_cls_min": float(w_plus.mean(dim=0).min()),
                "w_plus_cls_max": float(w_plus.mean(dim=0).max()),
                "w_plus_frac_gt1": float((w_plus > 1.0).double().mean()),
                "w_minus": float(w_minus.mean()),
                "n": feats.shape[0],
            }

    # --- report -----------------------------------------------------------
    for name in gauges:
        print(f"\n=== gauge: {name} ===")
        print(
            f"{'task':>4} {'K':>2} {'m(Theta)':>11} {'+/-':>9} "
            f"{'kappa':>10} {'log(1-k)/K':>12} {'+/-':>9} "
            f"{'k_staged':>10} {'+/-':>9} {'k_st min':>9} {'k_st max':>9} "
            f"{'1-k_+':>10} {'k_-':>9} {'inv err':>9} "
            f"{'E_k':>9} {'w+':>8} {'w-':>8} "
            f"{'w+ cls lo':>10} {'w+ cls hi':>10} {'frac w+>1':>10}"
        )
        for s in range(n_tasks):
            r = results[name][s]
            print(
                f"{s:>4} {r['K']:>2} {r['m_theta']:>11.3e} {r['m_theta_se']:>9.2e} "
                f"{r['kappa']:>10.7f} {r['log_nc_per_class']:>12.4f} "
                f"{r['log_nc_se']:>9.2e} "
                f"{r['kappa_staged']:>10.6f} {r['kappa_staged_se']:>9.2e} "
                f"{r['kappa_staged_min']:>9.6f} {r['kappa_staged_max']:>9.6f} "
                f"{1.0 - r['kappa_plus']:>10.3e} {r['kappa_minus']:>9.3e} "
                f"{r['invariance_err']:>9.2e} "
                f"{r['excess']:>9.3f} "
                f"{r['w_plus']:>8.3f} {r['w_minus']:>8.3f} "
                f"{r['w_plus_cls_min']:>10.3f} {r['w_plus_cls_max']:>10.3f} "
                f"{r['w_plus_frac_gt1']:>10.4f}"
            )

    # --- kill criterion ---------------------------------------------------
    def spearman(a: List[float], b: List[float]) -> float:
        def rank(v):
            order = sorted(range(len(v)), key=lambda i: v[i])
            out = [0.0] * len(v)
            for pos, idx in enumerate(order):
                out[idx] = float(pos)
            return out

        ra, rb = rank(a), rank(b)
        ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
        num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
        den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
        return num / den if den else float("nan")

    print("\n=== KILL CRITERION (PR-4, registered before running) ===")
    verdicts = []
    for metric, other in [
        (m, o)
        for o in ("per_task", "test_batch")
        for m in ("log_nc_per_class", "m_theta", "kappa_staged")
    ]:
        a = [results["pooled"][s][metric] for s in range(n_tasks)]
        b = [results[other][s][metric] for s in range(n_tasks)]
        rho = spearman(a, b)
        if rho < 0.8:
            verdict = "FLIPS -- diagnostic measures the normalisation. STOP."
        elif rho < 0.9:
            verdict = "WARNING -- gauge-sensitive; hedge all downstream claims"
        else:
            verdict = "OK -- ordering survives the gauge change"
        verdicts.append((metric, rho, verdict))
        print(f"  {metric:>18}: Spearman(pooled, {other:<10}) = {rho:+.4f}   {verdict}")

    print("\n=== VARIATION CHECK (is there anything to order?) ===")
    for metric, se_key in (
        ("log_nc_per_class", "log_nc_se"),
        ("m_theta", "m_theta_se"),
        ("kappa_staged", "kappa_staged_se"),
    ):
        vals = [results["pooled"][s][metric] for s in range(n_tasks)]
        spread = max(vals) - min(vals)
        typ_se = sum(results["pooled"][s][se_key] for s in range(n_tasks)) / n_tasks
        flat = spread < typ_se
        print(
            f"  {metric:>18}: across-task spread={spread:.4e}  "
            f"mean within-task SE={typ_se:.4e}  "
            f"{'FLAT -- does not vary' if flat else 'varies'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
