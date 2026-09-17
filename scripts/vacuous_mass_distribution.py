"""Is the vacuous (ignorance) mass a useful signal? Offline, over a checkpoint.

PR-4 computed the aggregate ignorance ``m(Theta)`` as **one number per task** and
killed it as a cross-task forgetting readout: the ordering is gauge-dependent
(Spearman 0.152 between the pooled and per-task ``mu``), and block-centring forces
``W+ = W-`` so ``m(Theta)`` and ``kappa`` are two views of the same two numbers.
See ``docs/woe-cl/pr4-ignorance-conflict.md``.

This asks the two questions that survive that verdict, both about *distributions*
rather than task means:

**(A) The aggregate mass, per sample.** ``m~(Theta) = exp(-W-)/S``. A per-task
mean says nothing about whether the *spread* carries information, and the natural
use of an ignorance mass is per-sample: high ignorance should mean "the model does
not know", so it should predict its own errors. Measured as AUROC for
misclassification, against the readout any classifier already has for free --
softmax max-probability. An ignorance mass that cannot beat softmax confidence on
the model's own errors is not a signal, whatever its theory says.

**(B) The per-feature vacuous mass**, which is the follow-up PR-4 Sec 6 names as
legitimate and does not exist yet. Each ``w_jk`` is one simple support function,
whose mass on the frame is

    v_jk = exp(-|w_jk|)   in (0, 1],   v_jk = 1  <=>  feature j says nothing
                                                      about class k

This is the un-aggregated readout. It cannot inherit ``W+ = W-``, because that
identity holds only for the sum over ``j``; and unlike ``m(Theta)`` -- which is
``exp(-25..-50)`` and therefore numerically dead -- it is bounded in ``(0, 1]``
with real spread. The vacuous *fraction* it defines is the same quantity the
``I_1`` sparsity result already moves (0.092 -> 0.286, README E4), so it is known
to respond to something.

Both are run under the three ``mu`` gauges and PR-4's registered kill criterion is
re-applied to each, because a readout that reorders when the gauge changes is
measuring the normalisation. Nothing here touches anything the model optimises.

Usage:
    python scripts/vacuous_mass_distribution.py [--checkpoint PATH]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.resnet1d import ResNet1D  # noqa: E402
from model.woe_si import compute_weights_of_evidence  # noqa: E402
from scripts.ds_ignorance_conflict import (  # noqa: E402
    DEFAULT_CKPT,
    block_centre,
    build_args,
    mass_quantities,
    subset_loader,
)
from utils import misc_utils  # noqa: E402

# |w| below this leaves >= 0.9 of the simple support function's mass on the frame,
# i.e. the feature is saying essentially nothing about that class. Matches the
# threshold `evidence_distribution_from_run.py` uses for its vacuous fraction.
VACUITY_THRESHOLD = 0.9


def auroc(scores: torch.Tensor, positive: torch.Tensor) -> float:
    """Rank-based AUROC of ``scores`` for the binary label ``positive``.

    Ties get mid-ranks, so a constant score returns exactly 0.5 rather than
    something that depends on sort order -- which matters here, since a saturated
    readout is one of the outcomes being tested for.
    """
    n_pos = int(positive.sum())
    n_neg = int(positive.numel() - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores, dtype=torch.float64)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float64)
    # Mid-ranks for ties.
    sorted_scores = scores[order]
    start = 0
    for end in range(1, scores.numel() + 1):
        if end == scores.numel() or sorted_scores[end] != sorted_scores[start]:
            if end - start > 1:
                ranks[order[start:end]] = ranks[order[start:end]].mean()
            start = end
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def spearman(a: List[float], b: List[float]) -> float:
    """Spearman rho, matching ``ds_ignorance_conflict``'s implementation."""

    def rank(v: List[float]) -> List[float]:
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


@torch.no_grad()
def features_and_labels(net, loader, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Penultimate features and their labels, eval-mode BN, no grad."""
    feats, labels = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        feats.append(net.forward_features(xb, bn_training=False).double().cpu())
        labels.append(yb.cpu())
    return torch.cat(feats), torch.cat(labels)


def quantiles(values: torch.Tensor, points=(0.0, 0.1, 0.5, 0.9, 1.0)) -> List[float]:
    """Quantiles as plain floats, for printing a distribution rather than a mean."""
    qs = torch.tensor(points, dtype=values.dtype)
    return [float(v) for v in torch.quantile(values, qs)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join(ROOT, DEFAULT_CKPT))
    parser.add_argument("--per-task-n", type=int, default=2048)
    parser.add_argument("--test-limit", type=int, default=4096)
    parser.add_argument(
        "--centering",
        default="prop2_uniform",
        choices=["prop2_uniform", "centered_uniform"],
        help="prop2_uniform is Denoeux Prop 2 / Eq 38, under which "
        "sum_j w_jk = z_k exactly; centered_uniform is the project default "
        "every recorded result was measured under.",
    )
    cli = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    print(f"checkpoint: {cli.checkpoint}  (task {payload['task']})")
    print(f"centering: {cli.centering}")

    fc_w = state["net.model.fc.weight"].double()
    fc_b = state["net.model.fc.bias"].double()
    n_outputs, J = fc_w.shape

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

    net = ResNet1D(n_outputs, build_args(classes_per_task)).to(device)
    net.model.load_state_dict(
        {
            k[len("net.model.") :]: v
            for k, v in state.items()
            if k.startswith("net.model.")
        },
        strict=False,
    )
    net.eval()
    print(f"J={J}  n_outputs={n_outputs}  classes_per_task={classes_per_task}")

    # --- gauges (same construction and seeded random draw as PR-4) ---------
    per_task_mu: Dict[int, torch.Tensor] = {}
    for s in range(n_tasks):
        sub, _ = subset_loader(
            loader, loader.train_dataset, s, cli.per_task_n, "train", 12345
        )
        feats, _ = features_and_labels(net, sub, device)
        per_task_mu[s] = feats.mean(dim=0)
    mu_pooled = torch.stack([per_task_mu[s] for s in range(n_tasks)]).mean(dim=0)
    gauges = {
        "pooled": lambda s: mu_pooled,
        "per_task": lambda s: per_task_mu[s],
        "final_task": lambda s: per_task_mu[n_tasks - 1],
    }

    # --- readout ----------------------------------------------------------
    agg: Dict[str, Dict[int, Dict[str, float]]] = {g: {} for g in gauges}
    vac: Dict[str, Dict[int, Dict[str, float]]] = {g: {} for g in gauges}
    dist_rows: List[Tuple[int, List[float], List[float]]] = []

    for s in range(n_tasks):
        sub, _ = subset_loader(
            loader, loader.test_dataset, s, cli.test_limit, "test", 999
        )
        feats, labels = features_and_labels(net, sub, device)
        cols = misc_utils.current_task_class_indices(s, classes_per_task, n_outputs)
        K_s = cols.numel()
        w_blk, b_blk = block_centre(fc_w[cols], fc_b[cols])

        # Within-block prediction. Block-centring shifts every logit in the block
        # by the same per-sample constant, so it leaves softmax and argmax alone.
        logits = feats @ fc_w[cols].T + fc_b[cols]
        predicted = logits.argmax(dim=1)
        target = torch.full_like(labels, -1)
        for position, column in enumerate(cols.tolist()):
            target[labels == column] = position
        valid = target >= 0
        wrong = (predicted != target)[valid]
        confidence = torch.softmax(logits, dim=1).max(dim=1).values[valid]
        # Logit margin (top1 - top2): the cheapest thing log(1-kappa) has to beat.
        top2 = logits.topk(2, dim=1).values
        margin = (top2[:, 0] - top2[:, 1])[valid]
        # Low confidence should predict error, so the score is its negation.
        auroc_softmax = auroc(-confidence, wrong)

        for name, mu_of in gauges.items():
            w = compute_weights_of_evidence(
                feats, w_blk, b_blk, mu_of(s), centering_mode=cli.centering
            )
            w_plus = torch.relu(w).sum(dim=2)
            w_minus = torch.relu(-w).sum(dim=2)
            q = mass_quantities(w_plus, w_minus)

            log_m = torch.log(q["m_theta"].clamp(min=1e-300)) / K_s
            ignorance = -log_m[valid]  # higher = more ignorant
            # log(1-kappa) is, to leading order, -sum_{k != k*} w+_k: a runner-up
            # support measure. Does it carry anything the logit margin does not?
            log_nc = (q["log_nonconflict"] / K_s)[valid]
            agg[name][s] = {
                "K": K_s,
                "auroc_margin": auroc(-margin, wrong),
                "corr_lognc_margin": float(
                    torch.corrcoef(torch.stack([log_nc, margin.double()]))[0, 1]
                ),
                "auroc_ignorance": auroc(ignorance, wrong),
                "auroc_softmax": auroc_softmax,
                "auroc_nonconflict": auroc(-(q["log_nonconflict"] / K_s)[valid], wrong),
                "mean_log_m": float(log_m.mean()),
                "corr_with_W_minus": float(
                    torch.corrcoef(torch.stack([log_m, -q["W_minus"] / K_s]))[0, 1]
                ),
                "error_rate": float(wrong.double().mean()),
                "n": int(valid.sum()),
            }

            # (B) per-feature vacuous mass, mean over the batch -> (K_s, J)
            vacuity = torch.exp(-w.abs()).mean(dim=0)
            vac[name][s] = {
                "mean": float(vacuity.mean()),
                "frac_vacuous": float((vacuity > VACUITY_THRESHOLD).double().mean()),
                "q10": quantiles(vacuity.flatten())[1],
                "q50": quantiles(vacuity.flatten())[2],
                "q90": quantiles(vacuity.flatten())[3],
            }
            if name == "pooled":
                dist_rows.append((s, quantiles(log_m), quantiles(vacuity.flatten())))

    # --- report -----------------------------------------------------------
    print("\n=== (A) aggregate ignorance, per sample (gauge: pooled) ===")
    print("log m~(Theta) / K -- distribution over test samples, per task")
    print(f"{'task':>4} {'min':>10} {'p10':>10} {'p50':>10} {'p90':>10} {'max':>10}")
    for s, log_m_q, _ in dist_rows:
        print(f"{s:>4} " + " ".join(f"{v:>10.3f}" for v in log_m_q))

    print("\nDoes it predict the model's own errors? (AUROC, 0.5 = no signal)")
    print(
        f"{'task':>4} {'err rate':>9} {'ignorance':>10} {'log(1-k)':>10} "
        f"{'margin':>10} {'softmax':>10} {'corr(m,-W-)':>12} {'r(lnc,marg)':>12}"
    )
    for s in range(n_tasks):
        r = agg["pooled"][s]
        print(
            f"{s:>4} {r['error_rate']:>9.4f} {r['auroc_ignorance']:>10.4f} "
            f"{r['auroc_nonconflict']:>10.4f} {r['auroc_margin']:>10.4f} "
            f"{r['auroc_softmax']:>10.4f} "
            f"{r['corr_with_W_minus']:>12.5f} {r['corr_lognc_margin']:>12.4f}"
        )

    def mean_of(key: str) -> float:
        return sum(agg["pooled"][s][key] for s in range(n_tasks)) / n_tasks

    print(
        f"{'mean':>4} {'':>9} {mean_of('auroc_ignorance'):>10.4f} "
        f"{mean_of('auroc_nonconflict'):>10.4f} {mean_of('auroc_margin'):>10.4f} "
        f"{mean_of('auroc_softmax'):>10.4f} {'':>12} "
        f"{mean_of('corr_lognc_margin'):>12.4f}"
    )

    print("\n=== (B) per-feature vacuous mass exp(-|w_jk|) (gauge: pooled) ===")
    print(
        f"{'task':>4} {'min':>8} {'p10':>8} {'p50':>8} {'p90':>8} {'max':>8} "
        f"{'mean':>8} {'frac>' + str(VACUITY_THRESHOLD):>10}"
    )
    for s, _, vac_q in dist_rows:
        r = vac["pooled"][s]
        print(
            f"{s:>4} "
            + " ".join(f"{v:>8.4f}" for v in vac_q)
            + f" {r['mean']:>8.4f} {r['frac_vacuous']:>10.4f}"
        )

    print("\n=== KILL CRITERION (PR-4's, re-applied to each readout) ===")
    for label, table, key in (
        ("aggregate m(Theta)", agg, "mean_log_m"),
        ("aggregate AUROC", agg, "auroc_ignorance"),
        ("per-feature vacuity", vac, "mean"),
        ("per-feature vacuous frac", vac, "frac_vacuous"),
    ):
        a = [table["pooled"][s][key] for s in range(n_tasks)]
        b = [table["per_task"][s][key] for s in range(n_tasks)]
        rho = spearman(a, b)
        if rho < 0.8:
            verdict = "FLIPS -- measures the normalisation"
        elif rho < 0.9:
            verdict = "WARNING -- gauge-sensitive"
        else:
            verdict = "OK -- survives the gauge change"
        print(f"  {label:>26}: Spearman(pooled, per_task) = {rho:+.4f}   {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
