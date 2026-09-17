"""Gate B1 Arm 1: within-run head comparison (post-hoc linear probe).

Both heads are scored on the SAME extracted post-LayerNorm features, so the
only thing that differs between them is the readout. Offline: no re-training,
no auxiliary loss.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate_b1_lib import (  # noqa: E402
    SEEDS,
    Harness,
    extract_features,
    macro_f1_from_global_preds,
)
from model.evidential_modules import pignistic_probability  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "logs/eucr/gate_b1"
TASK = 0


def _to_global(model, local_scores, task):
    start, end = model.backbone._task_offsets[int(task)]
    out = local_scores.new_full(
        (local_scores.size(0), model.backbone.num_classes), -1e9
    )
    out[:, start:end] = local_scores
    return model._mask(out, task)


@torch.no_grad()
def ds_predict(model, z, task=TASK):
    """DS-head predictions from already-extracted features."""
    head = model.backbone.ds_heads[str(task)]
    mass = head(z.to(next(model.parameters()).device))
    if model.backbone.head == "pignistic":
        scores = pignistic_probability(
            mass, scale=model.backbone._scale_for(mass.size(-1) - 1)
        )
    else:
        n = mass.size(-1) - 1
        scores = model.backbone.dm_heads[str(task)](mass)[:, :n]
    return torch.argmax(_to_global(model, scores, task), dim=1).cpu()


@torch.no_grad()
def linear_predict(model, z, W, b, task=TASK):
    scores = z.to(W.device) @ W.t() + b
    return torch.argmax(_to_global(model, scores, task), dim=1).cpu()


def _solve(z, target, n_classes, l2, iters=400):
    """Deterministic multinomial logistic fit: CPU, float64, L2 on the mean loss.

    The first version ran LBFGS on GPU in float32 and was NOT reproducible: the
    features are near-separable, so the unregularised optimum is flat along many
    directions and float32 accumulation order picked a different point on each
    run. Delta_head moved by up to 0.16 between identical runs. CPU/float64 plus
    a real ridge makes the optimum unique.
    """
    z = z.double()
    W = torch.zeros(n_classes, z.size(1), dtype=torch.float64, requires_grad=True)
    b = torch.zeros(n_classes, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS(
        [W, b],
        max_iter=iters,
        history_size=30,
        tolerance_grad=1e-12,
        tolerance_change=1e-14,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z @ W.t() + b, target) + l2 * (W * W).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


L2_LADDER = [1e-1, 1e-2, 1e-3, 1e-4]


def fit_probe(z, y, n_classes, offset, device, l2=None, iters=400):
    """Fit the probe, selecting the ridge on a held-out slice of task-0 TRAIN.

    Selection never touches test data. Returns (W, b, chosen_l2, ladder_fits).
    """
    z = z.cpu()
    target = (y - offset).cpu().long()
    n = z.size(0)
    g = torch.Generator().manual_seed(12345)
    perm = torch.randperm(n, generator=g)
    cut = int(0.8 * n)
    tr, va = perm[:cut], perm[cut:]

    ladder = {}
    best, best_acc = None, -1.0
    for cand in L2_LADDER:
        Wc, bc = _solve(z[tr], target[tr], n_classes, cand, iters)
        acc = float(
            ((z[va].double() @ Wc.t() + bc).argmax(1) == target[va]).double().mean()
        )
        ladder[cand] = acc
        if acc > best_acc:
            best, best_acc = cand, acc
    chosen = l2 if l2 is not None else best
    W, b = _solve(z, target, n_classes, chosen, iters)
    return W.float().to(device), b.float().to(device), chosen, ladder


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {}
    for seed in SEEDS:
        h = Harness(seed)
        device = next(h.model.parameters()).device
        n_cls = h.args.classes_per_task[TASK]
        offset = h.model.backbone._task_offsets[TASK][0]

        # --- fit the probe on ck0 task-0 TRAIN features (batch regime, as trained)
        h.at(0, "batch")
        z_tr, y_tr = extract_features(h.model, h.train_tasks[TASK], TASK, "batch")
        W, b, chosen_l2, ladder = fit_probe(z_tr, y_tr, n_cls, offset, device)

        r = {"l2": chosen_l2, "l2_ladder_val_acc": ladder}
        for ck in (0, 3):
            h.at(ck, "batch")
            for regime in ("running", "batch"):
                z, y = extract_features(h.model, h.test_tasks[TASK], TASK, regime)
                r[f"ck{ck}_{regime}_ds"] = macro_f1_from_global_preds(
                    ds_predict(h.model, z), y, h.model, TASK, h.args
                )
                r[f"ck{ck}_{regime}_linear"] = macro_f1_from_global_preds(
                    linear_predict(h.model, z, W, b), y, h.model, TASK, h.args
                )

        r["gap_ds"] = r["ck3_batch_ds"] - r["ck3_running_ds"]
        r["gap_linear"] = r["ck3_batch_linear"] - r["ck3_running_linear"]
        r["delta_head"] = r["gap_ds"] - r["gap_linear"]
        r["sanity_margin"] = r["ck0_batch_linear"] - r["ck0_batch_ds"]
        rows[seed] = r
        print(f"seed {seed}: {json.dumps(r)}", flush=True)
        del h

    (OUT / "arm1.json").write_text(json.dumps(rows, indent=2))

    print("\n=== Gate B1 Arm 1 ===")
    print(
        f"{'seed':>5} {'DS ck0':>8} {'lin ck0':>8} {'gap DS':>8} {'gap lin':>8} {'Delta':>8}"
    )
    for s in SEEDS:
        r = rows[s]
        print(
            f"{s:5d} {r['ck0_batch_ds']:8.4f} {r['ck0_batch_linear']:8.4f} "
            f"{r['gap_ds']:8.4f} {r['gap_linear']:8.4f} {r['delta_head']:8.4f}"
        )

    kills = [s for s in SEEDS if rows[s]["sanity_margin"] < -0.05]
    print(
        f"\nSanity kill (linear ck0 < DS ck0 - 0.05): "
        f"{'FIRED on seeds ' + str(kills) if kills else 'not triggered'}"
    )
    passing = [s for s in SEEDS if rows[s]["delta_head"] > 0.2]
    print(
        f"H1 threshold (Delta_head > 0.2 in >= 4/5): {len(passing)}/5 -> "
        f"{'SUPPORTED' if len(passing) >= 4 else 'NOT SUPPORTED'}"
    )
    deltas = [rows[s]["delta_head"] for s in SEEDS]
    print(f"Delta_head mean {sum(deltas)/len(deltas):+.4f}")


if __name__ == "__main__":
    main()
