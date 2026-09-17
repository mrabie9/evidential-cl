"""Gate B1 Arm 2: shift and scale projection.

Estimates the post-LayerNorm feature shift between normalisation regimes on one
half of task-0 held-out, applies it to the other half, and re-scores. Runs the
registered instrument check (sum_c delta_c / gamma_c == 0) first.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate_b1_arm1 import TASK, ds_predict, fit_probe, linear_predict  # noqa: E402
from gate_b1_lib import (  # noqa: E402
    SEEDS,
    Harness,
    extract_features,
    pooled_macro_f1,
)

OUT = Path(__file__).resolve().parents[1] / "logs/eucr/gate_b1"
CKPT = 3


def corrections(z_r_a, z_b_a):
    """Estimate on half A. delta is applied additively (mu_batch - mu_running)."""
    mu_r, mu_b = z_r_a.mean(0), z_b_a.mean(0)
    sd_r, sd_b = z_r_a.std(0).clamp_min(1e-8), z_b_a.std(0)
    return mu_r, mu_b, mu_b - mu_r, sd_b / sd_r


def apply_condition(z, name, mu_r, mu_b, delta, sigma_ratio):
    if name == "unmodified":
        return z
    if name == "centred":
        return z + delta
    if name == "rescaled":
        # Registered as rescale WITHOUT centring: a pure per-channel gain.
        return z * sigma_ratio
    if name == "centred_then_rescaled":
        return (z - mu_r) * sigma_ratio + mu_b
    raise ValueError(name)


CONDITIONS = ["unmodified", "centred", "rescaled", "centred_then_rescaled"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {}
    for seed in SEEDS:
        h = Harness(seed)
        device = next(h.model.parameters()).device
        n_cls = h.args.classes_per_task[TASK]
        offset = h.model.backbone._task_offsets[TASK][0]

        h.at(0, "batch")
        z_tr, y_tr = extract_features(h.model, h.train_tasks[TASK], TASK, "batch")
        W, b, _l2, _ladder = fit_probe(z_tr, y_tr, n_cls, offset, device)

        h.at(CKPT, "batch")
        z_r, y = extract_features(h.model, h.test_tasks[TASK], TASK, "running")
        z_b, _ = extract_features(h.model, h.test_tasks[TASK], TASK, "batch")

        # Deterministic split-half: estimate on A, apply and score on B.
        n = z_r.size(0)
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(n, generator=g)
        a, bidx = perm[: n // 2], perm[n // 2 :]

        mu_r, mu_b, delta, sigma_ratio = corrections(z_r[a], z_b[a])

        gamma = h.model.backbone.feat_norm.weight.detach().float().cpu()
        ratio = delta / gamma
        r = {
            "instrument_sum_delta_over_gamma": float(ratio.sum()),
            "instrument_scale_ref": float(ratio.abs().sum()),
            "delta_norm": float(delta.norm()),
            "gamma_std": float(gamma.std()),
            "sigma_ratio_mean": float(sigma_ratio.mean()),
            "sigma_ratio_std": float(sigma_ratio.std()),
        }
        # delta decomposed against gamma (the direction LayerNorm annihilates).
        par = (delta @ gamma) / (gamma @ gamma) * gamma
        r["delta_parallel_frac"] = float(par.norm() / delta.norm())
        r["delta_orthogonal_frac"] = float((delta - par).norm() / delta.norm())

        f1_batch = {}
        model = h.model
        for head, predict in (
            ("ds", lambda z: ds_predict(model, z)),
            ("linear", lambda z: linear_predict(model, z, W, b)),
        ):
            f1_batch[head] = pooled_macro_f1(predict(z_b[bidx]), y[bidx])
            for cond in CONDITIONS:
                zc = apply_condition(z_r[bidx], cond, mu_r, mu_b, delta, sigma_ratio)
                r[f"{head}_{cond}"] = pooled_macro_f1(predict(zc), y[bidx])
            r[f"{head}_batch"] = f1_batch[head]
            span = f1_batch[head] - r[f"{head}_unmodified"]
            for cond in CONDITIONS[1:]:
                r[f"{head}_{cond}_recovery"] = (
                    (r[f"{head}_{cond}"] - r[f"{head}_unmodified"]) / span
                    if abs(span) > 1e-9
                    else float("nan")
                )
        rows[seed] = r
        print(f"seed {seed}: {json.dumps(r)}", flush=True)
        del h

    (OUT / "arm2.json").write_text(json.dumps(rows, indent=2))

    print("\n=== Gate B1 Arm 2 ===")
    ratios = [rows[s]["instrument_sum_delta_over_gamma"] for s in SEEDS]
    refs = [rows[s]["instrument_scale_ref"] for s in SEEDS]
    print("Instrument check sum_c delta_c/gamma_c (vs scale ref):")
    for s, v, ref in zip(SEEDS, ratios, refs):
        print(f"  seed {s:3d}: {v:+.3e}   ref {ref:8.2f}   relative {abs(v)/ref:.2e}")
    print(
        f"  -> {'PASS' if max(abs(v)/ref for v, ref in zip(ratios, refs)) < 1e-6 else 'FAIL'}"
    )

    for head in ("ds", "linear"):
        print(f"\n{head} head, recovery of the running->batch gap on half B:")
        print(
            f"{'seed':>5} {'running':>8} {'batch':>8} "
            + " ".join(f"{c[:14]:>15}" for c in CONDITIONS[1:])
        )
        for s in SEEDS:
            r = rows[s]
            print(
                f"{s:5d} {r[head+'_unmodified']:8.4f} {r[head+'_batch']:8.4f} "
                + " ".join(
                    f"{r[head+'_'+c+'_recovery']:14.1%} " for c in CONDITIONS[1:]
                )
            )
        for c in CONDITIONS[1:]:
            vals = [rows[s][head + "_" + c + "_recovery"] for s in SEEDS]
            print(f"  mean {c:24s} {sum(vals)/len(vals):.1%}")

    key = [rows[s]["ds_centred_then_rescaled_recovery"] for s in SEEDS]
    print(
        f"\nH2 threshold (DS centred_then_rescaled recovers > 70%): "
        f"mean {sum(key)/len(key):.1%} -> "
        f"{'SUPPORTED' if sum(key)/len(key) > 0.70 else 'NOT SUPPORTED'}"
    )
    par = [rows[s]["delta_parallel_frac"] for s in SEEDS]
    print(
        f"delta gamma-parallel fraction: mean {sum(par)/len(par):.2e} "
        "(identically zero under LayerNorm)"
    )


if __name__ == "__main__":
    main()
