"""Gate B1 rider: mass-flow decomposition on task-0 anchors.

Not gating on its own. Registered as a rival-distribution shape result, NOT an
evidential-uncertainty result: at the shipped activation_norm="max" the fused
ignorance mass is ~4e-4, so Dou_y = 1 - p_y identically.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate_b1_arm1 import TASK  # noqa: E402
from gate_b1_lib import SEEDS, Harness, extract_features  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "logs/eucr/gate_b1"


@torch.no_grad()
def masses(model, z, task=TASK):
    return model.backbone.ds_heads[str(task)](
        z.to(next(model.parameters()).device)
    ).cpu()


def rival_stats(m, y_local):
    """Bel_y, ignorance, and the shape of the renormalised rival distribution."""
    beliefs = m[..., :-1]
    omega = m[..., -1]
    n = beliefs.size(0)
    bel_y = beliefs[torch.arange(n), y_local]
    rivals = beliefs.clone()
    rivals[torch.arange(n), y_local] = 0.0
    total = rivals.sum(1, keepdim=True).clamp_min(1e-12)
    p = rivals / total
    entropy = -(p.clamp_min(1e-12).log() * p).sum(1)
    top2 = p.topk(2, dim=1)
    return {
        "bel_y": bel_y,
        "omega": omega,
        "rival_entropy": entropy,
        "rival_top2_gap": top2.values[:, 0] - top2.values[:, 1],
        "modal_rival": top2.indices[:, 0],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {}
    for seed in SEEDS:
        h = Harness(seed)
        offset = h.model.backbone._task_offsets[TASK][0]
        noise_local = None
        nl = getattr(h.model, "noise_label", None)
        if nl is not None:
            noise_local = int(nl) - offset

        h.at(0, "batch")
        z0, y = extract_features(h.model, h.test_tasks[TASK], TASK, "batch")
        y_local = (y - offset).long()
        m0 = masses(h.model, z0)
        correct0 = m0[..., :-1].argmax(1) == y_local
        keep = correct0.clone()
        if noise_local is not None:
            keep &= y_local != noise_local
        idx = keep.nonzero(as_tuple=True)[0]

        s0 = rival_stats(m0[idx], y_local[idx])
        r = {
            "n_anchors": int(idx.numel()),
            "n_total": int(y.numel()),
            "omega_ck0_mean": float(s0["omega"].mean()),
        }

        h.at(3, "batch")
        gamma_feat = None
        per_regime = {}
        for regime in ("running", "batch"):
            z3, _ = extract_features(h.model, h.test_tasks[TASK], TASK, regime)
            if regime == "running":
                z3_r = z3
            else:
                gamma_feat = z3
            m3 = masses(h.model, z3[idx])
            s3 = rival_stats(m3, y_local[idx])
            flipped = m3[..., :-1].argmax(1) != y_local[idx]
            per_regime[regime] = (s3, flipped)
            r[f"omega_ck3_{regime}_mean"] = float(s3["omega"].mean())
            r[f"flip_rate_{regime}"] = float(flipped.float().mean())
            r[f"delta_bel_y_{regime}"] = float(
                (s3["bel_y"] - s0["bel_y"])[flipped].mean()
            )
            r[f"delta_omega_{regime}"] = float(
                (s3["omega"] - s0["omega"])[flipped].mean()
            )
            r[f"rival_entropy_{regime}"] = float(s3["rival_entropy"][flipped].mean())
            r[f"rival_top2_gap_{regime}"] = float(s3["rival_top2_gap"][flipped].mean())

        # H3: is the modal rival's class aligned with the feature shift delta?
        delta = gamma_feat.mean(0) - z3_r.mean(0)
        proto_w = h.model.backbone.ds_heads[str(TASK)].ds1.w.detach().float().cpu()
        align = torch.nn.functional.cosine_similarity(
            proto_w, delta.unsqueeze(0).expand_as(proto_w), dim=1
        )
        beta = h.model.backbone.ds_heads[str(TASK)].ds2.beta.detach().float().cpu()
        proto_class = (beta**2).argmax(0)  # class each prototype most supports
        n_cls = int(proto_class.max()) + 1
        class_align = torch.tensor(
            [
                (
                    align[proto_class == c].mean()
                    if (proto_class == c).any()
                    else float("nan")
                )
                for c in range(n_cls)
            ]
        )
        order = torch.argsort(class_align, descending=True)
        rank = {int(c): i for i, c in enumerate(order.tolist())}
        s3r, flipped_r = per_regime["running"]
        modal = s3r["modal_rival"][flipped_r]
        if modal.numel():
            modal_class = int(torch.bincount(modal, minlength=n_cls).argmax())
            r["modal_rival_class"] = modal_class
            r["modal_rival_align_rank"] = rank[modal_class]
            r["n_classes"] = n_cls
            r["modal_in_top_quartile"] = bool(rank[modal_class] < max(1, n_cls // 4))
            r["class_alignment"] = [round(float(v), 4) for v in class_align]
        rows[seed] = r
        print(f"seed {seed}: {json.dumps(r)}", flush=True)
        del h

    (OUT / "rider.json").write_text(json.dumps(rows, indent=2))

    print("\n=== Gate B1 rider: mass flow on flipped task-0 anchors ===")
    print(
        f"{'seed':>5} {'anchors':>8} {'flip run':>9} {'flip bat':>9} "
        f"{'dBel run':>9} {'dOmega r':>9} {'H_riv run':>10} {'H_riv bat':>10}"
    )
    for s in SEEDS:
        r = rows[s]
        print(
            f"{s:5d} {r['n_anchors']:8d} {r['flip_rate_running']:9.3f} "
            f"{r['flip_rate_batch']:9.3f} {r['delta_bel_y_running']:+9.4f} "
            f"{r['delta_omega_running']:+9.2e} {r['rival_entropy_running']:10.4f} "
            f"{r['rival_entropy_batch']:10.4f}"
        )
    lower = [
        s
        for s in SEEDS
        if rows[s]["rival_entropy_running"] < rows[s]["rival_entropy_batch"]
    ]
    print(
        f"\nH3 part 1 (rival entropy lower under running): {len(lower)}/5 seeds {lower}"
    )
    top = [s for s in SEEDS if rows[s].get("modal_in_top_quartile")]
    print(
        f"H3 part 2 (modal rival class in top quartile by delta alignment): "
        f"{len(top)}/5 seeds {top}"
    )
    print(
        "  modal class / alignment rank per seed: "
        + " ".join(
            f"{s}:{rows[s].get('modal_rival_class')}@{rows[s].get('modal_rival_align_rank')}"
            for s in SEEDS
        )
    )
    ok = len(lower) >= 4 and len(top) >= 4
    print(f"H3 -> {'SUPPORTED' if ok else 'NOT SUPPORTED'}")
    om = [rows[s]["omega_ck0_mean"] for s in SEEDS]
    print(
        f"\nDeclared degeneracy check: mean fused omega at ck0 = {sum(om)/len(om):.3e}"
    )


if __name__ == "__main__":
    main()
