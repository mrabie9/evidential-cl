#!/usr/bin/env python
"""Curate logs/ablations into logs/ablations/<mode>/<algo>/<ID>/[lrX/]<run>/<seed>/.

One directory per published ablation row, holding exactly the runs that back that row
in the current tables:

    docs/ablation_studies.tex  tab:ablation_optim      -> til/{gem,res-er,cmaml,bcl-dual,ctn}
    docs/comb_ablations.tex    tab:comb_til_ablation   -> til/gem_bob
    docs/se_cil_ablations.tex  tab:cil_ablation        -> cil/{res-er,cmaml,bcl-dual}

MANIFEST promotes those runs into the tree; DEMOTE evicts everything that used to live
there and no longer backs a published number (bf16-AMP runs superseded by fp32 reruns,
pre-split-CE C-MAML pools, the lr replicas the tables no longer report, and B2). Demoted
runs go back to logs/<config-stem>/ under their original names -- nothing is deleted.

An lr level is kept only where one algo's rows sit at different learning rates, which
after this curation is GEM alone (G0/G1 at 0.01 vs the not-re-run G2 at 0.03).

Both directions are idempotent: a run already at its destination is skipped. Every
executed operation is journalled to logs/ablations/.organise_journal.jsonl so a move can
be reversed. Dry-run by default; pass --go to execute, --check to only report the config
signature of each curated row.

Usage:
    la-maml_env/bin/python scripts/organise_ablations.py [--go] [--check]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import time

REPO = "/home/lunet/wsmr11/repos/evidential-cl"
LOG = os.path.join(REPO, "logs")
ABL = os.path.join(LOG, "ablations")
JOURNAL = os.path.join(ABL, ".organise_journal.jsonl")

# (mode, algo, row_id, lr|None, [source globs relative to logs/], op)
# op: "move" (canonical home) or "copy" (run shared by two rows; canonical home is the
# row that moves it -- G1 and E1 are the same er_ring configuration).
MANIFEST: list[tuple[str, str, str, str | None, list[str], str]] = [
    # ---------------- TIL leave-one-out, tab:ablation_optim ----------------
    # GEM: G0/G1 recomputed in fp32 at lr 0.01; G2 deliberately not re-run and stays at
    # its bf16 lr 0.03 pool (table note: its -38 effect cannot turn on a ~1.4 pt precision
    # difference). Hence the lr level survives here and nowhere else.
    ("til", "gem", "G0", "0.01", ["gem/g0_noamp_lr01_se_til-*"], "move"),
    ("til", "gem", "G1", "0.01", ["er_ring/ering_static_lr01_noamp_se_til-*"], "copy"),
    ("til", "gem", "G2", "0.03", ["ablations/gem/G2/lr0.03/gp-ablation_se_til-*",
                                  "ablations/gem/G2/lr0.03/gem_noqp_s7-13-21-*"], "move"),
    # Res-ER: all three rows fp32, lr 0.01, inner_steps 2, --eralg4_joint_er on E0.
    ("til", "res-er", "E0", None, ["eralg4/e0_noamp_joint_se_til-*"], "move"),
    ("til", "res-er", "E1", None, ["er_ring/ering_static_lr01_noamp_se_til-*"], "move"),
    ("til", "res-er", "E2", None, ["er_ring/ering_dynring_lr01_noamp_se_til-*"], "move"),
    # C-MAML: M0/M1/M3/M4 re-run after the split-CE reduction fix, all with
    # --cmaml_joint_er. M2 (replay removed) is not re-run -- both fixes are inert with an
    # empty buffer -- so its pre-fix pool stays and only its delta is re-paired.
    ("til", "cmaml", "M0", None, ["cmaml/m0_cmaml_joint_splitfix_se_til-*"], "move"),
    ("til", "cmaml", "M1", None, ["cmaml/m1_firstorder_joint_splitfix_se_til-*"], "move"),
    ("til", "cmaml", "M2", None, ["ablations/cmaml/M2/cmaml_cwfix_noreplay_se_til-*",
                                  "ablations/cmaml/M2/cmaml_noreplay_s7-13-21-*"], "move"),
    ("til", "cmaml", "M3", None,
     ["cmaml_single_inner/m3_singleinner_joint_splitfix_se_til-*"], "move"),
    ("til", "cmaml", "M4", None, ["cmaml_alpha0/m4_alpha0_joint_splitfix_se_til-*"], "move"),
    # BCL-Dual: unaffected by the precision and loss-reduction reruns; the curated pools
    # already back the table.
    ("til", "bcl-dual", "B0", None, ["ablations/bcl-dual/B0/*"], "move"),
    ("til", "bcl-dual", "B1", None, ["ablations/bcl-dual/B1/*"], "move"),
    ("til", "bcl-dual", "B3", None, ["ablations/bcl-dual/B3/*"], "move"),
    ("til", "bcl-dual", "B4", None, ["ablations/bcl-dual/B4/*"], "move"),
    ("til", "bcl-dual", "B5a", None, ["ablations/bcl-dual/B5a/*"], "move"),
    ("til", "bcl-dual", "B5b", None, ["ablations/bcl-dual/B5b/*"], "move"),
    # CTN: reported at its native lr 0.03 and ALWAYS without AMP (bf16 costs CTN ~3 F1
    # and inflates its variance). Every row here had a bf16 twin of the same expt name
    # sitting beside it; those are demoted below.
    ("til", "ctn", "T0", None, ["ablations/ctn/T0/lr0.03/ctn_bm_se_noamp_til-2026-07-11_05-41-35-*",
                                "ablations/ctn/T0/lr0.03/ctn_full_s7-13-21-2026-07-14_*",
                                "ablations/ctn/T0/lr0.03/ctn_full_topup_lr03-*"], "move"),
    ("til", "ctn", "T1", None, ["ablations/ctn/T1/lr0.03/film-ablation_se_noamp_til-*",
                                "ablations/ctn/T1/lr0.03/ctn_nofilm_s7-13-21-2026-07-14_*",
                                "ablations/ctn/T1/lr0.03/ctn_nofilm_topup_lr03-*"], "move"),
    ("til", "ctn", "T2", None, ["ablations/ctn/T2/lr0.03/distill-ablation_se_noamp_til-*",
                                "ablations/ctn/T2/lr0.03/ctn_nodistill_s7-13-21-2026-07-14_14-46-56-*",
                                "ablations/ctn/T2/lr0.03/ctn_nodistill_topup_lr03-*"], "move"),
    ("til", "ctn", "T3", None, ["ablations/ctn/T3/lr0.03/noreplay-ablation_se_noamp_til_s*",
                                "ablations/ctn/T3/lr0.03/ctn_noreplay_s7-13-21-2026-07-14_15-26-33-*"],
     "move"),

    # ---------------- TIL combination study, tab:comb_til_ablation ----------------
    # Table IDs map onto the gem_bob arm IDs used by scripts/compare_gembob.py:
    #   X0=A0  X1=A1(ring)  X2=A2(distill l=1)  X3=A3(bilevel matched)  X4=A5(meta)
    #   Z0=C1  Z0+X3=C2  Z0+X4=C3.  A1A3/A1A5/A2A3/A2A5 are the branch-design pairs that
    # the additivity read uses but the table does not print. All fp32 (_noamp); the bf16
    # and lr0.03 replicas stay in logs/gem_bob*/.
    ("til", "gem_bob", "A0", None, ["gem_bob/gembob_a0_noamp-*"], "move"),
    ("til", "gem_bob", "A1", None, ["gem_bob_dynring/gembob_a1_dynring_noamp-*"], "move"),
    ("til", "gem_bob", "A2", None, ["gem_bob_distill/gembob_a2_distill_l1_noamp-*"], "move"),
    ("til", "gem_bob", "A3", None,
     ["gem_bob_bilevel/gembob_a3_bilevel_matched_noamp-*"], "move"),
    ("til", "gem_bob", "A5", None, ["gem_bob_meta/gembob_a5_meta_noamp-*"], "move"),
    ("til", "gem_bob", "C1", None, ["gem_bob_c1/gembob_c1_noamp-*"], "move"),
    ("til", "gem_bob", "C2", None, ["gem_bob_c2/gembob_c2_noamp-*"], "move"),
    ("til", "gem_bob", "C3", None, ["gem_bob_c3/gembob_c3_noamp-*"], "move"),
    ("til", "gem_bob", "A1A3", None, ["gem_bob_a1a3/gembob_a1a3_noamp-*"], "move"),
    ("til", "gem_bob", "A1A5", None, ["gem_bob_a1a5/gembob_a1a5_noamp-*"], "move"),
    ("til", "gem_bob", "A2A3", None, ["gem_bob_a2a3/gembob_a2a3_noamp-*"], "move"),
    ("til", "gem_bob", "A2A5", None, ["gem_bob_a2a5/gembob_a2a5_noamp-*"], "move"),

    # ---------------- CIL leave-one-out, tab:cil_ablation ----------------
    # Exactly the run dirs pinned by scripts/analyse_cil_ablations.py: the C-MAML rows all
    # carry --cmaml_joint_er, the BCL rows all carry the CIL train-mask fix, and E0 carries
    # --eralg4_joint_er. Their same-named predecessors (07-16/07-17/07-23) are superseded
    # and stay in logs/.
    ("cil", "res-er", "E0", None,
     ["eralg4/eralg4_resER_joint_se_cil-2026-07-23_13-51-02-4315"], "move"),
    # E1/E2 mean the same buffers as in TIL: E1 static per-task ring, E2 the fully-utilised
    # dynamic ring. Both are er_ring at E0's lr 0.01 / inner_steps 2, n=9.
    ("cil", "res-er", "E1", None, ["er_ring/er_ring_static_base_se_cil-*"], "move"),
    ("cil", "res-er", "E2", None, ["er_ring/er_ring_dynring_se_cil-*"], "move"),
    ("cil", "cmaml", "M0", None, ["cmaml/probe-fix_se_cil-2026-07-24_17-06-02-2511"], "move"),
    ("cil", "cmaml", "M1", None,
     ["cmaml/cmaml_secondorder_se_cil-2026-07-24_17-06-01-9403"], "move"),
    ("cil", "cmaml", "M2", None,
     ["cmaml_no_replay/cmaml_noreplay_se_cil-2026-07-24_21-59-47-2576"], "move"),
    ("cil", "cmaml", "M3", None,
     ["cmaml_single_inner/cmaml_singleinner_se_cil-2026-07-24_22-47-00-1856"], "move"),
    ("cil", "cmaml", "M4", None,
     ["cmaml_alpha0/cmaml_alpha0_se_cil-2026-07-25_02-01-57-1956"], "move"),
    ("cil", "bcl-dual", "B0", None,
     ["bcl_dual/fixed-cil-mask_se_cil-2026-07-24_16-58-55-9781"], "move"),
    ("cil", "bcl-dual", "B1", None,
     ["bcl_nodistill/bcl_nodistill_se_cil-2026-07-24_16-58-55-9259"], "move"),
    ("cil", "bcl-dual", "B3", None,
     ["bcl_nodualmem/bcl_nodualmem_se_cil-2026-07-24_22-33-42-3749"], "move"),
    ("cil", "bcl-dual", "B4", None,
     ["bcl_noreplay/bcl_noreplay_se_cil-2026-07-24_22-33-43-4427"], "move"),
    ("cil", "bcl-dual", "B5a", None,
     ["bcl_singlelevel/bcl_singlelevel_is2_se_cil-2026-07-25_01-52-54-5716"], "move"),
    ("cil", "bcl-dual", "B5b", None,
     ["bcl_singlelevel/bcl_singlelevel_is4_se_cil-2026-07-25_02-34-30-7886"], "move"),
]

# (glob of runs to evict, config stem to return them to, reason)
DEMOTE: list[tuple[str, str, str]] = [
    ("ablations/gem/G0/lr0.03/*", "gem", "GEM bf16 lr0.03 pool; table reports lr0.01 fp32"),
    ("ablations/gem/G0/lr0.01/gem_full_lr01abl-*", "gem",
     "GEM bf16 lr0.01; superseded by the fp32 g0_noamp rerun"),
    ("ablations/gem/G1/lr0.03/*", "er_ring", "Ring-ER bf16 lr0.03 pool; table reports lr0.01 fp32"),
    ("ablations/gem/G1/lr0.01/ering_rehearsal_lr01abl-*", "er_ring",
     "Ring-ER bf16 lr0.01; superseded by the fp32 ering_static rerun"),
    ("ablations/gem/G2/lr0.01/*", "gem", "G2 lr0.01 replica; table reports the lr0.03 pool"),
    ("ablations/res-er/E0/eralg4_resER_*", "eralg4",
     "Res-ER bf16, pre --eralg4_joint_er; superseded by e0_noamp_joint"),
    ("ablations/res-er/E1/er_ring_lr*", "er_ring", "Ring-ER bf16; superseded by ering_static fp32"),
    ("ablations/res-er/E2/er_ring_dynring_*", "er_ring",
     "dynamic-ring bf16; superseded by ering_dynring fp32"),
    ("ablations/cmaml/M0/cmaml_*", "cmaml", "C-MAML pre split-CE fix; superseded by m0_*_splitfix"),
    ("ablations/cmaml/M1/cmaml_*", "cmaml", "C-MAML pre split-CE fix; superseded by m1_*_splitfix"),
    ("ablations/cmaml/M3/cmaml_*", "cmaml_single_inner",
     "C-MAML pre split-CE fix; superseded by m3_*_splitfix"),
    ("ablations/cmaml/M4/cmaml_*", "cmaml_alpha0",
     "C-MAML pre split-CE fix; superseded by m4_*_splitfix"),
    ("ablations/bcl-dual/B2/*", "bcl_nobilevel",
     "B2 (n=3) is not a published row; B5a/B5b replaced it as the bilevel ablation"),
    ("ablations/ctn/T0/lr0.03/ctn_bm_se_noamp_til-2026-07-11_04-35-52-*", "ctn",
     "bf16 despite the expt name; the 05-41-35 twin is the fp32 run"),
    ("ablations/ctn/T0/lr0.03/ctn_full_s7-13-21-2026-07-13_*", "ctn",
     "bf16 twin of ctn_full_s7-13-21"),
    ("ablations/ctn/T0/lr0.01/*", "ctn", "CTN lr0.01 replica; table reports lr0.03"),
    ("ablations/ctn/T1/lr0.03/ctn_nofilm_s7-13-21-2026-07-13_*", "ctn_nofilm",
     "bf16 twin of ctn_nofilm_s7-13-21"),
    ("ablations/ctn/T1/lr0.01/*", "ctn_nofilm", "CTN lr0.01 replica; table reports lr0.03"),
    ("ablations/ctn/T2/lr0.03/ctn_nodistill_amp_s7-13-21-*", "ctn_nodistill",
     "bf16 twin of ctn_nodistill_s7-13-21"),
    ("ablations/ctn/T2/lr0.01/*", "ctn_nodistill", "CTN lr0.01 replica; table reports lr0.03"),
    ("ablations/ctn/T3/lr0.03/ctn_noreplay_s7-13-21-2026-07-14_02-15-24-*", "ctn_noreplay",
     "bf16 twin of ctn_noreplay_s7-13-21"),
    ("ablations/ctn/T3/lr0.01/*", "ctn_noreplay", "CTN lr0.01 replica; table reports lr0.03"),
]

# config keys that must agree across every run pooled into one row, with the value to
# assume when a run predates the flag (otherwise a missing key alone reads as a mismatch)
SIGNATURE = {"model": None, "lr": None, "opt_wt": None, "amp": True, "inner_steps": None,
             "n_memories": None, "memory_strength": None, "second_order": False,
             "er_dynamic_ring": False, "class_incremental": False}


def row_dir(mode: str, algo: str, rid: str, lr: str | None) -> str:
    return os.path.join(ABL, mode, algo, rid) + (f"/lr{lr}" if lr else "")


def resolve(patterns: list[str]) -> list[str]:
    """Expand source globs (relative to logs/) to existing run dirs."""
    hits = []
    for pat in patterns:
        hits += [d for d in sorted(glob.glob(os.path.join(LOG, pat))) if os.path.isdir(d)]
    return hits


def n_seeds(run: str) -> int:
    return len([d for d in glob.glob(run + "/*") if os.path.basename(d).isdigit()])


def signature(run: str) -> dict:
    """Config signature of a run dir, read from the first seed's parameters."""
    for p in sorted(glob.glob(os.path.join(run, "*", "training_parameters.json"))):
        d = json.load(open(p))
        return {k: d.get(k, dflt) for k, dflt in SIGNATURE.items()}
    return {}


def build_plan() -> tuple[list[dict], list[str]]:
    plan, missing = [], []
    for mode, algo, rid, lr, pats, op in MANIFEST:
        dest = row_dir(mode, algo, rid, lr)
        hits = resolve(pats)
        # A row is only unresolved if nothing matches AND its destination is still empty;
        # on a re-run the sources are already home (and, for a shared "copy" row, gone
        # from their original location), which is not an error.
        if not hits and not [d for d in glob.glob(dest + "/*") if os.path.isdir(d)]:
            missing.append(f"{mode}/{algo}/{rid}: {pats}")
        for src in hits:
            dst = os.path.join(dest, os.path.basename(src))
            if os.path.abspath(src) == os.path.abspath(dst):
                continue  # already home
            plan.append(dict(op=op, src=src, dst=dst, row=f"{mode}/{algo}/{rid}",
                             seeds=n_seeds(src)))
    for pat, stem, why in DEMOTE:
        for src in sorted(glob.glob(os.path.join(LOG, pat))):
            if not os.path.isdir(src):
                continue
            dst = os.path.join(LOG, stem, os.path.basename(src))
            if os.path.abspath(src) == os.path.abspath(dst):
                continue
            plan.append(dict(op="demote", src=src, dst=dst, row=why, seeds=n_seeds(src)))
    return plan, missing


def check() -> int:
    """Report each curated row: seed set and config signature agreement."""
    bad = 0
    for mode, algo, rid, lr, _pats, _op in MANIFEST:
        dest = row_dir(mode, algo, rid, lr)
        runs = sorted(d for d in glob.glob(dest + "/*") if os.path.isdir(d))
        seeds, sigs = set(), {}
        for r in runs:
            seeds |= {int(os.path.basename(s)) for s in glob.glob(r + "/*")
                      if os.path.basename(s).isdigit()}
            sigs[os.path.basename(r)] = signature(r)
        uniq = {json.dumps(s, sort_keys=True) for s in sigs.values()}
        flag = "" if len(uniq) <= 1 else "  <-- MIXED CONFIGS"
        if len(uniq) > 1:
            bad += 1
        rel = os.path.relpath(dest, ABL)
        print(f"{rel:34s} runs={len(runs)} n={len(seeds):2d} seeds={sorted(seeds)}{flag}")
        if uniq:
            for s in sorted(uniq):
                print(f"    {s}")
        if len(uniq) > 1:
            for name, s in sigs.items():
                print(f"    {name}: {json.dumps(s, sort_keys=True)}")
    print(f"\n{bad} row(s) with mixed configs.")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="execute the plan")
    ap.add_argument("--check", action="store_true", help="only verify the curated rows")
    args = ap.parse_args()

    if args.check:
        return check()

    plan, missing = build_plan()
    for p in plan:
        tag = {"move": "MOVE", "copy": "COPY", "demote": "DEMOTE"}[p["op"]]
        print(f"{tag:6s} [{p['seeds']}s] {os.path.relpath(p['src'], LOG)}"
              f"  ->  {os.path.relpath(p['dst'], LOG)}    ({p['row']})")
    n_mv = sum(1 for p in plan if p["op"] == "move")
    n_cp = sum(1 for p in plan if p["op"] == "copy")
    n_dm = sum(1 for p in plan if p["op"] == "demote")
    print(f"\n{n_mv} move, {n_cp} copy, {n_dm} demote; "
          f"{sum(p['seeds'] for p in plan)} seed-dirs. MISSING={missing}")
    if not args.go:
        print("\n(dry-run; pass --go to execute)")
        return 0
    if missing:
        print("ABORT: unresolved sources")
        return 1
    os.makedirs(ABL, exist_ok=True)
    with open(JOURNAL, "a") as jf:
        for p in plan:
            os.makedirs(os.path.dirname(p["dst"]), exist_ok=True)
            if os.path.exists(p["dst"]):
                print(f"SKIP (destination exists): {p['dst']}")
                continue
            if p["op"] == "copy":
                shutil.copytree(p["src"], p["dst"])
            else:
                shutil.move(p["src"], p["dst"])
            jf.write(json.dumps({"t": time.time(), **{k: p[k] for k in
                                                      ("op", "src", "dst", "row")}}) + "\n")
    # drop the directories the demotions emptied
    for _ in range(4):
        for d in sorted(glob.glob(ABL + "/*/*/*/*") + glob.glob(ABL + "/*/*/*")
                        + glob.glob(ABL + "/*/*") + glob.glob(ABL + "/*"), reverse=True):
            if os.path.isdir(d) and not os.listdir(d):
                os.rmdir(d)
    print(f"\nDONE. journal: {os.path.relpath(JOURNAL, REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
