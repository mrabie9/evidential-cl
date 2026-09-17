#!/usr/bin/env python
"""Paired significance tests for the gem_bob (GEM "best of best") add-one study.

Every arm is compared to the A0 baseline (gem_bob with all mechanism gates off, which
reproduces plain GEM exactly) over a common seed set, using the same protocol as the
leave-one-out grid in docs/ablation_studies.tex Sec. "Statistical protocol": paired t
with a 95% CI, an exact sign-flip permutation p (floor 1/2^(n-1) = 0.004 at n=9), and
Holm-Bonferroni correction, combined into the verdict of compare_ablation.verdict
against a smallest effect of interest of DELTA = 1 F1 point.

The stats primitives are imported from compare_ablation.py so both studies cannot drift
apart. Arms read the curated tree logs/ablations/til/gem_bob/<ID>/<run>/<seed>/results.txt
(scripts/organise_ablations.py), which holds the fp32 (--no-amp) replica -- the run set
behind tab:comb_til_ablation, whose IDs map onto the arm IDs here as
    X0=A0  X1=A1  X2=A2  X3=A3  X4=A5  Z0=C1  Z0+X3=C2  Z0+X4=C3.
The bf16 and lr0.03 replicas were never promoted and stay in logs/gem_bob*/; --legacy
reads those by expt_name prefix (newest run wins per seed) for the precision comparison.

HOLM FAMILY. Correction runs across the four MECHANISM arms (A1, A2, A3, A5) only. A4 is
the unmatched-budget rerun of A3, i.e. the same hypothesis at a different step count, so
including it would inflate the family size; it is reported as a diagnostic and the A3-vs-A4
gap is what separates structure from compute (see "budget masquerades as mechanism").

Usage:  la-maml_env/bin/python scripts/compare_gembob.py [--phase 1|2|all] [--verbose]
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compare_ablation import DELTA, holm, mean_std, paired, parse_results, verdict

LOG = "/home/lunet/wsmr11/repos/evidential-cl/logs"
TREE = f"{LOG}/ablations/til/gem_bob"
# --legacy only: main.py names each run directory after the CONFIG FILE STEM, not the model,
# so the un-promoted replicas scatter across logs/gem_bob/, logs/gem_bob_dynring/,
# logs/gem_bob_distill/, etc. Pool by expt_name (unique per arm) across all of them.
MODEL_DIR = f"{LOG}/gem_bob*"

BASELINE = ("A0", "gembob_a0", "baseline (all gates off == plain GEM)")

# (id, expt_name prefix, description). MECHANISM arms enter the Holm family.
# A2 uses distill_lambda=1 (gembob_a2_distill_l1), matching BCL-Dual's distill coefficient
# (self.reg=memory_strength=1). The original distill_lambda=0.1 run (gembob_a2_distill, eff
# weight 0.004) was underpowered on F1 (+0.3, inconclusive) and is retained as the A2' diagnostic;
# at lambda=1 (eff 0.04) distillation is load-bearing on F1 (+1.8, F1 66.2, reproducing the
# gem_distill headline) and BWT. The KL term is uncompensated (no T^2) in gem_bob and both parents.
PHASE1_MECHANISMS = [
    ("A1", "gembob_a1_dynring", "+ fully-utilised ring buffer (E2)"),
    ("A2", "gembob_a2_distill_l1", "+ KL distillation, lambda=1 (B1/T2, BCL-matched)"),
    ("A3", "gembob_a3_bilevel_matched", "+ bilevel, budget-matched (B5b)"),
    ("A5", "gembob_a5_meta", "+ meta-batch averaging K=3 (M3)"),
]
PHASE1_DIAGNOSTIC = [
    ("A2'", "gembob_a2_distill", "+ KL distillation, lambda=0.1 (eff 0.004, orig underweight)"),
    ("A4", "gembob_a4_bilevel_unmatched", "+ bilevel, 2x budget (B5a analogue)"),
]
# Phase 2 combines the Phase-1 load-bearing mechanisms (A1 ring, A2 distill lambda=1) and
# tests whether the two inert/inconclusive arms (A3 bilevel, A5 meta-batch) add anything on top.
PHASE2 = [
    ("C1", "gembob_c1", "A1+A2: ring + distill(1)"),
    ("C2", "gembob_c2", "A1+A2+A3: + bilevel (inner_steps 1, budget-matched)"),
    ("C3", "gembob_c3", "A1+A2+A5: + meta-batch K=3"),
]
# Phase 2b: the remaining pairs (branch design) -- each winner paired with each non-winner,
# scheduled to backfill the slots freed as C1/C2/C3 finish.
PHASE2B = [
    ("A1A5", "gembob_a1a5", "A1+A5: ring + meta"),
    ("A2A5", "gembob_a2a5", "A2+A5: distill + meta"),
    ("A2A3", "gembob_a2a3", "A2+A3: distill + bilevel"),
    ("A1A3", "gembob_a1a3", "A1+A3: ring + bilevel"),
]
# Each combo's parent arms. The honest additivity bar is the BEST parent: a combo is only
# worth its extra mechanism if it beats the strongest arm it already contains. Paired delta
# near 0 against the best parent => that added mechanism is redundant in this combination.
PHASE2_PARENTS = {
    "C1": ("gembob_c1", [("A1", "gembob_a1_dynring"), ("A2", "gembob_a2_distill_l1")]),
    "C2": ("gembob_c2", [("C1", "gembob_c1"), ("A3", "gembob_a3_bilevel_matched")]),
    "C3": ("gembob_c3", [("C1", "gembob_c1"), ("A5", "gembob_a5_meta")]),
    "A1A5": ("gembob_a1a5", [("A1", "gembob_a1_dynring"), ("A5", "gembob_a5_meta")]),
    "A2A5": ("gembob_a2a5", [("A2", "gembob_a2_distill_l1"), ("A5", "gembob_a5_meta")]),
    "A2A3": ("gembob_a2a3", [("A2", "gembob_a2_distill_l1"),
                             ("A3", "gembob_a3_bilevel_matched")]),
    "A1A3": ("gembob_a1a3", [("A1", "gembob_a1_dynring"),
                             ("A3", "gembob_a3_bilevel_matched")]),
}

# set from --legacy: None reads the curated tree, a string reads logs/gem_bob*/ with that
# expt-name suffix ("" = the bf16 lr0.01 study, "_lr03", "_noamp").
LEGACY = None


def arm_pool(arm_id, expt_prefix=None):
    """Pool per-seed (f1, bwt) for one arm.

    With LEGACY unset the arm is its curated directory logs/ablations/til/gem_bob/<ID>.
    With --legacy the arm is every run under logs/gem_bob*/ whose basename starts with
    <expt_prefix><LEGACY>-, so a topped-up rerun pools with the original. Newest run wins
    per seed either way.

    Returns:
        (pool, contributors) where pool maps seed -> (f1, bwt, mtime, dir).
    """
    if LEGACY is None:
        pattern, expt_prefix = f"{TREE}/{arm_id}/*/*", None
    else:
        pattern = f"{MODEL_DIR}/{expt_prefix}{LEGACY}-*/*"
    pool = {}
    for seed_dir in glob.glob(pattern):
        seed = os.path.basename(seed_dir)
        if not (
            seed.isdigit()
            and os.path.isdir(seed_dir)
            and os.path.exists(os.path.join(seed_dir, "results.txt"))
        ):
            continue
        # Guard against a prefix collision (e.g. "gembob_a1" matching "gembob_a1_alt"):
        # the character after the prefix must be the timestamp separator.
        run_name = os.path.basename(os.path.dirname(seed_dir))
        if expt_prefix and not run_name.startswith(f"{expt_prefix}{LEGACY}-"):
            continue
        f1, bwt = parse_results(seed_dir)
        if f1 is None:
            continue
        mtime = os.path.getmtime(seed_dir)
        seed_i = int(seed)
        if seed_i not in pool or mtime > pool[seed_i][2]:
            pool[seed_i] = (f1, bwt, mtime, seed_dir)
    contributors = {}
    for seed_i, (_, _, _, seed_dir) in pool.items():
        contributors.setdefault(
            os.path.basename(os.path.dirname(seed_dir)), []
        ).append(seed_i)
    return pool, sorted(contributors.items())


def report(arms, base_pool, base_id, title, holm_family=True, verbose=False):
    """Print one table of arms against the baseline, Holm-corrected within the table."""
    print(f"\n{title}")
    print(
        f"  {'ID':4s} {'F1':>14s} {'BWT':>14s} {'dF1':>7s} {'95% CI':>17s} "
        f"{'p_holm':>7s} {'p_perm':>7s}  {'verdict':13s}  description"
    )
    rows = []
    for arm_id, prefix, desc in arms:
        pool, contributors = arm_pool(arm_id, prefix)
        if not pool:
            print(f"  {arm_id:4s} {'-- no runs found --':>14s}   {desc}")
            continue
        f1_mu, f1_sd, n_f1 = mean_std([v[0] for v in pool.values()])
        bwt_mu, bwt_sd, _ = mean_std([v[1] for v in pool.values()])
        st = paired(base_pool, pool, key=0)
        rows.append((arm_id, desc, f1_mu, f1_sd, n_f1, bwt_mu, bwt_sd, st, contributors))

    if not rows:
        return
    p_ts = [r[7]["p_t"] for r in rows]
    adj = (
        holm([p if p == p else 1.0 for p in p_ts])
        if holm_family
        else [p if p == p else 1.0 for p in p_ts]
    )
    for (arm_id, desc, f1_mu, f1_sd, n_f1, bwt_mu, bwt_sd, st, contributors), ph in zip(
        rows, adj
    ):
        lo, hi = (c * 100 for c in st["ci"])
        v = verdict(ph, st["p_perm"], lo, hi) if st["n"] >= 2 else f"n={st['n']}"
        print(
            f"  {arm_id:4s} {f1_mu * 100:7.1f}+-{f1_sd * 100:4.1f}({n_f1}) "
            f"{bwt_mu * 100:7.1f}+-{bwt_sd * 100:4.1f} "
            f"{st['mean'] * 100:+7.1f} [{lo:+6.1f},{hi:+6.1f}] "
            f"{ph:7.3f} {st['p_perm']:7.3f}  {v:13s}  {desc}"
        )
        if verbose:
            print(f"       paired seeds (n={st['n']}): {st['seeds']}")
            for run_name, seeds in contributors:
                print(f"       from {run_name}: {sorted(seeds)}")
    if not holm_family:
        print("       (uncorrected: diagnostic arm, not part of the Holm family)")


def report_additivity(base_id, base_prefix, verbose=False):
    """For each combo present, print its paired F1 delta vs A0 and vs each parent arm.

    ``paired(ref, combo)`` returns ``combo - ref``, so a positive delta means the combo
    beats the reference. The row of interest is the combo against its BEST parent: if that
    delta's CI includes 0, the combination does not improve on the parent (the added
    mechanism is redundant there). No Holm correction here -- this is a descriptive
    additivity read, not a family of confirmatory tests.
    """
    print("\nADDITIVITY -- each combo vs A0 and vs its parent arms (paired F1)")
    print(
        f"  {'combo':18s} {'F1':>7s}   {'vs':6s} {'dF1':>7s} {'95% CI':>16s} {'p_perm':>7s}"
    )
    for combo_id, (combo_prefix, parents) in PHASE2_PARENTS.items():
        combo_pool, _ = arm_pool(combo_id, combo_prefix)
        if not combo_pool:
            continue
        c_mu, _, _ = mean_std([v[0] for v in combo_pool.values()])
        refs = [(base_id, base_prefix)] + parents
        first = True
        for ref_id, ref_prefix in refs:
            ref_pool, _ = arm_pool(ref_id, ref_prefix)
            if not ref_pool:
                continue
            st = paired(ref_pool, combo_pool, key=0)
            if st["n"] < 2:
                continue
            lo, hi = st["ci"][0] * 100, st["ci"][1] * 100
            label = combo_id
            head = (
                f"  {label:18s} {c_mu * 100:6.1f} " if first else f"  {'':18s} {'':6s} "
            )
            print(
                f"{head}  {ref_id:6s} {st['mean'] * 100:+6.1f} "
                f"[{lo:+6.1f},{hi:+6.1f}] {st['p_perm']:7.3f}"
            )
            first = False
    print("  (delta = combo - reference; positive => combo better. Best-parent row is the")
    print("   additivity test: CI excluding 0 => the added mechanism helps on top of it.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["1", "2", "all"])
    ap.add_argument("--verbose", action="store_true", help="print contributing run dirs")
    ap.add_argument(
        "--legacy",
        nargs="?",
        const="",
        default=None,
        metavar="SUFFIX",
        help="read an un-promoted replica from logs/gem_bob*/ instead of the curated "
        "tree: bare --legacy is the bf16 lr=0.01 study, '_lr03' the lr=0.03 one, "
        "'_noamp' the fp32 runs as they were logged before promotion.",
    )
    args = ap.parse_args()

    global LEGACY
    LEGACY = args.legacy
    src = f"logs/gem_bob*/<arm>{LEGACY}-*" if LEGACY is not None else "logs/ablations/til/gem_bob"

    base_id, base_prefix, base_desc = BASELINE
    base_pool, base_contributors = arm_pool(base_id, base_prefix)
    if not base_pool:
        print(f"No baseline runs found under {src}")
        return 1
    print(f"\nsource: {src}")
    f1_mu, f1_sd, n_f1 = mean_std([v[0] for v in base_pool.values()])
    bwt_mu, bwt_sd, _ = mean_std([v[1] for v in base_pool.values()])
    print(
        f"\nBaseline {base_id}: F1 {f1_mu * 100:.1f}+-{f1_sd * 100:.1f} (n={n_f1})  "
        f"BWT {bwt_mu * 100:.1f}+-{bwt_sd * 100:.1f}   [{base_desc}]"
    )
    if args.verbose:
        for run_name, seeds in base_contributors:
            print(f"       from {run_name}: {sorted(seeds)}")

    if args.phase in ("1", "all"):
        report(
            PHASE1_MECHANISMS,
            base_pool,
            base_id,
            "PHASE 1 -- add-one (Holm across the 4 mechanism arms)",
            holm_family=True,
            verbose=args.verbose,
        )
        report(
            PHASE1_DIAGNOSTIC,
            base_pool,
            base_id,
            "PHASE 1 -- budget diagnostic (compare against A3, not a mechanism claim)",
            holm_family=False,
            verbose=args.verbose,
        )
    if args.phase in ("2", "all"):
        report(
            PHASE2,
            base_pool,
            base_id,
            "PHASE 2 -- combinations of the surviving mechanisms (vs A0)",
            holm_family=True,
            verbose=args.verbose,
        )
        report(
            PHASE2B,
            base_pool,
            base_id,
            "PHASE 2b -- remaining pairs, branch design (vs A0)",
            holm_family=True,
            verbose=args.verbose,
        )
        report_additivity(base_id, base_prefix, verbose=args.verbose)

    print(
        f"\n  verdict vs delta={DELTA} F1 pt: load-bearing (p_holm & p_perm < 0.05) / "
        f"borderline (CI excludes 0) / inert (CI within +-{DELTA}) / inconclusive"
    )
    print("  metric: results.txt 'Final F1' (macro F1 including noise), sample stdev\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
