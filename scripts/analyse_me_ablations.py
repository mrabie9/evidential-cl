#!/usr/bin/env python3
"""Paired leave-one-out analysis for the MULTI-EPOCH ablation grids (TIL and CIL).

Reads the runs produced by scripts/run_me_ablations.sh -- the multi-epoch
(--n_epochs 10 --inner_steps 1 --no-amp) rerun of the two published single-epoch grids,
tab:ablation_optim (docs/ablation_studies.tex) and tab:cil_ablation
(docs/se_cil_ablations.tex). Both modes share one regime and one precision here, so the
TIL and CIL tables are read across as well as down.

Pooling is by (config stem, expt_name): a row is every seed under
logs/<stem>/<expt>-<timestamp>/<seed>/results.txt, with the newest run dir winning a
duplicate seed. The expt_name is unique per row, so no curated tree is needed.

Statistics follow sec:stat_protocol of docs/ablation_studies.tex: per-seed paired deltas
against the family baseline over their common seeds, mean effect with a 95% Student-t CI,
Holm-Bonferroni-corrected paired t-test within each family, and an exact sign-flip
permutation test. Verdicts are against a smallest effect of interest of 1 F1 point.

Metric: canonical macro f1_total = results.txt "Final F1". Also reported are BWT
(results.txt "Backward"), p_fa (seed_metrics val_fa) and forward zero-shot transfer.

Writes:
  docs/me_til_ablations.tex      tab:me_til_ablation
  docs/me_cil_ablations.tex      tab:me_cil_ablation
  docs/me_cil_ablations_fwt.tex  tab:me_cil_ablation_fwt
  docs/me_ablation_analysis.txt  the plain-text grid, both modes

Usage:  la-maml_env/bin/python scripts/analyse_me_ablations.py [--verbose]
"""
import argparse
import glob
import json
import os
from itertools import product

import numpy as np
from scipy import stats
from results_txt import f1_stats

REPO = "/home/lunet/wsmr11/repos/evidential-cl"
LOGS = os.path.join(REPO, "logs")
DELTA = 1.0  # smallest effect of interest, F1 points

# mode -> family -> [(row_id, mechanism label, config stem, expt_name), ...]; first row is
# the family baseline. G1 has no job of its own: it is the same er_ring configuration as
# E1, so both rows read the E1 pool, exactly as the single-epoch curated tree does.
GRIDS = {
    "til": {
        "GEM": [
            ("G0", "None (full method)", "gem", "g0_me_til"),
            ("G1", "Gradient projection $\\to$ episodic replay (ring-buffer)",
             "er_ring", "e1_me_til"),
            ("G2", "Episodic memory: projection and replay both off", "gem_noqp", "g2_me_til"),
        ],
        "Res-ER": [
            ("E0", "None (full method)", "eralg4", "e0_me_til"),
            ("E1", "Reservoir sampling $\\to$ static ring buffer (Ring-ER)",
             "er_ring", "e1_me_til"),
            ("E2", "Reservoir sampling $\\to$ fully-utilised Ring-ER",
             "er_ring", "e2_me_til"),
        ],
        "C-MAML": [
            ("M0", "None (full method)", "cmaml", "m0_me_til"),
            ("M1", "Second-order meta-gradient (Hessian term)", "cmaml", "m1_me_til"),
            ("M2", "Episodic replay ($B^{\\text{rep}}$ in the meta-loss)",
             "cmaml_no_replay", "m2_me_til"),
            ("M3", "Mini-batch averaging ($K_{\\text{meta}}\\!=\\!1$)",
             "cmaml_single_inner", "m3_me_til"),
            ("M4", "Meta-learning ($\\alpha\\!=\\!0$, no inner loop)",
             "cmaml_alpha0", "m4_me_til"),
        ],
        "BCL-Dual": [
            ("B0", "None (full method)", "bcl_dual", "b0_me_til"),
            ("B1", "\\gls{kl} distillation term $\\lambda_{\\text{KL}}\\,D_{\\text{KL}}(\\cdot)$",
             "bcl_nodistill", "b1_me_til"),
            ("B3", "Second memory buffer (reverts to a single buffer)",
             "bcl_nodualmem", "b3_me_til"),
            ("B4", "Episodic replay (both channels)", "bcl_noreplay", "b4_me_til"),
            ("B5a", "Bilevel structure $\\to$ single loop", "bcl_singlelevel", "b5a_me_til"),
            ("B5b", "Bilevel structure $\\to$ single loop, matched steps/batch",
             "bcl_singlelevel", "b5b_me_til"),
        ],
        "CTN": [
            ("T0", "None (full method)", "ctn", "t0_me_til"),
            ("T1", "\\gls{film} task conditioning", "ctn_nofilm", "t1_me_til"),
            ("T2", "\\gls{kl} distillation term $\\lambda_{\\text{KL}}\\,D_{\\text{KL}}(\\cdot)$",
             "ctn_nodistill", "t2_me_til"),
            ("T3", "Episodic replay", "ctn_noreplay", "t3_me_til"),
        ],
    },
    "cil": {
        "Res-ER": [
            ("E0", "None (full method)", "eralg4", "e0_me_cil"),
            ("E1", "Reservoir sampling $\\to$ static ring buffer (Ring-ER)",
             "er_ring", "e1_me_cil"),
            ("E2", "Reservoir sampling $\\to$ fully-utilised Ring-ER",
             "er_ring", "e2_me_cil"),
        ],
        "C-MAML": [
            ("M0", "None (full method)", "cmaml", "m0_me_cil"),
            ("M1", "Second-order meta-gradient (Hessian term)", "cmaml", "m1_me_cil"),
            ("M2", "Episodic replay ($B^{\\text{rep}}$ in the meta-loss)",
             "cmaml_no_replay", "m2_me_cil"),
            ("M3", "Mini-batch averaging ($K_{\\text{meta}}\\!=\\!1$)",
             "cmaml_single_inner", "m3_me_cil"),
            ("M4", "Meta-learning ($\\alpha\\!=\\!0$, no inner loop)",
             "cmaml_alpha0", "m4_me_cil"),
        ],
        "BCL-Dual": [
            ("B0", "None (full method)", "bcl_dual", "b0_me_cil"),
            ("B1", "\\gls{kl} distillation term $\\lambda_{\\text{KL}}\\,D_{\\text{KL}}(\\cdot)$",
             "bcl_nodistill", "b1_me_cil"),
            ("B3", "Second memory buffer (reverts to a single buffer)",
             "bcl_nodualmem", "b3_me_cil"),
            ("B4", "Episodic replay (both channels)", "bcl_noreplay", "b4_me_cil"),
            ("B5a", "Bilevel structure $\\to$ single loop", "bcl_singlelevel", "b5a_me_cil"),
            ("B5b", "Bilevel structure $\\to$ single loop, matched steps/batch",
             "bcl_singlelevel", "b5b_me_cil"),
        ],
    },
}

IDX = dict(f1=0, bwt=1, diag=2, pfa=3, fwt=4)


def _forward_zs(seed_dir):
    """Forward-transfer zero-shot: mean over tasks 1..9 of each task's pre-train zero-shot
    total F1 (metrics/task{t}.npz 'zero_shot_total_f1'). This is the trained_zs term of
    FWT = trained_zs - untrained_zs; the untrained baseline is a per-seed constant common
    to every method, so it cancels in the paired within-seed delta and the verdicts are
    baseline-invariant. NaN when the metrics are absent."""
    md = os.path.join(seed_dir, "metrics")
    vals = []
    for t in range(1, 10):
        p = os.path.join(md, f"task{t}.npz")
        if not os.path.exists(p):
            return np.nan
        z = np.load(p, allow_pickle=True)
        if "zero_shot_total_f1" not in z.files:
            return np.nan
        vals.append(float(z["zero_shot_total_f1"]))
    return float(np.mean(vals)) if vals else np.nan


def parse_seed(seed_dir):
    """Return (f1_total, bwt, diag, pfa, fwt) as fractions for one seed directory."""
    txt = open(os.path.join(seed_dir, "results.txt"), errors="ignore").read()
    f1_values = f1_stats(txt)
    f1 = f1_values["Final F1"]
    bwt = f1_values["Backward"]
    diag = f1_values.get("Diagonal F1", np.nan)
    pfa = np.nan
    smj = os.path.join(seed_dir, "seed_metrics.json")
    if os.path.exists(smj):
        try:
            pfa = float(json.load(open(smj)).get("val_fa", np.nan))
        except Exception:
            pass
    return f1, bwt, diag, pfa, _forward_zs(seed_dir)


def load_row(stem, expt):
    """Pool every seed under logs/<stem>/<expt>-<timestamp>/. Returns {seed: tuple};
    later (newer) run dirs win a duplicate seed."""
    dirs = sorted(d for d in glob.glob(os.path.join(LOGS, stem, expt + "-*")) if os.path.isdir(d))
    seedmap = {}
    for d in dirs:
        for sd in sorted(os.listdir(d)):
            if sd.isdigit() and os.path.exists(os.path.join(d, sd, "results.txt")):
                try:
                    seedmap[int(sd)] = parse_seed(os.path.join(d, sd))
                except Exception:
                    pass  # a seed still running has a partial results.txt
    return seedmap


def series(seedmap, seeds, key):
    return np.array([seedmap[s][IDX[key]] for s in seeds]) * 100


def allser(seedmap, key):
    return series(seedmap, sorted(seedmap), key)


def perm_pvalue(delta):
    """Exact two-sided sign-flip permutation p on the mean of paired deltas."""
    n = len(delta)
    obs = abs(delta.mean())
    count = 0
    for signs in product([1, -1], repeat=n):
        if abs((np.array(signs) * delta).mean()) >= obs - 1e-12:
            count += 1
    return count / (2 ** n)


def paired_stats(base, abl):
    d = abl - base
    n = len(d)
    if n == 0:
        return dict(mean=np.nan, ci=(np.nan, np.nan), t_p=np.nan, p_perm=np.nan, d=d, n=0)
    mean = d.mean()
    if n < 2 or not np.all(np.isfinite(d)):
        return dict(mean=mean, ci=(np.nan, np.nan), t_p=np.nan, p_perm=np.nan, d=d, n=n)
    se = d.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    ci = (mean - tcrit * se, mean + tcrit * se)
    if se == 0:
        t_p = 0.0 if mean != 0 else 1.0
    else:
        _, t_p = stats.ttest_rel(abl, base)
    return dict(mean=mean, ci=ci, t_p=t_p, p_perm=perm_pvalue(d), d=d, n=n)


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values, preserving input order."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(np.nan_to_num(pvals, nan=1.0))
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        if val != val:  # a row with too few paired seeds to test yet
            adj[idx] = np.nan
            continue
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def verdict(p_holm, p_perm, ci):
    """Four-way verdict of sec:stat_protocol, keyed to the glossary macros used by the
    published TIL table: ss = significant, ms = marginal, ns = insignificant."""
    if not (ci[0] == ci[0]):
        return "n/a"
    excl0 = ci[0] > 0 or ci[1] < 0
    if p_holm == p_holm and p_perm == p_perm and p_holm < 0.05 and p_perm < 0.05:
        return "ss"
    if excl0:
        return "ms"
    if ci[0] > -DELTA and ci[1] < DELTA:
        return "ns"
    return "underpowered"


VERDICT_TEX = {"ss": r"\gls{ss}", "ms": r"\gls{ms}", "ns": r"\gls{ns}",
               "underpowered": "underpowered", "n/a": "---"}


def family_stats(rows):
    """Baseline summary plus per-ablation paired stats, pairing each ablation against the
    baseline on their COMMON seeds so n grows as top-ups land."""
    base_id, _, bstem, bexpt = rows[0]
    base_map = load_row(bstem, bexpt)
    fs = dict(base_id=base_id, base_map=base_map, base_n=len(base_map), abl=[])
    for rid, label, stem, expt in rows[1:]:
        amap = load_row(stem, expt)
        common = sorted(set(base_map) & set(amap))
        # Fewer than two paired seeds cannot be tested, but the mean delta is still real
        # and must be shown as such -- scoring zeros here would print a fabricated +0.0
        # for a row that is tens of points below its baseline.
        fs["abl"].append(dict(
            rid=rid, label=label, n=len(common), map=amap,
            sf=paired_stats(series(base_map, common, "f1"), series(amap, common, "f1")),
            sb=paired_stats(series(base_map, common, "bwt"), series(amap, common, "bwt")),
            sw=paired_stats(series(base_map, common, "fwt"), series(amap, common, "fwt")),
        ))
    for metric, pkey, vkey in (("sf", "holm", "verdict"), ("sw", "holm_fwt", "verdict_fwt"),
                               ("sb", "holm_bwt", "verdict_bwt")):
        adj = holm([a[metric]["t_p"] for a in fs["abl"]]) if fs["abl"] else []
        for i, a in enumerate(fs["abl"]):
            a[pkey] = adj[i]
            a[vkey] = verdict(adj[i], a[metric]["p_perm"], a[metric]["ci"])
    return fs


def ms(arr):
    """mean, sample std over a pooled series, NaN-safe."""
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    return arr.mean(), (arr.std(ddof=1) if arr.size > 1 else 0.0)


# ---------------------------------------------------------------- text report


def text_report(mode, verbose=False):
    lines = [f"\n########## {mode.upper()}  (multi-epoch: 10 epochs, inner_steps 1, fp32) ##########"]
    for fam, rows in GRIDS[mode].items():
        fs = family_stats(rows)
        lines.append(f"\n===== {fam} =====")
        # n is "paired/pooled": the seeds shared with the baseline (what the test uses)
        # over the seeds this row has landed so far.
        lines.append(f"{'ID':<4} {'F1':>11} {'BWT':>11} {'FWT':>11} {'pfa':>6} {'n':>6} | "
                     f"{'dF1 [95% CI]':>22} {'pHolm':>7} {'pPerm':>7} {'verdict':>13}")
        bm = fs["base_map"]
        f1m, f1s = ms(allser(bm, "f1"))
        bwm, bws = ms(allser(bm, "bwt"))
        fwm, fws = ms(allser(bm, "fwt"))
        pfm, _ = ms(allser(bm, "pfa"))
        lines.append(f"{fs['base_id']:<4} {f1m:>6.1f}±{f1s:<4.1f} {bwm:>6.1f}±{bws:<4.1f} "
                     f"{fwm:>6.1f}±{fws:<4.1f} {pfm:>6.1f} {'-/' + str(fs['base_n']):>6} | "
                     f"{'(baseline)':>22}")
        for a in fs["abl"]:
            f1m, f1s = ms(allser(a["map"], "f1"))
            bwm, bws = ms(allser(a["map"], "bwt"))
            fwm, fws = ms(allser(a["map"], "fwt"))
            pfm, _ = ms(allser(a["map"], "pfa"))
            s, ci = a["sf"], a["sf"]["ci"]
            d = f"{s['mean']:+.1f} [{ci[0]:+.1f},{ci[1]:+.1f}]"
            npair = f"{a['n']}/{len(a['map'])}"
            lines.append(f"{a['rid']:<4} {f1m:>6.1f}±{f1s:<4.1f} {bwm:>6.1f}±{bws:<4.1f} "
                         f"{fwm:>6.1f}±{fws:<4.1f} {pfm:>6.1f} {npair:>6} | {d:>22} "
                         f"{a['holm']:>7.3f} {s['p_perm']:>7.3f} {a['verdict']:>13}")
            if verbose:
                lines.append(f"      seeds={sorted(a['map'])}  per-seed dF1={np.round(s['d'], 2).tolist()}")
        if verbose:
            lines.append(f"      BWT verdicts: " + ", ".join(
                f"{a['rid']}:{a['verdict_bwt']}({a['sb']['mean']:+.1f})" for a in fs["abl"]))
    return "\n".join(lines)


# ---------------------------------------------------------------- LaTeX


def pmv(mean, std):
    if not (mean == mean):
        return "---"
    return f"\\pmv{{{mean:.1f}}}{{{std:.1f}}}"


def pcell(p):
    if not (p == p):
        return "---"
    return "$<\\!0.001$" if p < 0.001 else f"${p:.3f}$"


def dcell(s):
    ci = s["ci"]
    if not (ci[0] == ci[0]):
        return "---"
    return f"${s['mean']:+.1f}$~[${ci[0]:+.1f},{ci[1]:+.1f}$]"


REGIME_SENTENCE = (
    r"All rows run the multi-epoch regime (10 epochs per task, one inner step per batch) "
    r"in fp32, and every setting that is not the ablated mechanism is matched across the "
    r"two settings: the same precision, the same split-\gls{bn} replay forward on the "
    r"replay rows (\texttt{--eralg4\_joint\_er}, \texttt{--cmaml\_joint\_er}), the same "
    r"split loss reduction, and the same row semantics (M0 is the second-order full "
    r"method and M1 removes the Hessian term in both). "
)

PROTOCOL_SENTENCE = (
    r"$\fcl$ and BWT are mean\,$\pm$\,sample std over the seeds pooled for that row; "
    r"$\overline{\Delta}$ is paired seed-for-seed against the family baseline over their "
    r"common seeds. Effects are assessed by the protocol of \cref{sec:stat_protocol}: a "
    r"95\% Student-$t$ \gls{ci}, a Holm--Bonferroni-corrected paired $t$-test within each "
    r"family, an exact sign-flip permutation test, and a smallest effect of interest of "
    r"$\delta{=}1$ $\fcl$ point, combined into one of four verdicts (\gls{ss}, \gls{ms}, "
    r"\gls{ns}, underpowered)."
)


def emit_table(mode, path, label, caption, wide):
    L = []
    env = "table*" if wide else "table"
    L.append(rf"\begin{{{env}}}[t]")
    L.append(r"    \centering")
    L.append(r"    \caption{" + caption + r"}")
    L.append(r"    \label{" + label + r"}")
    L.append(r"    \setlength{\tabcolsep}{5pt}")
    L.append(r"    \begin{threeparttable}")
    if wide:
        L.append(r"    \begin{adjustbox}{max width=0.9\textwidth, center}")
        L.append(r"    \begin{tabular}{@{}c p{6.5cm} cccccc l@{}}")
    else:
        L.append(r"    \resizebox{\columnwidth}{!}{%")
        L.append(r"    \begin{tabular}{@{}c cccccc l@{}}")
    ncol = 9 if wide else 8
    L.append(r"    \toprule")
    head = "    ID & "
    if wide:
        head += "Mechanism removed / altered & "
    head += (r"$\fcl$ & BWT & $n$ & $\overline{\Delta}_{\fcl}$~[95\% CI] & "
             r"$p_{\text{Holm}}$ & $p_{\text{perm}}$ & Verdict \\")
    L.append(head)
    L.append(r"    \midrule")
    fams = list(GRIDS[mode].items())
    for fi, (fam, rows) in enumerate(fams):
        L.append(rf"    \multicolumn{{{ncol}}}{{@{{}}l}}{{\textit{{{fam}}}}}\\")
        fs = family_stats(rows)
        f1m, f1s = ms(allser(fs["base_map"], "f1"))
        bwm, bws = ms(allser(fs["base_map"], "bwt"))
        cells = [fs["base_id"]]
        if wide:
            cells.append("None (full method)")
        cells += [pmv(f1m, f1s), pmv(bwm, bws), str(fs["base_n"]), "---", "---", "---", "---"]
        L.append("    " + " & ".join(cells) + r" \\")
        for a in fs["abl"]:
            f1m, f1s = ms(allser(a["map"], "f1"))
            bwm, bws = ms(allser(a["map"], "bwt"))
            cells = [a["rid"]]
            if wide:
                cells.append(a["label"])
            cells += [pmv(f1m, f1s), pmv(bwm, bws), str(a["n"]), dcell(a["sf"]),
                      pcell(a["holm"]), pcell(a["sf"]["p_perm"]), VERDICT_TEX[a["verdict"]]]
            L.append("    " + " & ".join(cells) + r" \\")
        if fi < len(fams) - 1:
            L.append(r"    \addlinespace")
    L.append(r"    \bottomrule")
    L.append(r"    \end{tabular}%")
    L.append(r"    \end{adjustbox}" if wide else r"    }")
    L.append(r"    \end{threeparttable}")
    L.append(rf"\end{{{env}}}")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path


def emit_fwt_table(mode, path, label, caption):
    L = []
    L.append(r"\begin{table*}[t]")
    L.append(r"    \centering")
    L.append(r"    \caption{" + caption + r"}")
    L.append(r"    \label{" + label + r"}")
    L.append(r"    \setlength{\tabcolsep}{5pt}")
    L.append(r"    \begin{threeparttable}")
    L.append(r"    \begin{adjustbox}{max width=0.9\textwidth, center}")
    L.append(r"    \begin{tabular}{@{}c p{6.5cm} cccccc l@{}}")
    L.append(r"    \toprule")
    L.append(r"    ID & Mechanism removed / altered & FWT & $\fcl$ & $n$ & "
             r"$\overline{\Delta}_{\text{FWT}}$~[95\% CI] & $p_{\text{Holm}}$ & "
             r"$p_{\text{perm}}$ & Verdict \\")
    L.append(r"    \midrule")
    fams = list(GRIDS[mode].items())
    for fi, (fam, rows) in enumerate(fams):
        L.append(r"    \multicolumn{9}{@{}l}{\textit{" + fam + r"}}\\")
        fs = family_stats(rows)
        fwm, fws = ms(allser(fs["base_map"], "fwt"))
        f1m, f1s = ms(allser(fs["base_map"], "f1"))
        L.append(f"    {fs['base_id']} & None (full method) & {pmv(fwm, fws)} & "
                 f"{pmv(f1m, f1s)} & {fs['base_n']} & --- & --- & --- & --- \\\\")
        for a in fs["abl"]:
            fwm, fws = ms(allser(a["map"], "fwt"))
            f1m, f1s = ms(allser(a["map"], "f1"))
            L.append(f"    {a['rid']} & {a['label']} & {pmv(fwm, fws)} & {pmv(f1m, f1s)} & "
                     f"{a['n']} & {dcell(a['sw'])} & {pcell(a['holm_fwt'])} & "
                     f"{pcell(a['sw']['p_perm'])} & {VERDICT_TEX[a['verdict_fwt']]} \\\\")
        if fi < len(fams) - 1:
            L.append(r"    \addlinespace")
    L.append(r"    \bottomrule")
    L.append(r"    \end{tabular}%")
    L.append(r"    \end{adjustbox}")
    L.append(r"    \end{threeparttable}")
    L.append(r"\end{table*}")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path


TIL_CAPTION = (
    r"Multi-epoch \gls{til} ablations for GEM, Res-ER, C-MAML, BCL-Dual and CTN. Each row "
    r"removes or perturbs a single mechanism from the full method (row~0) and reports "
    r"$\fcl$ and \gls{bwt}. This is the multi-epoch counterpart of the single-epoch grid in "
    r"Table~\ref{tab:ablation_optim}: same rows, same mechanisms, ten epochs per task "
    r"instead of one. " + REGIME_SENTENCE +
    r"The GEM family is single-learning-rate here (G2 at $0.01$ alongside G0/G1), so the "
    r"mixed-rate caveat of Table~\ref{tab:ablation_optim} does not apply. G1 and E1 are the "
    r"same Ring-ER configuration, reported once per family. ``$\to$'' denotes ``replaced "
    r"by''. " + PROTOCOL_SENTENCE
)

CIL_CAPTION = (
    r"Multi-epoch \gls{cil} ablations --- the class-incremental complement of "
    r"Table~\ref{tab:me_til_ablation}, on the three replay-based families. Row IDs denote "
    r"the same mechanisms as in the \gls{til} table. Reported are macro-$\fcl$ "
    r"($f1_{\text{total}}$) and backward transfer (BWT). " + REGIME_SENTENCE +
    r"The two grids are therefore directly comparable row by row, which the single-epoch "
    r"pair (Tables~\ref{tab:ablation_optim} and~\ref{tab:cil_ablation}) was not: those "
    r"differed in precision and in the sign convention of the M1 row. " + PROTOCOL_SENTENCE
)

CIL_FWT_CAPTION = (
    r"Forward-transfer (FWT) view of the multi-epoch \gls{cil} ablations "
    r"(Table~\ref{tab:me_cil_ablation}), significance-tested on FWT rather than $\fcl$. FWT "
    r"is the mean over tasks $1$--$9$ of each task's pre-train zero-shot macro "
    r"$f1_{\text{total}}$ (the trained\_zs term of $\text{FWT}=\text{trained\_zs}-"
    r"\text{untrained\_zs}$; the untrained baseline is a per-seed constant common to every "
    r"method, so it cancels in the paired within-seed $\overline{\Delta}$ and the verdicts "
    r"are baseline-invariant). Protocol as in \cref{sec:stat_protocol}."
)


def print_topup():
    """Report the rows whose verdict is still underpowered and emit the top-up command.

    A row qualifies when its interval spans zero and is wider than the SESOI -- the one
    case sec:stat_protocol says to extend to n=9. Rows already verdicted ss/ms/ns are
    final at n=6 and are deliberately NOT re-run. A row's baseline is topped up whenever
    any ablation in its family is, since the deltas are paired against it.
    """
    for mode in GRIDS:
        need, immature = {}, []
        for fam, rows in GRIDS[mode].items():
            fs = family_stats(rows)
            immature += [f"{a['rid']}(n={a['n']})" for a in fs["abl"] if a["n"] < 6]
            fam_rows = [a["rid"] for a in fs["abl"] if a["verdict"] == "underpowered"]
            if fam_rows:
                need[fam] = [rows[0][0]] + fam_rows
        if immature:
            print(f"[{mode}] NOT YET AT n=6, verdicts provisional: " + ", ".join(immature))
        if not need:
            print(f"[{mode}] no underpowered rows among those at n>=6")
            continue
        ids = sorted({r.lower() for v in need.values() for r in v})
        print(f"[{mode}] underpowered: " +
              "; ".join(f"{fam}: {', '.join(v[1:])}" for fam, v in need.items()))
        print(f"[{mode}] baselines pulled in for pairing: " +
              ", ".join(v[0] for v in need.values()))
        print(f'  SEED_GROUP_SPEC="1,2,3" MODES={mode} ROWS_ONLY={",".join(ids)} '
              f"MAX_PARALLEL=3 bash scripts/run_me_ablations.sh")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true", help="per-seed deltas and BWT verdicts")
    ap.add_argument("--topup", action="store_true",
                    help="list rows whose verdict is still underpowered and print the "
                         "run_me_ablations.sh command that takes them to n=9")
    args = ap.parse_args()

    if args.topup:
        print_topup()
        return

    report = text_report("til", args.verbose) + "\n" + text_report("cil", args.verbose)
    print(report)
    out_txt = os.path.join(REPO, "docs", "me_ablation_analysis.txt")
    with open(out_txt, "w") as f:
        f.write(report + "\n")

    written = [out_txt]
    written.append(emit_table("til", os.path.join(REPO, "docs", "me_til_ablations.tex"),
                              "tab:me_til_ablation", TIL_CAPTION, wide=True))
    written.append(emit_table("cil", os.path.join(REPO, "docs", "me_cil_ablations.tex"),
                              "tab:me_cil_ablation", CIL_CAPTION, wide=False))
    written.append(emit_fwt_table("cil", os.path.join(REPO, "docs", "me_cil_ablations_fwt.tex"),
                                  "tab:me_cil_ablation_fwt", CIL_FWT_CAPTION))
    print()
    for p in written:
        print("[written]", os.path.relpath(p, REPO))


if __name__ == "__main__":
    main()
