#!/usr/bin/env python3
"""Paired leave-one-out analysis for the CIL ablation grid (docs/se_cil_results.tex
complement). Implements the statistical protocol of docs/ablation_studies.tex
(sec:stat_protocol): per-seed paired deltas vs the family baseline, mean effect with
95% Student-t CI, two-sided paired t-test with Holm-Bonferroni correction within each
family, and an exact sign-flip permutation test. Verdict against a smallest-effect-of-
interest delta = 1 F1 point.

Metric: canonical macro f1_total = results.txt "Final F1" (== seed_metrics val_cls_f1).
Also reports BWT (results.txt "Backward") and pfa (seed_metrics val_fa). Values are
fractions on disk; reported in points (x100).
"""
import glob
import json
import os
import re
from itertools import product

import numpy as np
from scipy import stats

REPO = "/home/lunet/wsmr11/repos/evidential-cl"
ABL = os.path.join(REPO, "logs", "ablations", "cil")
SEEDS = [0, 1, 2, 3, 39, 55, 7, 13, 21]
DELTA = 1.0  # smallest effect of interest, F1 points

# family -> list of (row_id, label, curated dir under logs/ablations/cil). First = baseline.
#
# The curation is the pinning. C-MAML and BCL-Dual were re-run 2026-07-24/25 against two
# code changes that move their baselines -- the C-MAML rows all carry --cmaml_joint_er
# (baseline expt "probe-fix"), the BCL rows all carry the CIL train-mask fix (baseline expt
# "fixed-cil-mask") -- and only those reruns were promoted into the tree. Their same-named
# 07-16/07-17/07-23 predecessors stay in logs/<stem>/: pooling them would silently pair a
# post-fix ablation against a pre-fix baseline, which inverted every C-MAML meta verdict to
# "load-bearing". E0 likewise carries --eralg4_joint_er (the pre-flag eralg4_resER_se_cil
# runs, F1 10.0, are superseded); E1/E2 (er_ring) are unaffected by all three. E1 and E2 mean
# the same buffers as in tab:ablation_optim -- static per-task ring and fully-utilised
# dynamic ring -- both at E0's lr 0.01 / inner_steps 2. Every row carries the full n=9 seeds.
FAMILIES = {
    "Res-ER": [
        ("E0", "None (full method)", "res-er/E0"),
        ("E1", "Reservoir sampling $\\to$ static ring buffer (Ring-ER)", "res-er/E1"),
        ("E2", "Reservoir sampling $\\to$ fully-utilised Ring-ER", "res-er/E2"),
    ],
    "C-MAML": [
        ("M0", "None (full method)", "cmaml/M0"),
        ("M1", "Second-order meta-gradient (Hessian term)", "cmaml/M1"),
        ("M2", "Episodic replay buffer ($B^{\\text{rep}}$)", "cmaml/M2"),
        ("M3", "Meta-batch averaging ($K_{\\text{meta}}\\!=\\!1$)", "cmaml/M3"),
        ("M4", "Inner-loop adaptation ($\\alpha_{\\text{init}}\\!=\\!0$)", "cmaml/M4"),
    ],
    "BCL-Dual": [
        ("B0", "None (full method)", "bcl-dual/B0"),
        ("B1", "\\gls{kl} distillation term", "bcl-dual/B1"),
        ("B3", "Second (meta) memory buffer", "bcl-dual/B3"),
        ("B4", "Episodic replay (both channels)", "bcl-dual/B4"),
        ("B5a", "Bilevel $\\to$ single loop (half budget)", "bcl-dual/B5a"),
        ("B5b", "Bilevel $\\to$ single loop (matched budget)", "bcl-dual/B5b"),
    ],
}

IDX = dict(f1=0, bwt=1, diag=2, pfa=3, fwt=4)


def _forward_zs(sd):
    """Forward-transfer zero-shot: mean over tasks 1..9 of each task's pre-train zero-shot
    total_f1 (metrics/task{t}.npz 'zero_shot_total_f1'). This is the trained_zs term of
    FWT = trained_zs - untrained_zs; the untrained baseline is a per-seed constant common to
    all methods, so it cancels in the paired within-seed delta (verdicts are baseline-invariant).
    Returns a fraction, or NaN if metrics are absent."""
    md = os.path.join(sd, "metrics")
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


def parse_seed(run_dir, seed):
    """Return (f1_total, bwt, diag, pfa, fwt) in fractions for one seed."""
    sd = os.path.join(run_dir, str(seed))
    txt = open(os.path.join(sd, "results.txt")).read()
    f1 = float(re.search(r"Final F1:\s*([-\d.]+)", txt).group(1))
    bwt = float(re.search(r"Backward:\s*([-\d.]+)", txt).group(1))
    diag = float(re.search(r"Diagonal F1:\s*([-\d.]+)", txt).group(1))
    pfa = np.nan
    smj = os.path.join(sd, "seed_metrics.json")
    if os.path.exists(smj):
        m = json.load(open(smj))
        pfa = float(m.get("val_fa", np.nan))
    fwt = _forward_zs(sd)
    return f1, bwt, diag, pfa, fwt


def load_expt(relpath):
    """Pool seed subdirs across every run dir curated into logs/ablations/cil/<relpath>
    (base run plus any topup). Returns {seed: (f1, bwt, diag, pfa, fwt)} in fractions;
    later run dirs win on a duplicate seed."""
    dirs = sorted(d for d in glob.glob(os.path.join(ABL, relpath, "*")) if os.path.isdir(d))
    if not dirs:
        raise FileNotFoundError(f"no run dir under {os.path.join(ABL, relpath)}")
    seedmap = {}
    for d in dirs:
        for sd in sorted(os.listdir(d)):
            if sd.isdigit() and os.path.exists(os.path.join(d, sd, "results.txt")):
                seedmap[int(sd)] = parse_seed(d, int(sd))
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
    d = abl - base  # per-seed deltas, points
    n = len(d)
    mean = d.mean()
    sd = d.std(ddof=1)
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    ci = (mean - tcrit * se, mean + tcrit * se)
    if se == 0:
        t_p = 0.0 if mean != 0 else 1.0
    else:
        _, t_p = stats.ttest_rel(abl, base)
    p_perm = perm_pvalue(d)
    return dict(mean=mean, ci=ci, t_p=t_p, p_perm=p_perm, d=d)


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values, preserving input order."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def verdict(p_holm, p_perm, ci):
    excl0 = ci[0] > 0 or ci[1] < 0
    if p_holm < 0.05 and p_perm < 0.05:
        return "load-bearing"
    if excl0:
        return "borderline"
    if ci[0] > -DELTA and ci[1] < DELTA:
        return "inert"
    return "inconclusive"


def fmt(x):
    return f"{x:+.1f}"


def family_stats(data, fam, rows):
    """Compute baseline summaries and per-ablation paired stats for one family, pairing
    each ablation against the baseline on their COMMON seeds (so n grows as topups land)."""
    base_map = data[(fam, rows[0][0])]
    fs = dict(base_id=rows[0][0],
              base_f1=allser(base_map, "f1"), base_bwt=allser(base_map, "bwt"),
              base_pfa=allser(base_map, "pfa"), base_fwt=allser(base_map, "fwt"),
              base_n=len(base_map), abl=[])
    for rid, label, _relpath in rows[1:]:
        amap = data[(fam, rid)]
        common = sorted(set(base_map) & set(amap))
        sf = paired_stats(series(base_map, common, "f1"), series(amap, common, "f1"))
        sb = paired_stats(series(base_map, common, "bwt"), series(amap, common, "bwt"))
        sw = paired_stats(series(base_map, common, "fwt"), series(amap, common, "fwt"))
        fs["abl"].append(dict(rid=rid, label=label, n=len(common), sf=sf, sb=sb, sw=sw,
                              f1=allser(amap, "f1"), bwt=allser(amap, "bwt"),
                              pfa=allser(amap, "pfa"), fwt=allser(amap, "fwt")))
    # Holm within family, computed separately per metric
    holm_f1 = holm(np.array([a["sf"]["t_p"] for a in fs["abl"]])) if fs["abl"] else []
    holm_fwt = holm(np.array([a["sw"]["t_p"] for a in fs["abl"]])) if fs["abl"] else []
    for i, a in enumerate(fs["abl"]):
        a["holm"] = holm_f1[i]
        a["verdict"] = verdict(a["holm"], a["sf"]["p_perm"], a["sf"]["ci"])
        a["holm_fwt"] = holm_fwt[i]
        a["verdict_fwt"] = verdict(a["holm_fwt"], a["sw"]["p_perm"], a["sw"]["ci"])
    return fs


def main():
    data = {}
    for fam, rows in FAMILIES.items():
        for rid, label, relpath in rows:
            data[(fam, rid)] = load_expt(relpath)

    lines = []
    for fam, rows in FAMILIES.items():
        fs = family_stats(data, fam, rows)
        lines.append(f"\n===== {fam} =====  (F1 = macro f1_total; FWT = forward zero-shot, tasks 1-9)")
        lines.append(f"{'ID':<4} {'F1':>10} {'FWT':>10} {'BWT':>10} {'pfa':>6} {'n':>3} | "
                     f"{'dF1[CI]':>20} {'F1-verdict':>13} | {'dFWT[CI]':>20} {'FWT-verdict':>13}")
        lines.append(f"{fs['base_id']:<4} {fs['base_f1'].mean():>6.1f}±{fs['base_f1'].std(ddof=1):<3.1f} "
                     f"{fs['base_fwt'].mean():>6.1f}±{fs['base_fwt'].std(ddof=1):<3.1f} "
                     f"{fs['base_bwt'].mean():>6.1f}±{fs['base_bwt'].std(ddof=1):<3.1f} "
                     f"{fs['base_pfa'].mean():>6.1f} {fs['base_n']:>3} | {'(baseline)':>20} {'':>13} | "
                     f"{'':>20} {'':>13}")
        for a in fs["abl"]:
            ci, cw = a["sf"]["ci"], a["sw"]["ci"]
            d1 = fmt(a['sf']['mean']) + '[' + fmt(ci[0]) + ',' + fmt(ci[1]) + ']'
            dw = fmt(a['sw']['mean']) + '[' + fmt(cw[0]) + ',' + fmt(cw[1]) + ']'
            lines.append(
                f"{a['rid']:<4} {a['f1'].mean():>6.1f}±{a['f1'].std(ddof=1):<3.1f} "
                f"{a['fwt'].mean():>6.1f}±{a['fwt'].std(ddof=1):<3.1f} "
                f"{a['bwt'].mean():>6.1f}±{a['bwt'].std(ddof=1):<3.1f} {a['pfa'].mean():>6.1f} "
                f"{a['n']:>3} | {d1:>20} {a['verdict']:>13} | {dw:>20} {a['verdict_fwt']:>13}  ({a['label']})")

    out = "\n".join(lines)
    print(out)
    with open(os.path.join(REPO, "docs", "cil_ablation_analysis.txt"), "w") as f:
        f.write(out + "\n")
    emit_latex(data)
    emit_fwt_latex(data)
    print("\n[written] docs/cil_ablation_analysis.txt")
    print("[written] docs/se_cil_ablations.tex")
    print("[written] docs/se_cil_ablations_fwt.tex")


def pmv(mean, std):
    return f"\\pmv{{{mean:.1f}}}{{{std:.1f}}}"


# per-row LaTeX tweaks: (row_id) -> tnote marker on the ID cell
TNOTE = {"E0": "a", "M0": "b", "B0": "c", "B5a": "d", "B5b": "d"}


def emit_latex(data):
    L = []
    L.append(r"\begin{table}[t]")
    L.append(r"    \centering")
    L.append(r"    \caption{Ablations in the single-epoch \gls{cil} regime --- the "
             r"class-incremental complement of Table~\ref{tab:ablation_optim}, on the three "
             r"replay-based families of Table~\ref{tab:results_single-epoch_cil}. "
             r"Row IDs denote the same mechanisms as in Table~\ref{tab:ablation_optim}: each removes or "
             r"perturbs a single mechanism from the full method (row~0). Reported are "
             r"macro-$\fcl$ ($f1_{\text{total}}$) and backward transfer (BWT). The picture is starkly "
             r"simpler than in \gls{til}: \emph{retained data is the only load-bearing mechanism}. "
             r"Deleting episodic replay drops both meta-learners onto the same \gls{cil} floor "
             r"($\fcl\,{=}\,3.0\,{\pm}\,0.1$ for M2 and B4 alike), and replacing reservoir with a "
             r"per-task ring costs $10.0$ points (E1) or $9.7$ with the fully-utilised variant "
             r"(E2) --- a sign flip from \gls{til}, where the same two rows cost $1.5$ and $0.3$ "
             r"(note~a). Everything "
             r"else --- the second-order Hessian term, meta-batch averaging, inner-loop adaptation, "
             r"\gls{kl} distillation, the second memory, and the bilevel loop at matched budget --- is "
             r"null on $\fcl$, several of them nominally \emph{positive}. $\fcl$ and BWT are "
             r"mean\,$\pm$\,sample std over $n{=}9$ seeds; effects are assessed by the paired "
             r"significance protocol of \S\ref{sec:stat_protocol} (paired $t$, exact permutation, and "
             r"Holm-adjusted $p$-values with confidence intervals), combined into the per-row verdict "
             r"of the final column. Rows verdicted \emph{inconclusive} are null point estimates whose "
             r"interval is marginally wider than the $\delta\,{=}\,1$ point smallest effect of interest, "
             r"not evidence of an effect.}")
    L.append(r"    \label{tab:cil_ablation}")
    L.append(r"    \setlength{\tabcolsep}{5pt}")
    L.append(r"    \begin{threeparttable}")
    L.append(r"    \resizebox{\columnwidth}{!}{%")
    L.append(r"    \begin{tabular}{@{}c ccccc l@{}}")
    L.append(r"    \toprule")
    L.append(r"    ID & $\fcl$ & BWT & "
             r"$\overline{\Delta}_{\fcl}$~[95\% CI] & $p_{\text{Holm}}$ & $p_{\text{perm}}$ & Verdict \\")
    L.append(r"    \midrule")
    fam_list = list(FAMILIES.items())
    for fi, (fam, rows) in enumerate(fam_list):
        L.append(r"    \multicolumn{7}{@{}l}{\textit{" + fam + r"}}\\")
        fs = family_stats(data, fam, rows)
        b_tn = f"\\tnote{{{TNOTE[fs['base_id']]}}}" if fs["base_id"] in TNOTE else ""
        L.append(f"    {fs['base_id']}{b_tn} & "
                 f"{pmv(fs['base_f1'].mean(), fs['base_f1'].std(ddof=1))} & "
                 f"{pmv(fs['base_bwt'].mean(), fs['base_bwt'].std(ddof=1))} & --- & --- & --- & --- \\\\")
        for a in fs["abl"]:
            s = a["sf"]
            ph = a["holm"]
            ci = s["ci"]
            phs = "<\\!0.001" if ph < 0.001 else f"{ph:.3f}"
            dcell = f"${s['mean']:+.1f}$~[${ci[0]:+.1f},{ci[1]:+.1f}$]"
            tn = f"\\tnote{{{TNOTE[a['rid']]}}}" if a["rid"] in TNOTE else ""
            L.append(f"    {a['rid']}{tn} & {pmv(a['f1'].mean(), a['f1'].std(ddof=1))} & "
                     f"{pmv(a['bwt'].mean(), a['bwt'].std(ddof=1))} & "
                     f"{dcell} & ${phs}$ & ${s['p_perm']:.3f}$ & {a['verdict']} \\\\")
        if fi < len(fam_list) - 1:
            L.append(r"    \addlinespace")
    L.append(r"    \bottomrule")
    L.append(r"    \end{tabular}%")
    L.append(r"    }")
    L.append(r"    \begin{tablenotes}[flushleft]\footnotesize")
    L.append(r"    \begin{minipage}{\columnwidth}")
    L.append(r"    \item[a] Res-ER (E) runs at the lr$=0.01$ config-matched point "
             r"(\texttt{cil/eralg4} default) with the two-forward replay loop "
             r"(\texttt{--eralg4\_joint\_er}), which normalises the replay and current-task rows with "
             r"their own \gls{bn} statistics rather than the pooled mixture; this lifts the E0 baseline "
             r"from $\fcl\,{=}\,10.0$ to $13.5$. E1 (\texttt{er\_ring}) already scores the two blocks in "
             r"separate forwards by construction, so this is the first E0/E1 pair matched on that axis. "
             r"The baseline still sits below the tuned Res-ER of "
             r"Table~\ref{tab:results_single-epoch_cil} ($\fcl\,{=}\,15.8$), to which \gls{cil} is more "
             r"learning-rate-sensitive than \gls{til}; the E1/E2 verdicts are paired within-lr "
             r"comparisons and are unaffected. Both rings also lower $p_{fa}$ "
             r"($75.8\!\to\!23.0$ for E1, $27.7$ for E2) while collapsing $\fcl$: a per-task ring "
             r"under-covers the label space that reservoir keeps live, and keeping it full (E2) "
             r"recovers only $0.3$ points of the $10.0$. This is a sign flip from \gls{til}, where "
             r"the two rings sit within $1.5$ of reservoir and the dynamic one is the stronger "
             r"buffer.")
    L.append(r"    \item[b] The C-MAML rows all use the split-BatchNorm replay forward "
             r"(\texttt{--cmaml\_joint\_er}), which normalises the replay and current-task rows with "
             r"their own statistics instead of the pooled mixture. This is worth $+2.4$ $\fcl$ over the "
             r"pooled-forward baseline ($10.2\,{\to}\,12.6$, paired over the same 9 seeds, "
             r"$p_{\text{perm}}\,{=}\,0.004$) and reproduces in \gls{cil} the \gls{bn}-mixing retention "
             r"defect first isolated on Res-ER. The M1--M4 verdicts are paired within this setting.")
    L.append(r"    \item[c] The BCL-Dual rows all carry a correction to the training-time logit mask. "
             r"Previously the training forward applied the \gls{til} mask (current task's classes only), "
             r"making the training objective a within-task discrimination problem even under \gls{cil}; "
             r"it now masks cumulatively over all classes seen so far, matching the evaluation protocol. "
             r"Removing that task oracle moves the B0 baseline from $11.8\,{\pm}\,0.9$ to "
             r"$4.8\,{\pm}\,0.3$ ($\overline{\Delta}\,{=}\,-7.0$, $p_{\text{perm}}\,{=}\,0.004$), so "
             r"\gls{cil} BCL-Dual figures predating this correction are superseded. The honest baseline "
             r"sits only $1.8$ points above the no-replay floor, which compresses every within-family "
             r"effect below.")
    L.append(r"    \item[d] The bilevel ablation is run at half budget (B5a, matching B0's inner loop) "
             r"and at matched budget (B5b, double inner steps), as in Table~\ref{tab:ablation_optim}: the "
             r"$-0.8$ at half budget is optimisation budget, leaving a null matched-budget structural "
             r"remainder ($-0.2$, inert). B5a is also the one ablation that \emph{improves} BWT "
             r"($+1.0$): the single loop forgets less because it learns the current task less.")
    L.append(r"    \end{minipage}")
    L.append(r"    \end{tablenotes}")
    L.append(r"    \end{threeparttable}")
    L.append(r"\end{table}")
    with open(os.path.join(REPO, "docs", "se_cil_ablations.tex"), "w") as f:
        f.write("\n".join(L) + "\n")


def emit_fwt_latex(data):
    """Companion table: the same leave-one-out grid, significance-tested on forward transfer
    (FWT). Mirrors ablation_studies.tex; FWT is primary, $\\fcl$ is the context column."""
    L = []
    L.append(r"\begin{table*}[t]")
    L.append(r"    \centering")
    L.append(r"    \caption{Forward-transfer (FWT) view of the single-epoch \gls{cil} ablations "
             r"(Table~\ref{tab:cil_ablation}), significance-tested on FWT rather than $\fcl$. FWT is the "
             r"forward-transfer zero-shot score: the mean over tasks $1$--$9$ of each task's pre-train "
             r"zero-shot macro $f1_{\text{total}}$ (the trained\_zs term of $\text{FWT}=\text{trained\_zs}"
             r"-\text{untrained\_zs}$; the untrained baseline is a per-seed constant common to every "
             r"method, so it cancels in the paired within-seed $\overline{\Delta}$ and the verdicts are "
             r"baseline-invariant). FWT tracks $\fcl$ closely and amplifies it: replay dominates here too, "
             r"and more strongly ($-14.7$ for M2 against $-9.6$ on $\fcl$). FWT is also the sharper "
             r"readout: a zero-shot score averaged over nine future tasks has tighter intervals than the "
             r"final-model $\fcl$, so the three C-MAML meta rows that $\fcl$ can only call inconclusive "
             r"(M1, M3, M4) resolve here as unambiguously inert, and B1 firms from borderline to "
             r"load-bearing --- with the opposite sign to the one previously reported (note~d). "
             r"$n{=}9$ seeds; protocol as in \S\ref{sec:stat_protocol}.}")
    L.append(r"    \label{tab:cil_ablation_fwt}")
    L.append(r"    \setlength{\tabcolsep}{5pt}")
    L.append(r"    \begin{threeparttable}")
    L.append(r"    \resizebox{2\columnwidth}{!}{%")
    L.append(r"    \begin{tabular}{@{}c p{6.5cm} ccccc l@{}}")
    L.append(r"    \toprule")
    L.append(r"    ID & Mechanism removed / altered & FWT & $\fcl$ & "
             r"$\overline{\Delta}_{\text{FWT}}$~[95\% CI] & $p_{\text{Holm}}$ & $p_{\text{perm}}$ & Verdict \\")
    L.append(r"    \midrule")
    fam_list = list(FAMILIES.items())
    for fi, (fam, rows) in enumerate(fam_list):
        L.append(r"    \multicolumn{8}{@{}l}{\textit{" + fam + r"}}\\")
        fs = family_stats(data, fam, rows)
        # only note 'd' (distillation, on B1) is defined in this table; the F1-table markers
        # (a/b/c) are not repeated here, so the baseline carries no tnote.
        L.append(f"    {fs['base_id']} & None (full method) & "
                 f"{pmv(fs['base_fwt'].mean(), fs['base_fwt'].std(ddof=1))} & "
                 f"{pmv(fs['base_f1'].mean(), fs['base_f1'].std(ddof=1))} & --- & --- & --- & --- \\\\")
        for a in fs["abl"]:
            s = a["sw"]
            ph = a["holm_fwt"]
            ci = s["ci"]
            phs = "<\\!0.001" if ph < 0.001 else f"{ph:.3f}"
            dcell = f"${s['mean']:+.1f}$~[${ci[0]:+.1f},{ci[1]:+.1f}$]"
            tn = "\\tnote{d}" if a["rid"] == "B1" else ""
            L.append(f"    {a['rid']} & {a['label']}{tn} & {pmv(a['fwt'].mean(), a['fwt'].std(ddof=1))} & "
                     f"{pmv(a['f1'].mean(), a['f1'].std(ddof=1))} & "
                     f"{dcell} & ${phs}$ & ${s['p_perm']:.3f}$ & {a['verdict_fwt']} \\\\")
        if fi < len(fam_list) - 1:
            L.append(r"    \addlinespace")
    L.append(r"    \bottomrule")
    L.append(r"    \end{tabular}%")
    L.append(r"    }")
    L.append(r"    \begin{tablenotes}[flushleft]\footnotesize")
    L.append(r"    \begin{minipage}{\textwidth}")
    L.append(r"    \item[d] The \gls{kl} distillation term is not merely inert under \gls{cil} but mildly "
             r"\emph{harmful} to forward transfer: removing it \emph{gains} $+0.8$ FWT (load-bearing, "
             r"$p_{\text{perm}}\,{=}\,0.004$) alongside $+0.4$ $\fcl$ and a higher $p_{fa}$. Distilling "
             r"from a teacher whose head only ever saw a subset of the label space transfers that "
             r"restriction to the student, so dropping the constraint frees the head to spread mass over "
             r"unseen classes. Note this reverses the earlier \gls{cil} reading of B1, which was measured "
             r"against the pre-correction baseline of Table~\ref{tab:cil_ablation}, note~c.")
    L.append(r"    \end{minipage}")
    L.append(r"    \end{tablenotes}")
    L.append(r"    \end{threeparttable}")
    L.append(r"\end{table*}")
    with open(os.path.join(REPO, "docs", "se_cil_ablations_fwt.tex"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
