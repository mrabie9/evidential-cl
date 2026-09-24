#!/usr/bin/env python3
"""LaTeX table of one mode's ablation grid, single-pass (1e) and offline (5e) side by side.

Columns: ID | mechanism altered | 1e: Delta F1 [95% CI] | p_Holm | verdict
                                | 5e: Delta F1 [95% CI] | p_Holm | verdict

Statistics come from scripts/analyse_ablations_noise_removed.py (headline macro F1, paired
against the family baseline over shared seeds, Holm within family, SS/MS/NS/INC at
delta=1). Rows measured at n=12 (topped up) carry a dagger; untested rows print "--".

Usage:
  la-maml_env/bin/python scripts/tex_ablation_table.py --mode cil \
      --tag-1e nrm1e_otm --tag-5e nrm_otm --out table.tex
"""
import argparse
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "analyser", os.path.join(HERE, "analyse_ablations_noise_removed.py"))
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)


def tex_label(label):
    """Analyser mechanism label -> LaTeX."""
    added = label.startswith("[ADDED] ")
    if added:
        label = label[len("[ADDED] "):]
    label = (label.replace("->", r"$\rightarrow$").replace("&", r"\&")
             .replace("%", r"\%").replace("_", r"\_")
             .replace("alpha", r"$\alpha$").replace("beta", r"$\beta$"))
    return (r"\textit{(added)} " + label) if added else label


def fmt_p(p):
    return "$<$0.001" if p < 0.001 else "{:.3f}".format(p)


def regime_cells(stats, rid):
    """Three cells for one regime: delta [CI], p_Holm, verdict."""
    if rid not in stats:
        return ["--", "--", "--"]
    n, mean, lo, hi, p_holm, _p_perm, verd = stats[rid]
    dag = r"$^\dagger$" if n > 6 else ""
    delta = r"${:+.2f}$ [${:+.2f}$, ${:+.2f}$]{}".format(mean, lo, hi, dag)
    if verd == "SS":
        verd = r"\textbf{SS}"
    return [delta, fmt_p(p_holm), verd]


def build(mode, tag1, tag5):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{{{} ablations. $\Delta$F1 is the paired change in headline macro F1 "
        r"(points) against the family baseline; 95\% $t$ CI; $p_{{\mathrm{{Holm}}}}$ "
        r"corrected within family. Verdicts at $\delta=1$: SS significant, MS marginal, "
        r"NS negligible, INC inconclusive. $n=6$ seeds; $^\dagger$topped up to $n=12$.}}"
        .format(mode.upper()),
        r"\label{{tab:{}_ablations_1e_5e}}".format(mode),
        r"\resizebox{\textwidth}{!}{%  needs \usepackage{graphicx}",
        r"\begin{tabular}{llccccccc}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{Single-pass (1 epoch)} & & "
        r"\multicolumn{3}{c}{Offline (5 epochs)} \\",
        r"\cmidrule(lr){3-5} \cmidrule(lr){7-9}",
        r"ID & Mechanism altered & $\Delta$F1 [95\% CI] & $p_{\mathrm{Holm}}$ & Verdict "
        r"& & $\Delta$F1 [95\% CI] & $p_{\mathrm{Holm}}$ & Verdict \\",
        r"\midrule",
    ]
    families = list(A.GRIDS[mode].items())
    for fi, (family, rows) in enumerate(families):
        per_tag = []
        for tag in (tag1, tag5):
            pools = {rid: A.pool(mode, stem, pid, tag) for rid, _l, stem, pid in rows}
            per_tag.append(A.family_stats(rows, pools))
        lines.append(r"\multicolumn{{9}}{{l}}{{\textit{{{}}}}} \\".format(family))
        for rid, label, _stem, _pid in rows:
            if rid == rows[0][0]:
                cells = ["baseline", "", ""] + [""] + ["baseline", "", ""]
            else:
                cells = regime_cells(per_tag[0], rid) + [""] + regime_cells(per_tag[1], rid)
            lines.append("{} & {} & {} \\\\".format(rid, tex_label(label), " & ".join(cells)))
        if fi < len(families) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="cil", choices=sorted(A.GRIDS))
    ap.add_argument("--tag-1e", required=True)
    ap.add_argument("--tag-5e", required=True)
    ap.add_argument("--out", default=None, help="write here as well as stdout")
    args = ap.parse_args()
    tex = build(args.mode, args.tag_1e, args.tag_5e)
    print(tex)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(tex)


if __name__ == "__main__":
    main()
