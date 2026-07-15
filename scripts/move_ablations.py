#!/usr/bin/env python
"""Reorganise the ablation run dirs into logs/ablations/<algo>/<ID>/[lrX/]<run>.

Each ablation row is pooled from >=1 timestamped run dir (seeds 0/39/55 from one
job, 7/13/21 from another). G/T rows exist at two lrs -> lr subfolders.

Runs named '*_bm_*' belong to the budget-match study (shared as ablation baselines
by config-match); those are COPIED (originals left in place). Everything else is
MOVED. Dry-run by default; pass --go to execute.
"""
import argparse, glob, os, shutil, sys

LOG = "/home/lunet/wsmr11/repos/evidential-cl/logs"
DEST = f"{LOG}/ablations"

# algo_dir, ID, lr(None|"0.01"|"0.03") -> list of (stem, run-dir prefix)
MANIFEST = [
 ("gem","G0","0.03",[("gem","gem_full_s7-13-21"),("gem","gem_replay_se_til"),("gem","cudnn_se_til-2026-06-30")]),
 ("gem","G0","0.01",[("gem","gem_full_lr01abl")]),
 ("gem","G1","0.03",[("er_ring","gem-ablation_lr30_se_til")]),
 ("gem","G1","0.01",[("er_ring","ering_rehearsal_lr01abl")]),
 ("gem","G2","0.03",[("gem","gem_noreplay_s7-13-21")]),
 ("gem","G2","0.01",[("gem","gem_noreplay_lr01abl")]),
 ("res-er","E0",None,[("eralg4","eralg4_resER_s0-39-55"),("eralg4","eralg4_resER_s7-13-21")]),
 ("res-er","E1",None,[("er_ring","er_ring_dynring_bm_lr01_is2_se_til"),("er_ring","er_ring_dynring_s7-13-21")]),
 ("cmaml","M0",None,[("cmaml","cmaml_cwfix_so_se_til"),("cmaml","cmaml_secondorder_s7-13-21")]),
 ("cmaml","M1",None,[("cmaml","cmaml_cwfix_se_til-2026-07-09_18-15-39"),("cmaml","cmaml_full_s7-13-21")]),
 ("cmaml","M2",None,[("cmaml_no_replay","cmaml_cwfix_noreplay_se_til"),("cmaml_no_replay","cmaml_noreplay_s7-13-21")]),
 ("cmaml","M3",None,[("cmaml_single_inner","cmaml_cwfix_singleinner_se_til"),("cmaml_single_inner","cmaml_singleinner_s7-13-21")]),
 ("cmaml","M4",None,[("cmaml_alpha0","cmaml_cwfix_alpha0_se_til"),("cmaml_alpha0","cmaml_alpha0_s7-13-21")]),
 ("bcl-dual","B0",None,[("bcl_dual","bcl_b1_se_til"),("bcl_dual","bcl_dual_s7-13-21")]),
 ("bcl-dual","B1",None,[("bcl_nodistill","bcl_b1_nodistill_se_til"),("bcl_nodistill","bcl_nodistill_s7-13-21")]),
 ("bcl-dual","B2",None,[("bcl_nobilevel","bcl_nobilevel_s7-13-21")]),
 ("bcl-dual","B3",None,[("bcl_nodualmem","bcl_b1_nodualmem_se_til"),("bcl_nodualmem","bcl_nodualmem_s7-13-21")]),
 ("bcl-dual","B4",None,[("bcl_noreplay","bcl_b1_noreplay_se_til"),("bcl_noreplay","bcl_noreplay_s7-13-21")]),
 ("bcl-dual","B5a",None,[("bcl_singlelevel","bcl_b1_singlelevel_is2_se_til"),("bcl_singlelevel","bcl_singlelevel_is2_s7-13-21")]),
 ("bcl-dual","B5b",None,[("bcl_singlelevel","bcl_b1_singlelevel_is4_se_til"),("bcl_singlelevel","bcl_singlelevel_is4_s7-13-21")]),
 ("ctn","T0","0.03",[("ctn","ctn_bm_se_noamp_til"),("ctn","ctn_full_s7-13-21")]),
 ("ctn","T0","0.01",[("ctn","ctn_full_lr01abl")]),
 ("ctn","T1","0.03",[("ctn_nofilm","film-ablation_se_noamp_til"),("ctn_nofilm","ctn_nofilm_s7-13-21")]),
 ("ctn","T1","0.01",[("ctn_nofilm","ctn_nofilm_lr01abl")]),
 ("ctn","T2","0.03",[("ctn_nodistill","distill-ablation_se_noamp_til"),("ctn_nodistill","ctn_nodistill_s7-13-21")]),
 ("ctn","T2","0.01",[("ctn_nodistill","ctn_nodistill_lr01abl")]),
 ("ctn","T3","0.03",[("ctn_noreplay","noreplay-ablation_se_noamp_til_s"),("ctn_noreplay","ctn_noreplay_s7-13-21")]),
 ("ctn","T3","0.01",[("ctn_noreplay","ctn_noreplay_lr01abl")]),
]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--go", action="store_true"); a = ap.parse_args()
    plan, missing, seeds_tot = [], [], 0
    for algo, rid, lr, srcs in MANIFEST:
        ddir = f"{DEST}/{algo}/{rid}" + (f"/lr{lr}" if lr else "")
        for stem, pref in srcs:
            matches = sorted(glob.glob(f"{LOG}/{stem}/{pref}*"))
            matches = [m for m in matches if os.path.isdir(m)]
            if not matches:
                missing.append(f"{rid}: {stem}/{pref}*"); continue
            for src in matches:
                base = os.path.basename(src)
                op = "COPY" if "_bm_" in base else "MOVE"
                nseed = len([d for d in glob.glob(src+"/*") if os.path.basename(d).isdigit()])
                seeds_tot += nseed
                plan.append((op, src, f"{ddir}/{base}", nseed))
    for op, src, dst, ns in plan:
        print(f"{op} [{ns}s] {src.replace(LOG+'/','')}  ->  {dst.replace(LOG+'/','')}")
    print(f"\n{len(plan)} run-dirs, {seeds_tot} seed-dirs total. MISSING={missing}")
    if not a.go:
        print("\n(dry-run; pass --go to execute)"); return
    if missing:
        print("ABORT: unresolved sources"); sys.exit(1)
    for op, src, dst, ns in plan:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if op == "COPY": shutil.copytree(src, dst)
        else: shutil.move(src, dst)
    print(f"\nDONE: {sum(1 for p in plan if p[0]=='MOVE')} moved, {sum(1 for p in plan if p[0]=='COPY')} copied.")

if __name__ == "__main__":
    main()
