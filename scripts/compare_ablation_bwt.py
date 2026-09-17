#!/usr/bin/env python
"""BWT-vs-F1 paired significance across the ablation grid, to test whether the
BWT tests are informative (i.e. flag mechanisms the F1 tests miss). Reads the same
curated rows as compare_ablation.py (logs/ablations/til/), so every family is at the
configuration its table row reports; the superseded lr replicas that once let this
script run G/T at both learning rates now live back in logs/<config-stem>/."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from compare_ablation import arm_pool, paired, holm, mean_std, FAMILIES

DELTA = 1.0  # smallest effect of interest, in points (same for F1 and BWT)

def verdict(ph, pp, lo, hi, delta=DELTA):
    if ph == ph and pp == pp and ph < 0.05 and pp < 0.05:
        return "load-bearing"
    if lo > 0 or hi < 0:
        return "borderline"
    if lo >= -delta and hi <= delta:
        return "inert"
    return "inconclusive"

# family -> (baseline (id,relpath), [(id,relpath) ablations]); relpath under logs/ablations/til/.
GRID = {name: (spec["baseline"], spec["ablations"]) for name, spec in FAMILIES.items()}

hdr = f"{'row':4s} {'BWTbase→abl':16s} {'ΔBWT':7s} {'95% CI':16s} {'pHolm':6s} {'pperm':6s} {'BWT verdict':13s} | {'F1 verdict':12s}"
for fam,(base,abls) in GRID.items():
    bid,bpath = base
    bpool,_,_ = arm_pool(bpath)
    bwm,bws,nb = mean_std([v[1] for v in bpool.values()])
    print(f"\n==== {fam} ====   baseline {bid}: BWT {bwm*100:+.1f}±{bws*100:.1f} (n={nb})")
    print(hdr); print("-"*104)
    stF=[]; stB=[]
    for aid,apath in abls:
        ap,_,_ = arm_pool(apath)
        stF.append((aid, paired(bpool,ap,key=0), ap))
        stB.append((aid, paired(bpool,ap,key=1), ap))
    phF = holm([s[1]["p_t"] if s[1]["p_t"]==s[1]["p_t"] else 1.0 for s in stF])
    phB = holm([s[1]["p_t"] if s[1]["p_t"]==s[1]["p_t"] else 1.0 for s in stB])
    for i,(aid,stb,ap) in enumerate(stB):
        _,stf,_ = stF[i]
        bwm,bws,na = mean_std([v[1] for v in ap.values()])
        loB,hiB = stb["ci"]; loF,hiF = stf["ci"]
        vB = verdict(phB[i], stb["p_perm"], loB*100, hiB*100) if stb["n"]>=2 else "n<2"
        vF = verdict(phF[i], stf["p_perm"], loF*100, hiF*100) if stf["n"]>=2 else "n<2"
        print(f"{aid:4s} {bwm*100:+6.1f}(n={na})   {stb['mean']*100:+6.1f} "
              f"[{loB*100:+5.1f},{hiB*100:+5.1f}] {phB[i]:6.3f} {stb['p_perm']:6.3f} {vB:13s} | {vF:12s}")
print(f"\nδ={DELTA} pt for both metrics. BWT verdict uses the same rule as F1 (load-bearing / borderline / inert / inconclusive).")
