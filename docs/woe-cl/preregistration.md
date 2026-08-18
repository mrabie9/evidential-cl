# Pre-registered decision rules

Thresholds written down *before* the deciding runs completed, so the reading is
not chosen after seeing the numbers. Each entry records the date, the quantity,
the rule, and what each outcome licenses saying.

---

## PR-1 — C6 distillation vehicle gate (registered 2026-08-18, arm incomplete)

**Background.** C6 read the null between `logit` (distil `z`) and `evidence_sym`
(distil `w`) as evidence about what a replay buffer should carry. Both arms
recovered only ~19% of the gap CE rehearsal closes, so the null may reflect a
vehicle that does not deliver rather than two equivalent targets. `ce_logit` /
`ce_evidence_sym` (DER++ analogues) test whether distillation delivers at all.

**Quantity.** `V` = paired mean improvement of `ce_logit` over the `ce`
rehearsal bar (0.4466 at seed 0), at n=3, on the `woe_si_injection` host.

**Seed spread on this host.** sd ≈ 0.0031–0.0053; take sd ≈ 0.004.

**Rule.** To detect a `w`-vs-`z` difference at n=3 requires δ ≳ 2·sd ≈ 0.008.
For such a difference to be a plausible *fraction* of what the distillation term
contributes, the term must contribute several times that.

| `V` | verdict | what may be claimed |
|---|---|---|
| ≥ **+0.020** | vehicle cleared | run the `w`-vs-`z` head-to-head at n=3; a null there is informative |
| +0.010 to +0.020 | vehicle marginal | **do not** re-run the head-to-head. Report: distillation is marginal on this host and the `w`-vs-`z` question is not answerable here |
| < +0.010 | vehicle fails | C6 Claim 1 is reported as **untested**, not refuted |

**Note.** Passing technically is not clearing usefully. A vehicle contributing
under 0.01 has no dynamic range in which `w` and `z` could differ detectably, so
re-running the comparison would only buy a second uninterpretable null.

**Status at registration.** Three of six cells in: `ce_logit` 0.4506 / 0.4309 /
0.4553 against the 0.4466 bar. Best margin +0.009 at n=1 — currently *below* the
lowest band. Registered before the arm completed and before any n=3 replication.

---

## PR-2 — B6 under a floored SI denominator (registered 2026-08-18, run queued)

**Background.** `Omega_i^t = |omega_i^t| / ((Delta_i^t)^2 + xi)` with xi = 1e-3,
so the path-length correction only operates while `Delta^2 >> xi`
(`sqrt(xi)` = 0.0316). If the anchor pushes `|Delta|` below that, the denominator
floors and `Omega` becomes a displacement measure largely independent of the
scalar tracked along the way — which is a mundane competing explanation for B6's
null, and B6 is what licenses "WoE-SI is instrument, not method".

**Quantity.** Spearman correlation between `Omega` recomputed at xi = 1e-6 and
at xi = 1e-3, from the same dump; plus the `Omega`-mass share held by parameters
with `Delta^2 < xi`, reported separately for the stiff set (`b >= 1`) and the
inert bulk.

**Rule.**

| outcome | verdict |
|---|---|
| Spearman ≥ 0.95 **and** floored mass share of the stiff set < 0.10 | the floor is not reshaping `Omega`; B6 survives independently of it, and this is stated in one line |
| Spearman < 0.90 **or** floored stiff mass ≥ 0.25 | the floor is load-bearing; B6's null must be re-tested by *retraining* the scalar comparison at a smaller xi before the campaign's framing can rest on it |
| in between | inconclusive; retrain the B6 comparison anyway |

**Why mass-weighted and stiff-split.** A floored inert bulk is nearly harmless —
the parameters doing the protecting are still correctly normalised. A floored
stiff set is fatal. A count-based fraction cannot distinguish these and would
read a benign case as the fatal one.

**Secondary, already instrumented.** `delta_floored_frac` is reported per
consolidation, so its trend in `t` is observable. The anchor-feedback loop
predicts it rises monotonically in `t`; flat-and-high from task 0 instead means
xi was mis-sized from the start and the anchor is incidental.
