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

**Amendment 1 (same day).** Those three cells are the **lambda grid**
(0.1 / 1.0 / 10.0), all seed 0 — not three seeds. Their 0.024 spread is a lambda
response and says nothing about seed variance, so it does not impugn the sd the
bands were derived from. But a second problem does: **that sd was measured on
`woe_si_lc`** (anchored, lr 0.003) and this gate runs on `woe_si_injection`
(anchor off, lr 0.001, 256 memories). The bands inherit a number from the wrong
host. Seeds 39 and 55 of the `ce` bar are queued; if the injection host's sd
differs materially from 0.004, **the bands are recomputed from the measured
value at the same multiples** (2.5x sd for the lower edge, 5x for the upper) and
that recomputation is registered before `V` is read.

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

**Amendment 1 (same day) — the fail branch as first written cannot answer the
question.** `xi` enters the anchor, so small-`xi` arms train under a different
penalty landscape: different `Omega` scale, different effective anchor strength,
different trajectories. Comparing tuned-lambda-at-1e-3 against
untuned-lambda-at-1e-6 is the same class of error the A7 commensurability gate
caught (raw against normalised at a shared lambda), and this project has three
recorded instances of `Omega` scale failing to predict optimal lambda — a
measured ratio places a *grid*, never a cell. **The fail branch therefore
requires a full lambda grid at the new `xi`, per scalar, before any comparison is
read.** Cost, registered now rather than discovered when the branch fires: 3
scalars x 5 lambda x 3 seeds = 45 runs, ~15 min each at 3 concurrent, i.e. ~4
hours. If that is not affordable, the honest report is that B6 could not be
disambiguated from the floor, **not** a weaker version of B6.

**Why mass-weighted and stiff-split.** A floored inert bulk is nearly harmless —
the parameters doing the protecting are still correctly normalised. A floored
stiff set is fatal. A count-based fraction cannot distinguish these and would
read a benign case as the fatal one.

**Secondary, already instrumented.** `delta_floored_frac` is reported per
consolidation, so its trend in `t` is observable. The anchor-feedback loop
predicts it rises monotonically in `t`; flat-and-high from task 0 instead means
xi was mis-sized from the start and the anchor is incidental.


---

## PR-2a — the no-retrain screen for PR-2's fail branch (registered 2026-08-18, runs queued)

**Purpose.** PR-2's fail branch is expensive (~45 runs) and confounded unless
lambda is re-swept. This is the cheap direct test of the *confound itself* — does
the floor erase between-scalar differences in `Omega`? — with no retraining and
therefore no lambda problem.

**Method.** One dump per candidate tracked scalar (`i2`, `ce`, `phi2`, `z2`), each
carrying `numerator` and `delta_sq` separately at the run's own tuned lambda.
Rebuild `Omega` offline at xi = 1e-3 and 1e-6 for each, and compare the scalars'
`Omega` profiles *to one another* at each xi.

**Quantity.** `D(xi)` = mean pairwise Spearman distance between the scalars'
cumulative `Omega` vectors, computed at each xi.

**Rule.**

| outcome | verdict |
|---|---|
| `D(1e-6)` ≈ `D(1e-3)` | the floor is **not** erasing between-scalar structure; strong evidence B6 survives, and the expensive fail branch is not triggered |
| `D(1e-6)` >> `D(1e-3)` | the floor *is* collapsing distinct scalars onto a common displacement measure; PR-2's fail branch fires with its lambda grid |

**Stated limitation, registered up front.** A null here shows the floor does not
erase differences in `Omega`; it does *not* show that separated `Omega` would
produce separated accuracy. So PR-2a can **exonerate** the floor but cannot by
itself **convict** it, and it cannot fully replace the retrain in the convicting
branch. It is a screen, not a substitute.

---

## PR-2 / PR-2a — Amendment 2 (2026-08-18, written after the single-scalar dump, before the between-scalar dumps landed)

**The registered statistic could not fire, and has been replaced.** PR-2a's
`D(xi)` was a mean pairwise Spearman *distance* between scalars. Spearman
distances are bounded and compressed near the top of their range, so if the
scalars are genuinely different constructs `D(1e-3)` may already be large and
`D(1e-6) >> D(1e-3)` cannot cleanly occur. Replaced with
**`Spearman(Omega@xi, numerator)`** — computable from a *single* dump, not
range-compressed, and a direct test of the floor rather than of its consequences.
`scripts/xi_floor_check.py`.

**Result: the floor is real and total.** RMS `|Delta|` = 1.03e-3 against
`sqrt(xi)` = 3.16e-2, so `Delta^2` ~ 1e-6 against `xi` = 1e-3 — mis-sized by
~1000x. `Spearman(Omega, numerator) = 0.999977`. **SI's path-length denominator
has been inert throughout this project**; every `Omega` reported anywhere in
`README.md` is a raw path integral scaled by `1/xi`, not a curvature-like
quantity. `delta_floored_frac` is 1.0000 from task 0 *with the anchor off*, so
the anchor is incidental — branch 3 in its widest form, and the "self-reinforcing
loop" reading of the per-task suppression is withdrawn.

**PR-2's registered fail condition fired**: floored stiff-set mass = 0.9581,
against a convict threshold of 0.25. Spearman(Omega@1e-3, Omega@1e-6) = 0.907,
which is the "inconclusive" band on its own.

**But a measurement not in the registration points the other way, and this
amendment is written before the deciding dumps land so that it is not a post-hoc
escape.** The rule was a *proxy* for "the floor makes `Omega` a displacement
measure, so the tracked scalar does no work" — the competing explanation for B6.
Measured directly: **`Spearman(Omega, total |Delta|) = -0.060`.** Displacement
explains none of `Omega`'s rank structure. `Omega = |sum_steps h.Delta| / xi`
still carries the tracked scalar through `h`; an inert denominator does not make
it scalar-blind. The proxy fired; the thing it was proxying for did not.

**Registered handling.** The fail branch is **not** cancelled on this basis --
that would be arguing out of a rule because its outcome is unwelcome. Instead the
decision defers to the *direct* between-scalar test, whose runs (`ce`, `phi2`,
`z2` dumps) were queued before this was written:

| `min` pairwise `Spearman(Omega_a, Omega_b)` across scalars | verdict |
|---|---|
| >= 0.95 | the scalars produce the same `Omega`; B6's null **is** the floor; retrain branch fires |
| <= 0.80 | the scalars produce genuinely different `Omega`; B6 survives the floor; retrain **not** required, and the -0.060 above is the supporting evidence |
| 0.80-0.95 | retrain branch fires |

**Retrain branch, repriced.** Seeds are needed only at each scalar's best lambda,
not at every grid point: sweep lambda at one seed to locate each peak (3 x 5),
then three seeds at the peak (3 x 3, reusing the sweep's seed) = **24 runs, ~2
hours**, not 45/~4h. Registered because the escape hatch leaves B6 permanently
ambiguous, which is the worst available outcome, and halving the cost is the
difference between affordable and not.

**Independent of all branches**, the xi finding is a result in its own right and
is reported regardless: a project-wide implementation defect that changes what
every `Omega` in this document *is*.
