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

---

## PR-3 — B6 under a frozen `mu` reference (registered 2026-08-21, before implementation)

**Background.** B6's null (`i2` ties `ce`) is the campaign's spine, so it must
survive an implementation objection. `I_2` decomposes as

```
I_2 = sum_k [ (w+_k)^2 + (w-_k)^2 ]
    = 1/2 ||z||^2  +  1/2 sum_k ( sum_j |w_jk| )^2
```

with `w_jk = beta_jk (phi_j - mu_j) + beta_0k/J`. Because `sum_j w_jk = z_k`
identically, **`mu` cancels from the first term and appears only in the second**.
The first term is the logit-norm half that ties with CE; the second is the
DS-specific half. So the entire DS-specific content of `I_2` is measured against
`woe_feature_mean` -- a within-current-task EMA (momentum 0.9, ~10-step effective
window, seeded from the first batch, zeroed at every task boundary,
`woe_si.py:2660-2669` and `2308-2309`). If the tie is caused by that reference
lagging rather than by DS content being inert, B6 is a statement about the
implementation, not the theory.

**Intervention.** `--woe_mu_mode {ema,frozen_pretask}`. Under `frozen_pretask`,
`mu^(t)` is the unweighted mean of `phi` over task *t*'s **full** training set,
computed in a pre-pass before task *t*'s first gradient step, held fixed for the
task. Pre-pass runs in **both** arms (it is RNG- and BatchNorm-neutral), so the
arms are the same trajectory except for which `mu` the evidence reads.

**Host.** The B6 anchor-only host, unchanged: `configs/models/til/woe_si_lc.yaml`,
`--n_epochs 1 --inner_steps 2 --lr 0.003 --woe_omega_transform abs
--woe_anchor_mode proximal --woe_omega_accum sum --woe_importance_scalar i2`,
`woe_lc_lambda=0`, `xi=1e-3`, 10-task TIL one-shot.

**Seed spread on this host.** Two **on-peak** n=3 cells: `i2`@2.4e5 sd 0.0031,
`ce`@75 sd 0.0030. Pooled **sigma = 0.0031**. The third available n=3 cell,
`i2`@1.2e5 (sd 0.0153), is deliberately **excluded**: it is off-peak, and
off-peak cells sit on a slope where seed-to-seed trajectory variation maps onto
larger accuracy differences. Pooling it would inflate sigma ~3x and widen the
inconclusive band until a real effect landed inside it. The comparison here is
peak-to-peak, so the on-peak basis is the matching one. Bands are set at
multiples of sigma, per the convention registered in PR-1.

**Primary statistic.** `D' = mean final F1 (i2 + frozen_pretask, at its peak
lambda, n=3)  -  mean final F1 (i2 + ema, at its peak lambda, n=3)`, the latter
being **0.5008 +/- 0.0031**. `D'` isolates the single knob under test. The
question this run exists to answer is "is the tie a gauge artefact", which is
exactly whether freezing `mu` moves `i2` -- not whether frozen-`i2` clears CE.

| `|D'|` | verdict |
|---|---|
| <= **2 sigma ~ 0.006** | tie persists. The null is **not** about the `mu` reference. B6 survives the implementation objection; one line in the paper |
| 0.006 to **4 sigma ~ 0.012** | inconclusive. Extend to n=5 on both arms before concluding anything |
| > 0.012 | B6's null was partly a gauge artefact. DS content is **not** established as inert; the campaign's spine needs re-examining before anything is written |

**Secondary statistic, reported but not deciding.** `D = mean final F1
(i2 + frozen_pretask, peak, n=3) - 0.4994` (the `ce` bar, n=3, lambda 75, this
host). `D` moves two things at once -- the tracked scalar *and* the `mu` mode --
so it cannot isolate the objection. **If `D'` is null the objection is closed
regardless of where `D` lands.**

**Registered prior.** The two halves of `I_2` were measured collinear at
**cos 0.996** on the penalty (`omega-agreement-does-not-predict-accuracy`). If the
`mu`-dependent half moves in near-lockstep with the `mu`-free logit half, there is
little room for the `mu` reference to be what causes the tie, so the tie is
*expected* to persist. Stated here so it is a prior and not a post-hoc "we
expected that". It does not license skipping the run: the objection still has to
be closed, and a registered prior that is confirmed is evidence, whereas an
unregistered one is a story.

**Consequence for the paper's sentence, decided now rather than after the
result.** If `D'` is null, the claim written is **not** the vague "DS content is
inert" but the sharper:

> the DS-specific term of `I_2` is collinear with the logit norm on this
> architecture, and is therefore inert *here*

which is more informative, more falsifiable, and carries a prediction: **the null
should break on an architecture whose features disagree more** (lower cosine
between the halves). That prediction is the follow-up this campaign earns; the
vague version earns nothing.

**Also inadmissible, registered in advance.** The Omega Spearman between arms at
matched lambda is reported as a **mechanism descriptor only**. Relating Omega
agreement to accuracy gaps is inadmissible on this project
(`omega-agreement-does-not-predict-accuracy`), and that ruling is not suspended
because the arms here differ by `mu` rather than by scalar.

**Diagnostics dumped alongside** (all inside the same runs, no extra cells):
per task at end of task, `||mu_ema - mu_frozen|| / ||mu_frozen||` plus both raw
norms (the June checkpoints show `||mu||` varying 2.5x across tasks -- 42.6 at
task 1 against ~17 elsewhere -- so the ratio alone would hide the scale); the two
halves of `I_2` averaged over the batch under each `mu` mode; and the matched-lambda
Omega Spearman.

**Verification gate.** With the flag absent or `ema`, a regression run must
reproduce the recorded `i2` figures **0.5206 / 0.5008 / -0.0198** bit-identically.
Nothing else launches until it does, and whether it did is stated explicitly in
the report.

---

## PR-3 — Amendment 1 (2026-08-21, written after the E0 gate failed and **before any `frozen_pretask` run existed**)

**The registered verification gate cannot be satisfied, and is replaced.** PR-3
required the `ema` arm to reproduce the recorded `i2` control bit-identically.
It did not: **0.5224 / 0.4984 / -0.0240** against **0.5206 / 0.5008 / -0.0198**.

**What the failure was not.** Three competing explanations were tested before
touching the design, using the per-task `[LC] split` lines as a trajectory
fingerprint:

| test | result | rules out |
|---|---|---|
| two runs, pre-pass off, same command | **bit-identical** | run-to-run nondeterminism |
| pre-pass off, checkpoints on vs off | **bit-identical** | checkpoint writing |
| pre-pass off, full run | **0.5206 / 0.5008 / -0.0198** | the uncommitted tree having moved the bar -- today's code reproduces 2026-08-18 exactly, so every README figure drawn against 0.5008 stands |

**What it was.** The pre-pass itself. Both trajectories are individually
reproducible across separate launches -- so the pass shifts the run
*deterministically*, it does not add noise. Saving and restoring CPU+CUDA RNG
around the pass changed nothing (the with-pass run reproduced its own earlier
numbers to the digit), so the mechanism is **not** a stolen random draw. The
likeliest remaining candidate is workspace-dependent cuDNN algorithm selection --
`cudnn.deterministic` fixes the algorithm *given* the available workspace, not
across allocator states, and the pass moves tens of GB through the caching
allocator. **This is unproven and is recorded as a hypothesis, not a finding.**
Two isolated CUDA+AMP probes (3 forwards, 2-channel and 3-ADC) were inert, which
is consistent with a scale-dependent mechanism but does not establish one.

**Replacement gate.** `frozen_pretask` cannot run without the pre-pass -- the
pass *is* the intervention -- so the arms can only share a trajectory if **both**
pay for it. Bit-identity to a run from three days earlier was only ever a proxy
for "the arms differ by one knob"; running the pass in both arms serves that goal
directly, while running it in neither is impossible. Registered in its place:

1. **Reproducibility.** The with-pass trajectory reproduces across separate
   launches. *Necessary but thin*: two runs agreeing shows reproducibility under
   the conditions they ran in. Because the suspected mechanism is allocator
   state, which depends on what else is on the GPU, **one grid cell is repeated
   at a different job concurrency**. If the trajectory moves with concurrency,
   the entire comparison is pinned to a fixed `MAXJOBS` and that is stated -- and
   it is far cheaper to learn this before ten cells than after.
2. **Common mode.** Both arms run the pre-pass. The perturbation is therefore
   paid identically on both sides and cancels from `D'`.
3. **Common-mode behaviour, per seed.** At seed 0 the pass moves the control by
   -0.0024 = **0.77 sigma**, i.e. less than one seed sd. One draw is not enough.
   The n=3 runs at the peak give `ema + prepass` at three seeds: **if their mean
   sits more than 1 sigma (0.0031) from the recorded 0.5008, the perturbation is
   not behaving as common-mode noise and `D'` must be re-examined before it is
   read.** Registered in advance precisely because it is the branch that would be
   tempting to skip.

**What this amendment costs, written down now so it is not discovered later.**
With both arms perturbed, `D'` is clean but **neither arm is comparable to any
recorded figure in `README.md` any more**. PR-3 is a within-experiment
comparison and is unaffected. But frozen-arm or `ema + prepass` numbers **must
not** be quoted against the campaign's other results -- the `ce` bar, the A6/A7
tables, the B6 scalar comparison -- because those were measured without the pass.
The secondary statistic `D` (frozen against the `ce` bar of 0.4994) crosses
exactly this boundary and is therefore **downgraded further**: it was already
non-deciding, and it is now known to carry a ~0.8 sigma vehicle offset on one
side only. It is reported for completeness and nothing may be concluded from it.

**Method note earned here.** The gate caught something the unit tests could not,
because those tests verify an argument about a code path while the gate measures
the run. This is the second time in this campaign that a discipline registered in
advance has caught what an after-the-fact read would have absorbed (the first
being PR-2a's statistic, which could not fire and had to be replaced). Both
belong in the paper's methods section: they are the kind of thing that makes a
negative-results paper credible.

---

## PR-3 — Amendment 2 (2026-08-21, sign asymmetry; written after two **off-peak** grid cells, before any peak cell or any n=3)

**The registered bands are stated on `|D'|` and that is under-specified.** PR-3's
table reads `|D'| > 4 sigma` as "B6's null was partly a gauge artefact; DS content
is **not** established as inert". That reading only follows if `frozen_pretask` is
*better* than `ema`. A large **negative** `D'` -- the frozen reference being
worse -- licenses a different conclusion, and the rule as written would have
mapped it onto the wrong one.

**Occasion for noticing.** The first matched pair, at the off-peak `lambda = 6e4`:

| arm | diagonal | final | BWT |
|---|---|---|---|
| `ema` | 0.5695 | 0.5034 | -0.0661 |
| `frozen_pretask` | 0.5572 | 0.4880 | -0.0692 |

`D'(6e4) = -0.0154`. This is **not** the deciding statistic -- it is one seed at
an off-peak lambda, where the neighbouring cell's measured seed sd is 0.0153, so
this is ~1 sd of the local noise and predicts nothing. It is recorded only as the
occasion for fixing the rule, and the fix is registered before any cell that
could decide anything has run.

**Amended bands, by sign.**

| `D'` | verdict |
|---|---|
| `|D'| <= 2 sigma ~ 0.006` | tie persists. The null is not about the `mu` reference. B6 survives; one line in the paper. **Unchanged.** |
| `0.006 < |D'| <= 0.012` | inconclusive; extend to n=5 before concluding. **Unchanged.** |
| `D' > +0.012` | `frozen_pretask` **better**. B6's null was partly a gauge artefact; DS content is not established as inert; the campaign's spine needs re-examining. **Unchanged.** |
| `D' < -0.012` | `frozen_pretask` **worse**. See below -- a *different* conclusion, newly registered. |

**What a large negative `D'` licenses, and what it does not.** It closes the
registered objection, but more weakly than a null does, and the difference must
be stated rather than elided. The objection was "the tie is caused by the `mu`
reference lagging". If a better-conditioned reference -- unweighted, computed
over the task's *full* training set rather than a ~10-step EMA window -- makes
retention *worse*, then reference quality is not what is holding `i2` down, and
B6 survives. But it does **not** license "the EMA is the right reference": the
frozen reference carries a defect of its own, registered at the start of this run
and not discovered afterwards, namely that `mu^(t)` is measured on the model as it
stands at task `t`'s *start* and the features then drift away from it over the
task. `frozen_pretask` trades a **lagging** reference for a **stale** one. A
negative `D'` is therefore consistent with two readings that this experiment
cannot separate:

1. DS content is inert, and neither reference matters; or
2. staleness costs more than lag, and some third reference (a mid-task or
   end-of-task pass, i.e. two passes per task) would do better than both.

**Registered handling of that ambiguity.** If `D' < -0.012`, the paper's sentence
is the null-conclusion sentence already committed to in PR-3 -- the DS-specific
term is collinear with the logit norm and inert here -- **plus** an explicit
statement that the frozen alternative was tested and was worse, with the
staleness confound named. What may **not** be written is any claim that the EMA
reference is well chosen, or that the `mu` question is settled. A
`frozen_posttask` arm is the follow-up that would separate (1) from (2); it is
**not** run here and is not implied.

**Supporting diagnostic, already registered in PR-3 and now load-bearing.** The
per-task `||mu_ema - mu_frozen|| / ||mu_frozen||` trace distinguishes the readings
in part: the gap shrinks monotonically as the backbone settles (1.161 at task 0,
0.331 at task 1, 0.194 at task 2), so the EMA's departure from the true task mean
is largest at task 0 -- where the network trains from random init and there is
nothing yet to forget -- and smallest over the later tasks where retention is
actually decided. That is a measurement, not an argument, and it is reported
whichever way `D'` lands.

---

## PR-3 — Amendment 3 (2026-08-21, the deciding statistic moves to matched lambda; written before any peak cell and before any n=3)

**`D'` read peak-to-peak is not a valid comparison, and the error is in
Amendment 1's premise rather than in its arithmetic.** Amendment 1 justified
running the pre-pass in both arms on the grounds that its cost is then
"common-mode" and cancels from a difference. That holds for a shared **additive**
perturbation. The first three `ema` cells suggest it may not be additive:

| lambda | `ema` no pre-pass (recorded) | `ema` with pre-pass | shift |
|---|---|---|---|
| 6.0e4 | 0.4795 | 0.5034 | +0.0239 |
| 1.2e5 | 0.5036 | 0.5102 | +0.0066 |
| 2.4e5 | 0.5008 | 0.4984 | -0.0024 |

If the pre-pass changes the *function* mapping `lambda` to retention rather than
offsetting it, then a difference cancels it **only where both arms sit at the
same `lambda`**. `D'` read peak-to-peak does not: should `ema` peak at 1.2e5 and
`frozen_pretask` at 2.4e5, `D'` would contain the peak-vs-peak effect the
experiment is after **plus** a differential reshape of roughly
`+0.0066 - (-0.0024) = +0.009`, with no way to separate them afterwards. That is
a validity defect, not a variance one.

**Status of the reshape itself: a possibility, not a finding.** Three points in a
monotone sequence at one seed, two of them slope cells whose measured seed sd is
0.0153 -- against which the largest shift, +0.0239, is 1.6 sd. It is not
established and is not claimed. It is registered because the deciding statistic
must be **robust to it either way**, and matched-lambda reading is; peak-to-peak
is not. That robustness, not the reshape's reality, is the argument for the
change.

**Amended deciding statistic.**

> `D' = final F1 (frozen_pretask) - final F1 (ema)`, both at **`lambda = 2.4e5`**,
> both with the pre-pass, n=3 paired by seed (0/39/55).

`lambda = 2.4e5` is chosen for three reasons fixed before the peaks are known:
it is the confirmed `i2` peak on this host without the pre-pass; it is the cell
with the measured on-peak seed sd of **0.0031**, which is the sigma the bands are
built on; and it is the only cell with a matched no-pre-pass n=3 behind it, which
is what makes Amendment 1's clause 3 checkable at all. The bands and their sign
split (Amendment 2) carry over unchanged: tie at `|D'| <= 0.006`, inconclusive to
0.012, and the two distinct readings above that by sign.

**Demoted to descriptors, reported and not deciding:** the peak-to-peak
difference; each arm's own peak location; and the paired difference at every
other grid `lambda`. The full per-lambda paired difference is reported precisely
because it is the evidence for or against the reshape, and readers should be able
to see it rather than take the matched-lambda choice on trust.

**Consequence for E3, which gets cheaper rather than dearer.** The plan of "n=3
at each arm's own peak, plus a separate n=3 `ema+prepass` at 2.4e5 for clause 3"
is withdrawn. With the deciding statistic at matched `lambda`, E3 is **n=3 for
both arms at 2.4e5 -- four runs** (seeds 39 and 55 for each arm; seed 0 exists),
and Amendment 1's clause 3 falls out of the `ema` half of that same set rather
than being bolted on. The clause is then satisfied at the `lambda` where it means
something, against the matched no-pre-pass n=3 of 0.5008 +/- 0.0031.

---

## PR-3 — Amendment 4 (2026-08-21, the n=5 extension is waived in advance; written while E3 is running and **before any seed-39 or seed-55 result exists**)

**The registered "inconclusive -> extend to n=5" branch is withdrawn, and the
withdrawal is registered before it can fire.** Amendments 2 and 3 leave the
deciding statistic as the matched-lambda paired difference at 2.4e5, n=3. Seed 0
gives `D' = -0.0075` (2.4 sigma), inside the inconclusive band.

**Why the extension is not worth buying.** With sd ~0.0031 per arm the paired sd
is ~0.0044, so the standard error on a three-seed mean is ~0.0025. For the n=3
mean to clear the -0.012 boundary the two remaining seeds must average ~-0.014,
roughly 3x the seed-0 value; for it to return inside 2 sigma they must average
~-0.003. Both are possible; the modal outcome is a mean near -0.008, i.e. the
inconclusive band again and an automatic n=5 trigger.

Now read what the branches actually license:

| outcome | what may be said |
|---|---|
| tie (`|D'| <= 0.006`) | the objection is closed; B6 survives cleanly |
| frozen worse (`D' < -0.012`) | the objection is closed **weakly**; B6 survives; the `mu` question is explicitly unsettled |
| inconclusive (between) | both of the above, hedged |

**Every branch closes the objection.** The difference between them is one
qualifier in a methods paragraph, and none of them threatens the campaign's
spine. Four more runs to sharpen a qualifier is poor value, and the report is
honest either way: *the objection was tested; frozen `mu` was consistently
non-better across five lambda and three seeds; B6's null is not attributable to
the `mu` reference.*

**The branch that would have mattered is already effectively excluded.** Only
`D' > +0.012` -- frozen `mu` *better* -- would have put the spine back in
question. Four matched pairs at seed 0 are negative at every lambda on the grid
(-0.0075 to -0.0154), so a strongly positive n=3 mean is not a live possibility.
That, rather than the exact band, is tonight's information.

**Registered handling.** E3 runs to n=3 because it is already launched and cheap.
**Whatever band it lands in, PR-3 stops there.** No n=5. The write-up is the
"objection tested and closed, `mu` question open" version, with the band reported
as measured and not sharpened.

---

### PR-3 finding, independent of `D'`: a frozen reference makes the anchor over-anchor

This is registered as a **result in its own right**, because it does not depend on
which band `D'` lands in and would otherwise be lost as a by-product of a null.

At the deciding lambda, seed 0:

| arm | diagonal | BWT | final |
|---|---|---|---|
| `ema` | 0.5224 | -0.0240 | 0.4984 |
| `frozen_pretask` | 0.5068 | -0.0159 | 0.4909 |
| delta | **-0.0156** | **+0.0081** | -0.0075 |

**Freezing `mu` buys stability and pays for it in plasticity, at a net loss.**
Forgetting falls by a third; the diagonal falls twice as far as the retention
gain.

**Correction to this entry, made when the grid completed.** The sentence
originally written here -- "the same trade appears at every lambda on the grid" --
was written before the last cell landed and is **wrong at one of the five**. The
full seed-0 grid:

| lambda | d(diagonal) | d(BWT) | d(final) |
|---|---|---|---|
| 6.0e4 | -0.0123 | **-0.0031** | -0.0154 |
| 1.2e5 | -0.0183 | +0.0077 | -0.0106 |
| 2.4e5 | -0.0156 | +0.0081 | -0.0075 |
| 4.8e5 | -0.0165 | +0.0080 | -0.0085 |
| 9.6e5 | -0.0119 | +0.0057 | -0.0060 |

Accurately: **the plasticity cost is universal** -- the diagonal falls at all five
lambda, by 0.0119 to 0.0183, with no trend -- while **the stability gain holds at
four of five** and *reverses* at the weakest anchor, where frozen `mu` is worse on
both axes.

That reversal sharpens the mechanism rather than weakening it. At `lambda = 6e4`
the anchor is too weak to consolidate much of anything (`ema` BWT -0.0661, the
most forgetting anywhere on the grid), so there is no over-anchoring to be had and
the frozen reference is simply a worse reference. The stability gain appears only
once the anchor is strong enough to bite, and then holds at +0.0057 to +0.0081
across a 8x span of lambda while the diagonal cost stays flat. A mechanism that
switches on with anchor strength is better evidence for the over-anchoring reading
than one that is present uniformly, because it is what over-anchoring predicts and
a constant offset would not be.

The reading: a fixed reference makes the anchor *more conservative*. It
consolidates against a `mu` that no longer tracks the drifting features, so more
of the network reads as important, and over-anchoring costs the current task more
than it saves the old ones. This is precisely what the **staleness** confound
registered at the start of PR-3 predicts -- `frozen_pretask` trades a *lagging*
reference for a *stale* one -- and it is the signature that separates
"staleness costs more than lag" from "DS content is inert".

It does **not** settle that separation, and no claim is made that it does. What it
does is convert `frozen_posttask` (a two-pass variant measuring `mu` at the end of
the task, or mid-task) from a loose end into the specific, motivated follow-up:
if over-anchoring is the mechanism, a reference that tracks the features should
recover the lost diagonal without giving back the BWT.

---

## PR-4 — Ignorance vs conflict as a forgetting readout (registered 2026-08-21, **before any number is computed**)

**What this is.** An offline read-out diagnostic over trained checkpoints. It does
not touch the training objective, the tracked scalar, or `woe_feature_mean`, and
changes what no model optimises. Distinct from PR-3, which asked whether the `mu`
*inside* `I_2` explains B6's null (answer: no).

**Hypothesis, stated so it can fail.** Probability collapses "no support" and
"conflicting support" onto the same posterior; a mass function does not
(Denoeux Sec 5.1.1, Fig 8). So: a task degraded because its evidence has **decayed**
should show elevated `m(Theta)`; a task degraded because later tasks actively
**contradict** it should show elevated `kappa`. If neither varies meaningfully
across tasks, or the ordering is an artefact of the gauge, the diagnostic dies.

**Kill criterion (registered before running, gates everything downstream).**
The per-task ordering by `kappa/K_s` and by `m(Theta)` is recomputed under three
gauges: (1) balanced-pooled `mu` [primary], (2) per-task `mu`, (3) final-task `mu`.

> **If the ordering of tasks flips between gauge (1) and gauge (2), the
> diagnostic is measuring the normalisation and not the model. Phase 1 stops and
> that is the reported result. No Phase 2.**

"Flips" is operationalised in advance as **Spearman rho < 0.8 between the task
orderings under gauges (1) and (2)**, on either `kappa/K_s` or `m(Theta)`. A
second, weaker warning is recorded if `rho` falls in [0.8, 0.9): the ordering is
reported but every downstream claim is hedged as gauge-sensitive.

**A second kill condition, on variation rather than gauge.** If the across-task
spread of `kappa/K_s` (or of `m(Theta)`) is smaller than its within-task standard
error, the quantity is flat and there is nothing to order. Reported as "does not
vary", not as a weak ordering.

**Comparison rules, fixed now.**
* **Cross-sectional only.** At fixed `t` under a single `mu`, "task 0 shows more
  conflict than task `t`" is within one gauge and valid. "Conflict on task 0 rose
  between `t=3` and `t=9`" is **not**, if `mu` was recomputed at each `t`. Any
  longitudinal version freezes `mu` at the final measurement point and recomputes
  all earlier checkpoints under that one gauge.
* `kappa` is normalised by `K_s` before any cross-task comparison, because class
  counts are `[6,7,6,7,7,6,6,6,7,6]` and an aggregate over classes is not
  comparable between a 6-class and a 7-class block.
* **The excess is reported alongside `kappa`**, per class
  `E_k = sum_j |w_jk| - |z*_k| >= 0`. Since `sum_j w_jk = z*_k` identically, `E_k`
  isolates the gauge-dependent part from the gauge-invariant logit, so a reader
  can see how much of the conflict signal is decomposition rather than model.
  Nothing about disagreement can be gauge-invariant -- disagreement is a statement
  about a decomposition -- so the gauge is declared and its contribution shown
  rather than escape being claimed.

**Phase 2 is not designed and will not be designed until Phase 1 lands.** Its
question is registered only as: does the `kappa` / `m(Theta)` split predict a
task's accuracy drop better than the drop is predicted by BWT alone? If the split
is jointly no more informative than the accuracy drop itself, that is the honest
negative and is reported as a real result about DS in CL.

**Implementation facts checked before writing anything** (the spec asked these be
flagged rather than guessed):
* `compute_weights_of_evidence` (`woe_si.py:213-268`) centres the **features**
  (`phi - mu`) and adds the uniform offset `beta_0k / J`. It does **not** centre
  `beta` across classes. So the block-centring of step 3 is genuinely new and
  **does not conflict** with the existing implementation; the existing function is
  reused for step 4 with a pre-centred `beta` passed in.
* `_active_class_indices` (`woe_si.py:2672-2690`) returns exactly
  `range(offset1, offset2)` under TIL with no noise column configured on this
  schedule, which is the block `C_s` required. Reused.
* Checkpoints: the PR-3 `ema` arm at 2.4e5 seed 0 has **all ten**
  (`logs/woe_si_lc/mufz_ema_lam240000.0_s0-*/0/checkpoints`). The June fallback is
  not needed.
* `det_head` reads the same `phi` and is excluded; only `fc` is the classifier.
* Final-checkpoint `Omega` covers tasks 0-8 only, because `on_task_end` is never
  called from `main.py`. Irrelevant here -- this readout uses `fc` and `phi` only --
  and recorded so it is not misused.

**One derivation flag.** The multi-category `m(Theta)` and `kappa` are derived
here from the conjunctive combination of the `2K` simple support functions the
construction specifies, **not transcribed** from the paper's numbered results
(Eqs 20-21 / Proposition 1), which are not available to check against in this
session. The derivation is written out in the script so the correspondence can be
verified. If it disagrees with the paper, the paper is right and the numbers are
recomputed.

---

## PR-5 — Conflict *seeking* (registered 2026-08-28, before any run completed)

**Background.** E1/E2 charge the Least-Commitment objective with a positive
lambda, so `conflict = 2 * sum_k w+_k w-_k` is **penalised**. Minimising it drives
every class toward one-sided evidence (a class with enormous `w+` and zero `w-`
scores zero), which permits many features to vote the same way — exactly the
redundancy that leaves no room for later tasks. The hypothesis registered here is
the sign flip: **reward** conflict, so each class score is a small residue of
large opposing evidence and features are forced to take differing positions.
Call the wished-for consequence *feature specialisation*.

**The literal form is ill-posed, and that is a prediction, not a caveat.**
`conflict` is quadratic in the readout scale and unbounded above, and
cross-entropy only ever sees `w+_k - w-_k = z'_k`. So a negative lambda on the
raw term rewards an inflation direction CE is structurally blind to; the arm
should degenerate. It is run anyway (two cells) so the claim is shown rather than
asserted. The arm that carries the hypothesis is the bounded one:

    kappa_k = (1 - e^{-w+_k/tau})(1 - e^{-w-_k/tau})  in [0, 1],  tau = 4

the exact Dempster conflict (Eqs 21/31), averaged over classes, no `J^p` divisor.
It keeps the "both channels large" incentive and saturates, so the reward runs
out instead of running away. `tau = 4` is set from the measured `w+` of 4.1–5.3
on this project's runs; at `tau = 1` the transform already sits at kappa ≈ 0.98
and there is nothing left to seek.

**Host and bar.** `woe_si_lc` (replay-free, proximal anchor 1e6, `lr 0.003`,
one-shot `n_epochs 1 / inner_steps 2`), seed 0 — the same host E1 and E2 ran on.
The bar is the `woe_lc_lambda = 0` control **re-measured on the current working
tree**, not the recorded 0.4847: `model/woe_si.py`, `model/eralg4.py` and others
are modified on this branch and the recorded number predates them.

**Seed spread on this host.** sd ≈ 0.004 (PR-1), so n=3 detects δ ≳ 2·sd ≈ 0.008
and any n=1 gap under ~0.015 is inside the band that produced the E1 `i2`
λ=100 mirage.

**Quantity.** `S` = best `kappa` cell's final F1 at seed 0 minus the re-measured
λ=0 control's final F1, over the bracket λ ∈ {−0.5, −2, −8} (and λ = +2 as the
bounded-penalty control, which says whether the *sign* matters or the term is
merely inert).

**Rule.**

| `S` | verdict | what may be claimed |
|---|---|---|
| ≥ **+0.015** | screen passed | run n=3 (seeds 0, 39, 55) at the winning lambda; the claim needs paired mean ≥ **+0.008** there before anything is said |
| 0 to +0.015 | no effect | reported as **no effect**. Do **not** run n=3 — this is the mirage band |
| < 0 | negative | conflict-seeking is reported as harmful on this host, with the shape of the lambda response |

**A mechanism condition on any positive result.** A cell that improves the bar
without moving the thing it was supposed to move is not evidence for the
hypothesis. Any `S ≥ +0.015` must be accompanied by **both**: (1) the run's
step-weighted `conflict` share rising against the λ=0 control's 84.6%, and (2) a
feature-usage readout moving toward specialisation rather than away. If the
accuracy moves and the conflict share does not, the arm is reported as "a term
that helped, mechanism unidentified", and the specialisation reading is **not**
claimed.

**Registered failure modes that are still results.** The raw-`conflict` negative
cells diverging (NaN, or a diagonal collapse with the term dominating CE) is the
predicted outcome and is reported as the well-posedness argument for `kappa`. A
flat `kappa` response across three decades of lambda is reported as the term
being inert on this host, which is what E1 and E2 both found for the penalty
direction and would make the sign flip a null in the same place.

**Operating point, measured after registration and before any arm completed**
(`WOE_LC_DEBUG=1`, `woe_si_lc`, `lr 0.003`, λ=−1, first 12 steps of task 0):
`kappa` = **0.200**, against CE 1.76–2.02 — so the registered bracket
{−0.5, −2, −8} spans 5% / 22% / 89% of the cross-entropy, i.e. numerically inert
to nearly dominant, which is what brackets a peak. **The grid is not re-centred**
on this measurement; it is recorded because it also confirms two things the
design assumed: `kappa` = 0.20 leaves 5x headroom before saturation (so the
reward is live, not flat), and it rose 0.200 → 0.223 over those 12 steps, so the
term moves the quantity it names. Per-task conflict share on that same run came
in at 0.89 / 0.93 / 0.99 / 0.89 against the λ=0 record of 0.846 step-weighted.

**Outcome (2026-08-28, read against the bands above).** `S = 0.4247 − 0.4788 =`
**−0.054** — the `< 0` band. Conflict-seeking is reported as **harmful** on this
host, monotone in lambda (final F1 0.4788 / 0.4247 / 0.3733 / 0.3529 at
λ = 0 / −0.5 / −2 / −8), and harmful on *both* axes: the diagonal falls
0.5696 → 0.4872 while BWT worsens −0.0908 → −0.1552. **No n=3 was run**, per the
rule. The re-measured control came in at 0.4788 against the recorded 0.4847,
a 0.0059 drift well inside the seed band — so the decision to re-measure changed
nothing, but the bar is the current-tree number.

The registered failure modes both fired as predicted. The raw-`conflict` cells
degenerated exactly as the well-posedness argument said they would: loss to
−2.2e7 against a CE near 2, F1 0.076 / 0.061, BWT ≈ 0 because nothing was
learned. And the mechanism condition — which would only have gated a *positive*
result — is satisfiable in the negative direction too: at λ = −8 the loss
`CE − 8*kappa` reads −6.21, so `kappa >= 0.90` against the control's starting
0.20. The term moved the quantity it names, most of the way to its ceiling, and
cost accuracy anyway. Written up as README section E5.
