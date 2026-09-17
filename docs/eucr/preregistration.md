# EUCR pre-registration

Decision rules fixed **before** the corresponding runs. Written 2026-08-25, after
the trainability study (`eucr-trainability.tex`) and the DS-formalism gate, and
before Gate B0 / PR-E1 were executed.

Provenance note: Gate B0, PR-E1 and PR-E2 were proposed in review discussion and
never written down. That is the failure mode this file exists to prevent, so they
are recorded here in full, with statistics and thresholds, before any of them is
run.

---

## Gate B0 — is forgetting carried by normalisation statistics or by weights?

**Motivation.** BatchNorm running statistics are buffers, so `snapshot` and
`penalty` skip them and task *t+1* overwrites them wholesale. For a head that
reads feature *direction* (cosine distance to prototypes) and is measurably
unstable under running statistics, this is an unregularised forgetting channel
that no quadratic weight penalty can reach. If it dominates, PR-E1's restoration
condition has to include statistics, or PR-E1 measures the wrong thing.

**Procedure.** Load the task-1 checkpoint. Evaluate task-0 held-out F1 twice,
weights untouched at their task-1 values:

1. normally (BatchNorm running statistics);
2. with every BatchNorm module switched to batch statistics at inference
   (`module.train()` on the norm layers only; `momentum = 0` and buffer
   save/restore so the evaluation does not mutate the checkpoint).

**Statistic.**

```
S = [F1_batchstats(theta_1) - F1_runningstats(theta_1)]
    / [F1_runningstats(theta_0*) - F1_runningstats(theta_1)]
```

the fraction of the task-0 loss recovered by changing the normalisation regime
alone. Denominator is the forgetting actually incurred between the task-0 and
task-1 checkpoints.

**Decision rule.** Evaluated per seed on 3 seeds (0, 39, 55):

- `S >= 0.5` in **at least 2 of 3** seeds → forgetting here is substantially
  carried by normalisation statistics rather than weights; PR-E1's primary
  restoration condition becomes **weights plus statistics**.
- `S < 0.2` in at least 2 of 3 seeds → the weight question is clean; PR-E1
  condition 1 (weights only) is primary.
- Otherwise → inconclusive; run B1 before PR-E1.

**Cost.** Runs on existing checkpoints. No new training, no consolidation state.

---

## PR-E1 — does any per-coordinate importance measure localise forgetting?

**The predicted object.** Not "is EUCR's importance good" but the more general:
does *any* per-coordinate importance measure localise forgetting on this
architecture. A null here is an architecture-level finding that implicates the
whole importance family, EWC included.

**Statistic.** On task-0 held-out data,

```
R(k) = [F1(theta_restored) - F1(theta_1)] / [F1(theta_0*) - F1(theta_1)]
```

where `theta_restored` takes the top-k% of **shared backbone** coordinates by a
measure's rank back to their task-0 values and leaves the rest at task-1 values.
Unbounded above and below. Restoration condition (weights only, or weights plus
normalisation statistics) is set by Gate B0 under the rule above.

**Grid.** k in {0.1, 0.3, 1, 3, 10, 30}%. **Primary at k = 1%.**

**Measures.** MAS-on-entropy (the shipped EUCR importance — note `discord` and
`both` are bit-identical, so they are one arm, not two); Fisher from the CE head;
`|dtheta . g|`. **Controls:** `|dtheta|`, `|theta_0*|`, random.

**Pass condition.**

```
R_MAS-entropy(1%) - R_random(1%) >= 0.10   in at least 2 of 3 seeds
```

**Secondary.** `k_half` (smallest k with `R >= 0.5`), censored at 30%, compared
across measures by rank test. Fisher and `|dtheta . g|` are reported, not gating.

**Sanity kill.** If `R_random(1%) >= 0.5` the restoration is not selective at that
k; re-centre the grid *before* comparing any measure. This check runs first.

**What each outcome licenses.**

- MAS-entropy fails, Fisher succeeds → a statement about the *functional* being
  differentiated, not about importance methods.
- Everything fails, including displacement → kills the importance family on this
  architecture, EWC included. A real architecture-level finding.
- Everything succeeds → the anchor was worth calibrating after all, and the
  lambda question reopens (see the calibration proposal in the design note:
  choose lambda by penalty-gradient-norm ratio, report the ratio not lambda).

**Cost.** Needs a re-run: the checkpoint writer must serialise `importance` and
`theta_star`, which current checkpoints (`{task, state_dict}`) do not carry.

---

## PR-E2 — **skipped**

Different predictand from the DS gate, and not closed by it. Skipped under the
standing rule: it would need a new run to produce per-sample correctness
matrices, and those are not in the existing artefacts. With the OOD null in hand
and a collinearity risk against entropy, it is not worth a run.

---

## Folded into the PR-E1 re-run

Not gated, but logged from the same run rather than a separate pass:

- **Prototype utilisation**: per-prototype argmax counts and activation variance
  on held-out data. Outstanding from earlier review and still unrun. Directly
  tests the "no prototype ever approaches a feature" mechanism proposed for the
  OOD null (observed cosine distances span only [0.60, 1.34]).

## Standing analysis rules

- Report **signed** AUROC, not `|AUROC - 0.5|`. Most novelty scores here are
  anti-correlated, and the absolute value hides that; the anti-correlation is a
  finding. Where `|AUROC - 0.5|` is quoted, compare it against the null floor
  `E|Z| = 0.8 * sigma`, not against 0.
- Any novelty experiment on this data trains **without the noise class**. With
  ~50% noise per task, the noise output absorbs unfamiliar inputs and the
  experiment measures "does this look like noise" instead. Measured: excluding
  noise rows from the ID set is *not* sufficient; the noise output must not exist.
- All results are 3 seeds (0, 39, 55) and read from every per-seed
  `results.txt`, never seed 0 alone.

---

## Amendment 1 (2026-08-25) — Gate B0's statistic is ill-conditioned; result stands

**Executed.** Gate B0 ran on 8 seeds (0, 39, 55 from `f_newconfig`; 1, 2, 3, 7, 11
from `b0_extra`).

**Problem found on execution.** `S` divides by the forgetting actually incurred,
and on 2 of the first 3 seeds that denominator is degenerate: seed 0 forgot
0.008 F1 (nothing to recover) and seed 39 forgot **-0.136** (task-0 F1 *rose*
after training task 1 — backward transfer). `S` is meaningless there, and seed
0's `S = 5.04` is a small recovery over a smaller denominator, not evidence.
The rule as written would have fired on that degenerate value.

**Amendment.** `S` is reported only for seeds with real forgetting
(denominator >= 0.05 F1). Seeds outside that are reported and excluded from the
count, not silently kept. Alongside `S`, report the two absolute quantities,
which are well conditioned regardless of denominator:
`F1(theta_1, batch stats)` and `F1(theta_1, running stats)`.

**Result under the amended reading.** Six of eight seeds show real forgetting
(0.078 to 0.381 F1). On those, `S` = 0.46, 0.81, 0.88, 0.91, 0.92, 1.61
(median 0.88) — normalisation statistics carry roughly 80-90% of task-0
forgetting. The absolute quantities are cleaner still:
`F1(theta_1, batch)` is 0.73-0.80 across all eight seeds, while
`F1(theta_1, running)` ranges 0.45-0.80. Batch statistics give a stable, high
task-0 F1 regardless of seed; the running statistics inject the variance and, in
the worst seed, a 0.32 F1 collapse.

**Verdict: PASS, weights+statistics.** PR-E1's primary restoration condition is
weights plus normalisation statistics. The weight-only condition is retained as a
secondary arm, since it is now the more interesting comparison: it measures what
is left once the dominant channel is excluded.

**Consequence beyond PR-E1.** Most of EUCR's measured catastrophic forgetting is
a BatchNorm buffer artefact, not weight drift. A quadratic weight penalty cannot
reach buffers, so the inert anchor was never going to help — and this partly
explains BWT of -0.24 to -0.30 in the benchmark table. Recorded as a finding, not
as a gate outcome.

---

## Amendment 2 (2026-08-25) — Gate B0 extended; remedies tested out of sequence

The pre-registered order was B0 -> re-run -> PR-E1 -> G. Gate B0's result made the
BatchNorm remedy the highest-value experiment available, so it was run first.
Sequences update on evidence; statistic *definitions* do not, and none were
changed after seeing a value.

### B0 extended: more sequence positions, a non-EUCR control, batch-size sweep

Gate B0's single cell (task 0, checkpoint 1) cannot distinguish "normalisation
carries forgetting" from "the backbone barely forgets". Three extensions, all
offline on existing checkpoints. **Like-for-like comparison** throughout: batch
statistics at both ends, so the transductive advantage cancels.

EUCR, task 0, 5 seeds, mean F1 (running / batch statistics):

| checkpoint | running | batch |
|---|---|---|
| ck0 (just trained) | 0.776 | 0.844 |
| ck1 (one task later) | 0.614 | 0.753 |
| ck3 (end of sequence) | 0.107 | 0.672 |

Genuine forgetting (batch@ck0 - batch@ck3) = **0.172**. Measured forgetting
(running@ck0 - running@ck3) = **0.669**. So **~74% of EUCR's end-of-sequence
task-0 forgetting is a normalisation artefact** and ~26% is real. Task 1 behaves
the same way.

**It is EUCR-specific, not architectural.** An EWC control on the same backbone,
same loader, same eval path shows a running-vs-batch gap of **<= 0.013 F1 at
every cell**. The benchmark table is *not* mismeasured for other models. The
mechanism is the cosine-prototype head reading feature *direction*: a shift in
normalisation statistics rotates the feature cloud onto one prototype (cf. the
directional-coherence measurement, 0.07 under batch stats against 0.16-0.42
under running stats). A linear head with a bias is far more robust.

**Not a transduction artefact.** Batch-statistic evaluation was swept over
test batch sizes 32/64/128/512. The ck3 recovery is stable and only mildly
batch-size dependent (seed 1: 0.731/0.742/0.748/0.769), so even at batch 32 the
recovery is 0.61-0.73 against a running-statistic 0.12.

**There is also a within-task mismatch.** At ck0, immediately after training
task 0, batch statistics already beat running statistics by ~0.07 F1
(0.776 -> 0.844). That is a plain train/eval BN mismatch, not a continual one,
and it bounds what any cross-task statistics policy can recover.

### Remedies (4 tasks, 10 epochs, 3 seeds). Diagonal reported alongside BWT.

| arm | diagonal F1 | final F1 | BWT |
|---|---|---|---|
| `running` (shipped) | 0.5864 +/- 0.0421 | 0.2843 +/- 0.0291 | -0.302 |
| `freeze` after task 0 | 0.3644 +/- 0.0435 | 0.2128 +/- 0.0850 | **-0.152** |
| `per_task` (select by task id) | 0.5864 +/- 0.0421 | **0.3279 +/- 0.1413** | -0.259 |

`freeze` is the textbook trap: it improves BWT by 0.15 and pays 0.22 diagonal for
it, ending 0.07 *worse* on final F1. Reporting BWT alone would have made it look
like the best arm on the table. `per_task` leaves the diagonal untouched by
construction (right after task t, the running statistics *are* task t's) and buys
+0.044 final F1, but with a large seed spread.

`per_task` recovers much less than batch-statistic evaluation does, and the ck0
row above says why: it fixes cross-task overwriting but not the within-task
train/eval mismatch, which is roughly 0.07 F1 on its own.

**Framing rule for the writeup.** Batch-statistic evaluation is a *measurement*
change, not a method. Where it recovers forgetting, the claim is that the
forgetting was never there -- not that it was fixed. Only `freeze` and `per_task`
are interventions, and both must be reported with diagonal and BWT together.

**Prior work.** This is a confirmation in a new setting, not a discovery.
BatchNorm statistics biased toward the current task are a known continual-learning
failure mode: Continual Normalization (Pham et al., ICLR 2022, arXiv:2203.16102),
task-specific BatchNorm / CLBN, and the related LayerNorm-tuning line. Cite these;
the contribution here is the *magnitude* on a direction-reading evidential head
(74% of measured forgetting) and the EWC control showing it is head-specific
rather than architectural.

### Consequence for PR-E1

Residual genuine forgetting under batch-statistic evaluation is **0.172 F1 at
end-of-sequence** but only ~0.02 after one task. PR-E1 must therefore run at
**end-of-sequence** (task 0 at checkpoint 3), where the denominator is large
enough to have power; the after-task-1 position answers nothing at n=3.

---

## Amendment 3 (2026-08-25) — PR-E1 needs no re-run; two product measures added

**The re-run was over-scoped.** PR-E1 was blocked on serialising `importance`
and `theta_star`. Neither is needed:

* `finalize_task_after_training` runs at `main.py:1542` and the checkpoint saves
  at `main.py:1671`, so `task_0.pt`'s `state_dict` **is** `theta_star` for task 0.
* `importance` is a deterministic forward/backward accumulation over the task's
  data (`cons.compute_importance`), so it is recomputable from that checkpoint.

The whole panel therefore runs offline on checkpoints that already exist. The
re-run is still wanted for prototype utilisation logging, which is a separate
non-gating diagnostic, but it does not gate PR-E1.

**Two measures added, reported not gating.** `fisher * dtheta^2` and
`mas_entropy * dtheta^2`. These are elementwise products of arrays the panel
already builds, so they cost nothing. They are also better motivated than
importance alone for a *restoration* criterion: `importance * dtheta^2` is the
second-order Taylor term for the loss increase, i.e. exactly the quantity the
EWC/MAS penalty sums. Ranking by importance alone ignores how far a coordinate
actually moved.

The gating comparison is **unchanged**: `R_mas_entropy(1%) - R_random(1%) >= 0.10`
in at least 2 of 3 seeds. The products are additional reported columns. Adding
non-gating columns before execution is recorded here so it is auditable.

**Specification detail the original left open.** Under the weights+statistics
condition, restoring all BatchNorm statistics alone already recovers most of the
measured gap (Gate B0), which would leave the weight ranking almost no headroom.
`R(k)` is therefore measured *relative to the statistics-restored baseline*:

```
R(k) = [F1(stats restored + top-k% weights) - F1(stats restored, 0% weights)]
       / [F1(theta_0*) - F1(stats restored, 0% weights)]
```

so it asks what restoring weights adds *on top of* restoring statistics. That is
the 0.172 F1 denominator from Amendment 2, and it is the quantity with power.

---

## PR-E1 result (2026-08-25) — gate PASSES, conclusion is still negative

Run offline on `b0_extra` checkpoints, seeds 1/3/11, task 0 at end-of-sequence,
restoration = weights + BatchNorm statistics, `R(k)` relative to the
statistics-restored baseline (Amendment 3). Per-seed denominators 0.354 / 0.689 /
0.533 F1.

**Sanity kill: not triggered.** `R_random(1%)` = 0.014 / 0.002 / 0.002.

**Gate: PASS, 3/3 seeds.** `R_mas_entropy(1%) - R_random(1%)` = 0.575 / 0.187 /
0.223, all >= 0.10.

`R(k)` at the primary k = 1%, mean +/- sd over 3 seeds, ranked:

| measure | R(1%) |
|---|---|
| `|dtheta|`  (**control**) | **+0.699 +/- 0.104** |
| `fisher x dtheta^2` | +0.682 +/- 0.161 |
| `mas_entropy x dtheta^2` | +0.579 +/- 0.020 |
| `fisher` | +0.515 +/- 0.073 |
| `mas_entropy` (shipped) | +0.334 +/- 0.222 |
| `|dtheta . g|` | +0.318 +/- 0.248 |
| `|theta_0*|` (control) | +0.103 +/- 0.152 |
| `random` (control) | +0.006 +/- 0.007 |

**The displacement control beats every importance measure.** Forgetting on this
architecture *is* localisable -- 1% of coordinates restores 70% of it -- but the
coordinates are identified by *how far they moved*, not by any importance
functional. The shipped `mas_entropy` is less than half as good as plain
`|dtheta|` and is the second-worst non-trivial measure on the panel.

**The two product measures isolate why.** Multiplying either importance by
`dtheta^2` moves it most of the way to `|dtheta|` (`mas` 0.334 -> 0.579; `fisher`
0.515 -> 0.682), and neither product beats `|dtheta|`. So essentially all of the
gain in `importance x dtheta^2` comes from the `dtheta^2` factor. The importance
factor is not adding information over displacement; on `mas_entropy` it dilutes it.

**Caveat on the statistic, stated because it cuts against the headline.**
`R(k)` rewards L2 proximity to `theta_0*`, and ranking by `|dtheta|` maximises
that proximity at any k by construction, so part of the control's advantage is
mechanical rather than a claim about localisation. The products are the fair
test, since they hold the `dtheta^2` factor fixed and vary only the importance
term -- and they say the importance term buys nothing.

**What this licenses.** Not the "everything fails" branch: localisation works.
The finding is narrower and more specific -- on this architecture the importance
*family* is dominated by a displacement baseline, which applies to EWC's Fisher
as well as to EUCR's MAS-on-entropy. Combined with Gate B0 (74% of measured
forgetting is a normalisation artefact that no weight penalty can reach), the
case for EUCR's consolidation mechanism as a method contribution is closed.

---

## Gate B1 (registered 2026-08-27) — is normalisation-statistic sensitivity specific to the cosine-prototype head?

**Motivation.** Amendment 2 measured a running-vs-batch task-0 F1 gap of **0.565**
at ck3 for the DS head (0.107 -> 0.672) against **<= 0.013** at every cell for an
EWC control on the same backbone, same loader, same eval path. That contrast was
run out of sequence and has no registered arm behind it. It is now the candidate
headline claim -- "direction-reading heads amplify normalisation-statistic drift
by an order of magnitude over linear heads, under identical backbones and
identical CL methods" -- so it needs one. Under test: the sensitivity is a
property of a head reading feature *direction*, not of the backbone, the
normalisation layer, or the CL method.

This gate is registered as a **new arm, not an amendment to PR-E1**. PR-E1 has
already reported (see above); amending a closed arm after its result is the
failure mode this file exists to prevent. Gate B0 constrains nothing here: its
registered decision rule outputs only which restoration condition PR-E1 uses, so
no headline claim is committed against head-specificity.

**Checkpoints.** `b0_extra`, seeds 1, 2, 3, 7, 11 -- the set the 0.565 denominator
comes from. Not `amp_off_{0,39,55}`, which is the 3-seed Gate B set. **No new
seeds and no re-training in any arm** (except the registered Arm 1 fallback).

### Hypotheses

**H1 (head-specificity).** On identical weights, batches and features, a linear
head shows a materially smaller running-vs-batch F1 gap than the DS head.

**H2 (mechanism).** The gap is caused by a shift in the *post-LayerNorm* pooled
features between regimes -- `features = self.feat_norm(self.do(out))`,
`eucr_backbone.py:384`, the tensor both heads actually read -- decomposable into
a mean offset `delta` and a per-channel scale mismatch `sigma_ratio`, neither of
which cosine similarity is invariant to. Cosine is scale-invariant but not
shift-invariant; a linear head absorbs a constant input shift into its bias
(`ce_heads` is `nn.Linear(feat_dim, count)` with bias, `eucr_backbone.py:193`).

**H3.** Under running statistics, task-0 rival mass concentrates on the task-0
prototype best aligned with `delta`; under batch statistics, on the same anchors,
it spreads. Per-task heads plus task-incremental class masking mean mass cannot
leave task 0's own prototype set, so this is a **within-task angular claim**, not
a cross-task basin claim.

### What LayerNorm forbids, exactly

LayerNorm centres and scales in the **pre-affine** space, so the direction it
annihilates is `1` before the affine, which maps to `gamma` after it. Projecting
a post-LayerNorm `delta` against `1` would test the wrong direction whenever
`gamma` is non-uniform.

Write `u = (z - beta) / gamma` for the pre-affine normalised features. LayerNorm
guarantees per sample

```
(1/d) sum_c u_c = 0        and        (1/d) sum_c u_c^2 = 1
```

Both hold in **either** normalisation regime, so for the difference of regime
means the constraint is exact, not approximate:

```
sum_c delta_c / gamma_c = 0
```

The `gamma`-parallel component of `delta` is therefore **identically zero**, not
merely predicted to contribute little recovery. Measuring anything else means the
estimator is reading the wrong tensor or the wrong axis, so this runs as an
**instrument check before the arm**, not as a result within it.

The same argument pins the per-sample second moment. The surviving `sigma`
mismatch can therefore only be a **redistribution of variance across channels**,
never a global rescale -- which bounds what Arm 2's variance-matched condition
can possibly be attributing.

### Arm 1: post-hoc linear probe (offline)

Load ck0, freeze the backbone, fit a linear head on task-0 *training* features.
Load ck3 weights, score that same frozen head under both regimes. Same weights,
same batches, same features, no auxiliary loss, no re-training. This reproduces
the DS head's defining property -- fit during task 0, then frozen while the
backbone drifts -- without the aux-loss confound, and it is a *within-run* head
comparison, strictly stronger than the cross-model EWC control.

**Sanity kill, runs first.** If `F1_linear(ck0, batch) < F1_DS(ck0, batch) - 0.05`
the heads are not accuracy-matched and `Delta_head` is uninterpretable: a head
with no accuracy cannot lose any. Improve the probe, or fall back below, before
comparing gaps.

**Fallback if the probe cannot reach parity.** `eucr_ce_aux_weight: 0.1`,
tolerance **+/- 0.05** on the DS ck3 gap (~9% of the 0.565 effect, inside the
threshold margin), paired by seed. Registered now so the fallback is not a
post-hoc choice. This variant requires re-training and its DS numbers are not the
shipped model's; report the shipped-default DS gap alongside as the anchor.

**Statistic.** `Delta_head = gap_DS - gap_linear` at ck3, paired by seed.

**Threshold.** H1 supported if `Delta_head > 0.2` in **at least 4 of 5** seeds.
Per-seed table reported, not only the mean.

### Arm 2: shift and scale projection (offline)

All estimation in **post-LayerNorm** feature space.

**Instrument check, before any scoring.** Assert `sum_c delta_c / gamma_c = 0` to
numerical tolerance. Exact under LayerNorm; a non-zero value blocks the arm.

**Estimation.** Split task-0 held-out in half. Estimate `delta` and `sigma_ratio`
on half A; apply and score on half B. Estimating on the scored data would fit the
correction in the very regime being explained and make the recovery an upper
bound.

**Conditions, in registered order.** (1) unmodified; (2) centred only (subtract
`delta`); (3) rescaled only (apply `sigma_ratio` *without* centring); (4) centred
then rescaled. Order is registered because centring changes the per-channel std
that would then be measured, so "variance-matched" is otherwise ill-defined.

**Statistic.** Fraction of the 0.565 ck3 gap recovered.

**Threshold.** H2 supported if condition (4) recovers **> 70%**. Conditions (2)
and (3) individually are **descriptive, not gated** -- attributing between the two
halves is the object of the arm, and pre-committing to a split would be guessing.
Report `delta` decomposed into its `gamma`-parallel and `gamma`-orthogonal
components.

### Arm 3 — deferred to Gate B2

A cosine-argmin readout (`ds1` + `ds1_activate` + argmax, no
Belief/Omega/Dempster stack) would separate *cosine geometry* from *DS fusion*,
which Arm 1 cannot. It requires code that does not yet exist, and a threshold on
an unwritten readout is not a commitment. B1's claim is direction-reading versus
linear, which Arm 1 carries alone; cosine-versus-DS is a refinement and gets its
own gate.

`--eucr_distance_metric euclidean` is **not** an available substitute: it collapses
the model to F1 ~0.098, and normalisation sensitivity cannot be read off a model
that never learned.

### Rider: mass-flow decomposition (not gating on its own)

Task-0 anchors at ck0 and ck3, both regimes, same anchor set. Noise class excluded
via `signal_mask_exclude_noise`. Restricted to correct-at-ck0. **Byte-identical
batch composition and order at both checkpoints**, since batch-statistic scoring
is transductive and a sample's score depends on its batch. Decompose `Delta Bel_y`
over task-0 rivals; record top-2 rival gap and entropy of the renormalised rival
masses. Discount the **~0.07 within-task floor** (the ck0 running-vs-batch
mismatch, Amendment 2) before attributing anything to forgetting. Record ck2 as
well, since the collapse localises at the rcn->deeprad transition.

**Threshold for H3.** Rival-mass entropy lower under running than batch statistics
in at least 4 of 5 seeds, **and** the modal rival's prototype in the top quartile
by alignment with `delta` from Arm 2. The second condition is what makes this a
mechanism test rather than a description, and it is free given Arm 2.

**Declared degeneracy.** At the shipped default `activation_norm="max"`,
`m(Omega) ~ 4.2e-4` and therefore `Dou_y = 1 - p_y` identically, with the DS head
equal to a softmax to 1.8e-7. This rider is a **rival-distribution shape result**
and must not be read as an evidential-uncertainty result.

### Blocked is not failed

Arm 1's sanity kill and Arm 2's instrument check can both fail for boring
instrumental reasons. If either fires, the gate is recorded as **BLOCKED**, with
the failing quantity reported, and re-run after the instrument is fixed. It is
**not** recorded as a negative result on H1 or H2. A blocked gate written up as a
null is how a real effect gets buried.

### Declared up front

Every number here is transductive; batch statistics at both ends so the
transductive advantage cancels. **Both** running-statistic and batch-statistic
numbers are reported, with the gap itself as the object of study rather than
either one picked as true -- the standard inductive-protocol number is 0.669
end-of-sequence forgetting, the weights-only number is 0.172, and the difference
is the finding. All arms run offline on existing `b0_extra` checkpoints.

**Prior work.** BatchNorm statistics biased toward the current task are a known
continual-learning failure mode (Continual Normalization, Pham et al., ICLR 2022,
arXiv:2203.16102; task-specific BatchNorm / CLBN). The contribution claimed here
is not that, but the *head-specificity*: an order-of-magnitude amplification on a
direction-reading head against a linear head on identical features, with a
geometric mechanism.

---

## Gate B1 result (2026-08-27) — H1, H2 and H3 all fail; the headline claim does not survive

Run offline on `b0_extra`, seeds 1/2/3/7/11, task 0, ck0 and ck3. Scripts:
`scripts/gate_b1_lib.py`, `gate_b1_repro.py`, `gate_b1_arm1.py`, `gate_b1_arm2.py`,
`gate_b1_rider.py`. Raw per-seed JSON in `logs/eucr/gate_b1/`.

### Instrument checks: both PASS

**Reproduction of Amendment 2.** Running-statistic cells reproduce essentially
exactly; batch-statistic cells sit 0.02-0.04 low, which is the transduction the
protocol declares -- a batch-stat score depends on batch composition, so only the
deterministic running-stat cells can reproduce to 4 decimals.

| cell | here | Amendment 2 |
|---|---|---|
| ck0 running | 0.7726 | 0.776 |
| ck0 batch | 0.8264 | 0.844 |
| ck3 running | 0.1077 | 0.107 |
| ck3 batch | 0.6347 | 0.672 |

ck3 gap 0.527 (published 0.565); genuine forgetting 0.192 (published 0.172).
Per-seed denominators are used throughout rather than the published constant.

**LayerNorm constraint.** `sum_c delta_c / gamma_c` = 1e-9 to 2e-8 relative to
`sum_c |delta_c / gamma_c|`, all five seeds. Zero to float32 resolution, as the
pre-affine derivation requires. The `gamma`-parallel fraction of `delta` is
5.2e-4 +/- 4e-4. Note `gamma` is near-uniform here (std 0.0026 about a mean of
1.0000), so projecting against `1` instead of `gamma` would have given the same
answer in *this* model; the distinction remains necessary in general.

**A third instrument fault, found and fixed mid-gate.** Arm 1 was not
reproducible on first execution: `Delta_head` moved by up to 0.16 between
identical runs. Two independent causes, both now pinned in `gate_b1_lib.py`:

1. The batch regime evaluates with the backbone in *train* mode, so every
   BatchNorm **writes its running buffers** during a batch-statistic pass. A
   running-statistic pass afterwards was therefore no longer scoring the
   checkpoint. `extract_features` now snapshots and restores the buffers.
2. cuDNN benchmark mode (the parser default) re-picks convolution algorithms per
   run. At ck3 under running statistics the model sits near chance, where
   last-bit differences move many argmaxes. `pin_determinism()` disables it.

A third, in the probe itself: LBFGS in float32 on near-separable features has a
flat optimum and landed elsewhere on each run. The probe is now fitted on CPU in
float64 with a ridge selected on a held-out 20% slice of task-0 **train** (never
test). After all three fixes Arm 1 is bit-identical across runs; this was
verified before any result below was read.

### Arm 1 — H1 NOT SUPPORTED. The linear head collapses just as hard.

Both heads scored on the same extracted post-LayerNorm features.

| seed | DS ck0 | linear ck0 | gap DS | gap linear | `Delta_head` |
|---|---|---|---|---|---|
| 1 | 0.8391 | 0.8515 | 0.6430 | 0.6618 | -0.0188 |
| 2 | 0.8368 | 0.8503 | 0.4595 | 0.5302 | -0.0706 |
| 3 | 0.8333 | 0.8549 | 0.5077 | 0.5295 | -0.0218 |
| 7 | 0.8460 | 0.8513 | 0.5909 | 0.6112 | -0.0203 |
| 11 | 0.8420 | 0.8503 | 0.5798 | 0.6043 | -0.0245 |

**Sanity kill: not triggered** -- the probe is *better* than the DS head at ck0
(+0.0122 +/- 0.0062), so the comparison is accuracy-matched with margin to spare.

`Delta_head` = **-0.031 +/- 0.022**, **0/5** seeds above the registered 0.2, and
negative in all five. gap_DS 0.556 +/- 0.073 against gap_linear 0.587 +/- 0.057:
a plain linear head with a bias, on identical features and identical weights,
loses *slightly more* to the normalisation regime than the cosine-prototype DS
head does.

**The candidate headline is dead.** "Direction-reading heads amplify
normalisation-statistic drift by an order of magnitude over linear heads" is
false on this architecture. The `<= 0.013` EWC figure from Amendment 2 does not
isolate the head: EWC is a different training run with a different backbone, so
that contrast confounds head with backbone. Held at fixed backbone -- the only
comparison that can speak to head-specificity -- the effect is zero and slightly
the wrong way round. **Amendment 2's EWC control must not be reported as
evidence about heads.** What it shows is that EUCR's *backbone* is the
BatchNorm-unstable object, and the mechanism paragraph in Amendment 2 (a cosine
head reading feature direction) is unsupported.

Gate B2 (the cosine-argmin readout) is **withdrawn, not deferred**: it was
registered to separate cosine from DS fusion within a head-specificity claim
that no longer has a phenomenon to explain.

### Arm 2 — H2 NOT SUPPORTED. The shift is not a global affine one.

Estimated on half A of task-0 held-out, applied and scored on half B, pooled
macro F1 (per-batch averaging is order-dependent here because the deeprad files
are class-ordered, so a reordered subset cannot use it).

| condition | DS recovery | linear recovery |
|---|---|---|
| centred (`+delta`) | 18.8% +/- 20.7% | 11.4% |
| rescaled, uncentred | **-5.7%** +/- 7.1% | -4.3% |
| centred then rescaled | **26.2%** +/- 20.9% | 25.2% |

Against the registered > 70% threshold this fails outright. Matching the first
*and* second moments of the post-LayerNorm feature distribution recovers about a
quarter of the gap, so the running-vs-batch difference is **not** a shared
affine transport of the feature cloud: it moves samples individually. The
uncentred rescale is mildly *harmful*, consistent with cosine's scale-invariance
making a pure per-channel gain nearly inert on the DS head.

Descriptively the shift is nonetheless large: `||delta||` = 16.7 +/- 0.7 against
a mean-feature norm of order 6, and `sigma_batch / sigma_running` = 4.5 +/- 0.9
per channel. The features really do move a lot; a global correction just does not
capture how.

The two heads recover the gap almost identically (26.2% vs 25.2%), which is Arm
1's conclusion arriving by a second route.

### Rider — mass flow. H3 NOT SUPPORTED, but the decomposition is clean.

~8500 anchors per seed of 10000 (correct at ck0, noise excluded).

**Forgetting here is entirely confident error.** On flipped anchors,
`Delta Bel_y` = **-0.869 +/- 0.037** under running statistics with
`Delta m(Omega)` = **-1.5e-14**. Every unit of belief mass leaves the true class
for a rival; none goes to ignorance. Under batch statistics `Delta Bel_y` =
-0.699 with the same null ignorance channel. Flip rate 0.67 running / 0.25 batch.

**H3 fails on both conjuncts.** Rival-mass entropy is lower under running
statistics in only **3/5** seeds (registered: >= 4/5). The modal rival's class is
in the top quartile by `delta` alignment in **0/5** seeds -- and not at random:
its alignment rank is 5, 5, 5, 5, 2 out of 6, i.e. the mass concentrates on
consistently *anti*-aligned classes. Whatever selects the rival, it is not
alignment with the feature shift.

**Declared degeneracy, corrected.** Measured fused `m(Omega)` at ck0 is
**8.0e-15**, not the 4.2e-4 recorded at registration. Those are different
quantities and both are right: 4.2e-4 is the *per-prototype* `1 - s_p` for the
argmax prototype, while the fused mass is its product over all 60 prototypes.
The registration's conclusion is unchanged and strengthened -- `Dou_y = 1 - p_y`
holds to ~1e-14, so this is a rival-distribution shape result and nothing here
is an evidential-uncertainty result.

### What Gate B1 licenses

Not a blocked gate: every instrument check passed and the arms are reproducible.
Three registered hypotheses were tested and all three failed, so the finding is a
genuine negative rather than an instrument failure.

* Head-specificity is **false** on this architecture. The normalisation
  sensitivity belongs to the backbone, not to the direction-reading head, and
  Amendment 2's EWC contrast is confounded.
* The regime difference is **not** a global affine shift of the features, so the
  one-geometric-sentence explanation is unavailable.
* EUCR's task-0 forgetting is **confident error**, with the ignorance channel
  numerically dead throughout.

Combined with Gate B0 and PR-E1, EUCR now has no surviving method-level claim:
the consolidation mechanism is dominated by a displacement baseline, 74% of its
measured forgetting is a normalisation artefact, and the artefact is not
attributable to its evidential head.
