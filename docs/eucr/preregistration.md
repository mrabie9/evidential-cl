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
