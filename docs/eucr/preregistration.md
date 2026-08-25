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
