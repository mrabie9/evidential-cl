# Outline B — the B6-independent draft

Written 2026-08-18, *before* PR-2 resolves, so that a bad branch means choosing
between two drafts rather than rebuilding a spine under time pressure.

**Outline A** (the current plan) claims: *WoE-SI is an instrument, not a method;
two independent probes of DS structure inside it (A7, B6) both return inert; the
live variables are scale and diffuseness.* It rests on B6, and PR-2 is a live
competing explanation for B6.

**Outline B** does not use B6 at all.

---

## Claim

A **screening criterion** for Dempster-Shafer-derived continual-learning
mechanisms, demonstrated retrospectively on two failures, bounded by a stated
exclusion, and tested prospectively once.

> **Locality condition.** A DS derivation that is true of a belief state *at an
> instant* transfers to continual learning only if whatever the derivation holds
> fixed does not lie on the path between its assumption and its claimed benefit.
> Where the held-fixed quantity is on that path, continual learning is precisely
> the setting that moves it, and the mechanism fails in a predictable direction.

The criterion is cheap to apply — it needs only the derivation, not an
implementation — which is what makes it worth publishing as a screen.

## Evidence

**Instance 1 — C6 Claim 2, the latent buffer.** `phi` is a sufficient statistic
for the weights of evidence *at the moment of storage*, so a DS reading says
store `phi`, not the input. Sufficiency is on the path: the claim is that a
stored `phi` supports the same evidence later. Continual learning is defined by
backbone drift, which is exactly what breaks it. Both feature arms land *below*
naive fine-tuning (0.2269 and 0.2803 against 0.3030) with healthy diagonals and
catastrophic BWT, and the matched-items control makes it attributable: doubling
stored features made it **worse**, so the deficit is staleness, not sample count.

**Instance 2 — E4, the `I_1` bridge.** Least commitment at p=1 is an L1 criterion
on the weights of evidence, proposed as a one-parameter bridge from
regularisation to architectural isolation. "Committing less frees capacity" holds
the availability of the freed capacity fixed, and that is on the path to the
claimed benefit. Nothing reserves it. The mechanism provably engages — effective
features 338.7 -> 253.7, vacuous fraction 0.092 -> 0.286 — and that is *why* it
hurts: at the harmful setting the **diagonal rises** (0.5744 vs 0.5206) while
BWT collapses to -0.2048. Sparsity without allocation raises inter-task
collision instead of lowering it.

**Exclusion — E3.** The evidential objective fails, and the criterion does *not*
cover it. Its error is static: `w+` and `w-` are the positive and negative parts
of one sum, so the loss asks for two things that cannot be moved independently.
No time-fixity assumption is involved. A criterion that covered every failure
would be doing no work; naming what it misses, and why, is what makes it a
screen rather than a description.

**Prospective test — disjointness.** The criterion's forward call, registered
with grounds and a timestamp *before* the run (see `preregistration.md`). This is
the section that separates a screening criterion from a post-hoc summary, and it
is the only part of Outline B not yet executed.

## What is deliberately absent, and why

| result | why it is not in Outline B |
|---|---|
| **B6** | PR-2 raises a competing explanation (a floored SI denominator makes `Omega` a displacement measure, which would produce B6's null whatever scalar was tracked). Outline B does not need it. |
| **A7** | Resolves to *inert*, not refuted: the combination rule does nothing once its own precondition is granted, and the apparent effect was scale. A real result, but it is a statement about instrumentation, and its own mechanism turned out to be anchor-induced incommensurability rather than anything cautious. Belongs in an appendix as a methodological cautionary tale. |
| **C6 Claim 1** | Heading for *untested* under PR-1: the distillation vehicle contributes under 0.01, so there is no dynamic range in which `w` and `z` could differ detectably. |

## Cost of the switch

Outline B needs the disjointness prediction registered and run, which was
already planned. It needs no new experiments to support its two instances, both
of which are complete with attributable mechanisms and controls. The loss
relative to Outline A is scope, not soundness: two demonstrated instances rather
than a claim about WoE-SI as a whole.

## Prior art the prospective section must clear

Hard projection into activation/gradient null spaces: **GPM**, **Adam-NSCL**,
**OWM**, **OGD**. Learned masks: **HAT**, **SupSup**. Soft parameter-level
masking by importance: **SPG** (Konishi et al., ICML 2023) — the nearest
neighbour, since it is soft, importance-driven, and explicitly not monopolising
capacity. The novelty burden therefore falls entirely on the derivation doing
visible work: it must **predict** the functional form, the exponent, the
normaliser, or the coupling to `Omega`. If it only says "penalise overlap", it is
SPG or soft HAT with a DS story attached, and should be dropped rather than
defended.
