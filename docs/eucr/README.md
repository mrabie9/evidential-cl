# EUCR

Evidential Uncertainty Channel Regularisation: a Dempster-Shafer prototype head
on a 1D ResNet, with per-task DS heads and a MAS-style consolidation anchor.

## Documents

| file | what it is |
|---|---|
| `eucr-trainability.tex` / `.pdf` | The design note. Architecture, the measurement and numerical faults that were fixed, and an accounting of which parts of the method contribute. Start here. |
| `preregistration.md` | Decision rules for Gate B0, PR-E1 and Gate B1, fixed before running, with amendments and results recorded. |
| `initial_ideas.md` | Original sketch, superseded. |

## Headline

Four tasks (`t2-rcn,t0-rcn,t0-deeprad,t1-deeprad`), 10 epochs, 10k samples/task,
seeds 0/39/55, batch-statistic evaluation.

| method | diagonal F1 | final F1 | BWT |
|---|---|---|---|
| LwF | 0.7069 ± 0.0055 | **0.6602 ± 0.0170** | −0.047 |
| EWC | 0.7113 ± 0.0034 | 0.4270 ± 0.0251 | −0.284 |
| EUCR (proximal, λ=5000) | 0.6019 ± 0.0042 | 0.5932 ± 0.0038 | −0.009 |
| EUCR, no anchor | 0.7173 ± 0.0068 | 0.3261 ± 0.0193 | −0.391 |

EUCR beats EWC on final F1 and loses to LwF; its retention is the best on the
table and its plasticity is its weakness.

## Four things to know before running it

1. **Evaluation protocol.** Every `ResNet1D`-based model here evaluates with
   *batch* normalisation statistics, because `ResNet1D.forward` forces train mode
   via `bn_training=True`. EUCR had no such wrapper and was scored with frozen
   running statistics — worth 0.31 final F1. `--eucr_bn_stats` now defaults to
   `batch` to match. Any reported number is transductive; say so.
2. **Use the proximal anchor.** The loss form is outside its stability window at
   every λ (Ω median 2.7e-4, max 1.7e4; the anchor gradient outweighs the task
   gradient 60x and the global clip then scales the task signal to 1.6%).
   `--eucr_anchor_mode proximal` is now the config default. Its useful λ is
   ~1/(2·lr) upward, i.e. 50–5000, not 0.09.
3. **The normalisation sensitivity is not the head's.** Gate B1 held the
   backbone fixed and swapped only the readout: a post-hoc linear probe on the
   same post-LayerNorm features loses *slightly more* to running statistics than
   the DS head does (gap 0.587 vs 0.556, 0/5 seeds favouring the head
   hypothesis). Amendment 2's `<= 0.013` EWC figure confounds head with
   backbone; do not cite it as evidence about heads.
4. **The evidential content is not what makes it work.** Compared as frontiers, a
   fixed random linear readout on the same stage features traces the same
   trade-off curve as the DS readout. Per-coordinate weighting is worth +0.017
   over uniform Ω; none of it needs to be learned or evidential.

## Reproducing

```bash
main.py --config configs/base.yaml --config configs/models/til/eucr.yaml \
  --task-order-files "t2-rcn,t0-rcn,t0-deeprad,t1-deeprad" \
  --samples_per_task 10000 --n_epochs 10 --expt_name <arm>
```

~2 min/arm, several fit concurrently on one A6000. Every invocation is a
three-seed sweep: read all three per-seed `results.txt`, never seed 0 alone.

## Ablation controls worth knowing about

`--eucr_uncertainty uniform` (Ω=1, plain L2-SP) and `random_proj` (MAS with a
fixed random head on the same stage features) are the two controls that establish
point 3 above. `--eucr_distance_metric euclidean` and `--use_groupnorm` both
destroy the model (F1 ~0.098); cosine and BatchNorm are load-bearing.
