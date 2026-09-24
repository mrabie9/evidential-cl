# Model Hyperparameters

The table below lists the hyperparameters that are explicitly defined for each model-specific configuration dataclass under `model/`. Generic architectural toggles such as `arch`, `n_layers`, `input_channels`, and related infrastructure flags are intentionally omitted per the request.

| Model | Hyperparameters (default) | Defined In |
| --- | --- | --- |
| `agem` | `lr` (0.001), `memory_strength` (0.0), `inner_steps` (1), `memories` (5120), `n_memories` (0), `alpha_init` (0.001) | `model/agem.py` |
| `anml` | `update_lr` (0.1), `meta_lr` (0.001), `inner_steps` (10), `replay_batch_size` (20), `memories` (5120), `rln` (7), `use_old_task_memory` (true) | `model/anml.py` |
| `anml_base` | `alpha_init` (0.001) | `model/anml_base.py` |
| `bcl_dual` | `memory_strength` (1.0), `temperature` (2.0), `alpha_init` (0.001), `lr` (0.001), `beta` (1.0), `n_memories` (2000), `replay_batch_size` (20), `inner_steps` (5), `adapt_inner_steps` (5) | `model/bcl_dual.py` |
| `ctn` | `memory_strength` (0.5), `temperature` (5.0), `task_emb` (64), `lr` (0.01), `ctx_lr` (0.05), `n_memories` (50), `replay_batch_size` (20), `inner_steps` (2) | `model/ctn.py` |
| `er_ring` | `memory_strength` (1.0), `temperature` (2.0), `alpha_init` (0.001), `lr` (0.001), `n_memories` (2000), `n_memories` (0), `replay_batch_size` (20), `inner_steps` (5) | `model/er_ring.py` |
| `eralg4` | `alpha_init` (0.001), `lr` (0.001), `opt_lr` (0.1), `learn_lr` (false), `inner_steps` (1), `memories` (5120), `replay_batch_size` (20), `second_order` (false), `use_old_task_memory` (true), `cifar_batches` (3) | `model/eralg4.py` |
| `ewc` | `inner_steps` (1), `lr` (0.03), `optimizer` (sgd), `momentum` (0.0), `weight_decay` (0.0), `lamb` (1.0), `clipgrad` (100.0) | `model/ewc.py` |
| `ft` | inherits `iid2` (`inner_steps` (1), `lr` (0.001)); trains each task on its own data only | `model/ft.py` |
| `gem` | `gamma` (0.0), `inner_steps` (1), `lr` (0.001), `n_memories` (0), `alpha_init` (0.001) | `model/gem.py` |
| `hat` | `inner_steps` (1), `lr` (0.0001), `optimizer` (sgd), `gamma` (0.75), `smax` (50) | `model/hat.py` |
| `icarl` | `memory_strength` (0.0), `n_memories` (0), `inner_steps` (1), `alpha_init` (0.001), `lr` (0.001), `n_epochs` (1) | `model/icarl.py` |
| `iid2` | `inner_steps` (1), `lr` (0.001) | `model/iid2.py` |
| `lamaml_base` | `alpha_init` (0.001), `opt_wt` (0.1), `opt_lr` (0.1), `inner_steps` (1; effective total passes = `inner_steps × n_meta` from merged CLI/YAML), `memories` (5120), `replay_batch_size` (20), `use_old_task_memory` (true), `learn_lr` (false), `second_order` (false), `sync_update` (false), `cifar_batches` (3) | `model/lamaml_base.py` |
| `lwf` | `inner_steps` (1), `lr` (0.001), `optimizer` (adam), `momentum` (0.0), `weight_decay` (0.0), `clipgrad` (100.0), `temperature` (2.0), `distill_lambda` (1.0) | `model/lwf.py` |
| `meralg1` | `alpha_init` (0.001), `lr` (0.001), `replay_batch_size` (20), `memories` (5120), `batches_per_example` (1), `beta` (1.0), `gamma` (0.0) | `model/meralg1.py` |
| `meta-bgd` | `alpha_init` (0.001), `bgd_optimizer` (bgd), `mean_eta` (1.0), `std_init` (0.05), `train_mc_iters` (5), `optimizer_params` (<complex>), `inner_steps` (1), `memories` (5120), `replay_batch_size` (20), `use_old_task_memory` (true), `cifar_batches` (3) | `model/meta-bgd.py` |
| `packnet` | `inner_steps` (1), `lr` (0.01), `optimizer` (sgd), `momentum` (0.9), `weight_decay` (0.0), `clipgrad` (100.0), `prune_perc` (0.5) | `model/packnet.py` |
| `rwalk` | `inner_steps` (1), `lr` (0.001), `optimizer` (adam), `momentum` (0.0), `weight_decay` (0.0), `clipgrad` (100.0), `lamb` (1.0), `alpha` (0.9), `eps` (0.01) | `model/rwalk.py` |
| `si` | `inner_steps` (1), `lr` (0.001), `optimizer` (adam), `momentum` (0.0), `weight_decay` (0.0), `clipgrad` (100.0), `si_c` (0.1), `si_epsilon` (0.01) | `model/si.py` |
| `ucl` | `inner_steps` (1), `lr` (0.001), `lr_rho` (0.01), `beta` (0.0002), `alpha` (0.3), `ratio` (0.125), `clipgrad` (10.0), `split` (true) | `model/ucl.py` |
| `ucl_bresnet` | `inner_steps` (1), `lr` (0.001), `lr_rho` (0.01), `beta` (0.0002), `alpha` (0.3), `ratio` (0.125), `clipgrad` (10.0), `split` (true), `eval_samples` (20) | `model/ucl_bresnet.py` |

Notes:
- `lamaml_base` is consumed by both the `lamaml` and `lamaml_cifar` models; its hyperparameters therefore apply to both variants.
- `anml_base` provides the neuromodulated learner component used within `anml`, so its `alpha_init` value augments the hyperparameters listed for `anml`.

## High-Impact Hyperparameters

1. **Learning rates (`lr`, `opt_lr`, `opt_wt`, `meta_lr`, `update_lr`)** – dominate optimization stability and convergence speed across all baselines and meta-learners.
2. **Replay capacity terms (`memories`, `n_memories`, `n_memories`, `n_memories`)** – directly bound how much past data a method can leverage, which strongly affects forgetting.
3. **Constraint/regularization strengths (`memory_strength`, `memory_strength`, `lamb`, `si_c`, `ratio`)** – govern how aggressively each method preserves prior knowledge versus adapting to new tasks.
4. **Per-parameter LR initialization (`alpha_init`)** – critical for learning-to-learn methods (La-MAML, ANML, MER, etc.) because it scales the entire fast adaptation process.
5. **Temperature-like controls (`temperature`, `temperature`, `temperature`)** – modulate soft targets/logits and can markedly change distillation behavior or context modulation sharpness.
6. **Meta-iteration counts** – `inner_steps` counts alternating fast/meta rounds per `observe` for CTN and BCL-Dual (legacy YAML may still set `inner_steps` × `n_meta`; it is folded into one `inner_steps`). BCL-Dual `adapt_inner_steps` controls SGD depth in `adapt()`. La-MAML exposes a single effective `inner_steps` (product of `inner_steps` × `n_meta` when loading `LamamlBaseConfig`).
7. **Replay sampling size (`replay_batch_size`, `batches_per_example`)** – affects the gradient signal from memory and the balance between new and replayed data each update.
