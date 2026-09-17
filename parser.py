# coding=utf-8
import os
import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import yaml

# YAML keys that are applied under a different argparse dest. Historical spellings
# of the inner-loop step count, kept so old configs keep working.
CONFIG_KEY_ALIASES: Dict[str, str] = {
    "glances": "inner_steps",
    "update_steps": "inner_steps",
}

# YAML keys that are deliberately inert: they document the resulting behaviour
# for a reader but must NOT be wired to an argument, because something else
# decides the value. Anything here needs a comment saying what that something is.
INTENTIONALLY_UNUSED_CONFIG_KEYS: Set[str] = {
    # model.ucl_bresnet._infer_ucl_split_from_loader derives `split` from the
    # loader name (CIL -> concatenated heads, TIL -> per-task heads) and
    # overwrites whatever the config says. Registering it would let a config
    # appear to set something it cannot.
    "split",
}


def get_parser():
    parser = argparse.ArgumentParser(description="Continual learning")
    parser.add_argument(
        "--expt_name", type=str, default="test_lamaml", help="name of the experiment"
    )

    # model details
    parser.add_argument(
        "--model", type=str, default="lamaml_cifar", help="algo to train"
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet1d",
        help="arch to use for training",
        choices=["resnet1d"],
    )
    parser.add_argument(
        "--n_hiddens",
        type=int,
        default=100,
        help="number of hidden neurons at each layer",
    )
    parser.add_argument(
        "--n_layers", type=int, default=2, help="number of hidden layers"
    )
    parser.add_argument(
        "--xav_init",
        default=False,
        action="store_true",
        help="Use xavier initialization",
    )

    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Debug mode with more frequent logging and smaller data splits",
    )
    parser.add_argument(
        "--use_detector_arch",
        default=False,
        action="store_true",
        help="Enable the detector architecture; when disabled, treat -1 class labels as an extra task class.",
    )
    parser.add_argument(
        "--use_groupnorm",
        default=False,
        action="store_true",
        help="Use GroupNorm in compatible backbones instead of BatchNorm.",
    )
    parser.add_argument(
        "--gem_disable_qp",
        default=False,
        action="store_true",
        help="Ablation: disable GEM's QP gradient-projection constraint (and the past-task "
        "replay-gradient pass it feeds); reduces GEM to plain fine-tuning at matched buffer size.",
    )
    parser.add_argument(
        "--ctn_disable_film",
        default=False,
        action="store_true",
        help="Ablation: disable CTN's task-embedding->FiLM modulation (use base features only), "
        "isolating the FiLM head's contribution to BWT.",
    )
    parser.add_argument(
        "--ctn_disable_distill",
        default=False,
        action="store_true",
        help="Ablation: disable CTN's KL-distillation replay term (loss3 against frozen soft "
        "targets); plain replay CE is kept. Isolates distillation's contribution to BWT.",
    )
    parser.add_argument(
        "--gem_margin",
        default=0.5,
        type=float,
        help="QP margin for the GEM gradient-projection constraint in ctn_gem (B2). Kept "
        "separate from CTN's memory_strength, which is the KL-distillation weight.",
    )
    parser.add_argument(
        "--gembob_dynamic_ring",
        action="store_true",
        help="gem_bob: fully-utilised ring buffer (ablation row E2). Re-splits the replay "
        "budget across only the tasks seen so far instead of pre-partitioning into n_tasks "
        "fixed 1/T slices, so the buffer is always full; converges to the same final split. "
        "Ported from --er_dynamic_ring.",
    )
    parser.add_argument(
        "--gembob_distill",
        action="store_true",
        help="gem_bob: KL distillation on frozen per-task soft targets (ablation rows B1/T2), "
        "weighted by --distill_lambda at --temperature. Gated separately from "
        "--distill_lambda because that flag defaults to 1.0, which would otherwise switch "
        "distillation on in the add-one baseline.",
    )
    parser.add_argument(
        "--gembob_bilevel",
        action="store_true",
        help="gem_bob: bilevel inner/outer round (ablation row B5b). Each round takes an "
        "inner step on the training objective (QP-projected) followed by an outer step on a "
        "held-out validation buffer (unprojected), then a Reptile interpolation at --beta. "
        "Costs 2 SGD steps per round: budget-match by halving --inner_steps.",
    )
    parser.add_argument(
        "--gembob_val_memories",
        default=512,
        type=int,
        help="gem_bob: total held-out validation budget for the --gembob_bilevel outer step, "
        "split evenly across tasks. Rows are removed from the training stream, not copied, "
        "so the validation buffer stays disjoint.",
    )
    parser.add_argument(
        "--gembob_meta_batches",
        default=1,
        type=int,
        help="gem_bob: meta-batch averaging (ablation row M3). Accumulates the current-batch "
        "CE gradient over K chunks before a single projected step, so it is budget-neutral. "
        "1 disables it (single full-batch pass). Separate from --meta_batches, which defaults "
        "to 3 and would otherwise switch this on in the add-one baseline.",
    )
    parser.add_argument(
        "--gem_lwf",
        action="store_true",
        help="gem_distill: add a Learning-without-Forgetting term — KL over previous-task "
        "classes on the CURRENT batch vs a frozen teacher snapshot (taken at each task "
        "boundary), weighted by --gem_lwf_lambda at --temperature. Unlike --distill_lambda "
        "(on buffer samples) this acts on the new task's data, testing whether it stacks "
        "with on-buffer distillation. Composable with distillation; subject to GEM's QP.",
    )
    parser.add_argument(
        "--gem_lwf_lambda",
        default=1.0,
        type=float,
        help="Weight of the gem_distill LwF term when --gem_lwf is set.",
    )
    parser.add_argument(
        "--balanced_replay",
        default=False,
        action="store_true",
        help="Use class-balanced reservoir sampling (CBRS) for the gem_distill buffer instead "
        "of the per-task FIFO ring, so replay/distillation are not dominated by the noise/"
        "frequent classes (A1).",
    )
    parser.add_argument(
        "--balance_signal_only",
        default=False,
        action="store_true",
        help="With --balanced_replay: exclude the noise/detection class from balancing so it "
        "stays at its natural rate (avoids starving detection); only signal classes are balanced.",
    )

    # optimizer parameters influencing all models
    parser.add_argument(
        "--inner_steps",
        default=1,
        type=int,
        help=(
            "Inner optimization passes per observe call: multi-pass training (ex-glances), "
            "alternating fast/meta rounds for CTN and BCL-Dual, ANML inner updates (ex-update_steps). "
            "La-MAML uses the effective total pass count (see LamamlBaseConfig: inner_steps × n_meta "
            "from merged args for backward-compatible YAML). CTN/BCL-Dual fold legacy "
            "inner_steps × n_meta from YAML into a single inner_steps count."
        ),
    )
    parser.add_argument(
        "--n_epochs", type=int, default=1, help="Number of epochs per task"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="the amount of items received by the algorithm at one time (set to 1 across all "
        + "experiments). Variable name is from GEM project.",
    )
    parser.add_argument(
        "--replay_batch_size",
        type=float,
        default=20,
        help="The batch size for experience replay.",
    )
    parser.add_argument(
        "--memories",
        type=int,
        default=5120,
        help="number of total memories stored in a reservoir sampling based buffer",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate (For baselines)"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        help="optimizer name for models that support switching",
    )
    parser.add_argument(
        "--prune_perc",
        type=float,
        default=0.75,
        help=(
            "PackNet: fraction of currently free (unowned) weights to drop after each task; "
            "the complement is kept and assigned to the completed task."
        ),
    )
    parser.add_argument(
        "--post_prune_epochs",
        type=int,
        default=0,
        help=(
            "PackNet: full passes over the task train loader after packing for optional finetune; "
            "gradients only on weights newly assigned to that task. 0 disables."
        ),
    )
    parser.add_argument(
        "--no_class_weighted_ce",
        dest="class_weighted_ce",
        action="store_false",
        help=(
            "Disable inverse-frequency class weights in cross-entropy "
            "(default: weighted CE matches ucl_bresnet minibatch weighting)."
        ),
    )
    parser.set_defaults(class_weighted_ce=True)
    parser.add_argument(
        "--eralg4_masked_loss",
        action="store_true",
        help="eralg4 (ER-reservoir): apply per-sample TIL/CIL logit masking in "
        "the training loss (as er_ring and C-MAML do). Now the DEFAULT; this "
        "flag is kept for script compatibility.",
    )
    parser.add_argument(
        "--eralg4_unmasked_loss",
        dest="eralg4_masked_loss",
        action="store_false",
        help="eralg4: ablation switch restoring the legacy unmasked global-softmax "
        "training loss (cross-task interference; ~-6 F1 / -3 BWT in single-epoch "
        "TIL). See docs/cmaml_bcl_ablations.md.",
    )
    parser.set_defaults(eralg4_masked_loss=True)
    parser.add_argument(
        "--eralg4_joint_er",
        action="store_true",
        help="eralg4 (ER-reservoir): PROBE flag. Use the sister-repo two-forward "
        "training loop (current live batch + replay in separate forwards, adapter "
        "and backbone co-trained jointly in one opt_wt step) instead of the default "
        "concatenated single-batch forward with a decoupled manual adapter step.",
    )
    parser.set_defaults(eralg4_joint_er=False)
    parser.add_argument(
        "--eralg4_grad_avg",
        type=int,
        default=1,
        help="eralg4 (ER-reservoir): PROBE flag. Average the ER loss over K "
        "independent stochastic forward passes of the same batch before each "
        "optimizer step -- the twin of C-MAML's --meta_batches. resnet1d keeps "
        "four Dropout(p=0.2) layers active on every forward, so a single-forward "
        "gradient aligns only ~0.69 with the noise-free gradient (K=3 reaches "
        "~0.85) and its noise-inflated norm trips --grad_clip_norm every step. "
        "Tests whether C-MAML's TIL edge over Res-ER is just this variance "
        "reduction. K=1 (default) is the historical behaviour.",
    )
    parser.add_argument(
        "--eralg4_lwf_lambda",
        type=float,
        default=0.0,
        help="eralg4 (ER-reservoir): weight on a Learning-without-Forgetting "
        "logit-distillation term (temperature-scaled KL against a frozen "
        "end-of-task teacher, over the columns of already-completed tasks) "
        "evaluated on the CURRENT task's incoming batch. 0 (default) disables "
        "it. Distinct from --er_distill, which distills the REPLAY rows inside "
        "each row's own task class slice: LwF needs no buffer and constrains "
        "the function where the new data actually is. Composable with "
        "--er_distill; both share the one end-of-task teacher snapshot. Runs "
        "through the shared model/lwf_regulariser.py used by si / rwalk / "
        "woe_si, so the numerics match model.lwf.",
    )
    parser.add_argument(
        "--eralg4_lwf_temperature",
        type=float,
        default=5.0,
        help="eralg4: softmax temperature for --eralg4_lwf_lambda. Matches "
        "model.lwf's default of 5.0.",
    )
    parser.add_argument(
        "--cmaml_joint_er",
        action="store_true",
        help="C-MAML / La-MAML (lamaml_cifar): PROBE flag, twin of "
        "--eralg4_joint_er. Split the meta-loss forward into two separate "
        "net.forward passes (replay rows and current rows) instead of the "
        "default single forward over the concatenated getBatch batch, so "
        "backbone BatchNorm normalizes replay and current rows with their own "
        "statistics. Logits are concatenated and scored with the identical mask "
        "+ single CE, so the ONLY change is the forward split -- isolating the "
        "concat BN-mixing retention effect (see eralg4-bn-mixing memory / "
        "docs/reduction_experiments.md).",
    )
    parser.set_defaults(cmaml_joint_er=False)
    parser.add_argument(
        "--cmaml_replay_loss_mode",
        choices=["split", "split_norm", "pooled"],
        default="split",
        help="C-MAML / La-MAML (lamaml_cifar): how the meta loss combines the "
        "replay and current blocks of the getBatch batch. 'split' (DEFAULT) "
        "scores the two blocks separately and returns "
        "current + --memory_loss_lambda * replay, matching eralg4's "
        "_weighted_multitask_loss and every other replay model in the repo, so "
        "the replay share is pinned at 1:1. 'pooled' is the legacy single CE "
        "over the concatenated rows: because the inverse-frequency class weights "
        "are computed over that pooled batch and old-task classes are rare in it, "
        "replay's share of the loss ESCALATES with task count (measured 0.34 at "
        "task 0 to 0.85 by task 9), costing ~3 F1 of plasticity in single-epoch "
        "TIL -- keep it only to reproduce runs logged before 2026-07-25. "
        "'split_norm' divides 'split' by (1 + memory_loss_lambda), pinning the "
        "share without doubling the loss scale. See docs/cmaml_vs_reser_til.md.",
    )
    parser.add_argument(
        "--gem_replay",
        action="store_true",
        help="GEM: add an ER-style replay CE on sampled buffer rows to the "
        "current-task loss (weighted by --gem_replay_lambda, batch size "
        "--replay_batch_size), in addition to the QP constraints those same "
        "memories define. The QP projection acts on the combined gradient. "
        "Tests whether training on the buffer stacks with constraining on it.",
    )
    parser.add_argument(
        "--gem_replay_lambda",
        type=float,
        default=1.0,
        help="Weight of the GEM replay CE term when --gem_replay is set.",
    )

    # experiment parameters
    parser.add_argument("--cuda", default=True, action="store_true", help="Use GPU")
    parser.add_argument(
        "--save_checkpoints",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Save per-task model checkpoints (use --no-save_checkpoints to disable).",
    )
    parser.add_argument(
        "--amp",
        dest="amp",
        action="store_true",
        help="Enable automatic mixed precision during training on CUDA.",
    )
    parser.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        help="Disable automatic mixed precision during training.",
    )
    parser.set_defaults(amp=True)
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="Autocast dtype when AMP is enabled.",
    )
    parser.add_argument(
        "--cudnn_benchmark",
        dest="cudnn_benchmark",
        action="store_true",
        help="Enable cuDNN benchmark mode for potentially faster convolutions.",
    )
    parser.add_argument(
        "--no-cudnn-benchmark",
        dest="cudnn_benchmark",
        action="store_false",
        help="Disable cuDNN benchmark mode.",
    )
    parser.set_defaults(cudnn_benchmark=True)
    parser.add_argument("--seed", type=int, default=0, help="random seed of model")
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,39,55",
        help=(
            "Comma-separated list of random seeds to sweep. When more than one "
            "seed is given, main.py re-invokes itself once per seed (fresh "
            "process each). Ignored when --single-seed is set."
        ),
    )
    parser.add_argument(
        "--single-seed",
        action="store_true",
        help=(
            "Run a single seed (the value of --seed), ignoring --seeds. "
            "Reproduces the legacy single-run behavior."
        ),
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default="",
        help=(
            "Internal: shared run timestamp passed from the multi-seed launcher "
            "to each child so all seeds group under one experiment directory. "
            "Not normally set by users."
        ),
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=100,
        help="frequency of checking the validation accuracy, in minibatches",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/",
        help="the directory where the logs will be saved",
    )
    parser.add_argument("--tf_dir", type=str, default="", help="(not set by user)")
    parser.add_argument(
        "--calc_test_accuracy",
        default=False,
        action="store_true",
        help="Calculate test accuracy along with val accuracy",
    )
    parser.add_argument(
        "--state_logging",
        default=False,
        action="store_true",
        help="Print high-level state messages to stdout for debugging",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help=(
            "Resume an interrupted experiment from its task checkpoints instead of "
            "starting from task 0. Provide the experiment log directory (its "
            "`checkpoints/` folder is reused and training continues after the latest "
            "`task_<i>.pt`) or the path to a specific `task_<i>.pt` checkpoint."
        ),
    )
    parser.add_argument(
        "--resume_task",
        type=int,
        default=None,
        help=(
            "Override the task index to resume training at. Loads `task_<resume_task-1>.pt` "
            "and trains tasks `resume_task` onward. Defaults to one past the latest "
            "available checkpoint. Ignored unless --resume is set."
        ),
    )

    # data parameters
    parser.add_argument(
        "--data_path",
        default="data/tiny-imagenet-200/",
        help="path where data is located",
    )
    parser.add_argument(
        "--task-order-files",
        dest="task_order_files",
        type=str,
        default="",
        help=(
            "Comma-separated list of IQ .npz file names or stems defining the task order. "
            "When provided, overrides the default alphabetical file order for IQ datasets."
        ),
    )
    parser.add_argument(
        "--loader",
        type=str,
        default="task_incremental_loader",
        help="data loader to use",
    )
    parser.add_argument(
        "--samples_per_task",
        type=int,
        default=-1,
        help="training samples per task (all if negative)",
    )
    parser.add_argument(
        "--task-order-seed",
        dest="task_order_seed",
        type=int,
        default=None,
        help=(
            "When set, randomly permute task presentation order after resolving "
            "--task-order-files / default alphabetical order, using this seed via "
            "numpy.random.Generator (independent of --seed). Omit for the base order."
        ),
    )
    parser.add_argument(
        "--classes_per_it", type=int, default=4, help="number of classes in every batch"
    )
    parser.add_argument(
        "--iterations", type=int, default=5000, help="number of classes in every batch"
    )
    parser.add_argument(
        "--dataset",
        default="tinyimagenet",
        type=str,
        help="Dataset to train and test on.",
    )
    parser.add_argument(
        "--workers",
        default=3,
        type=int,
        help="Number of workers preprocessing the data.",
    )
    parser.add_argument(
        "--validation",
        default=0.0,
        type=float,
        help="Validation split (0. <= x <= 1.).",
    )
    parser.add_argument(
        "--data_scaling",
        default="none",
        type=str,
        choices=["none", "normalize", "standardize"],
        help=(
            "Apply scaling to IQ data: 'normalize' uses min/max scaling and "
            "'standardize' applies z-score based on training data."
        ),
    )
    parser.add_argument(
        "--snr_range",
        dest="snr_range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN_DB", "MAX_DB"),
        help=(
            "Inclusive SNR window (in dB) used to filter samples for datasets that "
            "carry a per-sample SNR label: deeprad's 'lbl_tr'/'lbl_te' column 0 and "
            "uclresm's 'snr_db_tr'/'snr_db_te'. Only samples with "
            "MIN_DB <= SNR <= MAX_DB are retained. Files without an SNR label "
            "(e.g. rcn) are left unfiltered. Omit to disable filtering."
        ),
    )
    parser.add_argument(
        "--use_iq_aug_features",
        default=False,
        action="store_true",
        help=(
            "When enabled, append exactly one derived IQ channel at model input "
            "time: I**2 + Q**2 (power) or I*Q (cross)."
        ),
    )
    parser.add_argument(
        "--iq_aug_feature_type",
        type=str,
        default="power",
        choices=["power", "cross"],
        help="When `--use_iq_aug_features` is enabled, select which derived IQ "
        "channel to append: `power` => I**2 + Q**2, `cross` => I*Q.",
    )
    parser.add_argument(
        "-order",
        "--class_order",
        default="old",
        type=str,
        help="define classes order of increment ",
        choices=["random", "chrono", "old", "super"],
    )
    parser.add_argument(
        "-inc",
        "--increment",
        default=5,
        type=int,
        help="number of classes to increment by in class incremental loader",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=100000,
        help="batch size to use during testing.",
    )
    parser.add_argument(
        "--nc_per_task",
        type=int,
        default=None,
        help="number of classes per task (uniform). Ignored if nc_per_task_list is provided.",
    )
    parser.add_argument(
        "--nc_per_task_list",
        type=str,
        default="",
        help="comma-separated class counts per task (overrides nc_per_task)",
    )
    parser.add_argument(
        "--val_rate", type=int, default=10, help="frequency (in epochs) of validation"
    )

    # La-MAML parameters
    parser.add_argument(
        "--opt_lr", type=float, default=1e-1, help="learning rate for LRs"
    )
    parser.add_argument(
        "--opt_wt", type=float, default=1e-1, help="learning rate for weights"
    )
    parser.add_argument(
        "--alpha_init", type=float, default=1e-3, help="initialization for the LRs"
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.0,
        help="Momentum used by La-MAML async per-parameter weight updates",
    )
    parser.add_argument(
        "--learn_lr",
        default=False,
        action="store_true",
        help="model should update the LRs during learning",
    )
    parser.add_argument(
        "--sync_update",
        default=False,
        action="store_true",
        help="the LRs and weights should be updated synchronously",
    )

    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=2.0,
        help="Clip the gradients by this value",
    )
    parser.add_argument(
        "--meta_batches",
        default=3,
        type=int,
        help="Number of batches in inner trajectory",
    )
    parser.add_argument(
        "--use_old_task_memory",
        default=False,
        action="store_true",
        help="Use only old task samples for replay buffer data",
    )
    parser.add_argument(
        "--second_order",
        default=False,
        action="store_true",
        help="use second order MAML updates",
    )

    # memory parameters for GEM | AGEM | ICARL
    parser.add_argument(
        "--n_memories",
        type=int,
        default=5120,
        help="total replay-buffer capacity across all tasks",
    )
    parser.add_argument(
        "--mem_sampling",
        type=str,
        default="ring",
        choices=["ring", "reservoir"],
        help=(
            "Replay-buffer update policy (GEM | BCL-Dual): 'ring' keeps the most "
            "recent samples per task; 'reservoir' keeps a uniform random sample of "
            "the whole task stream."
        ),
    )
    parser.add_argument(
        "--memory_strength",
        default=0,
        type=float,
        help="memory strength (meaning depends on memory)",
    )
    parser.add_argument(
        "--memory_loss_lambda",
        type=float,
        default=1.0,
        help="AGEM: scales replay/memory loss regularization strength.",
    )
    parser.add_argument(
        "--er_distill",
        action="store_true",
        help="ER-ring: add a KL distillation penalty on replay samples (soft targets "
        "frozen at each task boundary), weighted by --memory_strength at --temperature. "
        "Isolates whether distillation accounts for BCL-Dual's edge over plain replay.",
    )
    parser.add_argument(
        "--er_lwf",
        action="store_true",
        help="ER-ring: add a Learning-without-Forgetting penalty — KL distillation over "
        "previous-task classes on the CURRENT batch against a frozen teacher snapshot "
        "(taken at each task boundary), weighted by --memory_strength at --temperature. "
        "Unlike --er_distill (on buffer samples) this acts on the new task's data, so it "
        "tests whether functional regularization has leverage independent of replay. "
        "Composable with --er_distill.",
    )
    parser.add_argument(
        "--er_replay_noise",
        action="store_true",
        help="ER-ring: include noise-labeled buffer samples in replay, scoring replay "
        "rows on task-masked global logits (C-MAML's meta-batch construction) instead "
        "of the task-local gather that must exclude the shared noise class. Isolates "
        "whether replaying noise explains C-MAML's lower p_fa vs plain replay.",
    )
    parser.add_argument(
        "--er_dynamic_ring",
        action="store_true",
        help="ER-ring: dynamically re-split the replay budget across only the tasks "
        "seen so far instead of pre-partitioning into n_tasks fixed slices. Task 0 "
        "occupies the whole buffer; at each task boundary every prior task is shrunk to "
        "n_memories/(seen_tasks) and the freed room is given to the new task, so the "
        "buffer is always fully utilised (eralg4-style) while converging to the same "
        "final per-task split. Isolates whether er_ring's deficit vs eralg4 is buffer "
        "under-utilisation on early tasks.",
    )
    parser.add_argument(
        "--no_bilevel",
        action="store_true",
        help="BCL-Dual: ablate the bilevel two-loop optimization. Each inner round "
        "collapses to a single fused gradient step on cls_lambda*loss1 + loss2 + loss3 "
        "(current CE + replay CE + KL distill); the separate validation-buffer outer "
        "step and the Reptile interpolation are dropped. Reduces BCL-Dual to plain "
        "experience replay + distillation. Budget-match by doubling --inner_steps "
        "(B0 takes 2 SGD steps/round). See docs/cmaml_bcl_ablations.md.",
    )
    parser.add_argument(
        "--bcl_global_reservoir",
        action="store_true",
        help="BCL-Dual: replace the per-task replay buffer with a single GLOBAL reservoir "
        "pool (eralg4/Res-ER's mechanism, ablation E0). One flat n_memories buffer admitted "
        "by Vitter reservoir over the whole stream, with a per-slot task id so distillation "
        "soft targets are still frozen per task. Combined with --val_fraction 0 (B3, no dual "
        "memory) this is the CIL 'best-of-best' combination: B3 + global reservoir.",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.2,
        help="BCL-Dual: fraction of each task's memory reserved for the validation "
        "(dual-memory) buffer used by the outer loss; 0 disables dual memory.",
    )
    parser.add_argument(
        "--steps_per_sample", default=1, type=int, help="training steps per batch"
    )

    # # parameters specific to MER
    # parser.add_argument('--gamma', type=float, default=1.0,
    #                     help='gamma learning rate parameter')
    # parser.add_argument('--beta', type=float, default=1.0,
    #                     help='beta learning rate parameter')
    # parser.add_argument('--s', type=float, default=1,
    #                     help='current example learning rate multiplier (s)')
    # parser.add_argument('--batches_per_example', type=float, default=1,
    #                     help='the number of batch per incoming example')

    # parameters specific to Meta-BGD
    parser.add_argument(
        "--bgd_optimizer",
        type=str,
        default="bgd",
        choices=["adam", "adagrad", "bgd", "sgd"],
        help="Optimizer.",
    )
    parser.add_argument(
        "--optimizer_params",
        default="{}",
        type=str,
        nargs="*",
        help="Optimizer parameters",
    )

    parser.add_argument(
        "--train_mc_iters",
        default=5,
        type=int,
        help="Number of MonteCarlo samples during training(default 10)",
    )
    parser.add_argument(
        "--std_init", default=5e-2, type=float, help="STD init value (default 5e-2)"
    )
    parser.add_argument(
        "--mean_eta", default=1, type=float, help="Eta for mean step (default 1)"
    )
    parser.add_argument("--fisher_gamma", default=0.95, type=float, help="")

    ## ANML parameters
    parser.add_argument(
        "--rln",
        type=int,
        default=7,
        help="number of hidden neurons in the representation layer",
    )
    parser.add_argument(
        "--meta_lr", type=float, default=0.001, help="outer learning rate"
    )
    parser.add_argument(
        "--update_lr", type=float, default=0.1, help="inner learning rate"
    )

    # CTN parameters
    parser.add_argument(
        "--ctx_lr", type=float, default=0.05, help="Context learning rate for CTN"
    )
    parser.add_argument(
        "--n_meta",
        type=int,
        default=1,
        help=(
            "La-MAML: folded into inner_steps (inner_steps × n_meta) in LamamlBaseConfig. "
            "CTN/BCL-Dual: legacy only—multiplied with inner_steps when loading model config "
            "to match the old nested schedule; omit or set to 1 for a single inner_steps value."
        ),
    )
    parser.add_argument(
        "--temperature", type=float, default=5, help="Temperature for CTN"
    )
    parser.add_argument(
        "--task_emb", type=int, default=64, help="Task embedding dimension for CTN"
    )

    # Parameters for HAT

    # Regularisation-based CL methods (EWC, SI, RWalk, UCL).
    #
    # These were previously read only from each learner's dataclass defaults:
    # `parser._apply_config_overrides` skips any YAML key that is not a
    # registered argument, so `si_c`, `lamb`, `alpha`, `beta`, `ratio` and
    # `lr_rho` in configs/models/til/*.yaml were silently discarded. They default
    # to None here so that an unset value still falls through to the learner's
    # own default (each `*Config.from_args` skips None), which keeps the two
    # methods that share the name `alpha` -- RWalk's Fisher EMA momentum and
    # UCL's mu-penalty strength -- from inheriting each other's default.
    parser.add_argument(
        "--anchor_mode",
        type=str,
        default="loss",
        choices=["loss", "proximal"],
        help=(
            "How EWC / SI / RWalk / UCL apply their quadratic anchor. 'loss' "
            "(default) adds it to the training loss and lets the optimiser "
            "descend it. 'proximal' applies its closed-form minimiser after the "
            "optimiser step, keeping it out of the backward pass and the "
            "gradient-norm clip budget; unconditionally stable at any importance "
            "scale. The two modes need separate penalty-strength sweeps."
        ),
    )
    parser.add_argument(
        "--si_c",
        type=float,
        default=None,
        help="SI penalty strength c (weight on the path-integral anchor).",
    )
    parser.add_argument(
        "--si_epsilon",
        type=float,
        default=None,
        help="SI damping term in the per-task importance normaliser.",
    )
    parser.add_argument(
        "--si_lwf_lambda",
        type=float,
        default=0.0,
        help=(
            "si: weight on a Learning-without-Forgetting logit-distillation term "
            "(temperature-scaled KL against a frozen end-of-task teacher on "
            "previously-seen classes). 0 (default) disables it. Orthogonal to "
            "the SI path-integral anchor -- the anchor constrains parameters, "
            "this constrains the function -- so both can be on at once. Setting "
            "this with si_c=0 gives an LwF control inside this module. Mirrors "
            "--woe_lwf_lambda."
        ),
    )
    parser.add_argument(
        "--si_lwf_temperature",
        type=float,
        default=5.0,
        help=(
            "si: softmax temperature for --si_lwf_lambda. Matches model.lwf's "
            "default of 5.0."
        ),
    )
    parser.add_argument(
        "--lamb",
        type=float,
        default=None,
        help="EWC / RWalk anchor-penalty strength lambda.",
    )
    parser.add_argument(
        "--rwalk_lwf_lambda",
        type=float,
        default=0.0,
        help=(
            "rwalk: weight on a Learning-without-Forgetting logit-distillation "
            "term (temperature-scaled KL against a frozen end-of-task teacher on "
            "previously-seen classes). 0 (default) disables it. Orthogonal to "
            "the F + s anchor -- the anchor constrains parameters, this "
            "constrains the function -- so both can be on at once. Setting this "
            "with lamb=0 gives an LwF control inside this module. Mirrors "
            "--woe_lwf_lambda."
        ),
    )
    parser.add_argument(
        "--rwalk_lwf_temperature",
        type=float,
        default=5.0,
        help=(
            "rwalk: softmax temperature for --rwalk_lwf_lambda. Matches "
            "model.lwf's default of 5.0."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="RWalk Fisher EMA momentum; UCL mu-regularisation strength.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=None,
        help="RWalk damping term in the parameter-importance score s.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Shared name, per-model meaning (None leaves each model's own default): "
        "UCL sigma-regularisation strength; BCL-Dual and gem_bob's Reptile-style "
        "meta-step amplification (new = before + (after-before)*beta), where beta=1 "
        "makes the interpolation an identity while the inner/outer two-loop structure "
        "still runs -- use --no_bilevel / --gembob_bilevel to ablate that structure.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=None,
        help="UCL initial posterior sigma as a ratio of the He init scale.",
    )
    parser.add_argument(
        "--lr_rho",
        type=float,
        default=None,
        help="UCL learning rate for the posterior rho (sigma) parameters.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="HAT mask-sparsity penalty weight; MER meta-update rate.",
    )
    parser.add_argument(
        "--smax",
        type=float,
        default=None,
        help="HAT maximum gate temperature s_max in the annealing schedule.",
    )
    parser.add_argument(
        "--distill_lambda",
        type=float,
        default=None,
        help="Shared name, per-model meaning (None leaves each model's own default): "
        "LwF's weight on the logit-distillation term; gem_distill's weight on the KL "
        "replay term added to GEM's current-task loss (0 disables it, recovering pure "
        "GEM); and, with --gembob_distill, the same term in gem_bob.",
    )
    parser.add_argument(
        "--eval_samples",
        type=int,
        default=None,
        help="UCL Monte-Carlo samples drawn per evaluation forward pass.",
    )

    # EUCR (Evidential Uncertainty Channel Regularisation) parameters
    parser.add_argument(
        "--reg_lambda",
        type=float,
        default=1000.0,
        help="EUCR consolidation penalty strength (lambda).",
    )
    parser.add_argument(
        "--probe_loss_weight",
        type=float,
        default=0.5,
        help="EUCR weight of the backbone deep-evidential-supervision (probe) loss.",
    )
    parser.add_argument(
        "--probe_stages",
        type=str,
        default="1,2,3,4",
        help="EUCR comma-separated backbone stages (1-4) that carry evidential probes.",
    )
    parser.add_argument(
        "--reg_granularity",
        type=str,
        default="channel",
        choices=["channel", "param"],
        help="EUCR granularity of evidential importance / regularisation.",
    )
    parser.add_argument(
        "--nu",
        type=float,
        default=0.9,
        help="EUCR Dempster-Shafer decision-making ignorance retention factor.",
    )
    parser.add_argument(
        "--proto_factor",
        type=int,
        default=20,
        help="EUCR number of Dempster-Shafer prototypes per class.",
    )
    parser.add_argument(
        "--kl_warmup_epochs",
        type=int,
        default=35,
        help="EUCR evidential-loss KL warm-up length (in epochs).",
    )
    parser.add_argument(
        "--importance_batches",
        type=int,
        default=None,
        help="EUCR max minibatches used for end-of-task importance estimation (all if unset).",
    )
    parser.add_argument(
        "--eucr_depth",
        type=int,
        default=18,
        choices=[18, 34],
        help="EUCR evidential ResNet-1D backbone depth.",
    )
    parser.add_argument(
        "--eucr_distance_metric",
        type=str,
        default="cosine",
        choices=["cosine", "euclidean"],
        help="EUCR Dempster-Shafer prototype distance metric (cosine is robust to "
        "LayerNorm'd features; euclidean collapses the input signal).",
    )
    parser.add_argument(
        "--eucr_uncertainty",
        type=str,
        default="both",
        choices=["nonspecificity", "discord", "both"],
        help="EUCR consolidation importance readout: nonspecificity (omega), "
        "discord (entropy of the pignistic probability), or both (DS total).",
    )
    parser.add_argument(
        "--eucr_head",
        type=str,
        default="dm",
        choices=["dm", "pignistic"],
        help="EUCR classification head: dm (expected-utility + evidential BCE) or "
        "pignistic (BetP probability + NLL, no KL warm-up).",
    )

    # WoE-SI (Weight-of-Evidence Synaptic Intelligence) parameters.
    parser.add_argument(
        "--woe_lambda",
        type=float,
        default=0.1,
        help="WoE-SI quadratic-penalty strength (analogue of SI's si_c).",
    )
    parser.add_argument(
        "--woe_xi",
        type=float,
        default=1e-3,
        help="WoE-SI damping term xi in the per-task importance normaliser.",
    )
    parser.add_argument(
        "--woe_centering_mode",
        type=str,
        default="centered_uniform",
        choices=["centered_uniform", "raw_uniform", "full_lc"],
        help=(
            "WoE-SI feature-centering / alpha scheme for the DS weights of "
            "evidence (Denoeux 2019 Eq 25/29). 'full_lc' is not implemented."
        ),
    )
    parser.add_argument(
        "--woe_mu_momentum",
        type=float,
        default=0.9,
        help="WoE-SI EMA momentum for the per-task running feature mean mu_j.",
    )
    parser.add_argument(
        "--woe_importance_stride",
        type=int,
        default=1,
        help="WoE-SI: compute the I_2 importance gradient every k optimiser steps.",
    )
    parser.add_argument(
        "--woe_conflict_weighting",
        action="store_true",
        help="WoE-SI: enable the kappa-style conflict-weighting ablation (default off).",
    )
    parser.add_argument(
        "--woe_reg_level",
        type=str,
        default="parameter",
        choices=["parameter", "channel", "output"],
        help=(
            "WoE-SI regularisation granularity / mechanism. 'parameter': per-weight "
            "SI path integral. 'channel': same path integral, omega collapsed to "
            "per-output-channel. 'output': DS evidence distillation against a frozen "
            "teacher (functional, not on the SI path-integral scale -> needs its own "
            "woe_lambda)."
        ),
    )

    parser.add_argument(
        "--woe_omega_winsorise",
        type=float,
        default=0.0,
        help=(
            "WoE-SI: cap cumulative Omega at this global quantile after each "
            "consolidation (e.g. 0.999). 0 (default) disables capping. The path "
            "integral is heavy tailed; a few outliers otherwise carry curvature "
            "the optimiser cannot integrate."
        ),
    )
    parser.add_argument(
        "--woe_anchor_mode",
        type=str,
        default="loss",
        choices=["loss", "proximal"],
        help=(
            "WoE-SI: how to apply the quadratic anchor. 'loss' adds it to the "
            "training loss (original). 'proximal' applies its closed-form "
            "minimiser after the optimiser step, keeping it out of the backward "
            "pass and the gradient-norm clip budget; unconditionally stable for "
            "any Omega. Ignored when woe_reg_level='output'."
        ),
    )

    parser.add_argument(
        "--woe_replay_mode",
        type=str,
        default="ce",
        choices=["ce", "evidence", "both"],
        help=(
            "woe_si_replay: what the reservoir contributes to the loss. 'ce' "
            "(default) rehearses stored samples with cross-entropy. 'evidence' "
            "stores each item's DS total evidence at insertion time and applies a "
            "one-sided penalty when that evidence later decays, leaving increases "
            "free. 'both' sums the two."
        ),
    )
    parser.add_argument(
        "--woe_evidence_lambda",
        type=float,
        default=1.0,
        help=(
            "woe_si_replay: weight on the evidence-decay penalty (used when "
            "woe_replay_mode is 'evidence' or 'both'). Not on the same scale as "
            "woe_replay_lambda; needs its own sweep."
        ),
    )
    parser.add_argument(
        "--woe_evidence_readout_only",
        action="store_true",
        help=(
            "woe_si_replay: detach backbone features in the evidence-decay "
            "penalty, so rehearsed items constrain only the linear readout and "
            "send no gradient into the backbone. With woe_replay_mode='evidence' "
            "the buffer becomes a pure distillation signal and the backbone is "
            "trained solely on the current task."
        ),
    )
    parser.add_argument(
        "--woe_evidence_scale",
        type=str,
        default="weight",
        choices=["weight", "belief"],
        help=(
            "Scale the functional evidence penalties are measured on -- both "
            "woe_si's woe_reg_level='output' distillation and woe_si_replay's "
            "evidence-decay hinge. "
            "'weight' (default) uses the raw weights of evidence, which are "
            "unbounded above -- a one-sided penalty on them can be satisfied by "
            "inflating the readout. 'belief' uses 1 - exp(-w/tau), the mass each "
            "channel commits, which saturates at 1 so inflation stops paying. The "
            "two scales differ by a factor of J^2 in normalisation, so "
            "woe_evidence_lambda does not transfer between them."
        ),
    )
    parser.add_argument(
        "--woe_importance_scalar",
        type=str,
        default="i2",
        choices=["i2", "z2", "phi2", "ce"],
        help=(
            "woe_si: which scalar the SI path integral tracks. 'i2' (default) "
            "is the Dempster-Shafer information content, i.e. WoE-SI proper. "
            "The rest are ablations: 'z2' squared active-logit norm, 'phi2' "
            "squared feature norm, 'ce' the task loss (= plain Synaptic "
            "Intelligence). They sit on different scales, so woe_lambda must "
            "be swept per scalar."
        ),
    )
    parser.add_argument(
        "--woe_evidence_distill_lambda",
        type=float,
        default=0.0,
        help=(
            "woe_si: weight on the DS evidence-distillation term running "
            "*alongside* a parameter anchor, instead of replacing it. This is "
            "the evidential counterpart of --woe_lwf_lambda: same frozen "
            "teacher, but the target is the per-class (w_plus, w_minus) rather "
            "than the logits. Incompatible with woe_reg_level='output', which "
            "already applies this term weighted by woe_lambda. Its scale is "
            "the J^2-normalised one, where the swept value was ~3."
        ),
    )
    parser.add_argument(
        "--woe_lwf_lambda",
        type=float,
        default=0.0,
        help=(
            "woe_si: weight on a Learning-without-Forgetting logit-distillation "
            "term (temperature-scaled KL against a frozen end-of-task teacher on "
            "previously-seen classes). 0 (default) disables it. Orthogonal to the "
            "I_2 parameter anchor -- the anchor constrains parameters, this "
            "constrains the function -- so both can be on at once. Setting this "
            "with woe_lambda=0 gives an LwF control inside this module."
        ),
    )
    parser.add_argument(
        "--woe_lwf_temperature",
        type=float,
        default=5.0,
        help=(
            "woe_si: softmax temperature for --woe_lwf_lambda. Matches "
            "model.lwf's default of 5.0."
        ),
    )
    parser.add_argument(
        "--woe_evidence_asymmetric",
        action="store_true",
        help=(
            "woe_si (woe_reg_level='output'): charge only *deterioration* of the "
            "teacher's evidence -- support for an old class falling, or evidence "
            "against it rising -- leaving improvement free, instead of the "
            "symmetric squared drift. Removes the upper arm that pinned the "
            "evidence scale, so pair it with woe_evidence_scale='belief', which "
            "is bounded; otherwise the constraint is satisfiable by inflating the "
            "readout. woe_si_replay's decay penalty is always one-sided and "
            "ignores this flag."
        ),
    )
    parser.add_argument(
        "--woe_evidence_belief_tau",
        type=float,
        default=1.0,
        help=(
            "Temperature in the belief map 1 - exp(-w/tau) (used "
            "when woe_evidence_scale='belief'). w_plus is a sum over J features "
            "and may sit on the flat tail of the curve where every drop looks "
            "negligible; set tau near the typical w_plus to move the operating "
            "point back onto the responsive region. 1.0 is the plain DS transform."
        ),
    )

    # WoE-SI + reservoir experience replay (model: woe_si_replay).
    parser.add_argument(
        "--woe_replay_memories",
        type=int,
        default=5120,
        help="woe_si_replay: reservoir buffer capacity (total stored exemplars).",
    )
    parser.add_argument(
        "--woe_replay_batch_size",
        type=int,
        default=20,
        help="woe_si_replay: number of replay exemplars drawn per optimiser step.",
    )
    parser.add_argument(
        "--woe_replay_lambda",
        type=float,
        default=1.0,
        help="woe_si_replay: weight on the reservoir-replay cross-entropy term.",
    )

    return parser


def _expanded_config_paths(config_sources: Sequence[str] | None) -> List[Path]:
    """Resolve config file and directory inputs into a concrete ordered list."""

    if not config_sources:
        return []

    paths: List[Path] = []
    for source in config_sources:
        if not source:
            continue
        path = Path(source).expanduser()
        if path.is_dir():
            candidates = list(path.glob("*.yaml")) + list(path.glob("*.yml"))
            for candidate in sorted(
                candidate for candidate in candidates if candidate.is_file()
            ):
                paths.append(candidate)
            continue
        if not path.exists():
            raise FileNotFoundError(f"Config source '{source}' does not exist")
        paths.append(path)
    return paths


def _apply_config_overrides(
    args: argparse.Namespace, config_paths: Iterable[Path]
) -> argparse.Namespace:
    """Apply YAML overrides from the provided config files to the namespace.

    A YAML key reaches a learner only if some ``add_argument`` in
    :func:`get_parser` declares that dest: the namespace is built by
    ``parser.parse_args([])`` and so contains exactly the registered dests and
    nothing else. Any other key is therefore not applicable, and this raises
    rather than skipping it. Silently dropping such keys is how
    ``configs/models/til/si.yaml``'s ``si_c: 0.4`` came to have no effect on any
    run for as long as it existed, while the file read as if it did.

    Args:
        args: Namespace of parser defaults to overwrite in place.
        config_paths: YAML files, applied in order; later files win.

    Returns:
        The same namespace, with every applicable key applied.

    Raises:
        ValueError: If any file contains a key that no argument declares and
            that is not listed in :data:`INTENTIONALLY_UNUSED_CONFIG_KEYS`.

    Usage:
        >>> _apply_config_overrides(args, [Path("configs/base.yaml")])
    """
    unrecognised: List[Tuple[str, str]] = []
    for path in config_paths:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        for key, value in data.items():
            if key in CONFIG_KEY_ALIASES:
                setattr(args, CONFIG_KEY_ALIASES[key], value)
                continue
            if hasattr(args, key):
                setattr(args, key, value)
                continue
            if key in INTENTIONALLY_UNUSED_CONFIG_KEYS:
                continue
            unrecognised.append((str(path), key))
    if unrecognised:
        listing = "\n".join(f"    {path}: {key}" for path, key in unrecognised)
        raise ValueError(
            "Config key(s) that no argparse argument declares, so they would "
            "have no effect on the run:\n"
            f"{listing}\n"
            "Register the argument in parser.get_parser(), remove the key, or "
            "add it to parser.INTENTIONALLY_UNUSED_CONFIG_KEYS with a comment "
            "explaining why it is inert."
        )
    return args


def parse_args_from_yaml(config_sources: Sequence[str] | str | None):
    """Load arguments from one or more YAML configuration files."""

    parser = get_parser()
    args = parser.parse_args([])
    if isinstance(config_sources, str) or isinstance(config_sources, os.PathLike):
        config_list: Sequence[str] = [str(config_sources)]
    else:
        config_list = config_sources or []
    config_paths = _expanded_config_paths(config_list)
    return _apply_config_overrides(args, config_paths)
