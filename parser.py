# coding=utf-8
import os
import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

import yaml


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

    # experiment parameters
    parser.add_argument("--cuda", default=True, action="store_true", help="Use GPU")
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
    """Apply YAML overrides from the provided config files to the namespace."""

    for path in config_paths:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        for key, value in data.items():
            if key == "glances":
                setattr(args, "inner_steps", value)
                continue
            if key == "update_steps":
                setattr(args, "inner_steps", value)
                continue
            if hasattr(args, key):
                setattr(args, key, value)
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
