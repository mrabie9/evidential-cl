import argparse
import time
import importlib
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

import parser as file_parser
from main import (
    eval_tasks,
    save_results,
    log_state,
    _split_eval_output,
)
from utils.training_forward import (
    model_forward_for_metric_loop,
    unpack_observe_result,
)
from metrics.metrics import confusion_matrix
from utils import misc_utils
from utils.training_metrics import (
    macro_f1,
    macro_precision,
    macro_recall,
)


def _load_model_from_results_checkpoint(
    model: torch.nn.Module, results_checkpoint_path: str
) -> None:
    """Load model weights from a saved ``results.pt`` bundle.

    Args:
        model: Model instance to receive checkpoint weights.
        results_checkpoint_path: Path to a ``results.pt`` file created by
            ``main.py``/``save_results``.

    Raises:
        SystemExit: If the checkpoint path is invalid or the file structure is
            not loadable as a model checkpoint.

    Usage:
        _load_model_from_results_checkpoint(model, "logs/.../results.pt")
    """
    checkpoint_path = Path(results_checkpoint_path).expanduser()
    if not checkpoint_path.exists():
        raise SystemExit("Resume checkpoint does not exist: {}".format(checkpoint_path))

    try:
        checkpoint_bundle = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint_bundle = torch.load(checkpoint_path, map_location="cpu")

    checkpoint_state_dict = None
    if isinstance(checkpoint_bundle, (list, tuple)) and len(checkpoint_bundle) >= 3:
        checkpoint_state_dict = checkpoint_bundle[2]
    elif isinstance(checkpoint_bundle, dict):
        checkpoint_state_dict = checkpoint_bundle.get("state_dict", checkpoint_bundle)

    if not isinstance(checkpoint_state_dict, dict):
        raise SystemExit(
            "Unsupported checkpoint format at {}. Expected results.pt tuple with state_dict at index 2.".format(
                checkpoint_path
            )
        )

    model_state_dict = model.state_dict()
    filtered_state_dict = {
        key: value
        for key, value in checkpoint_state_dict.items()
        if key in model_state_dict
    }
    skipped_unexpected_keys = sorted(
        set(checkpoint_state_dict) - set(filtered_state_dict)
    )
    incompatible = model.load_state_dict(filtered_state_dict, strict=False)

    print(
        "Loaded resume checkpoint: {} (matched keys: {} / {})".format(
            checkpoint_path,
            len(filtered_state_dict),
            len(model_state_dict),
        )
    )
    if skipped_unexpected_keys:
        print(
            "Skipped {} unexpected checkpoint key(s).".format(
                len(skipped_unexpected_keys)
            )
        )
    if incompatible.missing_keys:
        print(
            "Missing {} model key(s) in checkpoint.".format(
                len(incompatible.missing_keys)
            )
        )


def _default_main_config_chain() -> List[str]:
    """Return the default YAML config chain for single-round runs.

    This mirrors the behaviour in ``main.py`` so that, if no ``--config`` is
    given, the base configuration is applied automatically.

    Returns:
        List of YAML config file paths to apply in order.

    Usage:
        chain = _default_main_config_chain()
    """
    chain: List[str] = []
    base_cfg = Path("configs/base.yaml")
    if base_cfg.exists():
        chain.append(str(base_cfg))
    legacy = Path("config_all.yaml")
    if legacy.exists():
        chain.append(str(legacy))
    return chain


def _select_task_indices_from_order(task_names: List[str], order_arg: str) -> List[int]:
    """Select task indices based on the ``task_order_files`` argument.

    The task incremental IQ loader populates ``task_names`` from the IQ
    ``.npz`` filenames (stems). This helper resolves the names listed in
    ``task_order_files`` to task indices in that list.

    Args:
        task_names: List of task name stems provided by the loader.
        order_arg: Raw ``task_order_files`` string (comma-separated).

    Returns:
        List of task indices to include in the single-round experiment. If
        ``order_arg`` is empty, all tasks are returned.

    Usage:
        indices = _select_task_indices_from_order(loader.task_names, args.task_order_files)
    """
    if not task_names:
        return []
    if not order_arg:
        return list(range(len(task_names)))

    tokens = [token.strip() for token in order_arg.split(",") if token.strip()]
    if not tokens:
        return list(range(len(task_names)))

    stem_to_index = {stem: idx for idx, stem in enumerate(task_names)}
    selected_indices: List[int] = []
    for token in tokens:
        stem = os.path.splitext(token)[0]
        if stem not in stem_to_index:
            available = ", ".join(sorted(stem_to_index.keys()))
            raise SystemExit(
                f"--task-order-files references unknown task '{token}'. "
                f"Available tasks: {available}"
            )
        idx = stem_to_index[stem]
        if idx not in selected_indices:
            selected_indices.append(idx)
    return selected_indices


def build_single_round_loaders(
    args,
    loader,
) -> Tuple[DataLoader, DataLoader, List[int]]:
    """Build train and test loaders for a single-round (non-LL) experiment.

    Tasks are selected using ``args.task_order_files`` and the loader's
    ``task_names`` attribute. If multiple tasks are selected, their datasets
    are combined into a single effective task. When this happens, the function
    prints ``\"combining....\"`` to make the behaviour explicit.

    Args:
        args: Parsed experiment arguments / configuration.
        loader: An instance of the incremental loader created from
            ``dataloaders.<loader>.IncrementalLoader``.

    Returns:
        Tuple containing:

        - train_loader: DataLoader for the combined training data.
        - test_loader: DataLoader for the combined test/validation data.
        - selected_indices: List of integer task indices that were used.

    Usage:
        train_loader, test_loader, indices = build_single_round_loaders(args, loader)
    """
    task_names = getattr(loader, "task_names", [])
    selected_indices = _select_task_indices_from_order(
        task_names, getattr(args, "task_order_files", "")
    )
    if not selected_indices:
        raise SystemExit("No tasks selected for single-round experiment.")

    # Materialize all tasks once.
    train_loaders: List[DataLoader] = []
    test_loaders: List[DataLoader] = []
    all_task_infos: List[dict] = []

    loader._current_task = 0  # Reset to first task.
    for _ in range(loader.n_tasks):
        task_info, train_loader, _, test_loader = loader.new_task()
        all_task_infos.append(task_info)
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    selected_train_datasets: List[torch.utils.data.Dataset] = []
    selected_test_datasets: List[torch.utils.data.Dataset] = []

    for task_idx in selected_indices:
        task_train_loader = train_loaders[task_idx]
        task_test_loader = test_loaders[task_idx]

        selected_train_datasets.append(task_train_loader.dataset)
        selected_test_datasets.append(task_test_loader.dataset)

    if len(selected_indices) > 1:
        print("combining....")

    combined_train_dataset = (
        ConcatDataset(selected_train_datasets)
        if len(selected_train_datasets) > 1
        else selected_train_datasets[0]
    )
    combined_test_dataset = (
        ConcatDataset(selected_test_datasets)
        if len(selected_test_datasets) > 1
        else selected_test_datasets[0]
    )

    train_loader = DataLoader(
        combined_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    test_loader = DataLoader(
        combined_test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    return train_loader, test_loader, selected_indices


def run_single_round_training(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    args,
    task_index: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, float, Dict[str, np.ndarray]]:
    """Run a non-lifelong single-round training loop for ``n_epochs``.

    This loop is intentionally similar to the per-task inner loop in
    ``life_experience`` but operates on a single (possibly combined) task.

    Args:
        model: The model to train.
        train_loader: Combined training DataLoader.
        test_loader: Combined test/validation DataLoader.
        args: Parsed experiment arguments / configuration.
        task_index: Task id passed to ``model.observe``/eval for head and CIL
            logit-mask selection. Callers combining multiple tasks (the
            default) must pass the highest task index among the combined set
            so class-incremental masking covers every combined class instead
            of collapsing to task 0's block alone.

    Returns:
        Tuple of:

        - result_val_t: Tensor of task indices (single value here).
        - result_val_a: Tensor of per-eval validation recalls.
        - time_spent: Total wall-clock time spent in seconds.

    Usage:
        result_val_t, result_val_a, time_spent = run_single_round_training(model, train_loader, test_loader, args, task_index=3)
    """
    device = torch.device(
        "cuda" if getattr(args, "cuda", False) and torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    from tqdm import tqdm  # Imported lazily to keep top-level imports minimal.

    interactive_terminal = sys.stdout.isatty()

    result_val_a: List[List[float]] = []
    result_val_t: List[int] = []
    per_epoch_losses: List[float] = []
    per_epoch_train_recalls: List[float] = []
    per_epoch_train_precisions: List[float] = []
    per_epoch_val_cls_rec: List[float] = []
    per_epoch_val_cls_prec: List[float] = []
    per_epoch_train_f1: List[float] = []
    per_epoch_val_f1: List[float] = []

    time_start = time.time()

    current_task_index = task_index

    for epoch in range(args.n_epochs):
        model.real_epoch = epoch
        epoch_losses: List[float] = []
        epoch_recalls: List[float] = []
        epoch_precisions: List[float] = []
        epoch_f1s: List[float] = []

        progress_bar = tqdm(train_loader, disable=not interactive_terminal)
        for batch in progress_bar:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                xb, yb = batch
            elif isinstance(batch, (list, tuple)) and len(batch) == 3:
                xb, yb, _ = batch
            else:
                raise ValueError("Unexpected batch structure in single-round training.")

            xb = xb.to(device)
            y_cls = yb if torch.is_tensor(yb) else torch.as_tensor(yb)

            y_for_observe = y_cls.to(device)

            model.train()
            observe_result = model.observe(xb, y_for_observe, current_task_index)
            loss, cls_tr_rec, metric_logits = unpack_observe_result(observe_result)

            epoch_losses.append(float(loss))

            if metric_logits is not None:
                predictions = torch.argmax(metric_logits, dim=1).cpu()
            else:
                model.eval()
                with torch.no_grad():
                    logits = model_forward_for_metric_loop(
                        model, xb, current_task_index, args
                    )
                    predictions = torch.argmax(logits, dim=1).cpu()
                model.train()

            y_cls_for_metric = y_cls.cpu()
            if getattr(model, "split", False):
                offset1, _ = model.compute_offsets(current_task_index)
                y_cls_for_metric = y_cls_for_metric - offset1

            precision = macro_precision(predictions, y_cls_for_metric)
            f1 = macro_f1(predictions, y_cls_for_metric)
            cls_tr_rec = macro_recall(predictions, y_cls_for_metric)

            epoch_recalls.append(float(cls_tr_rec))
            epoch_precisions.append(float(precision))
            epoch_f1s.append(float(f1))

            progress_bar.set_description(
                "Ep: {}/{} | Loss: {:.3f} | Rec: {:.3f} | Prec: {:.3f} | F1: {:.3f}".format(
                    epoch + 1,
                    args.n_epochs,
                    float(loss),
                    float(cls_tr_rec),
                    float(precision),
                    float(f1),
                )
            )

        # Validation at end of epoch on the combined test loader. Passing
        # cil_mask_upto_task keeps class-incremental masking scoped to every
        # combined task's classes, not just current_task_index's own block.
        val_loaders = [test_loader]
        val_outputs = eval_tasks(
            model, val_loaders, args, cil_mask_upto_task=current_task_index
        )
        val_acc, val_prec, val_f1 = _split_eval_output(val_outputs)
        if isinstance(val_acc, (list, tuple)):
            val_acc_values = [float(v) for v in val_acc]
            cur_val_acc = float(val_acc[0])
        else:
            val_acc_values = [float(val_acc)]
            cur_val_acc = float(val_acc)
        cur_val_f1 = None
        cur_val_prec = None
        if val_f1 is not None:
            if isinstance(val_f1, (list, tuple)):
                cur_val_f1 = float(val_f1[0])
            else:
                cur_val_f1 = float(val_f1)
        if val_prec is not None:
            if isinstance(val_prec, (list, tuple)):
                cur_val_prec = float(val_prec[0])
            else:
                cur_val_prec = float(val_prec)

        result_val_a.append(val_acc_values)
        # Single-round training is one logical round regardless of the CIL
        # mask width (current_task_index): metrics.transfer_stats derives its
        # task count from result_val_t.max()+1, so it must stay a constant 0
        # here or the confusion-matrix reduction indexes out of bounds.
        result_val_t.append(0)

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        avg_rec = float(np.mean(epoch_recalls)) if epoch_recalls else float("nan")
        avg_prec = (
            float(np.mean(epoch_precisions)) if epoch_precisions else float("nan")
        )
        avg_f1 = float(np.mean(epoch_f1s)) if epoch_f1s else float("nan")

        per_epoch_losses.append(avg_loss)
        per_epoch_train_recalls.append(avg_rec)
        per_epoch_train_precisions.append(avg_prec)
        per_epoch_val_cls_rec.append(
            cur_val_acc if cur_val_acc is not None else float("nan")
        )
        per_epoch_val_cls_prec.append(
            cur_val_prec if cur_val_prec is not None else float("nan")
        )
        per_epoch_train_f1.append(avg_f1)
        per_epoch_val_f1.append(cur_val_f1 if cur_val_f1 is not None else float("nan"))

        print(
            "Epoch {}/{} | Avg Loss {:.4f} | Avg Rec {:.4f} | Avg Prec {:.4f} | Avg F1 {:.4f} | Val Rec {} | Val Prec {:.4f} | Val F1 {:.4f}".format(
                epoch + 1,
                args.n_epochs,
                avg_loss,
                avg_rec,
                avg_prec,
                avg_f1,
                val_acc_values,
                cur_val_prec if cur_val_prec is not None else float("nan"),
                cur_val_f1 if cur_val_f1 is not None else float("nan"),
            )
        )

    finalize_fn = getattr(model, "finalize_task_after_training", None)
    if callable(finalize_fn):
        finalize_fn(train_loader)

    result_val_t_tensor = torch.as_tensor(result_val_t, dtype=torch.long)
    max_len = max(len(row) for row in result_val_a)
    padded_val = torch.full((len(result_val_a), max_len), 0.0, dtype=torch.float)
    for row_idx, row in enumerate(result_val_a):
        padded_val[row_idx, : len(row)] = torch.as_tensor(row, dtype=torch.float)

    time_spent = time.time() - time_start

    metrics_payload: Dict[str, np.ndarray] = {
        "losses": np.asarray(per_epoch_losses, dtype=float),
        "tr_macro_rec": np.asarray(per_epoch_train_recalls, dtype=float),
        "train_macro_prec": np.asarray(per_epoch_train_precisions, dtype=float),
        "val_macro_rec": np.asarray(per_epoch_val_cls_rec, dtype=float),
        "val_macro_prec_per_epoch": np.asarray(per_epoch_val_cls_prec, dtype=float),
        "train_macro_rec": np.asarray(per_epoch_train_recalls, dtype=float),
        "train_macro_f1": np.asarray(per_epoch_train_f1, dtype=float),
        "val_macro_f1_per_epoch": np.asarray(per_epoch_val_f1, dtype=float),
        # For compatibility with scripts that expect a final-task vector.
        "val_macro_f1": np.asarray(
            [per_epoch_val_f1[-1]] if per_epoch_val_f1 else [], dtype=float
        ),
    }

    return result_val_t_tensor, padded_val, time_spent, metrics_payload


def main() -> None:
    """Entry point for non-lifelong (single-round) experiments.

    This script mirrors the high-level structure of ``main.py`` but trains on
    a single (possibly combined) task for ``n_epochs`` instead of running a
    full continual-learning schedule.

    Usage:
        python main_single_round.py \\
            --config configs/base.yaml \\
            --config configs/models/rwalk.yaml \\
            --config configs/non_ll_single_round.yaml
    """
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="FILE",
        help="YAML config fragment to apply (may be provided multiple times).",
    )
    config_parser.add_argument(
        "--config-dir",
        action="append",
        default=[],
        metavar="DIR",
        help="Directory of YAML fragments to apply in alphabetical order.",
    )
    config_parser.add_argument(
        "--no-config",
        action="store_true",
        help="Skip loading YAML configs and rely solely on CLI arguments.",
    )
    config_cli, remaining = config_parser.parse_known_args()

    config_chain: List[str] = []
    if not config_cli.no_config:
        config_chain.extend(config_cli.config_dir)
        config_chain.extend(config_cli.config)
        if not config_chain:
            config_chain = _default_main_config_chain()

    base_args = file_parser.parse_args_from_yaml(config_chain or None)
    parser = file_parser.get_parser()
    parser.add_argument(
        "--resume-results-pt",
        dest="resume_results_pt",
        type=str,
        default="",
        help=(
            "Path to a previous results.pt checkpoint. When provided, load model "
            "weights from that checkpoint before continuing single-round training."
        ),
    )
    args = parser.parse_args(remaining, namespace=base_args)

    args.lr = misc_utils.scale_learning_rate_for_batch_size(args.lr, args.batch_size)
    print("Running model (single-round): ", args.model)
    log_state(
        args.state_logging,
        "Single-round experiment '{}' starting with model '{}' (seed {})".format(
            args.expt_name, args.model, args.seed
        ),
    )

    # Task presentation order follows the training seed unless --task-order-seed
    # pins it. Resolved before the loader reads it and before log_dir() records it.
    misc_utils.resolve_task_order_seed(args)

    misc_utils.init_seed(args.seed)

    Loader = importlib.import_module("dataloaders." + args.loader)
    loader = Loader.IncrementalLoader(args, seed=args.seed)
    n_inputs, n_outputs, n_tasks = loader.get_dataset_info()
    args.get_samples_per_task = getattr(loader, "get_samples_per_task", None)
    args.classes_per_task = getattr(loader, "classes_per_task", None)
    print("Classes per task:", args.classes_per_task)

    timestamp = misc_utils.get_date_time()
    config_name = Path(config_chain[-1]).stem if config_chain else None
    args.log_dir, args.tf_dir = misc_utils.log_dir(args, timestamp, config_name)
    log_state(args.state_logging, "Logging to {}".format(args.log_dir))

    Model = importlib.import_module("model." + args.model)
    model = Model.Net(n_inputs, n_outputs, n_tasks, args)
    if getattr(args, "resume_results_pt", ""):
        _load_model_from_results_checkpoint(model, args.resume_results_pt)
        log_state(
            args.state_logging,
            "Resumed model weights from {}".format(args.resume_results_pt),
        )
    if args.cuda:
        try:
            model.cuda()
        except RuntimeError:
            pass
    print(args.cuda)
    print("Model device:", next(model.parameters()).device)
    log_state(
        args.state_logging,
        "Model initialized on device {}".format(next(model.parameters()).device),
    )

    train_loader, test_loader, selected_indices = build_single_round_loaders(
        args, loader
    )
    print("Single-round using task indices:", selected_indices)
    combined_task_index = max(selected_indices)

    result_val_t, result_val_a, time_spent, metrics_payload = run_single_round_training(
        model, train_loader, test_loader, args, task_index=combined_task_index
    )

    def _safe_last(values: np.ndarray | None) -> float | None:
        """Return last finite metric value from a NumPy vector."""
        if values is None or values.size == 0:
            return None
        last_value = float(values[-1])
        if np.isnan(last_value):
            return None
        return last_value

    summary_tr_parts: List[str] = []
    summary_te_parts: List[str] = []

    train_macro_rec = _safe_last(metrics_payload.get("tr_macro_rec"))
    train_macro_prec = _safe_last(metrics_payload.get("train_macro_prec"))
    train_macro_f1 = _safe_last(metrics_payload.get("train_macro_f1"))
    if train_macro_rec is not None:
        summary_tr_parts.append("macro_rec={:.4f}".format(train_macro_rec))
    if train_macro_prec is not None:
        summary_tr_parts.append("macro_prec={:.4f}".format(train_macro_prec))
    if train_macro_f1 is not None:
        summary_tr_parts.append("macro_f1={:.4f}".format(train_macro_f1))
    if summary_tr_parts:
        print("SUMMARY_TR " + " ".join(summary_tr_parts))

    val_macro_rec = _safe_last(metrics_payload.get("val_macro_rec"))
    val_macro_prec = _safe_last(metrics_payload.get("val_macro_prec_per_epoch"))
    val_macro_f1 = _safe_last(metrics_payload.get("val_macro_f1_per_epoch"))
    if val_macro_rec is not None:
        summary_te_parts.append("macro_rec={:.4f}".format(val_macro_rec))
    if val_macro_prec is not None:
        summary_te_parts.append("macro_prec={:.4f}".format(val_macro_prec))
    if val_macro_f1 is not None:
        summary_te_parts.append("macro_f1={:.4f}".format(val_macro_f1))
    if summary_te_parts:
        print("SUMMARY_TE " + " ".join(summary_te_parts))

    # Save per-epoch metrics under the same /metrics layout used in main.py.
    logs_dir = os.path.join(args.log_dir, "metrics")
    os.makedirs(logs_dir, exist_ok=True)
    np.savez(os.path.join(logs_dir, "task0.npz"), **metrics_payload)

    # Record a human-readable task order entry for this combined run.
    task_order_path = os.path.join(logs_dir, "task_order.txt")
    task_names = getattr(loader, "task_names", None)
    if task_names and selected_indices:
        combined_name = "+".join(
            task_names[i] for i in selected_indices if 0 <= i < len(task_names)
        )
    else:
        combined_name = "task0"
    with open(task_order_path, "a", encoding="utf-8") as f_task_order:
        f_task_order.write(str(combined_name) + "\n")

    dummy_test_t = torch.empty((0,), dtype=torch.long)
    dummy_test_a = torch.empty((0, 0), dtype=torch.float)
    # Single-round training tracks precision and F1 per epoch, not per task, so
    # the per-task matrices ``save_results`` expects are empty here.
    dummy_val_prec = torch.empty((0, 0), dtype=torch.float)
    dummy_val_f1 = torch.empty((0, 0), dtype=torch.float)
    _ = confusion_matrix(
        result_val_t, result_val_a, args.log_dir, "results_single_round.txt"
    )
    save_results(
        args,
        result_val_t,
        result_val_a,
        dummy_val_prec,
        dummy_val_f1,
        dummy_test_t,
        dummy_test_a,
        model,
        time_spent,
    )
    log_state(
        args.state_logging,
        "Single-round results saved; total runtime {:.2f}s".format(time_spent),
    )

    # Print and append total runtime for this single-round experiment.
    print("Total runtime: {:.2f} seconds".format(time_spent))
    results_txt_path = os.path.join(args.log_dir, "results.txt")
    try:
        with open(results_txt_path, "a", encoding="utf-8") as results_file:
            results_file.write("total_runtime_seconds: {:.3f}\n".format(time_spent))
    except OSError:
        # If results.txt cannot be written, fail silently to avoid breaking experiments.
        pass


if __name__ == "__main__":
    print("New Single-Round Experiment Starting...")
    main()
