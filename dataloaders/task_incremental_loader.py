import os
from typing import List, Sequence, TypeVar

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloaders.idataset import DummyArrayDataset
from dataloaders.iq_data_loader import IQDataGenerator

_TaskSeqElem = TypeVar("_TaskSeqElem")


def _normalize_label_array(labels, expected_len, source):
    """Ensure label arrays align with the sample dimension.

    Some of the IQ datasets store labels in a (n_classes, n_samples) layout or
    keep an extra singleton dimension which confuses sklearn utilities during
    validation splits.  This helper reshapes the labels to a 1-D vector of
    length ``expected_len`` and converts one-hot encodings to class indices.
    """

    if labels is None:
        return labels

    arr = np.asarray(labels)
    if arr.ndim == 2 and arr.shape[0] == expected_len and arr.shape[1] == 2:
        return arr.astype(np.int64, copy=False)
    if arr.ndim == 2 and arr.shape[0] == 2 and arr.shape[1] == expected_len:
        return np.ascontiguousarray(arr.T).astype(np.int64, copy=False)
    if arr.size == expected_len:
        arr = arr.reshape(expected_len)
    else:
        arr = np.squeeze(arr)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.shape and arr.shape[0] != expected_len:
            axis = next(
                (idx for idx, size in enumerate(arr.shape) if size == expected_len),
                None,
            )
            if axis is None:
                raise ValueError(
                    f"{source} labels have shape {arr.shape}, which is incompatible with "
                    f"{expected_len} samples."
                )
            if axis != 0:
                arr = np.moveaxis(arr, axis, 0)
        if arr.ndim > 1:
            arr = arr.reshape(expected_len, -1)
            if arr.shape[1] == 1:
                arr = arr[:, 0]
            else:
                arr = np.argmax(arr, axis=1)

    return arr.astype(np.int64, copy=False)


def _compute_scaling_offset_and_scale(
    data: np.ndarray,
    scaling_mode: str,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute offset and scale arrays for data normalization.

    Args:
        data: Training samples to derive scaling statistics.
        scaling_mode: Either ``normalize`` (min/max) or ``standardize`` (z-score).
        epsilon: Small value to avoid division by zero.

    Returns:
        Tuple of (offset, scale) arrays that can be applied as
        ``(data - offset) / scale``.
    """

    if scaling_mode not in ("normalize", "standardize"):
        raise ValueError(f"Unsupported scaling_mode '{scaling_mode}'.")

    if scaling_mode == "normalize":
        minimum_values = data.min(axis=0, keepdims=True)
        maximum_values = data.max(axis=0, keepdims=True)
        scale = np.maximum(maximum_values - minimum_values, epsilon)
        offset = minimum_values
    else:
        mean_values = data.mean(axis=0, keepdims=True)
        standard_deviation = data.std(axis=0, keepdims=True)
        scale = np.maximum(standard_deviation, epsilon)
        offset = mean_values

    return offset, scale


def _apply_data_scaling(
    training_samples: np.ndarray,
    test_samples: np.ndarray,
    scaling_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply normalization or standardization to IQ samples.

    Args:
        training_samples: Training samples array.
        test_samples: Test samples array.
        scaling_mode: ``normalize`` or ``standardize``.

    Returns:
        Tuple containing scaled training and test arrays.
    """

    if scaling_mode == "none":
        return training_samples, test_samples

    if np.iscomplexobj(training_samples) or np.iscomplexobj(test_samples):
        train_real = training_samples.real
        train_imag = training_samples.imag
        test_real = test_samples.real
        test_imag = test_samples.imag

        real_offset, real_scale = _compute_scaling_offset_and_scale(
            train_real, scaling_mode
        )
        imag_offset, imag_scale = _compute_scaling_offset_and_scale(
            train_imag, scaling_mode
        )

        scaled_train = (train_real - real_offset) / real_scale
        scaled_test = (test_real - real_offset) / real_scale
        scaled_train = scaled_train + 1j * ((train_imag - imag_offset) / imag_scale)
        scaled_test = scaled_test + 1j * ((test_imag - imag_offset) / imag_scale)
        return scaled_train.astype(np.complex64, copy=False), scaled_test.astype(
            np.complex64, copy=False
        )

    offset, scale = _compute_scaling_offset_and_scale(training_samples, scaling_mode)
    scaled_training = (training_samples - offset) / scale
    scaled_test = (test_samples - offset) / scale
    return scaled_training.astype(np.float32, copy=False), scaled_test.astype(
        np.float32, copy=False
    )


def _filter_samples_by_snr(
    samples: np.ndarray | None,
    labels,
    snr_labels: np.ndarray | None,
    snr_range: Sequence[float] | None,
    source: str,
) -> tuple[np.ndarray | None, object]:
    """Keep only samples whose SNR (dB) lies within ``snr_range`` inclusive.

    Datasets carry a per-sample SNR in dB under different keys: deeprad stores it
    in the first column of its 2-D ``lbl_tr``/``lbl_te`` arrays, while uclresm stores
    a 1-D ``snr_db_tr``/``snr_db_te`` vector. This helper masks ``samples`` and
    ``labels`` down to the rows whose SNR falls inside the requested window. When
    the dataset has no SNR label (``snr_labels is None``) or no window was requested
    (``snr_range is None``), the inputs are returned unchanged so that mixed task
    orders containing non-SNR-labelled files (e.g. rcn) still load.

    Args:
        samples: Sample array with the sample dimension first.
        labels: Label array aligned to ``samples`` along axis 0.
        snr_labels: Per-sample SNR label array. When 2-D, column 0 is read as the
            SNR in dB (deeprad ``lbl_*``); a 1-D array is used directly (uclresm
            ``snr_db_*``). When ``None`` filtering is skipped for this split.
        snr_range: Inclusive ``(min_db, max_db)`` bounds, or ``None`` to disable.
        source: Human-readable tag used in log/error messages.

    Returns:
        The filtered ``(samples, labels)`` pair, or the inputs unchanged when the
        filter does not apply.

    Raises:
        ValueError: If the window removes every sample for this split.

    Usage:
        >>> import numpy as np
        >>> x = np.arange(6).reshape(3, 2)
        >>> snr = np.array([[-5.0], [3.0], [12.0]])
        >>> xf, yf = _filter_samples_by_snr(x, np.array([0, 1, 2]), snr, (0, 10), "demo")
        >>> yf.tolist()
        [1]
    """

    if snr_range is None or samples is None or snr_labels is None:
        return samples, labels

    snr_min, snr_max = float(snr_range[0]), float(snr_range[1])
    snr_values = np.asarray(snr_labels)
    if snr_values.ndim == 2:
        snr_values = snr_values[:, 0]
    snr_values = snr_values.reshape(-1)

    if snr_values.shape[0] != samples.shape[0]:
        print(
            f"[snr_filter] {source}: SNR label length {snr_values.shape[0]} does not "
            f"match {samples.shape[0]} samples; skipping SNR filter."
        )
        return samples, labels

    keep_mask = (snr_values >= snr_min) & (snr_values <= snr_max)
    kept = int(keep_mask.sum())
    if kept == 0:
        raise ValueError(
            f"[snr_filter] {source}: SNR window [{snr_min}, {snr_max}] dB removed all "
            f"{snr_values.shape[0]} samples. Widen --snr_range."
        )
    print(
        f"[snr_filter] {source}: keeping {kept}/{snr_values.shape[0]} samples "
        f"with SNR in [{snr_min}, {snr_max}] dB."
    )
    return samples[keep_mask], np.asarray(labels)[keep_mask]


def _resolve_task_file_order(all_files: Sequence[str], order_arg: str) -> List[str]:
    """Resolve IQ .npz task file ordering from a CLI argument.

    Args:
        all_files: Iterable of discovered ``.npz`` filenames in ``data_path``.
        order_arg: Raw string from ``--task-order-files`` (comma-separated).

    Returns:
        A list of filenames in the desired task order. If a non-empty
        ``order_arg`` is provided, only the referenced files are included
        and any discovered but unreferenced files are ignored (with a
        warning).

    Raises:
        SystemExit: If the argument references unknown files, contains
        duplicates, or resolves to more files than were discovered.
    """
    if not all_files:
        return []

    discovered = list(all_files)
    if not order_arg:
        # Default behaviour: natural alphabetical order (current behaviour).
        return sorted(discovered)

    tokens = [token.strip() for token in order_arg.split(",") if token.strip()]
    if not tokens:
        raise SystemExit(
            "--task-order-files was provided but no valid filenames or stems were found after parsing."
        )

    stem_to_file: dict[str, str] = {}
    name_to_file: dict[str, str] = {}
    for filename in discovered:
        name_to_file[filename] = filename
        stem, _ = os.path.splitext(filename)
        stem_to_file[stem] = filename

    ordered_files: List[str] = []
    for token in tokens:
        if token in name_to_file:
            resolved = name_to_file[token]
        else:
            stem = os.path.splitext(token)[0]
            if stem in stem_to_file:
                resolved = stem_to_file[stem]
            else:
                available_stems = ", ".join(sorted(stem_to_file.keys()))
                raise SystemExit(
                    f"--task-order-files references unknown file or stem '{token}'. "
                    f"Available stems: {available_stems}"
                )
        if resolved in ordered_files:
            raise SystemExit(
                f"--task-order-files contains a duplicate reference to '{resolved}'. "
                "Each IQ .npz file must appear exactly once."
            )
        ordered_files.append(resolved)

    num_discovered = len(discovered)
    num_requested = len(ordered_files)
    if num_requested > num_discovered:
        raise SystemExit(
            "--task-order-files lists more files than were found in data_path "
            f"(requested {num_requested}, available {num_discovered})."
        )

    if num_requested < num_discovered:
        discovered_set = set(discovered)
        ordered_set = set(ordered_files)
        ignored = sorted(discovered_set - ordered_set)
        if ignored:
            print(
                "[WARNING] Ignoring task files not listed in --task-order-files: "
                + ", ".join(ignored)
            )

    return ordered_files


def permute_task_sequence(
    base_sequence: Sequence[_TaskSeqElem],
    task_order_seed: int | None,
) -> tuple[list[_TaskSeqElem], np.ndarray | None]:
    """Deterministically reorder tasks for continual-learning experiments.

    Args:
        base_sequence: Ordered tasks after resolving ``--task-order-files`` or default sort.
        task_order_seed: When not ``None``, permute using ``numpy.random.default_rng``;
            ``None`` keeps ``base_sequence`` order.

    Returns:
        A pair ``(reordered, perm)``. When ``perm`` is not ``None``,
        ``reordered[slot] == base_sequence[int(perm[slot])]`` for each slot.

    Usage:
        >>> permute_task_sequence(("only",), 123)[0]
        ['only']
    """

    items = list(base_sequence)
    num_tasks = len(items)
    if task_order_seed is None or num_tasks <= 1:
        return items, None
    rng = np.random.default_rng(task_order_seed)
    perm = rng.permutation(num_tasks)
    reordered = [items[int(perm[slot])] for slot in range(num_tasks)]
    return reordered, perm


def _maybe_move_sample_axis(x, y, source):
    """Ensure samples are the first dimension to match label length."""
    if x is None or not hasattr(x, "ndim"):
        return x
    if x.ndim < 2:
        return x
    if y is None:
        return x
    y_arr = np.asarray(y)
    if y_arr.ndim == 0:
        return x
    sample_len = y_arr.shape[0]
    if x.shape[0] == sample_len:
        return x
    if x.ndim >= 2 and x.shape[1] == sample_len:
        return np.moveaxis(x, 1, 0)
    if x.ndim >= 3 and x.shape[2] == sample_len:
        return np.moveaxis(x, 2, 0)
    return x


class IncrementalLoader:

    def __init__(
        self,
        args,
        shuffle=False,
        seed=1,
    ):
        self._args = args
        validation_split = args.validation
        increment = args.increment

        self.classes_per_task = []
        self.task_names: list[str] = []
        self._setup_data(
            class_order_type=args.class_order,
            seed=seed,
            increment=increment,
            validation_split=validation_split,
        )

        self._current_task = 0

        self._batch_size = args.batch_size
        self._test_batch_size = args.test_batch_size
        self._workers = args.workers
        self._shuffle = shuffle

        self._setup_test_tasks()

    @property
    def n_tasks(self):
        return len(self.test_dataset)

    def new_task(self):
        if self._current_task >= len(self.test_dataset):
            raise Exception("No more tasks.")

        p_tr, p_te = self.sample_permutations[self._current_task]
        print(
            f"Task {self._current_task}: {len(p_tr)} training samples, {len(p_te)} test samples."
        )
        x_train, y_train = (
            self.train_dataset[self._current_task][1][p_tr],
            self.train_dataset[self._current_task][2][p_tr],
        )
        x_test, y_test = (
            self.test_dataset[self._current_task][1][p_te],
            self.test_dataset[self._current_task][2][p_te],
        )

        train_loader = self._get_loader(x_train, y_train, mode="train")
        test_loader = self._get_loader(x_test, y_test, mode="test")

        task_name = None
        if 0 <= self._current_task < len(self.task_names):
            task_name = self.task_names[self._current_task]

        task_info = {
            "min_class": 0,
            "max_class": self.n_outputs,
            "increment": -1,
            "task": self._current_task,
            "task_name": task_name,
            "max_task": len(self.test_dataset),
            "n_train_data": len(x_train),
            "n_test_data": len(x_test),
        }

        self._current_task += 1

        return task_info, train_loader, None, test_loader

    def _setup_test_tasks(self):
        self.test_tasks = []
        for i in range(len(self.test_dataset)):
            # .append(x, y, mode="test")
            self.test_tasks.append(
                self._get_loader(
                    self.test_dataset[i][1], self.test_dataset[i][2], mode="test"
                )
            )

    def get_tasks(self, dataset_type="test"):
        """Return a list of DataLoaders, one per task, for use by eval_tasks."""
        if dataset_type == "test":
            dataset = self.test_dataset
            perm_index = 1
        elif dataset_type == "val":
            dataset = self.test_dataset
            perm_index = 1
        elif dataset_type == "train":
            dataset = self.train_dataset
            perm_index = 0
        else:
            raise NotImplementedError("Unknown mode {}.".format(dataset_type))

        if self._args.samples_per_task <= 0:
            if dataset_type in ("test", "val"):
                return list(self.test_tasks)
            loaders = []
            for task in dataset:
                loaders.append(
                    self._get_loader(
                        task[1],
                        task[2],
                        mode="train" if dataset_type == "train" else "test",
                    )
                )
            return loaders

        trimmed = []
        for task_id, task in enumerate(dataset):
            perms = self.sample_permutations[task_id]
            if isinstance(perms, (list, tuple)):
                perm = perms[perm_index]
            else:
                perm = perms if dataset_type == "train" else None
            if perm is None:
                trimmed.append((task[0], task[1], task[2]))
                continue
            trimmed.append((task[0], task[1][perm], task[2][perm]))
        return [
            self._get_loader(
                t[1], t[2], mode="train" if dataset_type == "train" else "test"
            )
            for t in trimmed
        ]

    def get_dataset_info(self):
        def _max_label_value(labels):
            if isinstance(labels, np.ndarray):
                arr = labels
                if arr.ndim == 2 and arr.shape[1] == 2:
                    arr = arr[:, 0]
                arr = arr[arr >= 0]
                return int(arr.max()) if arr.size else -1
            if torch.is_tensor(labels):
                tensor = labels
                if tensor.dim() == 2 and tensor.size(1) == 2:
                    tensor = tensor[:, 0]
                tensor = tensor[tensor >= 0]
                return int(tensor.max().item()) if tensor.numel() > 0 else -1
            return int(np.max(labels))

        if isinstance(self.train_dataset[0][1], np.ndarray):
            sample = self.train_dataset[0][1]
            if self._args.dataset.lower() == "iq":
                if sample.ndim == 2:
                    n_inputs = sample.shape[1] * (2 if np.iscomplexobj(sample) else 1)
                elif sample.ndim == 3:
                    if sample.shape[1] == 2:
                        n_inputs = 2 * sample.shape[2]
                    elif sample.shape[1] == 3 and sample.shape[2] % 2 == 0:
                        n_inputs = sample.shape[2]
                    else:
                        n_inputs = sample.shape[-1]
                elif sample.ndim == 4 and sample.shape[2] == 2:
                    n_inputs = 2 * sample.shape[3]
                else:
                    n_inputs = sample.shape[1] if sample.ndim > 1 else int(sample.size)
            else:
                n_inputs = sample.shape[1] * (2 if np.iscomplexobj(sample) else 1)
            n_outputs = 0
            for i in range(len(self.train_dataset)):
                n_outputs = max(n_outputs, _max_label_value(self.train_dataset[i][2]))
                n_outputs = max(n_outputs, _max_label_value(self.test_dataset[i][2]))
            self.n_outputs = n_outputs
            return n_inputs, n_outputs + 1, self.n_tasks
        else:
            n_inputs = self.train_dataset[0][1].size(1)
            n_outputs = 0
            for i in range(len(self.train_dataset)):
                n_outputs = max(n_outputs, _max_label_value(self.train_dataset[i][2]))
                n_outputs = max(n_outputs, _max_label_value(self.test_dataset[i][2]))
            self.n_outputs = n_outputs
            return n_inputs, n_outputs + 1, self.n_tasks

    def get_samples_per_task(self, task_id=None, split="train"):
        if task_id is None:
            task_id = self._current_task
        if self._args.samples_per_task > 0:
            return int(self._args.samples_per_task)
        if split not in ("train", "test"):
            raise ValueError(f"Unknown split '{split}' (expected 'train' or 'test').")
        perms = self.sample_permutations[task_id]
        if isinstance(perms, (list, tuple)):
            idx = 0 if split == "train" else 1
            return int(len(perms[idx]))
        if split == "train":
            return int(len(perms))
        test_data = self.test_dataset[task_id][1]
        if isinstance(test_data, np.ndarray):
            return int(test_data.shape[0])
        return int(test_data.size(0))

    def _get_loader(self, x, y, shuffle=False, mode="train"):
        if mode == "train":
            batch_size = self._batch_size
        elif mode == "test":
            batch_size = self._test_batch_size
        else:
            raise NotImplementedError("Unknown mode {}.".format(mode))

        if isinstance(x, np.ndarray):
            target_adc_channels = None
            if (
                str(getattr(self._args, "model", "")).lower() == "iid2"
                and str(getattr(self._args, "dataset", "")).lower() == "iq"
            ):
                target_adc_channels = 3
            dataset = IQDataGenerator(x, y, target_adc_channels=target_adc_channels)
        else:
            dataset = DummyArrayDataset(x, y)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # self._workers
        )

    def _setup_data(
        self, class_order_type=False, seed=1, increment=10, validation_split=0.0
    ):
        # FIXME: handles online loading of images
        torch.manual_seed(seed)

        data_files = [f for f in os.listdir(self._args.data_path) if f.endswith(".npz")]
        if data_files and self._args.dataset.lower() == "iq":
            data_files = _resolve_task_file_order(
                data_files, self._args.task_order_files
            )
            base_task_labels = [os.path.splitext(f)[0] for f in data_files]
            task_order_seed = getattr(self._args, "task_order_seed", None)
            data_files, task_perm = permute_task_sequence(data_files, task_order_seed)
            if task_perm is not None:
                presentation_labels = [os.path.splitext(f)[0] for f in data_files]
                print(
                    f"[task_order] task_order_seed={task_order_seed} "
                    f"base_order={base_task_labels} "
                    f"presentation_slot_to_base_index={task_perm.tolist()} "
                    f"presentation_order={presentation_labels}"
                )
            raw_datasets = []
            all_labels = []
            labels_offset = 0
            collapse_noise_across_tasks = (
                str(getattr(self._args, "model", "")).lower() == "iid2"
            )
            global_noise_label: int | None = None

            # Track human-readable task names based on file names.
            self.task_names = [os.path.splitext(f)[0] for f in data_files]

            # Load npz files and concatenate datasets
            for fname in data_files:
                data = np.load(os.path.join(self._args.data_path, fname))

                def _get(keys):
                    for k in keys:
                        if k in data:
                            return data[k]
                    return None

                x_train = _get(
                    ["x_train", "X_train", "Xtr", "xtr", "Xcv", "xcv", "x", "X"]
                )
                y_train = _get(["y_train", "Y_train", "ytr", "ycv", "y", "Y"])
                x_test = _get(["x_test", "X_test", "Xte", "xte"])
                y_test = _get(["y_test", "Y_test", "yte"])

                if x_train is not None and y_train is not None:
                    before = x_train.shape
                    x_train = _maybe_move_sample_axis(
                        x_train, y_train, f"{fname} train"
                    )
                    after = x_train.shape
                    if before != after:
                        print(f"{fname} train: moved sample axis {before} -> {after}")
                        # x_train = x_train[:,0,:]
                        # print("Dropped ADC channel from train data, new shape:", x_train.shape)
                if x_test is not None and y_test is not None:
                    before = x_test.shape
                    x_test = _maybe_move_sample_axis(x_test, y_test, f"{fname} test")
                    after = x_test.shape
                    if before != after:
                        print(f"{fname} test: moved sample axis {before} -> {after}")
                        # x_test = x_test[:,0,:]
                        # print("Dropped ADC channel from test data, new shape:", x_test.shape)

                # Optionally drop samples outside a requested SNR window. The SNR
                # source differs by dataset: deeprad stores it in column 0 of
                # lbl_tr/lbl_te, while uclresm stores a 1-D snr_db_tr/snr_db_te.
                # Applied after the sample axis has been moved to the front so that
                # samples, labels and SNR labels are all aligned along axis 0.
                snr_range = getattr(self._args, "snr_range", None)
                if snr_range is not None:
                    x_train, y_train = _filter_samples_by_snr(
                        x_train,
                        y_train,
                        _get(["lbl_tr", "lbl_train", "snr_db_tr"]),
                        snr_range,
                        f"{fname} train",
                    )
                    x_test, y_test = _filter_samples_by_snr(
                        x_test,
                        y_test,
                        _get(["lbl_te", "lbl_test", "snr_db_te"]),
                        snr_range,
                        f"{fname} test",
                    )

                if x_train is not None and y_train is not None:
                    y_train = _normalize_label_array(
                        y_train, x_train.shape[0], f"{fname} train"
                    )
                    print(
                        f"{fname} train: x={x_train.shape}, y={np.asarray(y_train).shape}"
                    )
                if x_test is not None and y_test is not None:
                    y_test = _normalize_label_array(
                        y_test, x_test.shape[0], f"{fname} test"
                    )
                    print(
                        f"{fname} test: x={x_test.shape}, y={np.asarray(y_test).shape}"
                    )

                if (
                    x_train is None
                    or y_train is None
                    or x_test is None
                    or y_test is None
                ):
                    missing = []
                    training_set = True
                    testing_set = True
                    if x_train is None:
                        missing.append("x_train")
                        training_set = False
                    if y_train is None:
                        missing.append("y_train")
                        training_set = False
                    if training_set:
                        from sklearn.model_selection import train_test_split

                        x_train, x_test, y_train, y_test = train_test_split(
                            x_train,
                            y_train,
                            test_size=validation_split,
                            random_state=42,
                            stratify=y_train,
                        )
                    if x_test is None:
                        missing.append("x_test")
                        testing_set = False
                    if y_test is None:
                        missing.append("y_test")
                        testing_set = False
                    available = ", ".join(sorted(data.keys()))
                    if not testing_set or not training_set:
                        raise ValueError(
                            f"Missing dataset entries ({', '.join(missing)}) in {fname}. "
                            f"Available keys: {available}"
                        )

                scaling_mode = getattr(self._args, "data_scaling", "none")
                if scaling_mode != "none":
                    x_train, x_test = _apply_data_scaling(x_train, x_test, scaling_mode)
                    print(f"{fname}: applied data scaling mode '{scaling_mode}'.")
                else:
                    raise ValueError(f"Unsupported data scaling mode '{scaling_mode}'.")

                size_tr = x_train.shape[0]
                size_te = (
                    min(x_test.shape[0], int(size_tr * validation_split))
                    if validation_split > 0.0
                    else x_test.shape[0]
                )
                x_test = x_test[:size_te]
                y_test = y_test[:size_te]

                if y_train.ndim == 2 and y_train.shape[1] == 2:
                    train_unique = np.unique_counts(y_train[:, 0])
                else:
                    train_unique = np.unique_counts(y_train)
                print(f"Loaded {fname}: Unique train labels: {train_unique}")

                y_train = np.asarray(y_train, dtype=np.int64)
                y_test = np.asarray(y_test, dtype=np.int64)

                print(
                    f"Noise labels ratio in {fname} train: {(y_train < 0).mean():.2f}, test: {(y_test < 0).mean():.2f}"
                )

                # Remap labels to a contiguous global range starting from 0
                if y_train.ndim == 2 and y_train.shape[1] == 2:
                    y_train_cls = y_train[:, 0]
                    y_train_det = y_train[:, 1]
                    y_test_cls = y_test[:, 0]
                    y_test_det = y_test[:, 1]
                    use_detector_arch = bool(
                        getattr(self._args, "use_detector_arch", False)
                    )
                    # print(f"Using detector architecture: {use_detector_arch}")
                    has_negatives = (y_train_cls < 0).any() or (y_test_cls < 0).any()

                    unique_labels = np.unique(y_train_cls[y_train_cls >= 0])
                    needs_remap = unique_labels.size > 0 and not np.array_equal(
                        unique_labels, np.arange(unique_labels.size)
                    )
                    y_train_cls_remap = y_train_cls.copy()
                    y_test_cls_remap = y_test_cls.copy()
                    mask_train = y_train_cls >= 0
                    mask_test = y_test_cls >= 0
                    if needs_remap:
                        y_train_cls_remap[mask_train] = (
                            unique_labels.searchsorted(y_train_cls[mask_train])
                            + labels_offset
                        )
                        y_test_cls_remap[mask_test] = (
                            unique_labels.searchsorted(y_test_cls[mask_test])
                            + labels_offset
                        )
                    else:
                        y_train_cls_remap[mask_train] = (
                            y_train_cls[mask_train] + labels_offset
                        )
                        y_test_cls_remap[mask_test] = (
                            y_test_cls[mask_test] + labels_offset
                        )
                    extra_class = 0
                    if (not use_detector_arch) and has_negatives:
                        if collapse_noise_across_tasks:
                            # Keep "noise-only" samples as -1 for now; we assign a single
                            # shared label across all tasks after loading all datasets.
                            y_train_cls_remap[~mask_train] = -1
                            y_test_cls_remap[~mask_test] = -1
                        else:
                            # Legacy behaviour: per-task noise label at the end of this task's range.
                            extra_class = 1
                            neg_label = labels_offset + unique_labels.size
                            y_train_cls_remap[~mask_train] = neg_label
                            y_test_cls_remap[~mask_test] = neg_label
                    if use_detector_arch:
                        y_train = np.stack([y_train_cls_remap, y_train_det], axis=1)
                        y_test = np.stack([y_test_cls_remap, y_test_det], axis=1)
                    else:
                        y_train = y_train_cls_remap
                        y_test = y_test_cls_remap
                else:
                    use_detector_arch = bool(
                        getattr(self._args, "use_detector_arch", False)
                    )
                    has_negatives = (y_train < 0).any() or (y_test < 0).any()
                    unique_labels = np.unique(y_train[y_train >= 0])
                    needs_remap = unique_labels.size > 0 and not np.array_equal(
                        unique_labels, np.arange(unique_labels.size)
                    )
                    y_train_remap = y_train.copy()
                    y_test_remap = y_test.copy()
                    mask_train = y_train >= 0
                    mask_test = y_test >= 0
                    if needs_remap:
                        y_train_remap[mask_train] = (
                            unique_labels.searchsorted(y_train[mask_train])
                            + labels_offset
                        )
                        y_test_remap[mask_test] = (
                            unique_labels.searchsorted(y_test[mask_test])
                            + labels_offset
                        )
                    else:
                        y_train_remap[mask_train] = y_train[mask_train] + labels_offset
                        y_test_remap[mask_test] = y_test[mask_test] + labels_offset
                    extra_class = 0
                    if (not use_detector_arch) and has_negatives:
                        if collapse_noise_across_tasks:
                            # Keep "noise-only" samples as -1 for now; we assign a single
                            # shared label across all tasks after loading all datasets.
                            y_train_remap[~mask_train] = -1
                            y_test_remap[~mask_test] = -1
                        else:
                            # Legacy behaviour: per-task noise label at the end of this task's range.
                            extra_class = 1
                            neg_label = labels_offset + unique_labels.size
                            y_train_remap[~mask_train] = neg_label
                            y_test_remap[~mask_test] = neg_label
                    y_train = y_train_remap
                    y_test = y_test_remap
                if collapse_noise_across_tasks:
                    # Only advance by non-noise classes. Noise label is global.
                    labels_offset += unique_labels.size
                else:
                    labels_offset += unique_labels.size + extra_class
                if y_train.ndim == 2 and y_train.shape[1] == 2:
                    remapped = np.unique(y_train[:, 0])
                else:
                    remapped = np.unique(y_train)
                print(
                    f"Loaded {fname}: Remapped labels: {remapped}. Size: {x_train.shape[0]})"
                )

                # 3D array[task, split (xtr/yte/xte/yte), data]
                raw_datasets.append((x_train, y_train, x_test, y_test))
                if y_train.ndim == 2 and y_train.shape[1] == 2:
                    all_labels.append(y_train[:, 0].reshape(-1))
                    all_labels.append(y_test[:, 0].reshape(-1))
                else:
                    all_labels.append(y_train.reshape(-1))
                    all_labels.append(y_test.reshape(-1))

            if not raw_datasets:
                raise ValueError(
                    "No IQ datasets were loaded. Please check the data path."
                )

            self.train_dataset, self.test_dataset = [], []
            for x_train, y_train, x_test, y_test in raw_datasets:
                self.train_dataset.append((None, x_train, y_train.astype(np.int64)))
                self.test_dataset.append((None, x_test, y_test.astype(np.int64)))

            if collapse_noise_across_tasks:
                # Assign the single shared noise label across all tasks.
                global_noise_label = int(labels_offset)
                self.noise_label = global_noise_label
                self._args.noise_label = global_noise_label
                for dataset in (self.train_dataset, self.test_dataset):
                    for task_index in range(len(dataset)):
                        labels = dataset[task_index][2]
                        if labels.ndim == 2 and labels.shape[1] == 2:
                            labels = labels.copy()
                            noise_mask = labels[:, 0] < 0
                            if noise_mask.any():
                                labels[noise_mask, 0] = global_noise_label
                            dataset[task_index] = (
                                dataset[task_index][0],
                                dataset[task_index][1],
                                labels.astype(np.int64),
                            )
                        else:
                            labels = labels.copy()
                            noise_mask = labels < 0
                            if noise_mask.any():
                                labels[noise_mask] = global_noise_label
                            dataset[task_index] = (
                                dataset[task_index][0],
                                dataset[task_index][1],
                                labels.astype(np.int64),
                            )

            self.sample_permutations = []
            for t in range(len(self.train_dataset)):
                N = self.train_dataset[t][1].shape[0]  # number of samples in task t
                if self._args.samples_per_task <= 0:
                    n = N
                else:
                    n = min(self._args.samples_per_task, N)
                # randomly shuffle data
                # print(f"Task {t}: {N} training samples, using {n} samples for training and testing.")
                p_tr = np.random.permutation(N)[:n]
                N = self.test_dataset[t][1].shape[0]
                p_te = np.random.permutation(N)[:n]
                self.sample_permutations.append([p_tr, p_te])

            # Track per-task class counts for downstream models.
            def _task_class_count(task):
                labels = task[2]
                if labels.ndim == 2 and labels.shape[1] == 2:
                    labels = labels[:, 0]
                if collapse_noise_across_tasks:
                    # Exclude shared noise label from per-task class count.
                    labels = labels[(labels >= 0) & (labels != self.noise_label)]
                else:
                    labels = labels[labels >= 0]
                return int(np.unique(labels).size)

            self.classes_per_task = [
                _task_class_count(task) for task in self.train_dataset
            ]
            print("Built classes_per_task:", self.classes_per_task)
            # Persist on args for convenience.
            self._args.classes_per_task = self.classes_per_task
        else:
            self.train_dataset, self.test_dataset = torch.load(
                os.path.join(self._args.data_path, self._args.dataset + ".pt")
            )

            # Fallback generic task names when loading from a pre-built .pt file.
            self.task_names = [f"task{idx}" for idx in range(len(self.train_dataset))]

            self.sample_permutations = []

            # for every task, accumulate a shuffled set of samples_per_task
            for t in range(len(self.train_dataset)):
                N = self.train_dataset[t][1].size(0)
                if self._args.samples_per_task <= 0:
                    n = N
                else:
                    n = min(self._args.samples_per_task, N)
                p_tr = torch.randperm(N)[0:n]

                N_test = self.test_dataset[t][1].size(0)
                if self._args.samples_per_task <= 0:
                    n_test = N_test
                else:
                    n_test = min(self._args.samples_per_task, N_test)
                p_te = torch.randperm(N_test)[0:n_test]
                self.sample_permutations.append([p_tr, p_te])
            self.classes_per_task = [
                (
                    int(torch.unique(task[2]).numel())
                    if hasattr(torch, "unique")
                    else len(np.unique(task[2]))
                )
                for task in self.train_dataset
            ]
            self._args.classes_per_task = self.classes_per_task

            task_order_seed_pt = getattr(self._args, "task_order_seed", None)
            num_tasks_pt = len(self.train_dataset)
            _, perm_pt = permute_task_sequence(
                list(range(num_tasks_pt)), task_order_seed_pt
            )
            if perm_pt is not None:
                base_names_pt = list(self.task_names)
                self.train_dataset = [
                    self.train_dataset[int(perm_pt[slot])]
                    for slot in range(num_tasks_pt)
                ]
                self.test_dataset = [
                    self.test_dataset[int(perm_pt[slot])]
                    for slot in range(num_tasks_pt)
                ]
                self.sample_permutations = [
                    self.sample_permutations[int(perm_pt[slot])]
                    for slot in range(num_tasks_pt)
                ]
                self.classes_per_task = [
                    self.classes_per_task[int(perm_pt[slot])]
                    for slot in range(num_tasks_pt)
                ]
                self.task_names = [
                    f"task{int(perm_pt[slot])}" for slot in range(num_tasks_pt)
                ]
                print(
                    f"[task_order] task_order_seed={task_order_seed_pt} "
                    f"base_order={base_names_pt} "
                    f"presentation_slot_to_base_index={perm_pt.tolist()} "
                    f"presentation_order={list(self.task_names)}"
                )
                self._args.classes_per_task = self.classes_per_task
