import os
import random

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import transforms

from dataloaders.idataset import _get_datasets, DummyDataset
from dataloaders.iq_data_loader import (
    IQDataGenerator,
    iq_numpy_batch_to_three_adc_channel_first,
)
from dataloaders.task_incremental_loader import (
    _apply_data_scaling,
    _maybe_move_sample_axis,
    _normalize_label_array,
    _resolve_task_file_order,
    permute_task_sequence,
)

# --------
# Datasets CIFAR and TINYIMAGENET
# --------


class _SharedCumulativeIQTestDataset(torch.utils.data.Dataset):
    """Cumulative CIL test set assembled from per-task datasets by reference.

    Wraps a :class:`~torch.utils.data.ConcatDataset` of per-task test datasets
    (each referencing its task's IQ array via a :class:`Subset`) so that the many
    retained per-task test loaders share a single resident copy of every task's
    data instead of holding independent concatenated copies (the previous code
    materialised an ``O(N^2)`` pile of unified/concatenated arrays across tasks).

    Sample identity and ordering are unchanged versus ``np.concatenate`` of the
    order-corrected per-task chunks, so batches (and therefore metrics) are
    bit-identical. A real ``y`` label array is exposed because the evaluation
    helpers (``_extract_task_labels`` / noise-label inference) introspect
    ``loader.dataset.y``; ``ConcatDataset`` alone does not provide it.
    """

    def __init__(self, concat_dataset: ConcatDataset, y: np.ndarray):
        self._dataset = concat_dataset
        self.y = y

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index):
        return self._dataset[index]


def _iq_test_arrays_can_stack_along_batch(x_chunks: list[np.ndarray]) -> bool:
    """Return True if IQ test tensors can be concatenated on axis 0.

    Mixed datasets (e.g. uclresm ``(N, 3, L)`` vs RCN ``(N, L)``) cannot be
    stacked into one array for cumulative CIL evaluation.
    """
    if len(x_chunks) <= 1:
        return True
    first = x_chunks[0]
    for arr in x_chunks[1:]:
        if arr.ndim != first.ndim or arr.shape[1:] != first.shape[1:]:
            return False
    return True


class IncrementalLoader:

    def __init__(
        self,
        opt,
        shuffle=True,
        seed=1,
    ):
        self._opt = opt
        dataset_name = opt.dataset
        validation_split = opt.validation
        self.increment = opt.increment

        self.task_names: list[str] = []
        self.noise_label: int | None = None
        self.sample_permutations: list = []

        self._iq = dataset_name.lower() == "iq"
        if self._iq:
            self._setup_iq_tasks(opt.data_path, validation_split, seed)
            self.train_transforms = []
            self.common_transforms = []
            self.top_transforms = []
        else:
            datasets = _get_datasets(dataset_name)
            self._setup_data(
                datasets,
                class_order_type=opt.class_order,
                seed=seed,
                increment=self.increment,
                validation_split=validation_split,
            )
            self.train_transforms = datasets[0].train_transforms
            self.common_transforms = datasets[0].common_transforms
            self.top_transforms = datasets[0].top_transforms
        self.validation_split = validation_split

        self._current_task = 0

        self._batch_size = opt.batch_size
        self._test_batch_size = opt.test_batch_size
        self._workers = opt.workers
        self._shuffle = shuffle

        if self._iq:
            self._setup_iq_test_tasks(validation_split)
        else:
            self._setup_test_tasks(validation_split)

        # Persist per-task class counts for downstream components.
        self.classes_per_task = list(self.increments)
        self._opt.classes_per_task = self.classes_per_task

    @property
    def n_tasks(self):
        return len(self.increments)

    def _iq_target_adc_channels(self):
        """Mirror the ``target_adc_channels`` decision made in ``_get_loader``.

        Kept in sync with the IQ branch of :meth:`_get_loader` so per-task test
        datasets convert samples identically to the merged loader they replace.
        """
        iq_dataset = str(getattr(self._opt, "dataset", "")).lower() == "iq"
        if iq_dataset and (
            str(getattr(self._opt, "model", "")).lower() == "iid2"
            or getattr(self._opt, "loader", "") == "class_incremental_loader"
        ):
            return 3
        return None

    def _iq_test_task_dataset(self, task_j):
        """Return ``(subset, permuted_labels)`` for task ``task_j`` (cached).

        The subset references ``self.iq_test[task_j]`` (no data copy) and applies
        the fixed evaluation permutation lazily via :class:`Subset`, so every
        cumulative loader that includes this task shares one resident copy.
        """
        cache = getattr(self, "_iq_test_task_dataset_cache", None)
        if cache is None:
            cache = {}
            self._iq_test_task_dataset_cache = cache
        cached = cache.get(task_j)
        if cached is not None:
            return cached
        x_j, y_j = self.iq_test[task_j]
        p_te_j = self.sample_permutations[task_j][1]
        base = IQDataGenerator(
            x_j, y_j, target_adc_channels=self._iq_target_adc_channels()
        )
        subset = Subset(base, list(p_te_j))
        permuted_labels = np.asarray(y_j)[p_te_j]
        cached = (subset, permuted_labels)
        cache[task_j] = cached
        return cached

    def _build_shared_cumulative_iq_test_loader(self, current_task):
        """Build the cumulative (tasks 0..current) IQ test loader by reference.

        Returns ``(loader, n_samples)`` on success, or ``None`` when per-task
        sample shapes are not uniform (mixed sequence lengths) — in which case
        the caller falls back to the explicit-concatenation path so behaviour is
        unchanged for those datasets.
        """
        subsets = []
        label_chunks = []
        sample_shape = None
        for task_j in range(current_task + 1):
            subset, permuted_labels = self._iq_test_task_dataset(task_j)
            if len(subset) > 0:
                first_sample, _ = subset[0]
                shape = tuple(np.asarray(first_sample).shape)
                if sample_shape is None:
                    sample_shape = shape
                elif shape != sample_shape:
                    # Non-uniform shapes cannot be collated into one batch; defer
                    # to the legacy unify/last-task-only handling.
                    return None
            subsets.append(subset)
            label_chunks.append(permuted_labels)

        dataset = _SharedCumulativeIQTestDataset(
            ConcatDataset(subsets), np.concatenate(label_chunks)
        )
        loader = DataLoader(
            dataset,
            batch_size=self._test_batch_size,
            shuffle=True,
            num_workers=0,
        )
        return loader, len(dataset)

    def new_task(self, memory=None):
        if self._current_task >= len(self.increments):
            raise Exception("No more tasks.")

        min_class = sum(self.increments[: self._current_task])
        max_class = sum(self.increments[: self._current_task + 1])

        # Set by the IQ shared-cumulative test path; when None the test loader is
        # built from the materialised ``x_test``/``y_test`` arrays below.
        precomputed_test_loader = None
        precomputed_n_test = None

        if self._iq:
            x_train, y_train = self.iq_train[self._current_task]
            val_pair = self.iq_val[self._current_task]
            if val_pair is not None:
                x_val, y_val = val_pair
            else:
                x_val, y_val = np.array([]), np.array([])
            # Cumulative CIL test: all samples from tasks 0..current (matches vision
            # path using high_range=max_class with low_range=0). Prefer a loader
            # that references each task's test array once (shared across the many
            # retained per-task loaders) instead of holding concatenated copies.
            shared_cumulative = self._build_shared_cumulative_iq_test_loader(
                self._current_task
            )
            if shared_cumulative is not None:
                (
                    precomputed_test_loader,
                    precomputed_n_test,
                ) = shared_cumulative
            else:
                # Legacy path: per-task shapes are non-uniform, so fall back to
                # explicit unification / last-task-only handling (unchanged).
                x_test_chunks: list = []
                y_test_chunks: list = []
                for task_j in range(self._current_task + 1):
                    x_j, y_j = self.iq_test[task_j]
                    p_te_j = self.sample_permutations[task_j][1]
                    x_test_chunks.append(x_j[p_te_j])
                    y_test_chunks.append(y_j[p_te_j])

                x_merged: np.ndarray | None = None
                y_merged: np.ndarray | None = None
                if _iq_test_arrays_can_stack_along_batch(x_test_chunks):
                    x_merged = np.concatenate(x_test_chunks, axis=0)
                    y_merged = np.concatenate(y_test_chunks, axis=0)
                else:
                    try:
                        x_three = [
                            iq_numpy_batch_to_three_adc_channel_first(c)
                            for c in x_test_chunks
                        ]
                    except ValueError:
                        x_three = None
                    if x_three is not None and _iq_test_arrays_can_stack_along_batch(
                        x_three
                    ):
                        x_merged = np.concatenate(x_three, axis=0)
                        y_merged = np.concatenate(y_test_chunks, axis=0)
                        print(
                            "Note: Unified mixed IQ test tensors to (N, 3, L) via "
                            "ADC0 I/Q + zero-padded ADC1/2 (IID2-style) for cumulative CIL."
                        )
                if x_merged is not None:
                    x_test, y_test = x_merged, y_merged
                else:
                    print(
                        "Warning: IQ tasks use incompatible shapes even after 3-ADC "
                        "unification (e.g. different sequence lengths). "
                        "Using the current task test split only."
                    )
                    x_test = x_test_chunks[-1]
                    y_test = y_test_chunks[-1]
            p_tr, _ = self.sample_permutations[self._current_task]
            x_train, y_train = x_train[p_tr], y_train[p_tr]
        else:
            x_train, y_train = self._select(
                self.data_train,
                self.targets_train,
                low_range=min_class,
                high_range=max_class,
            )
            x_val, y_val = self._select(
                self.data_val,
                self.targets_val,
                low_range=min_class,
                high_range=max_class,
            )
            x_test, y_test = self._select(
                self.data_test, self.targets_test, high_range=max_class
            )

        if memory is not None:
            data_memory, targets_memory = memory
            print("Set memory of size: {}.".format(data_memory.shape[0]))
            x_train = np.concatenate((x_train, data_memory))
            y_train = np.concatenate((y_train, targets_memory))

        train_loader = self._get_loader(x_train, y_train, mode="train")
        val_loader = (
            self._get_loader(x_val, y_val, mode="train") if len(x_val) > 0 else None
        )
        if precomputed_test_loader is not None:
            test_loader = precomputed_test_loader
        else:
            test_loader = self._get_loader(x_test, y_test, mode="test")

        task_name = None
        if (
            self._iq
            and self.task_names
            and 0 <= self._current_task < len(self.task_names)
        ):
            task_name = self.task_names[self._current_task]

        task_info = {
            "min_class": min_class,
            "max_class": max_class,
            "increment": self.increments[self._current_task],
            "task": self._current_task,
            "task_name": task_name,
            "max_task": len(self.increments),
            "n_train_data": x_train.shape[0],
            "n_test_data": (
                precomputed_n_test
                if precomputed_n_test is not None
                else x_test.shape[0]
            ),
        }

        self._current_task += 1

        return task_info, train_loader, val_loader, test_loader

    def _setup_test_tasks(self, validation_split):
        self.test_tasks = []
        self.val_tasks = []
        for i in range(len(self.increments)):
            min_class = sum(self.increments[:i])
            max_class = sum(self.increments[: i + 1])

            x_test, y_test = self._select(
                self.data_test,
                self.targets_test,
                low_range=min_class,
                high_range=max_class,
            )
            self.test_tasks.append(self._get_loader(x_test, y_test, mode="test"))

            if validation_split > 0.0:
                x_val, y_val = self._select(
                    self.data_val,
                    self.targets_val,
                    low_range=min_class,
                    high_range=max_class,
                )
                self.val_tasks.append(self._get_loader(x_val, y_val, mode="test"))

    def _setup_iq_test_tasks(self, validation_split):
        self.test_tasks = []
        self.val_tasks = []
        for i in range(len(self.increments)):
            x_test, y_test = self.iq_test[i]
            p_te = self.sample_permutations[i][1]
            x_test, y_test = x_test[p_te], y_test[p_te]
            self.test_tasks.append(self._get_loader(x_test, y_test, mode="test"))
            if validation_split > 0.0 and self.iq_val[i] is not None:
                x_val, y_val = self.iq_val[i]
                self.val_tasks.append(self._get_loader(x_val, y_val, mode="test"))

    def get_tasks(self, dataset_type="test"):
        if dataset_type == "val":
            if self.validation_split > 0.0:
                return self.val_tasks
            else:
                return self.test_tasks
        elif dataset_type == "test":
            return self.test_tasks
        else:
            raise NotImplementedError("Unknown mode {}.".format(dataset_type))

    def get_samples_per_task(self, task_id=None, split="train"):
        if task_id is None:
            task_id = self._current_task
        if self._opt.samples_per_task > 0:
            return int(self._opt.samples_per_task)
        if split not in ("train", "test"):
            raise ValueError(f"Unknown split '{split}' (expected 'train' or 'test').")
        if self._iq:
            perms = self.sample_permutations[task_id]
            idx = 0 if split == "train" else 1
            return int(len(perms[idx]))
        min_class = sum(self.increments[:task_id])
        max_class = sum(self.increments[: task_id + 1])
        if split == "train":
            idxes = np.where(
                np.logical_and(
                    self.targets_train >= min_class,
                    self.targets_train < max_class,
                )
            )[0]
            return int(len(idxes))
        idxes = np.where(
            np.logical_and(
                self.targets_test >= 0,
                self.targets_test < max_class,
            )
        )[0]
        return int(len(idxes))

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

        if self._iq:
            sample = self.iq_train[0][0]
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
            n_outputs = 0
            for i in range(len(self.iq_train)):
                n_outputs = max(n_outputs, _max_label_value(self.iq_train[i][1]))
                n_outputs = max(n_outputs, _max_label_value(self.iq_test[i][1]))
            self.n_outputs = n_outputs
            n_task = len(self.increments)
            return n_inputs, n_outputs + 1, n_task
        else:
            if self._opt.dataset == "tinyimagenet":
                n_inputs = 3 * 64 * 64
            else:
                n_inputs = (
                    self.data_train.shape[3]
                    * self.data_train.shape[1]
                    * self.data_train.shape[2]
                )
            n_outputs = self._opt.increment * len(self.increments)
            n_task = len(self.increments)
            return n_inputs, n_outputs, n_task

    def _select(self, x, y, low_range=0, high_range=0):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]

    def _get_loader(self, x, y, shuffle=True, mode="train"):
        if mode == "train":
            pretrsf = transforms.Compose([*self.top_transforms])
            trsf = transforms.Compose([*self.train_transforms, *self.common_transforms])
            batch_size = self._batch_size
        elif mode == "test":
            pretrsf = transforms.Compose([*self.top_transforms])
            trsf = transforms.Compose(self.common_transforms)
            batch_size = self._test_batch_size
        elif mode == "flip":
            trsf = transforms.Compose(
                [transforms.RandomHorizontalFlip(p=1.0), *self.common_transforms]
            )
            batch_size = self._test_batch_size
        else:
            raise NotImplementedError("Unknown mode {}.".format(mode))

        if self._iq:
            target_adc_channels = None
            iq_dataset = str(getattr(self._opt, "dataset", "")).lower() == "iq"
            if iq_dataset and (
                str(getattr(self._opt, "model", "")).lower() == "iid2"
                or getattr(self._opt, "loader", "") == "class_incremental_loader"
            ):
                target_adc_channels = 3
            dataset = IQDataGenerator(x, y, target_adc_channels=target_adc_channels)
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=0,
            )
        else:
            return DataLoader(
                DummyDataset(x, y, trsf, pretrsf, self._opt.dataset == "tinyimagenet"),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=self._workers,
            )

    def _setup_data(
        self,
        datasets,
        class_order_type=False,
        seed=1,
        increment=10,
        validation_split=0.0,
    ):
        # FIXME: handles online loading of images
        self.data_train, self.targets_train = [], []
        self.data_test, self.targets_test = [], []
        self.data_val, self.targets_val = [], []
        self.increments = []
        self.class_order = []

        current_class_idx = 0  # When using multiple datasets
        for dataset in datasets:

            if self._opt.dataset == "tinyimagenet":
                root_path = self._opt.data_path
                train_dataset = dataset.base_dataset(root_path + "train/")
                test_dataset = dataset.base_dataset(root_path + "val/")

                train_dataset.data = train_dataset.samples
                test_dataset.data = test_dataset.samples

                x_train, y_train = train_dataset.data, np.array(train_dataset.targets)
                x_val, y_val, x_train, y_train = self._list_split_per_class(
                    x_train, y_train, validation_split
                )
                x_test, y_test = test_dataset.data, np.array(test_dataset.targets)

                order = [i for i in range(len(np.unique(y_train)))]
                if class_order_type == "random":
                    random.seed(
                        seed
                    )  # Ensure that following order is determined by seed:
                    random.shuffle(order)
                    print("Class order:", order)
                elif class_order_type == "old" and dataset.class_order is not None:
                    order = dataset.class_order
                else:
                    print("Classes are presented in a chronological order")

            else:
                root_path = self._opt.data_path
                train_dataset = dataset.base_dataset(
                    root_path, train=True, download=True
                )
                test_dataset = dataset.base_dataset(
                    root_path, train=False, download=True
                )

                x_train, y_train = train_dataset.data, np.array(train_dataset.targets)
                x_val, y_val, x_train, y_train = self._split_per_class(
                    x_train, y_train, validation_split
                )
                x_test, y_test = test_dataset.data, np.array(test_dataset.targets)

                order = [i for i in range(len(np.unique(y_train)))]
                if class_order_type == "random":
                    random.seed(
                        seed
                    )  # Ensure that following order is determined by seed:
                    random.shuffle(order)
                    print("Class order:", order)
                elif class_order_type == "old" and dataset.class_order is not None:
                    order = dataset.class_order
                elif (
                    class_order_type == "super"
                    and dataset.class_order_super is not None
                ):
                    order = dataset.class_order_super
                else:
                    print("Classes are presented in a chronological order")

            self.class_order.append(order)

            y_train = self._map_new_class_index(y_train, order)
            y_val = self._map_new_class_index(y_val, order)
            y_test = self._map_new_class_index(y_test, order)

            y_train += current_class_idx
            y_val += current_class_idx
            y_test += current_class_idx

            current_class_idx += len(order)
            if len(datasets) > 1:
                self.increments.append(len(order))
            else:
                self.increments = [increment for _ in range(len(order) // increment)]

            self.data_train.append(x_train)
            self.targets_train.append(y_train)
            self.data_val.append(x_val)
            self.targets_val.append(y_val)
            self.data_test.append(x_test)
            self.targets_test.append(y_test)

            self.data_train = np.concatenate(self.data_train)
            self.targets_train = np.concatenate(self.targets_train)
            self.data_val = np.concatenate(self.data_val)
            self.targets_val = np.concatenate(self.targets_val)
            self.data_test = np.concatenate(self.data_test)
            self.targets_test = np.concatenate(self.targets_test)

    @staticmethod
    def _map_new_class_index(y, order):
        """Transforms targets for new class order."""

        return np.array(list(map(lambda x: order.index(x), y)))

    @staticmethod
    def _split_per_class(x, y, validation_split=0.0):
        """Splits train data for a subset of validation data.
        Split is done so that each class has a much data.
        """
        shuffled_indexes = np.random.permutation(x.shape[0])
        x = x[shuffled_indexes]
        y = y[shuffled_indexes]

        x_val, y_val = [], []
        x_train, y_train = [], []

        group_key = y[:, 0] if (y.ndim == 2 and y.shape[1] == 2) else y

        for class_id in np.unique(group_key):
            class_indexes = np.where(group_key == class_id)[0]
            nb_val_elts = int(class_indexes.shape[0] * validation_split)

            val_indexes = class_indexes[:nb_val_elts]
            train_indexes = class_indexes[nb_val_elts:]

            x_val.append(x[val_indexes])
            y_val.append(y[val_indexes])
            x_train.append(x[train_indexes])
            y_train.append(y[train_indexes])

        x_val, y_val = np.concatenate(x_val), np.concatenate(y_val)
        x_train, y_train = np.concatenate(x_train), np.concatenate(y_train)

        return x_val, y_val, x_train, y_train

    def _setup_iq_tasks(self, data_path, validation_split=0.0, seed=1):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.iq_train = []
        self.iq_val = []
        self.iq_test = []
        self.increments = []
        self.class_order = []
        self.sample_permutations = []

        collapse_noise_across_tasks = True

        data_files = [f for f in os.listdir(data_path) if f.endswith(".npz")]
        data_files = _resolve_task_file_order(
            data_files, getattr(self._opt, "task_order_files", "") or ""
        )
        base_task_labels = [os.path.splitext(f)[0] for f in data_files]
        task_order_seed = getattr(self._opt, "task_order_seed", None)
        data_files, task_perm = permute_task_sequence(data_files, task_order_seed)
        if task_perm is not None:
            presentation_labels = [os.path.splitext(f)[0] for f in data_files]
            print(
                f"[task_order] task_order_seed={task_order_seed} "
                f"base_order={base_task_labels} "
                f"presentation_slot_to_base_index={task_perm.tolist()} "
                f"presentation_order={presentation_labels}"
            )
        raw_datasets: list[tuple] = []
        labels_offset = 0

        self.task_names = [os.path.splitext(f)[0] for f in data_files]

        for fname in data_files:
            data = np.load(os.path.join(data_path, fname))

            def _get(keys):
                for k in keys:
                    if k in data:
                        return data[k]
                return None

            x_train = _get(["x_train", "X_train", "Xtr", "xtr", "Xcv", "xcv", "x", "X"])
            y_train = _get(["y_train", "Y_train", "ytr", "ycv", "y", "Y"])
            x_test = _get(["x_test", "X_test", "Xte", "xte"])
            y_test = _get(["y_test", "Y_test", "yte"])

            if x_train is not None and y_train is not None:
                before = x_train.shape
                x_train = _maybe_move_sample_axis(x_train, y_train, f"{fname} train")
                after = x_train.shape
                if before != after:
                    print(f"{fname} train: moved sample axis {before} -> {after}")
            if x_test is not None and y_test is not None:
                before = x_test.shape
                x_test = _maybe_move_sample_axis(x_test, y_test, f"{fname} test")
                after = x_test.shape
                if before != after:
                    print(f"{fname} test: moved sample axis {before} -> {after}")
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
                print(f"{fname} test: x={x_test.shape}, y={np.asarray(y_test).shape}")

            if x_train is None or y_train is None or x_test is None or y_test is None:
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

            scaling_mode = getattr(self._opt, "data_scaling", "none")
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
                f"Noise labels ratio in {fname} train: {(y_train < 0).mean():.2f}, "
                f"test: {(y_test < 0).mean():.2f}"
            )

            if y_train.ndim == 2 and y_train.shape[1] == 2:
                y_train_cls = y_train[:, 0]
                y_train_det = y_train[:, 1]
                y_test_cls = y_test[:, 0]
                y_test_det = y_test[:, 1]
                use_detector_arch = bool(getattr(self._opt, "use_detector_arch", False))
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
                    y_test_cls_remap[mask_test] = y_test_cls[mask_test] + labels_offset
                extra_class = 0
                if (not use_detector_arch) and has_negatives:
                    if collapse_noise_across_tasks:
                        y_train_cls_remap[~mask_train] = -1
                        y_test_cls_remap[~mask_test] = -1
                    else:
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
                use_detector_arch = bool(getattr(self._opt, "use_detector_arch", False))
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
                        unique_labels.searchsorted(y_train[mask_train]) + labels_offset
                    )
                    y_test_remap[mask_test] = (
                        unique_labels.searchsorted(y_test[mask_test]) + labels_offset
                    )
                else:
                    y_train_remap[mask_train] = y_train[mask_train] + labels_offset
                    y_test_remap[mask_test] = y_test[mask_test] + labels_offset
                extra_class = 0
                if (not use_detector_arch) and has_negatives:
                    if collapse_noise_across_tasks:
                        y_train_remap[~mask_train] = -1
                        y_test_remap[~mask_test] = -1
                    else:
                        extra_class = 1
                        neg_label = labels_offset + unique_labels.size
                        y_train_remap[~mask_train] = neg_label
                        y_test_remap[~mask_test] = neg_label
                y_train = y_train_remap
                y_test = y_test_remap
            if collapse_noise_across_tasks:
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

            raw_datasets.append((x_train, y_train, x_test, y_test))

        if not raw_datasets:
            raise ValueError("No IQ datasets were loaded. Please check the data path.")

        if collapse_noise_across_tasks:
            global_noise_label = int(labels_offset)
            self.noise_label = global_noise_label
            self._opt.noise_label = global_noise_label
            for task_index in range(len(raw_datasets)):
                x_tr, y_tr, x_te, y_te = raw_datasets[task_index]
                if y_tr.ndim == 2 and y_tr.shape[1] == 2:
                    y_tr = y_tr.copy()
                    noise_mask = y_tr[:, 0] < 0
                    if noise_mask.any():
                        y_tr[noise_mask, 0] = global_noise_label
                    y_te = y_te.copy()
                    noise_mask_te = y_te[:, 0] < 0
                    if noise_mask_te.any():
                        y_te[noise_mask_te, 0] = global_noise_label
                else:
                    y_tr = y_tr.copy()
                    noise_mask = y_tr < 0
                    if noise_mask.any():
                        y_tr[noise_mask] = global_noise_label
                    y_te = y_te.copy()
                    noise_mask_te = y_te < 0
                    if noise_mask_te.any():
                        y_te[noise_mask_te] = global_noise_label
                raw_datasets[task_index] = (
                    x_tr,
                    y_tr.astype(np.int64),
                    x_te,
                    y_te.astype(np.int64),
                )

        def _task_class_count(y_labels: np.ndarray) -> int:
            labels = y_labels
            if labels.ndim == 2 and labels.shape[1] == 2:
                labels = labels[:, 0]
            labels = labels[(labels >= 0) & (labels != self.noise_label)]
            return int(np.unique(labels).size)

        for x_train, y_train, x_test, y_test in raw_datasets:
            if validation_split > 0.0:
                x_val, y_val, x_train, y_train = self._split_per_class(
                    x_train, y_train, validation_split
                )
                self.iq_val.append((x_val, y_val))
            else:
                self.iq_val.append(None)

            self.iq_train.append((x_train, y_train))
            self.iq_test.append((x_test, y_test))

        self.increments = [_task_class_count(y) for _, y in self.iq_train]
        print("Built IQ classes_per_task (increments):", self.increments)

        for t in range(len(self.iq_train)):
            training_count = self.iq_train[t][0].shape[0]
            if self._opt.samples_per_task <= 0:
                n_tr = training_count
            else:
                n_tr = min(int(self._opt.samples_per_task), training_count)
            p_tr = np.random.permutation(training_count)[:n_tr]

            test_count = self.iq_test[t][0].shape[0]
            if self._opt.samples_per_task <= 0:
                n_te = test_count
            else:
                n_te = min(int(self._opt.samples_per_task), test_count)
            p_te = np.random.permutation(test_count)[:n_te]
            self.sample_permutations.append([p_tr, p_te])

    @staticmethod
    def _list_split_per_class(x, y, validation_split=0.0):
        """Splits train data for a subset of validation data.
        Split is done so that each class has a much data.
        """
        c = list(zip(x, y))
        random.shuffle(c)
        x, y = zip(*c)

        x_val, y_val = [], []
        x_train, y_train = [], []

        for class_id in np.unique(y):
            class_indexes = np.where(y == class_id)[0]
            nb_val_elts = int(class_indexes.shape[0] * validation_split)

            val_indexes = class_indexes[:nb_val_elts]
            train_indexes = class_indexes[nb_val_elts:]

            x_val_i = [x[i] for i in val_indexes]
            y_val_i = [y[i] for i in val_indexes]

            x_train_i = [x[i] for i in train_indexes]
            y_train_i = [y[i] for i in train_indexes]

            x_val.append(x_val_i)
            y_val.append(y_val_i)

            x_train.append(x_train_i)
            y_train.append(y_train_i)

        x_val, y_val = np.concatenate(x_val), np.concatenate(y_val)
        x_train, y_train = np.concatenate(x_train), np.concatenate(y_train)

        return x_val, y_val, x_train, y_train

    def get_idx_data(self, idx, batch_size, mode="test", data_source="train"):
        """Returns a custom loader with specific idxs only.
        :param idx: A list of data indexes that we want.
        :param mode: Various mode for the transformations applied on it.
        :param data_source: Whether to fetch from the train, val, or test set.
        :return: The raw data and a loader.
        """
        if data_source == "train":
            x, y = self.data_train, self.targets_train
        elif data_source == "val":
            x, y = self.data_val, self.targets_val
        elif data_source == "test":
            x, y = self.data_test, self.targets_test
        else:
            raise ValueError("Unknown data source <{}>.".format(data_source))
        y, sorted_idx = y.sort()

        sampler = torch.utils.data.sampler.SubsetRandomSampler(idx)
        trsf = transforms.Compose(self.common_transforms)

        _loader = DataLoader(
            DummyDataset(x[sorted_idx], y, trsf),
            sampler=sampler,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self._workers,
        )
        return _loader

    def get_custom_loader(self, class_indexes, mode="test", data_source="train"):
        """Returns a custom loader.
        :param class_indexes: A list of class indexes that we want.
        :param mode: Various mode for the transformations applied on it.
        :param data_source: Whether to fetch from the train, val, or test set.
        :return: The raw data and a loader.
        """
        if not isinstance(
            class_indexes, list
        ):  # TODO: deprecated, should always give a list
            class_indexes = [class_indexes]

        if data_source == "train":
            x, y = self.data_train, self.targets_train
        elif data_source == "val":
            x, y = self.data_val, self.targets_val
        elif data_source == "test":
            x, y = self.data_test, self.targets_test
        else:
            raise ValueError("Unknown data source <{}>.".format(data_source))

        data, targets = [], []
        for class_index in class_indexes:
            class_data, class_targets = self._select(
                x, y, low_range=class_index, high_range=class_index + 1
            )
            data.append(class_data)
            targets.append(class_targets)

        data = np.concatenate(data)
        targets = np.concatenate(targets)

        return data, self._get_loader(data, targets, shuffle=False, mode=mode)
