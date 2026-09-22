import os
from typing import Optional, Tuple

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def ensure_iq_two_channel(iq_array: np.ndarray) -> np.ndarray:
    """Convert raw IQ samples into a ``(…, 2, L)`` float32 array.

    The datasets we consume may store IQ data in a variety of layouts:

    * complex-valued arrays with shape ``(…, L)``
    * real-valued arrays where I/Q are interleaved along the last axis
      (``[…, 2 * L]``)
    * already separated channel-first or channel-last representations
      (``[…, 2, L]`` or ``[…, L, 2]``)

    This helper normalises the array to the channel-first convention used by
    the models: ``(…, 2, L)`` with ``float32`` dtype.  The function is fully
    vectorised and supports batched inputs directly.
    """

    arr = np.asarray(iq_array)

    if arr.ndim == 0:
        raise ValueError("IQ sample must have at least one dimension")

    if np.iscomplexobj(arr):
        stacked = np.stack((arr.real, arr.imag), axis=-2)
        return np.ascontiguousarray(stacked.astype(np.float32, copy=False))

    arr = arr.astype(np.float32, copy=False)

    # Already channel-first: (..., 2, L)
    if arr.ndim > 2 and arr.shape[-2] == 2:
        return np.ascontiguousarray(arr)

    # Channel-last: (..., L, 2) -> (..., 2, L)
    if arr.ndim > 2 and arr.shape[-1] == 2:
        return np.ascontiguousarray(np.swapaxes(arr, -1, -2))

    last_dim = arr.shape[-1]
    if last_dim % 2 != 0:
        raise ValueError(
            "Expected an even number of features to split interleaved IQ data; "
            f"got shape {arr.shape}"
        )

    new_shape = arr.shape[:-1] + (last_dim // 2, 2)
    arr = arr.reshape(new_shape)
    arr = np.swapaxes(arr, -1, -2)
    return np.ascontiguousarray(arr)


def deinterleave_iq_rows(rows: np.ndarray) -> np.ndarray:
    """Split rows of interleaved I/Q samples onto a dedicated I/Q axis.

    Args:
        rows: Array of shape ``(..., 2 * L)`` whose last axis interleaves I and
            Q as ``[i0, q0, i1, q1, ...]``.

    Returns:
        Float32 array of shape ``(..., 2, L)`` with ``[..., 0, :]`` holding I
        and ``[..., 1, :]`` holding Q.

    Raises:
        ValueError: If the last axis has an odd length.

    Usage:
        >>> deinterleave_iq_rows(np.arange(8.0).reshape(2, 4))[0, 0]
        array([0., 2.], dtype=float32)
    """
    arr = np.asarray(rows)
    if arr.shape[-1] % 2 != 0:
        raise ValueError(
            "Expected an even length to split interleaved I/Q; "
            f"got shape {tuple(arr.shape)}"
        )
    split = arr.reshape(arr.shape[:-1] + (arr.shape[-1] // 2, 2))
    return np.ascontiguousarray(
        np.swapaxes(split, -1, -2).astype(np.float32, copy=False)
    )


def iq_numpy_batch_to_three_adc_channel_first(x: np.ndarray) -> np.ndarray:
    """Reshape batched IQ features to ``(N, 3, 2, L)`` float32 (ADC0 = I/Q).

    This matches the tensor layout produced by :class:`IQDataGenerator` when
    ``target_adc_channels=3``: ADC0 carries I on ``[:, 0, 0]`` and Q on
    ``[:, 0, 1]``; ADC1 and ADC2 are zeros. ``ResNet1D._prepare_input`` passes
    this layout through untouched, so the I/Q axis is never inferred from a
    flat length.

    Args:
        x: Batch with one of the following shapes:
            ``(N, F)`` — flat interleaved I/Q (``F`` even);
            ``(N, 2, L)`` — channel-first I and Q;
            ``(N, L, 2)`` — channel-last I/Q;
            ``(N, 3, 2 * L)`` — 3 ADC rows of interleaved I/Q.

    Returns:
        Array of shape ``(N, 3, 2, L)``.

    Raises:
        ValueError: If ``x`` has an unsupported rank or an odd flat length.

    Usage:
        >>> x_flat = np.zeros((4, 1024), dtype=np.float32)
        >>> y = iq_numpy_batch_to_three_adc_channel_first(x_flat)
        >>> y.shape
        (4, 3, 2, 512)
    """
    arr = np.asarray(x)
    if arr.ndim == 2:
        batch, features = arr.shape
        if batch == 0:
            return np.zeros((0, 3, 2, 0), dtype=np.float32)
        if features % 2 != 0:
            raise ValueError(
                "Flat IQ requires an even feature count for I/Q pairs; "
                f"got shape {tuple(arr.shape)}"
            )
        out = np.zeros((batch, 3, 2, features // 2), dtype=np.float32)
        out[:, 0] = deinterleave_iq_rows(arr)
        return np.ascontiguousarray(out)

    if arr.ndim == 3:
        if arr.shape[0] == 0:
            return np.zeros((0, 3, 2, 0), dtype=np.float32)
        # Channel-last (N, L, 2) or (N, L, 3)
        if arr.shape[-1] == 2 and arr.shape[-2] != 2:
            arr = np.ascontiguousarray(
                np.swapaxes(arr.astype(np.float32, copy=False), 1, 2)
            )
            return iq_numpy_batch_to_three_adc_channel_first(arr)

        if arr.shape[1] == 3:
            return deinterleave_iq_rows(arr)

        if arr.shape[1] == 2:
            arr = arr.astype(np.float32, copy=False)
            n_batch, _two, length = arr.shape
            out = np.zeros((n_batch, 3, 2, length), dtype=np.float32)
            out[:, 0] = arr
            return np.ascontiguousarray(out)

    if arr.ndim == 4 and arr.shape[1] == 3 and arr.shape[2] == 2:
        return np.ascontiguousarray(arr.astype(np.float32, copy=False))

    raise ValueError(
        "iq_numpy_batch_to_three_adc_channel_first expects x of shape "
        f"(N, F), (N, 2, L), (N, L, 2), (N, 3, 2L), or (N, 3, 2, L); "
        f"got {tuple(arr.shape)}"
    )


class IQDataGenerator(Dataset):
    """Dataset for raw in-phase/quadrature (IQ) samples.

    Parameters
    ----------
    x: np.ndarray
        Complex valued IQ samples. Real and imaginary parts are converted to a
        two channel float representation.
    y: np.ndarray
        Labels for each sample.
    transform: callable, optional
        Optional transform to be applied on a sample.
    convert_to_spectrogram: bool, optional
        If ``True`` the complex sample is converted to a simple frequency
        domain representation using the magnitude of the FFT.  This can be
        useful when models expect spectrogram like inputs.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        transform: Optional[callable] = None,
        convert_to_spectrogram: bool = False,
        target_adc_channels: int | None = None,
    ) -> None:
        self.x = x
        self.y = y
        self.transform = transform
        self.convert_to_spectrogram = convert_to_spectrogram
        self.target_adc_channels = target_adc_channels

    def __len__(self) -> int:  # pragma: no cover - simple wrapper
        return self.x.shape[0]

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        iq_sample = self.x[index, :]

        if iq_sample.ndim == 3 and iq_sample.shape[0] == 3 and iq_sample.shape[1] == 2:
            # Already (ADC, I/Q, L): the I/Q axis is explicit, nothing to parse.
            iq_sample = iq_sample.astype(np.float32, copy=False)
        elif iq_sample.ndim == 2:
            if iq_sample.shape[0] in (2, 3):
                iq_sample = iq_sample.astype(np.float32, copy=False)
            elif iq_sample.shape[1] in (2, 3):
                iq_sample = np.swapaxes(iq_sample, 0, 1).astype(np.float32, copy=False)
            else:
                flat = iq_sample.reshape(-1)
                i = flat[0::2]
                q = flat[1::2]
                iq_sample = np.stack([i, q], axis=0).astype(np.float32)
            if iq_sample.ndim == 2 and iq_sample.shape[0] == 3:
                # ADC rows still carry interleaved I/Q; separating them here is
                # the loader's job, so the model never has to infer the layout.
                iq_sample = deinterleave_iq_rows(iq_sample)
        else:
            # Represent complex input as two channels: I and Q
            if np.iscomplexobj(iq_sample):
                i = iq_sample.real
                q = iq_sample.imag
            else:
                i = iq_sample[0::2]
                q = iq_sample[1::2]

            iq_sample = np.stack([i, q], axis=0).astype(np.float32)

        if self.target_adc_channels == 3:
            # Standardize to `(3, 2, L)`: ADC0 holds the de-interleaved I/Q,
            # ADC1/ADC2 are zeros. The I/Q axis stays explicit, so
            # `ResNet1D._prepare_input` passes this through without having to
            # guess how a flat axis splits into I and Q.
            if iq_sample.ndim == 2 and iq_sample.shape[0] == 2:
                padded = np.zeros((3, 2, iq_sample.shape[1]), dtype=np.float32)
                padded[0] = iq_sample
                iq_sample = padded
            elif iq_sample.ndim == 3 and iq_sample.shape[:2] == (3, 2):
                iq_sample = iq_sample.astype(np.float32, copy=False)
            else:
                raise ValueError(
                    f"Expected I/Q channels (2, L) or ADC channels (3, 2, L) for "
                    f"target_adc_channels=3; got shape {tuple(iq_sample.shape)}."
                )

        label = self.y[index]

        if self.convert_to_spectrogram:
            iq_sample = self._convert_to_spectrogram(iq_sample)

        if self.transform:
            iq_sample = self.transform(iq_sample)

        if isinstance(label, (list, tuple)) and len(label) == 2:
            return iq_sample, (label[0], label[1])
        if isinstance(label, np.ndarray) and label.ndim == 1 and label.shape[0] == 2:
            return iq_sample, (label[0], label[1])
        return iq_sample, label

    def _convert_to_spectrogram(self, iq_sample: np.ndarray) -> np.ndarray:
        """Convert an IQ sample to a simple spectrogram representation."""
        # Compute magnitude of the FFT for each channel. This is a lightweight
        # approximation of a spectrogram and avoids external dependencies.
        spectrogram = np.abs(np.fft.fft(iq_sample, axis=-1))
        return spectrogram.astype(np.float32)


def load_data_iq(base_path: str, batch_size: int, args=None):
    """Utility to create dataloaders for IQ data.

    The loader expects ``train.npz`` and ``test.npz`` style files inside
    ``base_path``.  Datasets from different sources can be handled by
    inspecting the ``base_path`` name.
    """

    # Load train/test arrays depending on the dataset type
    if "radar" in base_path.lower():
        data = np.load(os.path.join(base_path, "radar_dataset.npz"))
        x_train, y_train = data["xtr"], data["ytr"]
        x_test, y_test = data["xte"], data["yte"]

        scaler_train = preprocessing.StandardScaler().fit(x_train)
        scaler_test = preprocessing.StandardScaler().fit(x_test)
        x_train = scaler_train.transform(x_train)
        x_test = scaler_test.transform(x_test)

    elif "usrp" in base_path.lower():
        train = np.load(os.path.join(base_path, "train.npz"))
        test = np.load(os.path.join(base_path, "test.npz"))
        x_train, y_train = train["X"], train["y"]
        x_test, y_test = test["X"], test["y"]

    elif "rfmls" in base_path.lower():
        train = np.load(os.path.join(base_path, "train.npz"))
        test = np.load(os.path.join(base_path, "test.npz"))
        x_train, y_train = train["X"], train["y"]
        x_test, y_test = test["X"], test["y"]

    else:
        train = np.load(os.path.join(base_path, "train.npz"))
        test = np.load(os.path.join(base_path, "test.npz"))
        x_train, y_train = train["X"], train["y"]
        x_test, y_test = test["X"], test["y"]

    # Split test set into validation and test
    x_test, x_val, y_test, y_val = train_test_split(
        x_test, y_test, test_size=0.7, random_state=42, stratify=y_test
    )

    # Build PyTorch dataloaders
    training_set = IQDataGenerator(x_train, y_train)
    train_loader = DataLoader(
        training_set, batch_size=batch_size, shuffle=True, num_workers=0
    )

    val_set = IQDataGenerator(x_val, y_val)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)

    test_set = IQDataGenerator(x_test, y_test)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=True, num_workers=0
    )

    return train_loader, val_loader, test_loader
