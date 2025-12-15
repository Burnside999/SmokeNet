# smokenet/data/loader.py

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import random_split

from smokenet.config import DataConfig

from .window import WindowDataset


def _load_csv(path: Path) -> torch.Tensor:
    array = np.loadtxt(path, delimiter=",")
    if array.ndim == 1:
        array = np.expand_dims(array, axis=0)
    return torch.from_numpy(array)


def _load_label(label_tensor: torch.Tensor) -> tuple[torch.Tensor, int | None]:
    if label_tensor.dim() == 1:
        fire_seq = label_tensor.long()
        fuel_label = None
    elif label_tensor.dim() == 2:
        # Accept both (T, 1) and (1, T) label shapes by rotating row vectors
        # into column vectors so the timestep dimension is always axis 0.
        if label_tensor.shape[0] == 1 and label_tensor.shape[1] > 1:
            label_tensor = label_tensor.transpose(0, 1)
        fire_seq = label_tensor[:, 0].long()
        fuel_label = (
            int(label_tensor[0, 1].item()) if label_tensor.shape[1] > 1 else None
        )
        if fuel_label is not None:
            # Ensure the provided fuel label is sequence-level and consistent.
            if not torch.all(label_tensor[:, 1] == label_tensor[0, 1]):
                raise ValueError(
                    "Fuel label column must be constant across the sequence"
                )
    else:
        raise ValueError("Label tensor must be 1D or 2D")

    return fire_seq, fuel_label


def _collect_pairs(data_dir: Path, label_dir: Path) -> Iterable[tuple[Path, Path, str]]:
    data_files: dict[str, Path] = {
        p.stem: p for p in data_dir.glob("*.csv") if p.is_file()
    }
    label_files: dict[str, Path] = {
        p.stem: p for p in label_dir.glob("*.csv") if p.is_file()
    }

    if data_files.keys() != label_files.keys():
        missing_in_labels = sorted(data_files.keys() - label_files.keys())
        missing_in_data = sorted(label_files.keys() - data_files.keys())
        raise ValueError(
            "Data and label files must have matching stems. "
            f"Missing labels for: {missing_in_labels}; missing data for: {missing_in_data}"
        )

    for stem in sorted(data_files.keys()):
        yield data_files[stem], label_files[stem], stem


def load_datasets(data_cfg: DataConfig) -> tuple[WindowDataset, WindowDataset]:
    data_dir = Path(data_cfg.data_dir)
    label_dir = Path(data_cfg.label_dir)

    if not data_dir.exists() or not label_dir.exists():
        raise FileNotFoundError("Data or label directory does not exist")

    signals: list[torch.Tensor] = []
    fire_labels: list[torch.Tensor] = []
    fuel_labels: list[int | None] = []
    sample_names: list[str] = []

    for data_path, label_path, stem in _collect_pairs(data_dir, label_dir):
        signal = _load_csv(data_path)
        label_tensor = _load_csv(label_path)
        fire_seq, fuel = _load_label(label_tensor)

        if (
            signal.shape[-1] != fire_seq.shape[0]
            and signal.shape[0] != fire_seq.shape[0]
        ):
            raise ValueError(
                f"Signal length ({signal.shape[-1]}) and label length ({fire_seq.shape[0]}) must match"
            )

        signals.append(signal)
        fire_labels.append(fire_seq)
        fuel_labels.append(fuel)
        sample_names.append(stem)

    if not signals:
        raise ValueError("No paired data/label files were found.")

    assert len(signals) == len(fire_labels) == len(fuel_labels) == len(sample_names), (
        "Signals, labels, and names counts must match"
    )

    dataset = WindowDataset(
        signals,
        fire_labels,
        fuel_labels,
        window_size=data_cfg.window_size,
        channels=data_cfg.channels,
        names=sample_names,
    )

    n_total = len(dataset)
    n_train = int(n_total * data_cfg.split_ratio)
    n_val = max(1, n_total - n_train)
    if n_train == 0:
        n_train = max(1, n_total - n_val)

    train_dataset, val_dataset = random_split(dataset, [n_train, n_total - n_train])
    return train_dataset, val_dataset
