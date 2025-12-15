# smokenet/data/loader.py

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import random_split

from smokenet.config import DataConfig
from .window import WindowDataset


def _load_tensor(path: Path) -> torch.Tensor:
    if path.suffix in {".pt", ".pth"}:
        return torch.load(path)
    # fallback to numpy array
    array = np.load(path)
    return torch.from_numpy(array)


def _load_label(value) -> Tuple[int, int]:
    if isinstance(value, dict):
        return int(value["fire"]), int(value["fuel"])
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return int(value[0]), int(value[1])
    raise ValueError("Label format must be a dict with fire/fuel or a 2-item sequence")


def _collect_pairs(data_dir: Path, label_dir: Path) -> Iterable[Tuple[Path, Path]]:
    data_files: Dict[str, Path] = {p.stem: p for p in data_dir.iterdir() if p.is_file()}
    label_files: Dict[str, Path] = {p.stem: p for p in label_dir.iterdir() if p.is_file()}

    for stem in sorted(data_files.keys() & label_files.keys()):
        yield data_files[stem], label_files[stem]


def load_datasets(data_cfg: DataConfig) -> Tuple[WindowDataset, WindowDataset]:
    data_dir = Path(data_cfg.data_dir)
    label_dir = Path(data_cfg.label_dir)

    if not data_dir.exists() or not label_dir.exists():
        raise FileNotFoundError("Data or label directory does not exist")

    signals: List[torch.Tensor] = []
    fire_labels: List[int] = []
    fuel_labels: List[int] = []

    for data_path, label_path in _collect_pairs(data_dir, label_dir):
        signal = _load_tensor(data_path)
        label_val = _load_tensor(label_path)
        fire, fuel = _load_label(label_val)

        signals.append(signal)
        fire_labels.append(fire)
        fuel_labels.append(fuel)

    if not signals:
        raise ValueError("No paired data/label files were found.")

    dataset = WindowDataset(
        signals,
        fire_labels,
        fuel_labels,
        window_size=data_cfg.window_size,
        channels=data_cfg.channels,
    )

    n_total = len(dataset)
    n_train = int(n_total * data_cfg.split_ratio)
    n_val = max(1, n_total - n_train)
    if n_train == 0:
        n_train = max(1, n_total - n_val)

    train_dataset, val_dataset = random_split(dataset, [n_train, n_total - n_train])
    return train_dataset, val_dataset