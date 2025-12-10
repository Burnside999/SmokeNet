# smokenet/data/dataset.py

from typing import List, Tuple
import torch
from torch.utils.data import Dataset

class SmokeDataset(Dataset):
    def __init__(
        self,
        signals: List[torch.Tensor],
        fire_labels: List[int],
        fuel_labels: List[int]
    ):
        assert len(signals) == len(fire_labels) == len(fuel_labels)
        self.signals = signals
        self.fire_labels = fire_labels
        self.fuel_labels = fuel_labels

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int, int]:
        x = self.signals[idx].float()      # (T, 3)
        y_fire = int(self.fire_labels[idx])
        y_fuel = int(self.fuel_labels[idx])
        return x, y_fire, y_fuel
