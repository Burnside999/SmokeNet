# smokenet/data/dataset.py


import torch
from torch.utils.data import Dataset


def _build_windows(signal: torch.Tensor, window_size: int) -> torch.Tensor:
    """Create T sliding windows ending at each timestep.

    Args:
        signal: Tensor shaped (C, T_raw).
        window_size: Number of timesteps per window.

    Returns:
        Tensor shaped (T_raw, C * window_size) where every timestep has a
        zero-padded left-aligned window flattened along the feature dimension.
    """

    if signal.dim() != 2:
        raise ValueError("Signal tensor must have shape (channels, T)")

    channels, T_raw = signal.shape
    windows = torch.zeros((T_raw, channels, window_size), dtype=signal.dtype)
    for t in range(T_raw):
        start = max(0, t - window_size + 1)
        end = t + 1
        window = signal[:, start:end]
        windows[t, :, -window.shape[1] :] = window

    return windows.reshape(T_raw, channels * window_size)


class WindowDataset(Dataset):
    def __init__(
        self,
        signals: list[torch.Tensor],
        fire_labels: list[torch.Tensor],
        fuel_labels: list[int | None],
        window_size: int,
        channels: int,
    ):
        if len(signals) != len(fire_labels) or len(signals) != len(fuel_labels):
            raise ValueError("Signals and labels must have the same length")

        self.window_size = window_size
        self.channels = channels
        self.signals = [self._validate_signal(sig) for sig in signals]
        self.fire_labels = fire_labels
        self.fuel_labels = fuel_labels
        self.fuel_available = all(f is not None for f in fuel_labels)

    def _validate_signal(self, signal: torch.Tensor) -> torch.Tensor:
        signal = signal.float()
        if signal.dim() != 2:
            raise ValueError("Each signal must be 2D (channels, T)")

        if signal.shape[0] == self.channels:
            return signal
        if signal.shape[1] == self.channels:
            return signal.transpose(0, 1)

        raise ValueError(
            f"Signal channel dimension mismatch: expected {self.channels}, "
            f"got shape {tuple(signal.shape)}"
        )

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, int | None]:
        signal = self.signals[idx]
        windows = _build_windows(signal, self.window_size)  # (T, channels*window)
        y_fire = self.fire_labels[idx].long()
        y_fuel = self.fuel_labels[idx]
        return windows, y_fire, y_fuel
