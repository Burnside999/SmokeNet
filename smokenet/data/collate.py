import torch


def smoke_collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor, int | None]]):
    """
    batch: list of (x, y_fire_seq, y_fuel)
    x: (T_i, feature_dim) where feature_dim depends on config
    y_fire_seq: (T_i,)
    """
    lengths = [x.shape[0] for x, _, _ in batch]
    T_max = max(lengths)

    xs = []
    y_fire = []
    y_fuel: list[int | None] = []
    mask = torch.zeros(len(batch), T_max, dtype=torch.float32)

    for i, (x, yf, yfu) in enumerate(batch):
        T = x.shape[0]
        pad_len = T_max - T
        if pad_len > 0:
            pad = torch.zeros(pad_len, x.shape[1], dtype=x.dtype)
            x_padded = torch.cat([x, pad], dim=0)
            y_pad = torch.zeros(pad_len, dtype=yf.dtype)
            y_padded = torch.cat([yf, y_pad], dim=0)
        else:
            x_padded = x
            y_padded = yf
        xs.append(x_padded)
        y_fire.append(y_padded)
        y_fuel.append(yfu)
        mask[i, :T] = 1.0

    xs = torch.stack(xs, dim=0)  # (B, T_max, feature_dim)
    lengths = torch.tensor(lengths, dtype=torch.long)
    y_fire = torch.stack(y_fire, dim=0).float()

    fuel_available = all(item is not None for item in y_fuel)
    if fuel_available:
        y_fuel_tensor: torch.Tensor | None = torch.tensor(y_fuel, dtype=torch.long)
    else:
        y_fuel_tensor = None

    return xs, lengths, y_fire, y_fuel_tensor, mask
