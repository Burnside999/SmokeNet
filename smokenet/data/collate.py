# smokenet/data/collate.py

from typing import List, Tuple
import torch

def smoke_collate_fn(
    batch: List[Tuple[torch.Tensor, int, int]]
):
    """
    batch: list of (x, y_fire, y_fuel)
    x: (T_i, 3)
    """
    lengths = [x.shape[0] for x, _, _ in batch]
    T_max = max(lengths)
    
    xs = []
    y_fire = []
    y_fuel = []
    
    for (x, yf, yfu) in batch:
        T = x.shape[0]
        pad_len = T_max - T
        if pad_len > 0:
            pad = torch.zeros(pad_len, x.shape[1], dtype=x.dtype)
            x_padded = torch.cat([x, pad], dim=0)
        else:
            x_padded = x
        xs.append(x_padded)
        y_fire.append(yf)
        y_fuel.append(yfu)
    
    xs = torch.stack(xs, dim=0)              # (B, T_max, 3)
    lengths = torch.tensor(lengths, dtype=torch.long)
    y_fire = torch.tensor(y_fire, dtype=torch.float32)  # BCEWithLogits
    y_fuel = torch.tensor(y_fuel, dtype=torch.long)     # CrossEntropy
    
    return xs, lengths, y_fire, y_fuel
