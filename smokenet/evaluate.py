# smokenet/evaluate.py


import torch
from torch.utils.data import DataLoader

from .utils.metrics import masked_binary_accuracy, multiclass_accuracy


@torch.no_grad()
def evaluate(model, dataloader: DataLoader, device: torch.device, fuel_enabled: bool):
    model.eval()
    total_fire_acc = 0.0
    total_fuel_acc = 0.0
    n_batches = 0

    for xs, lengths, y_fire, y_fuel, mask in dataloader:
        xs = xs.to(device)
        lengths = lengths.to(device)
        y_fire = y_fire.to(device)
        mask = mask.to(device)
        y_fuel = y_fuel.to(device) if y_fuel is not None else None

        fire_logits, fuel_logits = model(xs, lengths)

        total_fire_acc += masked_binary_accuracy(fire_logits, y_fire, mask)
        if fuel_enabled and fuel_logits is not None and y_fuel is not None:
            total_fuel_acc += multiclass_accuracy(fuel_logits, y_fuel)
        n_batches += 1

    if n_batches == 0:
        print("[Eval] Warning: validation dataloader is empty")
        return

    fire_acc = total_fire_acc / n_batches
    fuel_acc: float | None = total_fuel_acc / n_batches if fuel_enabled else None
    fuel_log = f" fuel_acc={fuel_acc:.4f}" if fuel_acc is not None else " fuel_acc=N/A"
    print(f"[Eval] fire_acc={fire_acc:.4f}{fuel_log}")
