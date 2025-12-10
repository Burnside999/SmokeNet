# smokenet/evaluate.py

import torch
from torch.utils.data import DataLoader

from .utils.metrics import binary_accuracy, multiclass_accuracy

@torch.no_grad()
def evaluate(model, dataloader: DataLoader, device: torch.device):
    model.eval()
    total_fire_acc = 0.0
    total_fuel_acc = 0.0
    n_batches = 0
    
    for xs, lengths, y_fire, y_fuel in dataloader:
        xs = xs.to(device)
        lengths = lengths.to(device)
        y_fire = y_fire.to(device)
        y_fuel = y_fuel.to(device)
        
        fire_logits, fuel_logits = model(xs, lengths)
        
        total_fire_acc += binary_accuracy(fire_logits, y_fire)
        total_fuel_acc += multiclass_accuracy(fuel_logits, y_fuel)
        n_batches += 1
    
    print(
        f"[Eval] fire_acc={total_fire_acc / n_batches:.4f} "
        f"fuel_acc={total_fuel_acc / n_batches:.4f}"
    )
