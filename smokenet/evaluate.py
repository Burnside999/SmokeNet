# smokenet/evaluate.py

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from .utils.metrics import masked_binary_accuracy, multiclass_accuracy


@dataclass
class EvaluationResult:
    fire_accuracy: float
    fuel_accuracy: float | None
    per_sample_fire: list[tuple[str, float]]
    per_sample_fuel: list[tuple[str, float]] | None


@torch.no_grad()
def evaluate(
    model, dataloader: DataLoader, device: torch.device, fuel_enabled: bool
) -> EvaluationResult:
    model.eval()
    total_fire_acc = 0.0
    total_fuel_acc = 0.0
    n_batches = 0
    per_sample_fire: list[tuple[str, float]] = []
    per_sample_fuel: list[tuple[str, float]] = []

    for xs, lengths, y_fire, y_fuel, mask, names in dataloader:
        xs = xs.to(device)
        lengths = lengths.to(device)
        y_fire = y_fire.to(device)
        mask = mask.to(device)
        y_fuel = y_fuel.to(device) if y_fuel is not None else None

        fire_logits, fuel_logits = model(xs, lengths)

        total_fire_acc += masked_binary_accuracy(fire_logits, y_fire, mask)

        probs = torch.sigmoid(fire_logits)
        preds = (probs > 0.5).long()
        correct = ((preds == y_fire.long()) * mask.long()).sum(dim=1)
        totals = mask.sum(dim=1).clamp(min=1)
        sample_acc = (correct / totals).cpu().tolist()
        per_sample_fire.extend(zip(names, sample_acc))

        if fuel_enabled and fuel_logits is not None and y_fuel is not None:
            total_fuel_acc += multiclass_accuracy(fuel_logits, y_fuel)
            fuel_preds = fuel_logits.argmax(dim=-1)
            sample_fuel_acc = (fuel_preds == y_fuel).float().cpu().tolist()
            per_sample_fuel.extend(zip(names, sample_fuel_acc))
        n_batches += 1

    if n_batches == 0:
        print("[Eval] Warning: validation dataloader is empty")
        return EvaluationResult(0.0, None, [], None)

    fire_acc = total_fire_acc / n_batches
    fuel_acc: float | None = total_fuel_acc / n_batches if fuel_enabled else None
    fuel_log = f" fuel_acc={fuel_acc:.4f}" if fuel_acc is not None else " fuel_acc=N/A"
    print(f"[Eval] fire_acc={fire_acc:.4f}{fuel_log}")

    return EvaluationResult(
        fire_accuracy=fire_acc,
        fuel_accuracy=fuel_acc,
        per_sample_fire=per_sample_fire,
        per_sample_fuel=per_sample_fuel if fuel_enabled else None,
    )
