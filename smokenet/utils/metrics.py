# smokenet/utils/metrics.py

import torch

def binary_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    logits: (B,)
    targets: (B,)
    """
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).long()
    targets = targets.long()
    correct = (preds == targets).sum().item()
    return correct / len(targets)

def multiclass_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    logits: (B, C)
    targets: (B,)
    """
    preds = logits.argmax(dim=-1)
    correct = (preds == targets).sum().item()
    return correct / len(targets)
