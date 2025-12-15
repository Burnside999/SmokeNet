# smokenet/utils/metrics.py

import torch

def masked_binary_accuracy(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> float:
    """
    Compute accuracy for variable-length sequences using a mask.

    Args:
        logits: (B, T)
        targets: (B, T)
        mask: (B, T) with 1.0 for valid steps, 0.0 for padding
    """
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).long()
    targets = targets.long()
    correct = ((preds == targets) * mask.long()).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0.0

def multiclass_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    logits: (B, C)
    targets: (B,)
    """
    preds = logits.argmax(dim=-1)
    correct = (preds == targets).sum().item()
    return correct / len(targets)
