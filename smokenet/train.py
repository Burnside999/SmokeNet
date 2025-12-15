# smokenet/train.py


import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import ModelConfig, TrainingConfig
from .evaluate import EvaluationResult, evaluate
from .models import build_model
from .utils.metrics import masked_binary_accuracy, multiclass_accuracy
from .utils.seed import set_seed


def train_one_epoch(
    model,
    dataloader: DataLoader,
    optimizer,
    device,
    lambda_fire: float,
    lambda_fuel: float,
    fuel_enabled: bool,
) -> tuple[float, float, float | None]:
    model.train()
    criterion_fire = torch.nn.BCEWithLogitsLoss(reduction="none")
    criterion_fuel = torch.nn.CrossEntropyLoss() if fuel_enabled else None

    total_loss = 0.0
    total_fire_acc = 0.0
    total_fuel_acc = 0.0
    n_batches = 0

    for xs, lengths, y_fire, y_fuel, mask, _names in dataloader:
        xs = xs.to(device)
        lengths = lengths.to(device)
        y_fire = y_fire.to(device)
        mask = mask.to(device)
        y_fuel = y_fuel.to(device) if y_fuel is not None else None

        fire_logits, fuel_logits = model(xs, lengths)

        loss_fire = criterion_fire(fire_logits, y_fire)
        loss_fire = (loss_fire * mask).sum() / mask.sum().clamp(min=1.0)
        loss = lambda_fire * loss_fire

        if fuel_enabled:
            if fuel_logits is None or y_fuel is None:
                raise ValueError(
                    "Fuel classification is enabled but the model did not return logits or labels."
                )
            loss_fuel = criterion_fuel(fuel_logits, y_fuel)
            loss = loss + lambda_fuel * loss_fuel

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_fire_acc += masked_binary_accuracy(
            fire_logits.detach(), y_fire.detach(), mask.detach()
        )
        if fuel_enabled and fuel_logits is not None and y_fuel is not None:
            total_fuel_acc += multiclass_accuracy(fuel_logits.detach(), y_fuel.detach())
        n_batches += 1

    fuel_acc = total_fuel_acc / n_batches if fuel_enabled else None
    return (
        total_loss / n_batches,
        total_fire_acc / n_batches,
        fuel_acc,
    )


def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    fuel_enabled: bool,
) -> tuple[torch.nn.Module, EvaluationResult]:
    set_seed(42)
    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")

    output_root = Path(train_cfg.output_root)
    weights_dir = output_root / "weights"
    figures_dir = output_root / "figures"
    weights_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(
        model_cfg.model_name,
        in_channels=model_cfg.in_channels,
        cnn_hidden=model_cfg.cnn_hidden,
        lstm_hidden=model_cfg.lstm_hidden,
        lstm_layers=model_cfg.lstm_layers,
        num_classes=model_cfg.num_fuel_classes,
        dropout=model_cfg.dropout,
        enable_fuel_classification=fuel_enabled,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    epoch_bar = tqdm(
        range(1, train_cfg.num_epochs + 1),
        desc="Epochs",
        unit="epoch",
    )

    history: dict[str, list[Any]] = {
        "train_fire_acc": [],
        "val_fire_acc": [],
        "train_fuel_acc": [],
        "val_fuel_acc": [],
    }
    best_fire_acc = -float("inf")
    best_result: EvaluationResult | None = None
    last_result: EvaluationResult | None = None

    for epoch in epoch_bar:
        loss, fire_acc, fuel_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train_cfg.lambda_fire,
            train_cfg.lambda_fuel,
            fuel_enabled,
        )
        val_result = evaluate(model, val_loader, device, fuel_enabled)
        last_result = val_result

        history["train_fire_acc"].append(fire_acc)
        history["val_fire_acc"].append(val_result.fire_accuracy)
        history["train_fuel_acc"].append(fuel_acc)
        history["val_fuel_acc"].append(val_result.fuel_accuracy)

        epoch_bar.set_postfix(
            {
                "loss": f"{loss:.4f}",
                "fire_acc": f"{fire_acc:.4f}",
                "val_fire": f"{val_result.fire_accuracy:.4f}",
                "fuel_acc": f"{fuel_acc:.4f}" if fuel_acc is not None else "N/A",
                "val_fuel": f"{val_result.fuel_accuracy:.4f}"
                if val_result.fuel_accuracy is not None
                else "N/A",
            }
        )

        if val_result.fire_accuracy > best_fire_acc:
            best_fire_acc = val_result.fire_accuracy
            best_result = val_result
            _save_checkpoint(
                weights_dir / "best.pt",
                model,
                epoch,
                train_cfg,
                model_cfg,
                val_result,
            )

        if epoch % train_cfg.ckpt_epoch == 0:
            _save_checkpoint(
                weights_dir / f"epoch_{epoch}.pt",
                model,
                epoch,
                train_cfg,
                model_cfg,
                val_result,
            )

    if last_result is None:
        raise RuntimeError("Validation was not performed during training")

    _save_checkpoint(
        weights_dir / "last.pt",
        model,
        train_cfg.num_epochs,
        train_cfg,
        model_cfg,
        last_result,
    )

    best_to_report = best_result or last_result
    _save_history(output_root, history)
    _plot_epoch_accuracy(
        history,
        figures_dir / "epoch_accuracy.png",
        fuel_enabled,
    )
    _plot_topk_accuracy(
        last_result,
        figures_dir,
        fuel_enabled,
    )

    return model, best_to_report


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    epoch: int,
    train_cfg: TrainingConfig,
    model_cfg: ModelConfig,
    eval_result: EvaluationResult,
):
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "training_config": train_cfg.__dict__,
        "model_config": model_cfg.__dict__,
        "metrics": {
            "fire_accuracy": eval_result.fire_accuracy,
            "fuel_accuracy": eval_result.fuel_accuracy,
        },
    }
    torch.save(payload, path)


def _save_history(output_root: Path, history: dict[str, list[Any]]):
    history_path = output_root / "metrics.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def _plot_epoch_accuracy(
    history: dict[str, list[Any]], plot_path: Path, fuel_enabled: bool
):
    epochs = list(range(1, len(history["train_fire_acc"]) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_fire_acc"], label="Train Fire Acc")
    plt.plot(epochs, history["val_fire_acc"], label="Val Fire Acc")

    if fuel_enabled:
        plt.plot(epochs, history["train_fuel_acc"], label="Train Fuel Acc")
        plt.plot(epochs, history["val_fuel_acc"], label="Val Fuel Acc")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Epoch Accuracy")
    plt.legend()
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def _plot_topk_accuracy(
    eval_result: EvaluationResult, figures_dir: Path, fuel_enabled: bool
):
    figures_dir.mkdir(parents=True, exist_ok=True)
    _plot_bar_chart(
        eval_result.per_sample_fire,
        10,
        "Top 10 Best Sequences (Fire Accuracy)",
        figures_dir / "top10_fire_best.png",
        reverse=True,
    )
    _plot_bar_chart(
        eval_result.per_sample_fire,
        10,
        "Top 10 Worst Sequences (Fire Accuracy)",
        figures_dir / "top10_fire_worst.png",
        reverse=False,
    )

    if fuel_enabled and eval_result.per_sample_fuel is not None:
        _plot_bar_chart(
            eval_result.per_sample_fuel,
            10,
            "Top 10 Best Sequences (Fuel Accuracy)",
            figures_dir / "top10_fuel_best.png",
            reverse=True,
        )
        _plot_bar_chart(
            eval_result.per_sample_fuel,
            10,
            "Top 10 Worst Sequences (Fuel Accuracy)",
            figures_dir / "top10_fuel_worst.png",
            reverse=False,
        )


def _plot_bar_chart(
    entries: list[tuple[str, float]],
    k: int,
    title: str,
    path: Path,
    reverse: bool,
):
    if not entries:
        return
    sorted_entries = sorted(entries, key=lambda x: x[1], reverse=reverse)[:k]
    labels, values = zip(*sorted_entries)
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(values)), values, color="skyblue")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
