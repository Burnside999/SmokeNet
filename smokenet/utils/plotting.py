# smokenet/utils/plotting.py

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from smokenet.evaluate import EvaluationResult


def plot_epoch_accuracy(
    history: dict[str, list[Any]], plot_path: Path, fuel_enabled: bool
) -> None:
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


def plot_topk_accuracy(
    eval_result: EvaluationResult, figures_dir: Path, fuel_enabled: bool
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    _plot_bar_chart(
        eval_result.per_sample_fire,
        10,
        "Best Sequences (Fire Accuracy)",
        figures_dir / "top10_fire_best.png",
        reverse=True,
    )
    _plot_bar_chart(
        eval_result.per_sample_fire,
        10,
        "Worst Sequences (Fire Accuracy)",
        figures_dir / "top10_fire_worst.png",
        reverse=False,
    )

    if fuel_enabled and eval_result.per_sample_fuel is not None:
        _plot_bar_chart(
            eval_result.per_sample_fuel,
            10,
            "Best Sequences (Fuel Accuracy)",
            figures_dir / "top10_fuel_best.png",
            reverse=True,
        )
        _plot_bar_chart(
            eval_result.per_sample_fuel,
            10,
            "Worst Sequences (Fuel Accuracy)",
            figures_dir / "top10_fuel_worst.png",
            reverse=False,
        )


def _plot_bar_chart(
    entries: list[tuple[str, float]],
    k: int,
    base_title: str,
    path: Path,
    reverse: bool,
) -> None:
    if not entries:
        return

    sorted_entries = sorted(entries, key=lambda x: x[1], reverse=reverse)[:k]
    labels, values = zip(*sorted_entries)
    entries_to_plot = len(values)
    title_prefix = f"Top {entries_to_plot}" if entries_to_plot < k else f"Top {k}"
    title = f"{title_prefix} {base_title}"

    plt.figure(figsize=(10, 5))
    plt.bar(range(entries_to_plot), values, color="skyblue")
    plt.xticks(range(entries_to_plot), labels, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
