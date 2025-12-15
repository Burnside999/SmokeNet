# main.py

import argparse

from torch.utils.data import DataLoader

from smokenet.config import load_config
from smokenet.data.collate import smoke_collate_fn
from smokenet.data.loader import load_datasets
from smokenet.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument(
        "--config", default="config/default.yaml", help="Path to config YAML file"
    )
    parser.add_argument(
        "--model", default=None, help="Path to model checkpoint (eval mode only)"
    )
    parser.add_argument("--batch-size", type=int, help="Override training batch size")
    parser.add_argument("--num-epochs", type=int, help="Override training epochs")
    parser.add_argument(
        "--learning-rate", type=float, help="Override optimizer learning rate"
    )
    parser.add_argument("--device", type=str, help="Override training device")
    parser.add_argument("--window-size", type=int, help="Override data window size")
    args = parser.parse_args()

    data_cfg, model_cfg, train_cfg = load_config(args.config)

    def warn_override(cfg, attr, value, label: str):
        if value is None:
            return
        old = getattr(cfg, attr)
        if value != old:
            print(f"[WARN] override{label}: {old} -> {value}")
        setattr(cfg, attr, value)

    warn_override(train_cfg, "batch_size", args.batch_size, "batch_size")
    warn_override(train_cfg, "num_epochs", args.num_epochs, "num_epochs")
    warn_override(train_cfg, "learning_rate", args.learning_rate, "learning_rate")
    warn_override(train_cfg, "device", args.device, "device")
    warn_override(data_cfg, "window_size", args.window_size, "window_size")
    # model accepts raw sensor channels; windowing handled by dataset
    model_cfg.in_channels = data_cfg.channels

    train_dataset, val_dataset = load_datasets(data_cfg)

    base_dataset = (
        train_dataset.dataset if hasattr(train_dataset, "dataset") else train_dataset
    )
    fuel_available = getattr(base_dataset, "fuel_available", False)
    fuel_enabled = model_cfg.enable_fuel_classification and fuel_available

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        collate_fn=smoke_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        collate_fn=smoke_collate_fn,
    )

    if args.mode == "train":
        if args.model:
            print("[WARN] --model only used in eval mode, ignored in train mode.")
        model, _ = train(train_loader, val_loader, model_cfg, train_cfg, fuel_enabled)
    else:
        if not args.model:
            print("[WARN] no --model provided, cannot load weights in eval mode.")
        else:
            print(f"[WARN] eval mode not implemented, skip loading model: {args.model}")


if __name__ == "__main__":
    main()
