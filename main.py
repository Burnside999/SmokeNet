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
    args = parser.parse_args()

    data_cfg, model_cfg, train_cfg = load_config()
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
        model, _ = train(train_loader, val_loader, model_cfg, train_cfg, fuel_enabled)
    else:
        # TODO: 从文件加载模型后 evaluate(...)
        pass


if __name__ == "__main__":
    main()
