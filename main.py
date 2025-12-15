# main.py

import argparse
import torch
from torch.utils.data import DataLoader

from smokenet.config import load_config
from smokenet.data.loader import load_datasets
from smokenet.data.collate import smoke_collate_fn
from smokenet.train import train
from smokenet.evaluate import evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    args = parser.parse_args()

    data_cfg, model_cfg, train_cfg = load_config()
    # match model input channels to windowed features
    model_cfg.in_channels = data_cfg.channels * data_cfg.window_size

    train_dataset, val_dataset = load_datasets(data_cfg)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        collate_fn=smoke_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        collate_fn=smoke_collate_fn
    )
    
    if args.mode == "train":
        model = train(train_loader, model_cfg, train_cfg)
        # TODO: 保存 model
        device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")
        evaluate(model.to(device), val_loader, device)
    else:
        # TODO: 从文件加载模型后 evaluate(...)
        pass

if __name__ == "__main__":
    main()
