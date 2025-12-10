# main.py

import argparse
import torch
from torch.utils.data import DataLoader

from smokenet.config import ModelConfig, TrainingConfig
from smokenet.data.dataset import SmokeDataset
from smokenet.data.collate import smoke_collate_fn
from smokenet.train import train
from smokenet.evaluate import evaluate

def load_your_data():
    signals = []
    fire_labels = []
    fuel_labels = []
    
    # ... 你自己的加载逻辑 ...
    
    dataset = SmokeDataset(signals, fire_labels, fuel_labels)
    # 这里就先拆个 80/20
    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = n - n_train
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )
    return train_dataset, val_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    args = parser.parse_args()
    
    train_dataset, val_dataset = load_your_data()
    
    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()
    
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
        # 这里你可以加保存 model 的逻辑
        device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")
        evaluate(model.to(device), val_loader, device)
    else:
        # eval 模式：从文件加载模型后 evaluate(...)
        pass

if __name__ == "__main__":
    main()
