# smokenet/train.py

from typing import Tuple
import torch
from torch.utils.data import DataLoader

from .config import ModelConfig, TrainingConfig
from .models import build_model
from .utils.seed import set_seed
from .utils.metrics import binary_accuracy, multiclass_accuracy

def train_one_epoch(
    model,
    dataloader: DataLoader,
    optimizer,
    device,
    lambda_fire: float,
    lambda_fuel: float
) -> Tuple[float, float, float]:
    model.train()
    criterion_fire = torch.nn.BCEWithLogitsLoss()
    criterion_fuel = torch.nn.CrossEntropyLoss()
    
    total_loss = 0.0
    total_fire_acc = 0.0
    total_fuel_acc = 0.0
    n_batches = 0
    
    for xs, lengths, y_fire, y_fuel in dataloader:
        xs = xs.to(device)
        lengths = lengths.to(device)
        y_fire = y_fire.to(device)
        y_fuel = y_fuel.to(device)
        
        fire_logits, fuel_logits = model(xs, lengths)
        
        loss_fire = criterion_fire(fire_logits, y_fire)
        loss_fuel = criterion_fuel(fuel_logits, y_fuel)
        loss = lambda_fire * loss_fire + lambda_fuel * loss_fuel
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_fire_acc += binary_accuracy(fire_logits.detach(), y_fire.detach())
        total_fuel_acc += multiclass_accuracy(fuel_logits.detach(), y_fuel.detach())
        n_batches += 1
    
    return (
        total_loss / n_batches,
        total_fire_acc / n_batches,
        total_fuel_acc / n_batches,
    )

def train(
    train_loader: DataLoader,
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig
):
    set_seed(42)
    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")
    
    model = build_model(
        model_cfg.model_name,
        in_channels=model_cfg.in_channels,
        cnn_hidden=model_cfg.cnn_hidden,
        lstm_hidden=model_cfg.lstm_hidden,
        lstm_layers=model_cfg.lstm_layers,
        num_fuel_classes=model_cfg.num_fuel_classes,
        dropout=model_cfg.dropout,
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay
    )
    
    for epoch in range(1, train_cfg.num_epochs + 1):
        loss, fire_acc, fuel_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train_cfg.lambda_fire,
            train_cfg.lambda_fuel
        )
        print(
            f"[Epoch {epoch}] "
            f"loss={loss:.4f} fire_acc={fire_acc:.4f} fuel_acc={fuel_acc:.4f}"
        )
    
    return model
