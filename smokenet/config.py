# smokenet/config.py

from dataclasses import dataclass

@dataclass
class ModelConfig:
    in_channels: int = 3
    cnn_hidden: int = 32
    lstm_hidden: int = 64
    lstm_layers: int = 1
    num_fuel_classes: int = 5
    dropout: float = 0.3
    model_name: str = "smokenet" # used in model factory

@dataclass
class TrainingConfig:
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lambda_fire: float = 1.0
    lambda_fuel: float = 1.0
    device: str = "cuda"  # or "cpu"
