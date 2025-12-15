# smokenet/config.py

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

@dataclass
class DataConfig:
    data_dir: str = "dataset/data"
    label_dir: str = "dataset/label"
    split_ratio: float = 0.8
    window_size: int = 16
    channels: int = 3

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataConfig":
        return cls(**data)


@dataclass
class ModelConfig:
    in_channels: int = 3
    cnn_hidden: int = 32
    lstm_hidden: int = 64
    lstm_layers: int = 1
    num_fuel_classes: int = 5
    dropout: float = 0.3
    model_name: str = "smokenet"  # used in model factory
    enable_fuel_classification: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        return cls(**data)


@dataclass
class TrainingConfig:
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lambda_fire: float = 1.0
    lambda_fuel: float = 1.0
    device: str = "cuda"  # or "cpu"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        return cls(**data)


def load_config(path: str = "config/default.yaml") -> Tuple[DataConfig, ModelConfig, TrainingConfig]:
    """Load YAML configuration and convert to dataclasses."""

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found at {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        raw_cfg: Dict[str, Any] = yaml.safe_load(f)

    data_cfg = DataConfig.from_dict(raw_cfg.get("data", {}))
    model_cfg = ModelConfig.from_dict(raw_cfg.get("model", {}))
    train_cfg = TrainingConfig.from_dict(raw_cfg.get("training", {}))

    return data_cfg, model_cfg, train_cfg