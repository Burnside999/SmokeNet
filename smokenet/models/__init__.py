# smokenet/models/__init__.py

from typing import Dict, Type

from .base import BaseTemporalModel
from .smokenet import SmokeNet

MODEL_REGISTRY: Dict[str, Type[BaseTemporalModel]] = {
    "smokenet": SmokeNet,
    # 将来你可以在这里加： "your_new_model": YourNewModel,
}

def build_model(name: str, **kwargs) -> BaseTemporalModel:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {name}")
    return MODEL_REGISTRY[name](**kwargs)
