# smokenet/models/__init__.py

from .base import BaseTemporalModel
from .smokenet import SmokeNet

MODEL_REGISTRY: dict[str, type[BaseTemporalModel]] = {
    "smokenet": SmokeNet,
}


def build_model(name: str, **kwargs) -> BaseTemporalModel:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {name}")
    return MODEL_REGISTRY[name](**kwargs)
