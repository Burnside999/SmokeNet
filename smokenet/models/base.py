# smokenet/models/base.py

from abc import ABC, abstractmethod
import torch.nn as nn
from typing import Tuple
import torch

class BaseTemporalModel(nn.Module, ABC):
    # Universal interface for all temporal models, 
    # input: (batch, T, C)
    # output: depends on the task, e.g. (batch, num_classes) for classification
    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError
