# smokenet/models/base.py

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseTemporalModel(nn.Module, ABC):
    # Universal interface for all temporal models,
    # input: (batch, T, C)
    # output: depends on the task, e.g. (batch, num_classes) for classification
    @abstractmethod
    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError
