# smokenet/models/smokenet.py


import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseTemporalModel


class SmokeNet(BaseTemporalModel):
    # Conv1d -> BiLSTM -> masked mean pooling -> two heads(fire / none-fire + classification)
    def __init__(
        self,
        in_channels: int = 3,
        cnn_hidden: int = 32,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        num_classes: int = 5,
        dropout: float = 0.3,
        enable_fuel_classification: bool = True,
    ):
        super().__init__()

        self.enable_fuel_classification = enable_fuel_classification

        # 1D-CNN: (B, C, T) -> (B, cnn_hidden, T)
        self.conv1 = nn.Conv1d(in_channels, cnn_hidden, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(cnn_hidden)

        # BiLSTM: input: (B, T, cnn_hidden), output: (B, T, 2*lstm_hidden)
        self.lstm = nn.LSTM(
            input_size=cnn_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)
        feat_dim = 2 * lstm_hidden  # bidirectional

        # fire / none-fire head
        self.fc_fire = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim // 2, 1),  # logits
        )

        # classification head
        if self.enable_fuel_classification:
            self.fc_fuel = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(feat_dim // 2, num_classes),
            )
        else:
            self.fc_fuel = None

    def forward(self, x, lengths) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x:       (batch, T, channels, window_size)
            lengths: (batch,)
        """
        batch_size, T_max, channels, window = x.shape

        # Merge batch and time to extract a feature vector per sliding window.
        x = x.reshape(batch_size * T_max, channels, window)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Global average over the window dimension -> (B*T, cnn_hidden)
        x = x.mean(dim=-1)
        x = x.reshape(batch_size, T_max, -1)

        # pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        fire_logits = self.fc_fire(out).squeeze(-1)  # (B, T_max)

        # masked mean pooling for sequence-level fuel classification
        batch_size, T_max, feat_dim = out.size()
        device = out.device
        mask = torch.arange(T_max, device=device)[None, :] < lengths[:, None]
        mask = mask.float().unsqueeze(-1)  # (B, T_max, 1)

        out_masked = out * mask
        sum_feat = out_masked.sum(dim=1)  # (B, feat_dim)
        len_feat = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
        h_seq = sum_feat / len_feat  # (B, feat_dim)

        h_seq = self.dropout(h_seq)

        fuel_logits: torch.Tensor | None
        if self.enable_fuel_classification and self.fc_fuel is not None:
            fuel_logits = self.fc_fuel(h_seq)  # (B, num_classes)
        else:
            fuel_logits = None

        return fire_logits, fuel_logits
