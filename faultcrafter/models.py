from __future__ import annotations

import torch
from torch import nn


class CNN1D(nn.Module):
    """Simple 1D CNN used as the fixed downstream fault evaluator."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.layer1 = self._block(1, 16, pool=False)
        self.layer2 = self._block(16, 32, pool=True)
        self.layer3 = self._block(32, 64, pool=True)
        self.layer4 = self._block(64, 128, pool=True)
        self.layer5 = self._block(128, 256, pool=True)
        self.layer6 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(4),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int, *, pool: bool) -> nn.Sequential:
        layers: list[nn.Module] = [
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.ndim != 3 or signal.shape[1] != 1:
            raise ValueError(f"expected [batch,1,length], got {tuple(signal.shape)}")
        hidden = self.layer1(signal)
        hidden = self.layer2(hidden)
        hidden = self.layer3(hidden)
        hidden = self.layer4(hidden)
        hidden = self.layer5(hidden)
        hidden = self.layer6(hidden)
        return self.classifier(hidden.flatten(start_dim=1))
