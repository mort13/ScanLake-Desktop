"""WordCNN: whole-word classifier for deposit/material names (from Rock Capture CNN).

Input (N, 1, 32, 256) grayscale in 0..1 -> output (N, num_classes) logits.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

INPUT_H, INPUT_W = 32, 256


class WordCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * (INPUT_H // 8) * (INPUT_W // 8), 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))  # 32 x 16 x 128
        x = self.pool(F.relu(self.conv2(x)))  # 64 x 8 x 64
        x = self.pool(F.relu(self.conv3(x)))  # 128 x 4 x 32
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)
