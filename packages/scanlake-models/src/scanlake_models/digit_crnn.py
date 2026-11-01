"""DigitCRNN: CNN + BiLSTM + CTC for digit strips (from Rock Capture CNN, state_dict compatible).

Input (N, 1, 32, 256) grayscale in 0..1 -> output (T=64, N, 14) log-probabilities.
CTC alphabet: blank = 0, then "0123456789.%m" at indices 1..13. The "m" (milli, as in
"25.26m SCU") was added later: models trained before it have 13 outputs and LEGACY_CHAR_CLASSES,
which their metadata states, so they keep working.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

INPUT_H, INPUT_W = 32, 256
TIME_STEPS = INPUT_W // 4  # two width halvings in the backbone
BLANK_IDX = 0
LEGACY_CHAR_CLASSES = "0123456789.%"  # Rock Capture CNN and the first ScanLake models
CHAR_CLASSES = LEGACY_CHAR_CLASSES + "m"
NUM_CLASSES = len(CHAR_CLASSES) + 1


class DigitCRNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32 x 16 x 128
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64 x 8 x 64
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # 128 x 4 x 64: pool height only, keep the time steps
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.rnn = nn.LSTM(
            input_size=128 * 4,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            dropout=0.3,
            batch_first=False,
        )
        self.fc = nn.Linear(256 * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)  # (N, 128, 4, T)
        n, c, h, t = features.shape
        sequence = features.view(n, c * h, t).permute(2, 0, 1)  # (T, N, 512)
        out, _ = self.rnn(sequence)
        return F.log_softmax(self.fc(out), dim=2)
