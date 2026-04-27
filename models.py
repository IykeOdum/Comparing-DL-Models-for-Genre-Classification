# models.py
# Three neural network architectures for music genre classification.
#
#   1. MLP         – flat baseline (flattened spectrogram features)
#   2. ShallowCNN  – 2–3 conv layers, minimal depth
#   3. DeepCNN     – 5 conv layers with residual-style connections
#
# All models accept input of shape (B, 1, N_MELS, T) and output (B, NUM_CLASSES).

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# ── Utility blocks ─────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → optional MaxPool."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        padding: int = 1,
        pool: bool = True,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── 1. MLP baseline ────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """Multilayer Perceptron operating on flattened log-Mel spectrogram.

    Architecture:
        Flatten → FC(2048) → BN → ReLU → Dropout
                → FC(1024) → BN → ReLU → Dropout
                → FC(512)  → BN → ReLU → Dropout
                → FC(NUM_CLASSES)

    The spectrogram is spatially averaged across the time axis before
    flattening to produce a fixed-size frequency-averaged representation,
    which keeps the input size tractable regardless of audio duration.
    """

    def __init__(
        self,
        n_mels: int = config.N_MELS,
        num_classes: int = config.NUM_CLASSES,
        dropout: float = config.DROPOUT_RATE,
    ):
        super().__init__()
        # We'll average over time so input_dim = n_mels
        input_dim = n_mels

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, N_MELS, T)
        x = x.squeeze(1)            # (B, N_MELS, T)
        x = x.mean(dim=-1)          # average over time → (B, N_MELS)
        return self.classifier(x)


# ── 2. Shallow CNN ─────────────────────────────────────────────────────────────

class ShallowCNN(nn.Module):
    """Two-stage CNN with three convolutional blocks.

    Architecture:
        Input (B, 1, 128, T)
        ConvBlock(1→32,  pool) → (B, 32, 64, T/2)
        ConvBlock(32→64, pool) → (B, 64, 32, T/4)
        ConvBlock(64→128,pool) → (B,128, 16, T/8)
        AdaptiveAvgPool → (B, 128, 1, 1)
        FC(128→256) → ReLU → Dropout → FC(256→NUM_CLASSES)

    Adaptive pooling decouples the classifier from the time dimension,
    so it works with any audio length.
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        dropout: float = config.DROPOUT_RATE,
    ):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(1,   32,  pool=True),   # 128→64, T→T/2
            ConvBlock(32,  64,  pool=True),   # 64→32,  T/2→T/4
            ConvBlock(64,  128, pool=True),   # 32→16,  T/4→T/8
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # → (B, 128, 1, 1)
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)       # (B, 128, h, w)
        x = self.pool(x)           # (B, 128, 1, 1)
        x = x.flatten(1)           # (B, 128)
        return self.classifier(x)


# ── 3. Deep CNN ────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Two ConvBlock units with a skip connection.

    If in_ch != out_ch, a 1×1 projection aligns the residual.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, out_ch, pool=False)
        self.conv2 = ConvBlock(out_ch, out_ch, pool=False)
        # Projection shortcut
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x)) + self.shortcut(x)


class DeepCNN(nn.Module):
    """Five-stage CNN with residual connections and progressive channel growth.

    Architecture:
        Input (B, 1, 128, T)
        Stem:  ConvBlock(1→32, pool)         → (B, 32,  64, T/2)
        Stage1: ResBlock(32→64)  + MaxPool   → (B, 64,  32, T/4)
        Stage2: ResBlock(64→128) + MaxPool   → (B, 128, 16, T/8)
        Stage3: ResBlock(128→256)+ MaxPool   → (B, 256,  8, T/16)
        Head:  AdaptiveAvgPool → Dropout → FC(256→512) → ReLU → Dropout → FC(512→NC)

    Total convolutional layers: 1 (stem) + 2×3 (residual stages) + head = 7 conv ops,
    but five distinct "blocks" of feature extraction depth.
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        dropout: float = config.DROPOUT_RATE,
    ):
        super().__init__()

        # Stem
        self.stem = ConvBlock(1, 32, kernel=7, padding=3, pool=True)   # large receptive field

        # Residual stages – each followed by MaxPool to downsample
        self.stage1 = nn.Sequential(ResidualBlock(32, 64),  nn.MaxPool2d(2, 2))
        self.stage2 = nn.Sequential(ResidualBlock(64, 128), nn.MaxPool2d(2, 2))
        self.stage3 = nn.Sequential(ResidualBlock(128, 256),nn.MaxPool2d(2, 2))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)           # (B, 32,  64, T/2)
        x = self.stage1(x)         # (B, 64,  32, T/4)
        x = self.stage2(x)         # (B, 128, 16, T/8)
        x = self.stage3(x)         # (B, 256,  8, T/16)
        x = self.pool(x)           # (B, 256, 1, 1)
        x = x.flatten(1)           # (B, 256)
        return self.classifier(x)


# ── Model registry ─────────────────────────────────────────────────────────────

MODELS = {
    "MLP":       MLP,
    "ShallowCNN": ShallowCNN,
    "DeepCNN":   DeepCNN,
}


def get_model(name: str) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODELS.keys())}")
    return MODELS[name]()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity check – verifies all three models handle a dummy batch
    dummy = torch.randn(4, 1, config.N_MELS, 1292)   # 4 samples, 30s @ 22050 Hz
    for name, ModelClass in MODELS.items():
        model = ModelClass()
        out = model(dummy)
        print(f"{name:12s}  output: {tuple(out.shape)}  params: {count_parameters(model):,}")