import torch
import torch.nn as nn
import numpy as np


class VibrationFeatureExtractor(nn.Module):
    def __init__(self):
        super(VibrationFeatureExtractor, self).__init__()

        # Wide kernel (e.g., 64) to capture long-period patterns in 1000-point signal
        self.features = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=32, kernel_size=64, stride=4, padding=32
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # Reduces the temporal dimension to exactly 5, regardless of input variations
            nn.AdaptiveAvgPool1d(5),
        )

        # Flattened size: 64 channels * 5 pooling size = 320
        self.feature_dim = 64 * 5

    def forward(self, x):
        # Input shape: (batch_size, 1, 1000)
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 320)
        return x


class ContrastiveModel(nn.Module):
    def __init__(self, encoder, embedding_dim=64, num_classes=2):
        super(ContrastiveModel, self).__init__()
        self.encoder = encoder

        # Projection head for Contrastive Learning (discarded after pre-training)
        self.projection_head = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )

        # Classification head for fine-tuning
        self.classifier_head = nn.Linear(self.encoder.feature_dim, num_classes)

        self.mode = "contrastive"  # Switch between 'contrastive' and 'classify'

    def set_mode(self, mode):
        assert mode in ["contrastive", "classify"]
        self.mode = mode

    def forward(self, x):
        features = self.encoder(x)
        if self.mode == "contrastive":
            return self.projection_head(features)
        else:
            return self.classifier_head(features)
