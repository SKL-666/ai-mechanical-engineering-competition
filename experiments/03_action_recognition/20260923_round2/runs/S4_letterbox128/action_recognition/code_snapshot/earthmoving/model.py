import torch.nn as nn


class TinyActionCNN(nn.Module):
    """From scratch; single-frame baseline or order-invariant multi-frame mean pooling."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten())
        self.head = nn.Linear(64 * 4 * 4, 5)

    def forward(self, x):
        b, c, t, h, w = x.shape
        features = self.encoder(x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w))
        return self.head(features.reshape(b, t, -1).mean(1))
