import torch.nn as nn
import torch.nn.functional as F


class CausalResidual(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.pad = 2*dilation
        self.conv = nn.Conv1d(channels, channels, 3, dilation=dilation)

    def forward(self, x):
        return F.relu(x + self.conv(F.pad(x, (self.pad, 0))))


class ActionNet(nn.Module):
    """Shared spatial backbone; last-frame or genuine ordered causal temporal aggregation."""
    def __init__(self, temporal=False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(1024, 64), nn.ReLU())
        self.temporal = nn.Sequential(*(CausalResidual(64, d) for d in [1, 2, 4])) if temporal else nn.Identity()
        self.head = nn.Conv1d(64, 5, 1)

    def timeline(self, x):
        b, c, t, h, w = x.shape
        feature = self.encoder(x.permute(0,2,1,3,4).reshape(b*t,c,h,w)).reshape(b,t,64).transpose(1,2)
        return self.head(self.temporal(feature))

    def forward(self, x):
        return self.timeline(x)[:, :, -1]
