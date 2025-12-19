import torch
import torch.nn as nn


class ChannelStandardScaler(nn.Module):
    """
    Per-channel standardization for field tensors.
    Stores mean/std as buffers so they are saved inside state_dict.
    Broadcasting shape: [1, C, 1, 1]
    """
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.eps = eps

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)

    def denorm(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.std + self.eps) + self.mean
