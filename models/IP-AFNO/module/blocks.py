import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F


class AFNO2DBlock(nn.Module):
    def __init__(self, channels, hidden_factor=2, hard_threshold=0.0):
        super().__init__()
        self.channels = channels
        self.hidden = channels * hidden_factor
        self.hard_threshold = hard_threshold

        self.linear1 = nn.Linear(2 * channels, self.hidden)
        self.linear2 = nn.Linear(self.hidden, 2 * channels)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels * 4, channels, kernel_size=1),
        )

        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape

        x_perm = x.permute(0, 2, 3, 1)
        x_norm = self.norm1(x_perm).permute(0, 3, 1, 2)

        x_fft = fft.rfft2(x_norm, norm="ortho")

        if self.hard_threshold > 0:
            ky = torch.fft.fftfreq(H, d=1.0).to(x.device)[:, None]
            kx = torch.fft.rfftfreq(W, d=1.0).to(x.device)[None, :]
            kk = torch.sqrt(ky ** 2 + kx ** 2)
            mask = (kk <= self.hard_threshold).float()
            x_fft = x_fft * mask

        real = x_fft.real
        imag = x_fft.imag
        x_cat = torch.cat([real, imag], dim=1)

        x_cat = x_cat.permute(0, 2, 3, 1)
        x_lin = self.linear1(x_cat)
        x_lin = F.gelu(x_lin)
        x_lin = self.linear2(x_lin)

        x_lin = x_lin.permute(0, 3, 1, 2)
        real2, imag2 = torch.chunk(x_lin, 2, dim=1)
        x_fft_new = torch.complex(real2, imag2)

        x_spec = fft.irfft2(x_fft_new, s=(H, W), norm="ortho")
        x1 = x + x_spec

        x1_perm = x1.permute(0, 2, 3, 1)
        x1_norm = self.norm2(x1_perm).permute(0, 3, 1, 2)

        x_mlp = self.mlp(x1_norm)
        out = x1 + x_mlp
        return out


class BCEncoder(nn.Module):
    """
    Encode BC field + rim mask [B, V+1, H, W] -> latent z [B, z_dim]
    """

    def __init__(self, in_channels: int, z_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),  # -> [B, 64, 1, 1]
        )
        self.proj = nn.Linear(64, z_dim)

    def forward(self, bc: torch.Tensor) -> torch.Tensor:
        # bc: [B, V+1, H, W]
        h = self.net(bc).squeeze(-1).squeeze(-1)  # [B, 64]
        z = self.proj(h)                          # [B, z_dim]
        return z


class CondAFNO2DBlock(nn.Module):
    """
    AFNO2DBlock + FiLM modulation: out = gamma(z) * out + beta(z)
    """
    def __init__(self, channels: int, z_dim: int, hidden_factor=2, hard_threshold=0.0):
        super().__init__()
        self.block = AFNO2DBlock(channels, hidden_factor=hidden_factor, hard_threshold=hard_threshold)
        self.film = nn.Linear(z_dim, 2 * channels)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W], z: [B, z_dim]
        h = self.block(x)

        gb = self.film(z)                 # [B, 2C]
        gamma, beta = gb.chunk(2, dim=-1) # [B, C], [B, C]
        gamma = 1.0 + gamma
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = beta.unsqueeze(-1).unsqueeze(-1)   # [B, C, 1, 1]

        return gamma * h + beta
