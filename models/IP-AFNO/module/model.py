import torch
import torch.nn as nn

from module.blocks import BCEncoder, CondAFNO2DBlock
from module.normalization import ChannelStandardScaler


class TC_AFNO_Intensity(nn.Module):
    def __init__(self, num_vars=11, num_times=3, H=100, W=100, num_blocks=4, film_zdim=64,
                 x_mean=None, x_std=None, y_mean=None, y_std=None, return_physical: bool = False):
        super().__init__()
        in_channels_main = num_vars * num_times
        in_channels_bc = (num_vars + 1)   # bc_in = [B_fill, bc_mask]

        self.num_vars = num_vars
        self.num_times = num_times

        self.stem = nn.Conv2d(in_channels_main + in_channels_bc, 64, kernel_size=3, padding=1)

        # --- BC encoder (FiLM conditioning) ---
        self.bc_encoder = BCEncoder(in_channels=num_vars + 1, z_dim=film_zdim)

        # --- AFNO stack with FiLM per block ---
        self.blocks = nn.ModuleList([
            CondAFNO2DBlock(channels=64, z_dim=film_zdim) for _ in range(num_blocks)
        ])

        self.out_conv = nn.Conv2d(64, num_vars, kernel_size=1)
        # --- Input scaler ---
        self.x_scaler = None
        if (x_mean is not None) and (x_std is not None):
            self.x_scaler = ChannelStandardScaler(x_mean, x_std)
        # --- Output scaler ---
        self.return_physical = return_physical
        self.y_scaler = None
        if (y_mean is not None) and (y_std is not None):
            self.y_scaler = ChannelStandardScaler(y_mean, y_std)

    def forward(self, x: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
        """
        x:  [B, T, V, H, W]
        bc: [B, V+1, H, W]   ([BC field, rim mask])
        """
        B, T, V, H, W = x.shape
        assert T == self.num_times and V == self.num_vars, "Unexpected input shape."
        expected_bc_ch = self.num_vars + 1
        assert (
            bc.ndim == 4
            and bc.shape[0] == B
            and bc.shape[1] == expected_bc_ch
        ), f"bc must be [B,{expected_bc_ch},H,W]."

        # Encode BC -> z
        z = self.bc_encoder(bc)  # [B, film_zdim]

        # Main AFNO trunk (inject BC as extra channels)
        x = x.reshape(B, T * V, H, W)             # [B, T*V, H, W]
        assert bc.shape[-2:] == (H, W), f"bc spatial {bc.shape[-2:]} != {(H,W)}"
        x = torch.cat([x, bc], dim=1)             # [B, T*V + (V+1), H, W]
        x = self.stem(x)

        for blk in self.blocks:
            x = blk(x, z)

        out = self.out_conv(x)  # [B, V, H, W]

        # If requested, convert normalized output back to physical units
        if self.return_physical and (self.y_scaler is not None):
            out = self.y_scaler.denorm(out)

        return out
