import argparse
import math

import pathlib

import numpy as np
import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in {"yes", "true", "t", "y", "1"}:
        return True
    if v in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train AFNO model (train + val only).")
    parser.add_argument("--data_path", default="/N/slate/kmluong/PROJECT2/level_2_data/wrf_tropical_cyclone_track_5_dataset_X.npy")
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--step_in", type=int, default=3, help="The number of frames used as input")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=str2bool, default=True)
    parser.add_argument("--num_vars", type=int, default=11)
    parser.add_argument("--num_times", type=int, default=3)
    parser.add_argument("--height", type=int, default=100)
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--num_blocks", type=int, default=4, help="Number of AFNO blocks, or hidden layers")
    parser.add_argument("--rim", type=int, default=3, help="Boundary pixels enforced from target.")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--film_zdim", type=int, default=64, help="FiLM conditioning embedding dim.")
    parser.add_argument("--checkpoint_dir", default="/N/slate/kmluong/PROJECT2/checkpoints")
    parser.add_argument("--checkpoint_name", default="best_model.pt")

    return parser.parse_args()


class TCDataset(Dataset):
    """Tiny wrapper that returns AFNO-friendly tensors."""

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        fields = torch.tensor(self.X[idx], dtype=torch.float32)
        target = torch.tensor(self.Y[idx], dtype=torch.float32)
        return {"fields": fields, "target_fields": target}


def make_smooth_phi(H: int, W: int, rim: int, device=None, dtype=torch.float32):
    """
    Smooth vanishing mask phi in [0,1] with width=rim pixels.
    phi = 0 on boundary and transitions to 1 with zero slope at rim.
    """
    if rim <= 0:
        return torch.ones(1, 1, H, W, device=device, dtype=dtype)

    yy = torch.arange(H, device=device, dtype=dtype).view(H, 1).repeat(1, W)
    xx = torch.arange(W, device=device, dtype=dtype).view(1, W).repeat(H, 1)

    d_top = yy
    d_left = xx
    d_bottom = (H - 1) - yy
    d_right = (W - 1) - xx
    d = torch.minimum(torch.minimum(d_top, d_bottom), torch.minimum(d_left, d_right))

    s = torch.clamp(d / float(rim), 0.0, 1.0)
    phi = torch.sin(0.5 * math.pi * s) ** 2
    return phi.view(1, 1, H, W)


def extract_bc_rim_from_y(y: torch.Tensor, rim: int):
    """
    Copy rim-thick boundary data from ground-truth y.
    Returns tensor with boundary filled and interior zeros.
    """
    if rim <= 0:
        return torch.zeros_like(y)

    B_fill = torch.zeros_like(y)
    B_fill[:, :, :rim, :] = y[:, :, :rim, :]
    B_fill[:, :, -rim:, :] = y[:, :, -rim:, :]
    B_fill[:, :, :, :rim] = y[:, :, :, :rim]
    B_fill[:, :, :, -rim:] = y[:, :, :, -rim:]
    return B_fill


def make_rim_mask_like(y: torch.Tensor, rim: int):
    """
    Binary mask (1 on enforced rim region, 0 interior) with same spatial size as y.
    """
    B, _, H, W = y.shape
    mask = torch.zeros((B, 1, H, W), device=y.device, dtype=y.dtype)
    if rim <= 0:
        return mask

    mask[:, :, :rim, :] = 1
    mask[:, :, -rim:, :] = 1
    mask[:, :, :, :rim] = 1
    mask[:, :, :, -rim:] = 1
    return mask


def create_tc_loaders(
    np_path,
    train_frac=0.7,
    val_frac=0.3,
    step_in=3,
    batch_size=8,
    num_workers=0,
    pin_memory=True,
):
    """
    Build train/val loaders from data shaped [N, T, H, W, C]; leftover slices go to validation.
    """

    data = np.load(np_path)
    assert data.ndim == 5, f"Expected [N, T, H, W, C], got {data.shape}"

    N, T, H, W, V = data.shape
    assert T >= step_in + 1, "Need at least step_in+1 temporal frames"
    assert 0 < train_frac < 1 and 0 < val_frac <= 1, "Fractions must be in (0, 1]"

    X_tmp = data[:, :step_in, ...]
    Y_tmp = data[:, step_in, ...]

    X = np.transpose(X_tmp, (0, 1, 4, 2, 3))
    Y = np.transpose(Y_tmp, (0, 3, 1, 2))

    N_train = int(N * train_frac)
    N_val = int(N * val_frac)

    if N_train + N_val < N:
        N_val = N - N_train
    if N_train == 0 or N_val == 0:
        raise ValueError("Insufficient samples for the requested split.")

    X_train, Y_train = X[:N_train], Y[:N_train]
    X_val, Y_val = X[N_train:N_train + N_val], Y[N_train:N_train + N_val]

    train_dataset = TCDataset(X_train, Y_train)
    val_dataset = TCDataset(X_val, Y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


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
        beta  = beta.unsqueeze(-1).unsqueeze(-1)   # [B, C, 1, 1]

        return gamma * h + beta


class TC_AFNO_Intensity(nn.Module):
    def __init__(self, num_vars=11, num_times=3, H=100, W=100, num_blocks=4, film_zdim=64):
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
        return out



def main():
    args = parse_args()

    train_loader, val_loader = create_tc_loaders(
        np_path=args.data_path,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        step_in=args.step_in,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    model = TC_AFNO_Intensity(
        num_vars=args.num_vars,
        num_times=args.num_times,
        H=args.height,
        W=args.width,
        num_blocks=args.num_blocks,
        film_zdim=args.film_zdim,
    ).to(device)


    phi = make_smooth_phi(
        H=args.height,
        W=args.width,
        rim=args.rim,
        device=device,
        dtype=torch.float32,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.MSELoss()
    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / args.checkpoint_name
    best_val_loss = float("inf")

    print(f"Using device: {device}", flush=True)
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        n_train_batches = 0

        for batch in train_loader:
            x = batch["fields"].to(device)
            y = batch["target_fields"].to(device)

            B_fill = extract_bc_rim_from_y(y, rim=args.rim)   # BC-rim (teacher forcing for now)
            bc_mask = make_rim_mask_like(y, rim=args.rim)
            bc_in = torch.cat([B_fill, bc_mask], dim=1)
            y_free = model(x, bc_in)                         # <-- FiLM conditioning happens here
            y_pred = phi * y_free + (1.0 - phi) * B_fill      # hard BC + smooth seam


            loss = loss_fn(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_train_batches += 1

        train_loss /= max(1, n_train_batches)

        model.eval()
        val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["fields"].to(device)
                y = batch["target_fields"].to(device)

                B_fill = extract_bc_rim_from_y(y, rim=args.rim)   # BC-rim (teacher forcing for now)
                bc_mask = make_rim_mask_like(y, rim=args.rim)
                bc_in = torch.cat([B_fill, bc_mask], dim=1)
                y_free = model(x, bc_in)                         # <-- FiLM conditioning happens here
                y_pred = phi * y_free + (1.0 - phi) * B_fill      # hard BC + smooth seam

                loss = loss_fn(y_pred, y)

                val_loss += loss.item()
                n_val_batches += 1

        val_loss /= max(1, n_val_batches)
        print(
            f"Epoch {epoch + 1}/{args.num_epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        , flush=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Saved best checkpoint to {checkpoint_path} (val_loss={val_loss:.6f})", flush=True)


if __name__ == "__main__":
    main()
