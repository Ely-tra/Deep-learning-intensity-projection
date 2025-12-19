import numpy as np
import argparse
import pathlib
import sys

import torch
import torch.nn as nn

MODULE_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from module.data import create_tc_loaders
from module.masks import extract_bc_rim_from_y, make_rim_mask_like, make_smooth_phi
from module.model import TC_AFNO_Intensity


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
    # ---- Compute output normalization stats from TRAIN set only ----
    # train_loader.dataset.Y is numpy array [N_train, V, H, W]
    Y_train_np = train_loader.dataset.Y
    y_mean = torch.from_numpy(Y_train_np.mean(axis=(0, 2, 3))).float()
    y_std = torch.from_numpy(Y_train_np.std(axis=(0, 2, 3))).float()
    print("y_std min/max:", y_std.min().item(), y_std.max().item(), flush=True)
    print("y_mean min/max:", y_mean.min().item(), y_mean.max().item(), flush=True)

    # ---- Compute input normalization stats from TRAIN set only ----
    # train_loader.dataset.X is numpy array [N_train, T, V, H, W]
    X_train_np = train_loader.dataset.X
    x_mean = torch.from_numpy(X_train_np.mean(axis=(0, 1, 3, 4))).float()
    x_std = torch.from_numpy(X_train_np.std(axis=(0, 1, 3, 4))).float()
    print("x_std min/max:", x_std.min().item(), x_std.max().item(), flush=True)
    print("x_mean min/max:", x_mean.min().item(), x_mean.max().item(), flush=True)

    model = TC_AFNO_Intensity(
        num_vars=args.num_vars,
        num_times=args.num_times,
        H=args.height,
        W=args.width,
        num_blocks=args.num_blocks,
        film_zdim=args.film_zdim,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        return_physical=False,   # IMPORTANT: train in normalized space
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

            # ---- Normalize input in training space ----
            if model.x_scaler is not None:
                x = model.x_scaler.norm(x)

            # ---- Normalize target in training space ----
            y = model.y_scaler.norm(y)   # [B, V, H, W]

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
                if model.x_scaler is not None:
                    x = model.x_scaler.norm(x)
                y = model.y_scaler.norm(y)

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
