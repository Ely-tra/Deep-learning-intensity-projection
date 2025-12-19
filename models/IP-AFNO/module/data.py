import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


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
