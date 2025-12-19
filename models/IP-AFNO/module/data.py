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
        fields = torch.from_numpy(self.X[idx]).float()
        target = torch.from_numpy(self.Y[idx]).float()

        return {"fields": fields, "target_fields": target}


def _compute_split_indices(N: int, train_frac: float, val_frac: float):
    """
    Ordered split (strip-first):
    train = first chunk, val = next chunk, test = remaining, all in original data order.
    """
    if not (0 < train_frac < 1) or not (0 < val_frac < 1):
        raise ValueError("train_frac and val_frac must be in (0, 1).")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be < 1 (need room for test).")

    n_train = int(round(N * train_frac))
    n_val   = int(round(N * val_frac))

    n_train = max(1, n_train)
    n_val   = max(1, n_val)
    if n_train + n_val >= N:
        n_val = max(1, N - n_train - 1)

    train_idx = np.arange(0, n_train, dtype=np.int64)
    val_idx   = np.arange(n_train, n_train + n_val, dtype=np.int64)
    test_idx  = np.arange(n_train + n_val, N, dtype=np.int64)

    if train_idx.size == 0 or val_idx.size == 0 or test_idx.size == 0:
        raise ValueError(f"Bad split sizes: train={train_idx.size}, val={val_idx.size}, test={test_idx.size}")
    return train_idx, val_idx, test_idx


def create_tc_loaders(
    np_path,
    train_frac=0.7,
    val_frac=0.2,
    step_in=3,
    batch_size=8,
    num_workers=0,
    pin_memory=True,
):
    """
    Create deterministic train/val loaders from data shaped [N, T, H, W, C].
    """
    # mmap_mode avoids loading the whole array into RAM immediately
    data = np.load(np_path, mmap_mode="r")
    if data.ndim != 5:
        raise ValueError(f"Expected [N, T, H, W, C], got {data.shape}")

    N, T, H, W, V = data.shape
    if T < step_in + 1:
        raise ValueError("Need at least step_in+1 temporal frames")

    X_tmp = data[:, :step_in, ...]      # [N, step_in, H, W, C]
    Y_tmp = data[:, step_in, ...]       # [N, H, W, C]

    X = np.transpose(X_tmp, (0, 1, 4, 2, 3))  # [N, step_in, C, H, W]
    Y = np.transpose(Y_tmp, (0, 3, 1, 2))     # [N, C, H, W]

    train_idx, val_idx, _ = _compute_split_indices(N, train_frac, val_frac)

    train_dataset = TCDataset(X[train_idx], Y[train_idx])
    val_dataset   = TCDataset(X[val_idx],   Y[val_idx])


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader

def create_test_loader(
    np_path,
    train_frac=0.7,
    val_frac=0.2,
    step_in=3,
    batch_size=8,
    num_workers=0,
    pin_memory=True,
):
    """
    Create deterministic test loader matching the same split used by create_tc_loaders
    when called with the same (np_path, train_frac, val_frac).
    """
    data = np.load(np_path, mmap_mode="r")
    if data.ndim != 5:
        raise ValueError(f"Expected [N, T, H, W, C], got {data.shape}")

    N, T, H, W, V = data.shape
    if T < step_in + 1:
        raise ValueError("Need at least step_in+1 temporal frames")

    X_tmp = data[:, :step_in, ...]
    Y_tmp = data[:, step_in, ...]

    X = np.transpose(X_tmp, (0, 1, 4, 2, 3))
    Y = np.transpose(Y_tmp, (0, 3, 1, 2))

    _, _, test_idx = _compute_split_indices(N, train_frac, val_frac)

    test_dataset = TCDataset(X[test_idx], Y[test_idx])

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    return test_loader


def create_tc_loaders_with_test(
    np_path,
    train_frac=0.7,
    val_frac=0.2,
    step_in=3,
    batch_size=8,
    num_workers=0,
    pin_memory=True,
):
    """
    Create deterministic train/val/test loaders from data shaped [N, T, H, W, C].
    """
    data = np.load(np_path, mmap_mode="r")
    if data.ndim != 5:
        raise ValueError(f"Expected [N, T, H, W, C], got {data.shape}")

    N, T, H, W, V = data.shape
    if T < step_in + 1:
        raise ValueError("Need at least step_in+1 temporal frames")

    X_tmp = data[:, :step_in, ...]
    Y_tmp = data[:, step_in, ...]

    X = np.transpose(X_tmp, (0, 1, 4, 2, 3))
    Y = np.transpose(Y_tmp, (0, 3, 1, 2))

    train_idx, val_idx, test_idx = _compute_split_indices(N, train_frac, val_frac)

    train_dataset = TCDataset(X[train_idx], Y[train_idx])
    val_dataset = TCDataset(X[val_idx], Y[val_idx])
    test_dataset = TCDataset(X[test_idx], Y[test_idx])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader, test_loader
