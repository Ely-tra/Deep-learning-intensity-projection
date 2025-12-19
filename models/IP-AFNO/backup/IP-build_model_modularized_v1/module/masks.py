import math

import torch


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
