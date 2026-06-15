"""
Mask-generation utilities for the masked-neighbor context path.

Three masking strategies on an (H, W) patch grid:
  - random:     i.i.d. Bernoulli per token (MAE-style baseline).
  - block:      small contiguous rectangular blocks; preferred for local defects.
  - horizontal: full-width horizontal stripe blocks (wider along EPI axis).

All functions return bool tensors where True = masked (target to reconstruct).
"""

from __future__ import annotations

import math
from typing import Tuple, Union

import torch


def build_mask(
    H: int,
    W: int,
    mask_ratio: float = 0.5,
    mask_type: str = "block",
    mask_block: Tuple[int, int] = (2, 2),
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Build a single binary mask for an (H, W) patch grid.

    Returns:
        BoolTensor of shape [H*W], True at positions to reconstruct.
    """
    assert 0.0 < mask_ratio < 1.0, f"mask_ratio must be in (0,1), got {mask_ratio}"
    N = H * W

    if mask_type == "random":
        # i.i.d. Bernoulli — each token independently masked with prob mask_ratio
        mask = torch.rand(N, device=device) < mask_ratio  # [N]

    elif mask_type == "block":
        bh, bw = mask_block
        assert H % bh == 0 and W % bw == 0, (
            f"Grid ({H},{W}) must be divisible by block size ({bh},{bw})"
        )
        gh, gw = H // bh, W // bw  # number of blocks along each axis
        num_blocks = gh * gw
        num_masked = max(1, round(mask_ratio * num_blocks))
        # Randomly select which blocks to mask
        perm = torch.randperm(num_blocks, device=device)
        block_mask = torch.zeros(num_blocks, dtype=torch.bool, device=device)
        block_mask[perm[:num_masked]] = True
        # Expand block mask to token mask: [gh, gw] -> [H, W]
        block_mask = block_mask.view(gh, gw)
        mask = block_mask.repeat_interleave(bh, dim=0).repeat_interleave(bw, dim=1)
        mask = mask.reshape(N)  # [H*W]

    elif mask_type == "horizontal":
        # Horizontal stripe blocks: each block spans the full width W, height = bh
        bh = mask_block[0]
        assert H % bh == 0, (
            f"Grid height ({H}) must be divisible by block height ({bh})"
        )
        num_stripes = H // bh
        num_masked = max(1, round(mask_ratio * num_stripes))
        perm = torch.randperm(num_stripes, device=device)
        stripe_mask = torch.zeros(num_stripes, dtype=torch.bool, device=device)
        stripe_mask[perm[:num_masked]] = True
        # Expand: [num_stripes] -> [H, W]
        mask = stripe_mask.unsqueeze(1).repeat_interleave(bh, dim=0).expand(-1, W)
        mask = mask.reshape(N)  # [H*W]

    else:
        raise ValueError(
            f"Unknown mask_type '{mask_type}'. Choose from: random, block, horizontal."
        )

    return mask


def build_batch_mask(
    B: int,
    H: int,
    W: int,
    mask_ratio: float = 0.5,
    mask_type: str = "block",
    mask_block: Tuple[int, int] = (2, 2),
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Build independent masks for a batch.

    Returns:
        BoolTensor of shape [B, H*W], True at positions to reconstruct.
    """
    masks = torch.stack(
        [build_mask(H, W, mask_ratio, mask_type, mask_block, device) for _ in range(B)]
    )
    return masks
