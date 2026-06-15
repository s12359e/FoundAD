#!/usr/bin/env python
"""Sanity check for masking utilities.

Run:
    python foundad/src/test_masking.py

Creates a dummy [B=2, N=1024, C=768] tensor (32x32 grid) and applies each
mask_type, printing masked-token counts and shapes.
"""

import sys, os
# Ensure the repo root is on the path so `from src.masking import ...` works
# when executed as `python foundad/src/test_masking.py` from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.masking import build_mask, build_batch_mask


def main():
    B, H, W, C = 2, 32, 32, 768
    N = H * W  # 1024
    dummy = torch.randn(B, N, C)

    print(f"Dummy tensor shape: {dummy.shape}  (B={B}, N={N}, C={C}, grid={H}x{W})")
    print("=" * 60)

    configs = [
        ("random",     0.5, (2, 2)),
        ("block",      0.5, (2, 2)),
        ("block",      0.4, (4, 4)),
        ("horizontal", 0.5, (2, 2)),
    ]

    for mask_type, ratio, block in configs:
        print(f"\nmask_type={mask_type!r}, mask_ratio={ratio}, mask_block={block}")

        # Single mask
        m = build_mask(H, W, ratio, mask_type, block)
        print(f"  build_mask     -> shape={m.shape}, dtype={m.dtype}, "
              f"masked={m.sum().item()}/{N} ({m.float().mean().item():.2%})")

        # Batch mask
        mb = build_batch_mask(B, H, W, ratio, mask_type, block)
        print(f"  build_batch_mask -> shape={mb.shape}, dtype={mb.dtype}")
        for i in range(B):
            cnt = mb[i].sum().item()
            print(f"    sample {i}: masked={cnt}/{N} ({cnt/N:.2%})")

        # Apply to dummy tensor: select masked tokens
        masked_tokens = dummy[mb]  # [total_masked, C]
        print(f"  dummy[mask]    -> shape={masked_tokens.shape}")

    print("\n" + "=" * 60)
    print("All checks passed.")


if __name__ == "__main__":
    main()
