#!/usr/bin/env python3
"""
Visualizes the 8 input channels fed to UNetFiLMPlus for a single .npz sample.
Uses the exact same feature-construction logic as training/testing (build_sample
from emulator.py), so what you see here is exactly what the model receives.

Usage:
    python3 visualize_channels.py --npz data_split/test/some_file.npz --out_dir channel_viz
    # or, to auto-pick a random file from a folder:
    python3 visualize_channels.py --npz_dir data_split/test --out_dir channel_viz
"""
import os
import glob
import random
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from testing_epoch11wt import build_sample  # reuses the exact same function


CHANNEL_NAMES = [
    "1. DEM (normalized)",
    "2. Slope",
    "3. Curvature",
    "4. NS coordinate",
    "5. WE coordinate",
    "6. Flow accumulation",
    "7. Distance to source",
    "8. h0 (source thickness, normalized)",
]

# A reasonable colormap per channel -- terrain-ish for DEM, coolwarm for
# curvature (diverging), viridis for the rest.
CMAPS = ["terrain", "viridis", "coolwarm", "viridis", "viridis",
         "viridis", "viridis", "magma"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=None, help="Single .npz file")
    ap.add_argument("--npz_dir", default=None, help="Folder to pick a random file from")
    ap.add_argument("--out_dir", default="channel_viz")
    ap.add_argument("--cell_size", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.npz and not args.npz_dir:
        raise ValueError("Provide --npz or --npz_dir")

    os.makedirs(args.out_dir, exist_ok=True)

    if args.npz:
        fpath = args.npz
    else:
        files = sorted(glob.glob(os.path.join(args.npz_dir, "*.npz")))
        random.seed(args.seed)
        fpath = random.choice(files)

    fname = os.path.splitext(os.path.basename(fpath))[0]
    print(f"Visualizing: {fpath}")

    dem, x, p, mask_gt, thick_gt = build_sample(fpath, args.cell_size)
    # x has shape (8, H, W) -- the exact 8-channel stack the model sees

    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    axs = axs.flatten()

    for i in range(8):
        im = axs[i].imshow(x[i], cmap=CMAPS[i])
        axs[i].set_title(CHANNEL_NAMES[i], fontsize=12)
        axs[i].axis("off")
        plt.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)

    fig.suptitle(f"8-channel model input -- {fname}\n"
                 f"(cohesion={p[0]:.1f}, density={p[1]:.1f}, volume={p[2]:.1f})",
                 fontsize=14)
    plt.tight_layout()

    out_path = os.path.join(args.out_dir, f"{fname}_channels.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
