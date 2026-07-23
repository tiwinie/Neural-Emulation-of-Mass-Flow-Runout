#!/usr/bin/env python3
"""
Runs a trained checkpoint on one or more .npz samples and saves side-by-side
plots comparing predicted mask/thickness against the ground truth.

Usage (single file):
    python3 predict_and_visualize.py \
        --checkpoint weights_test/checkpoint_epoch0011.pth \
        --npz data_split/val/some_file.npz \
        --out_dir predictions

Usage (a handful of random samples from a folder):
    python3 predict_and_visualize.py \
        --checkpoint weights_test/checkpoint_epoch0011.pth \
        --npz_dir data_split/val \
        --n_samples 5 \
        --out_dir predictions
"""
import os
import glob
import random
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display over SSH
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import torch

from emulator import (
    UNetFiLMPlus,
    pick_device,
    compute_slope,
    compute_curvature,
    compute_flow_accumulation,
    compute_dist_to_source,
    compute_ns_coord,
    compute_we_coord,
    safe_minmax,
)


def build_sample(npz_path: str, cell_size: float = 30.0):
    """Same feature-building logic as the training Dataset, for one file."""
    d = np.load(npz_path)

    dem = d["dem"].astype(np.float32)
    h0 = d["source"].astype(np.float32)
    thick_gt = d["h"].astype(np.float32)
    if thick_gt.ndim == 3:
        thick_gt = thick_gt[0]
    mask_gt = (thick_gt > 0).astype(np.float32)

    H, W = dem.shape
    slope = (compute_slope(dem, cell_size) / 90.0).astype(np.float32)
    curv = compute_curvature(dem, cell_size)
    curv = safe_minmax(np.clip(curv, -10, 10))
    flow_acc = compute_flow_accumulation(dem)
    ns_coord = compute_ns_coord(H, W)
    we_coord = compute_we_coord(H, W)
    dist = compute_dist_to_source(h0)
    dist = dist / (np.max(dist) + 1e-6)
    h0_norm = safe_minmax(h0)

    x = np.stack([dem, slope, curv, ns_coord, we_coord, flow_acc, dist, h0_norm], axis=0)
    p = np.array(
        [float(d["cohesion"]), float(d["density"]), float(d["volume"])],
        dtype=np.float32,
    )
    return dem, x, p, mask_gt, thick_gt


def load_checkpoint(checkpoint_path: str, base_ch: int, device: str):
    model = UNetFiLMPlus(in_channels=8, film_params=3, base_ch=base_ch)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  (note) missing keys: {missing}, unexpected keys: {unexpected}")
    model.to(device).eval()
    epoch = ckpt.get("epoch", "unknown")
    print(f"Loaded checkpoint from epoch {epoch}")
    return model


def segmentation_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, thresh: float = 0.5) -> dict:
    """
    Dice and F1 are the same formula for binary masks (2*TP / (2*TP + FP + FN)) --
    included separately since you asked for both by name.
    IoU (Jaccard) = TP / (TP + FP + FN), a stricter overlap measure than Dice.
    """
    pred_bin = (pred_mask > thresh).astype(np.float32)
    gt_bin = gt_mask.astype(np.float32)

    tp = float((pred_bin * gt_bin).sum())
    fp = float((pred_bin * (1 - gt_bin)).sum())
    fn = float(((1 - pred_bin) * gt_bin).sum())

    if tp + fp + fn == 0:
        # both prediction and ground truth are empty -> perfect match
        return {"dice": 1.0, "f1": 1.0, "iou": 1.0, "precision": 1.0, "recall": 1.0}

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    dice = (2 * tp) / (2 * tp + fp + fn)
    iou = tp / (tp + fp + fn)

    return {"dice": dice, "f1": dice, "iou": iou, "precision": precision, "recall": recall}


def thickness_metrics(pred_thick: np.ndarray, gt_thick: np.ndarray, region_mask: np.ndarray) -> dict:
    """MAE and RMSE on thickness, computed only inside the predicted+ground-truth region."""
    region = region_mask.astype(bool)
    if region.sum() == 0:
        return {"mae": 0.0, "rmse": 0.0}
    diff = pred_thick[region] - gt_thick[region]
    mae = float(np.abs(diff).mean())
    rmse = float(np.sqrt((diff ** 2).mean()))
    return {"mae": mae, "rmse": rmse}


def plot_sample(dem, mask_gt, thick_gt, pred_mask, pred_thick, out_png, title=""):
    ls = LightSource(azdeg=315, altdeg=45)
    hs = ls.shade(dem, cmap=plt.cm.gray, blend_mode="overlay", vert_exag=1)

    seg_m = segmentation_metrics(pred_mask, mask_gt)
    region = (pred_mask > 0.5) | (mask_gt > 0.5)
    thick_m = thickness_metrics(pred_thick, thick_gt, region)

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    axs[0].imshow(hs); axs[0].imshow(mask_gt, cmap="Reds", alpha=0.5)
    axs[0].set_title("Ground truth extent"); axs[0].axis("off")

    axs[1].imshow(hs); axs[1].imshow(pred_mask, cmap="Reds", alpha=0.5)
    axs[1].set_title(f"Predicted extent\nDice/F1={seg_m['f1']:.3f}  IoU={seg_m['iou']:.3f}")
    axs[1].axis("off")

    axs[2].imshow(hs)
    im2 = axs[2].imshow(thick_gt, cmap="magma", alpha=0.7, vmin=0)
    axs[2].set_title("Ground truth thickness"); axs[2].axis("off")
    plt.colorbar(im2, ax=axs[2], shrink=0.8, label="m")

    axs[3].imshow(hs)
    thick_masked = pred_thick * (pred_mask > 0.5)
    im3 = axs[3].imshow(thick_masked, cmap="magma", alpha=0.7, vmin=0)
    axs[3].set_title(f"Predicted thickness\nMAE={thick_m['mae']:.3f}m  RMSE={thick_m['rmse']:.3f}m")
    axs[3].axis("off")
    plt.colorbar(im3, ax=axs[3], shrink=0.8, label="m")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)

    metrics = {**seg_m, **thick_m}
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    ap.add_argument("--npz", default=None, help="Single .npz file to run on")
    ap.add_argument("--npz_dir", default=None, help="Folder of .npz files to sample from")
    ap.add_argument("--n_samples", type=int, default=5, help="How many files to pick from --npz_dir")
    ap.add_argument("--out_dir", default="predictions")
    ap.add_argument("--base_ch", type=int, default=32)
    ap.add_argument("--cell_size", type=float, default=30.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.npz and not args.npz_dir:
        raise ValueError("Provide either --npz (one file) or --npz_dir (a folder to sample from)")

    os.makedirs(args.out_dir, exist_ok=True)
    device = pick_device(args.device)
    print(f"Using device: {device}")

    model = load_checkpoint(args.checkpoint, args.base_ch, device)

    if args.npz:
        files = [args.npz]
    else:
        all_files = sorted(glob.glob(os.path.join(args.npz_dir, "*.npz")))
        random.seed(args.seed)
        files = random.sample(all_files, min(args.n_samples, len(all_files)))
        print(f"Sampled {len(files)} files from {args.npz_dir}")

    all_metrics = []
    for fpath in files:
        fname = os.path.splitext(os.path.basename(fpath))[0]
        dem, x, p, mask_gt, thick_gt = build_sample(fpath, args.cell_size)

        xt = torch.tensor(x[None], dtype=torch.float32, device=device)
        pt = torch.tensor(p[None], dtype=torch.float32, device=device)
        with torch.no_grad():
            seg_logits, thick_out = model(xt, pt)
            pred_mask = torch.sigmoid(seg_logits).squeeze().cpu().numpy()
            pred_thick = thick_out.squeeze().cpu().numpy()

        out_png = os.path.join(args.out_dir, f"{fname}_prediction.png")
        m = plot_sample(dem, mask_gt, thick_gt, pred_mask, pred_thick, out_png, title=fname)
        m["file"] = fname
        all_metrics.append(m)
        print(f"  {fname}: F1={m['f1']:.3f}  IoU={m['iou']:.3f}  "
              f"MAE={m['mae']:.3f}m  RMSE={m['rmse']:.3f}m -> saved {out_png}")

    if all_metrics:
        import csv
        csv_path = os.path.join(args.out_dir, "metrics_summary.csv")
        keys = ["file", "f1", "iou", "precision", "recall", "mae", "rmse"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for m in all_metrics:
                writer.writerow({k: m[k] for k in keys})
        print(f"\nSaved per-sample metrics -> {csv_path}")

    if len(files) > 1:
        avg_f1 = np.mean([m["f1"] for m in all_metrics])
        avg_iou = np.mean([m["iou"] for m in all_metrics])
        avg_mae = np.mean([m["mae"] for m in all_metrics])
        avg_rmse = np.mean([m["rmse"] for m in all_metrics])
        print(f"\nAverage across {len(files)} samples: "
              f"F1={avg_f1:.3f}  IoU={avg_iou:.3f}  MAE={avg_mae:.3f}m  RMSE={avg_rmse:.3f}m")


if __name__ == "__main__":
    main()
