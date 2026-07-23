
from __future__ import annotations
import os
import glob
import argparse
from typing import List, Tuple, Optional

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed on a remote/SSH machine
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Import your existing model + helpers.
# Assumes this file sits next to (or on the pythonpath of) the module containing
# UNetFiLMPlus, pick_device, etc. Adjust the import if your file has a different name.
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


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class LandslideDataset(Dataset):
    """
    Expects a directory of .npz files matching your actual sim output format:
        dem      : (H, W)    float32, already normalized 0-1
        source   : (H, W)    float32, h0 -- initial source thickness (unnormalized)
        h        : (1, H, W) float32, ground-truth FINAL thickness (target)
        volume   : scalar,   already normalized 0-1
        density  : scalar,   already normalized 0-1
        cohesion : scalar,   already normalized 0-1

    The 8-channel model input (dem, slope, curvature, ns, we, flow_acc, dist, h0)
    is built on the fly from `dem` and `source`, since only those two spatial
    fields are precomputed in your files. Slope/curvature use `cell_size` in
    the same map units as your DEM (defaults to 30, matching the original
    pipeline's Landsat/SRTM-scale assumption -- change if yours differs).

    The segmentation mask target is derived as (h > 0), since no explicit
    mask is stored.
    """

    def __init__(self, data_dir: str, cell_size: float = 30.0, augment: bool = False):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")
        self.cell_size = cell_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        d = np.load(self.files[idx])

        dem = d["dem"].astype(np.float32)          # already 0-1 normalized
        h0 = d["source"].astype(np.float32)         # unnormalized source thickness
        thick = d["h"].astype(np.float32)
        if thick.ndim == 3:                         # (1, H, W) -> (H, W)
            thick = thick[0]
        mask = (thick > 0).astype(np.float32)

        H, W = dem.shape

        # NOTE: slope/curvature are computed from the *normalized* dem here since
        # that's what's stored. If you have access to the raw-elevation dem
        # (meters) at prep time, prefer computing these from that instead --
        # slope/curvature magnitudes are sensitive to the units of the input.
        slope = (compute_slope(dem, self.cell_size) / 90.0).astype(np.float32)
        curv = compute_curvature(dem, self.cell_size)
        curv = safe_minmax(np.clip(curv, -10, 10))
        flow_acc = compute_flow_accumulation(dem)
        ns_coord = compute_ns_coord(H, W)
        we_coord = compute_we_coord(H, W)

        dist = compute_dist_to_source(h0)
        dist = dist / (np.max(dist) + 1e-6)

        # Normalize h0 the same way create_single_h0_adaptive-derived h0s are
        # scaled in the original pipeline: here we just min-max it per-sample,
        # since raw source thickness units (meters) vary a lot across sims.
        h0_norm = safe_minmax(h0)

        x = np.stack([dem, slope, curv, ns_coord, we_coord, flow_acc, dist, h0_norm], axis=0)

        p = np.array(
            [float(d["cohesion"]), float(d["density"]), float(d["volume"])],
            dtype=np.float32,
        )

        if self.augment:
            x, mask, thick = self._augment(x, mask, thick)

        return (
            torch.from_numpy(x.copy()),
            torch.from_numpy(p.copy()),
            torch.from_numpy(mask.copy()).unsqueeze(0),   # (1, H, W)
            torch.from_numpy(thick.copy()).unsqueeze(0),  # (1, H, W)
        )

    @staticmethod
    def _augment(x, mask, thick):
        # Simple flips/rotations valid for raster-style data.
        if np.random.rand() < 0.5:
            x, mask, thick = x[:, :, ::-1], mask[:, ::-1], thick[:, ::-1]
        if np.random.rand() < 0.5:
            x, mask, thick = x[:, ::-1, :], mask[::-1, :], thick[::-1, :]
        k = np.random.randint(0, 4)
        if k:
            x = np.rot90(x, k, axes=(1, 2))
            mask = np.rot90(mask, k)
            thick = np.rot90(thick, k)
        return x, mask, thick


# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------
def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Soft Dice loss on the predicted segmentation probability vs ground-truth mask.
    Complements BCE: BCE cares about per-pixel correctness, Dice cares about
    overlap of the (usually small) landslide region against the huge background --
    without it, BCE alone can get "lazy" and just predict all-background.
    """
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    target = target.flatten(1)
    intersection = (probs * target).sum(dim=1)
    union = probs.sum(dim=1) + target.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


class LandslideLoss(nn.Module):
    """
    seg loss:   BCEWithLogits + Dice, combined, on the predicted mask logits.
                BCE handles per-pixel accuracy; Dice handles class imbalance
                (landslide extent is usually a small fraction of the image).
    thick loss: L1 on thickness, computed only where ground truth OR prediction
                indicates landslide extent, so the network isn't penalized
                for "0 vs 0" everywhere off-slide (which would swamp the signal).
    """

    def __init__(self, seg_weight: float = 1.0, thick_weight: float = 1.0,
                 bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.seg_weight = seg_weight
        self.thick_weight = thick_weight
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, seg_logits, thick_pred, mask_gt, thick_gt):
        bce_loss = self.bce(seg_logits, mask_gt)
        dsc_loss = dice_loss(seg_logits, mask_gt)
        seg_loss = self.bce_weight * bce_loss + self.dice_weight * dsc_loss

        with torch.no_grad():
            region = ((mask_gt > 0.5) | (thick_gt > 0)).float()
        denom = region.sum().clamp_min(1.0)
        thick_loss = (torch.abs(thick_pred - thick_gt) * region).sum() / denom

        total = self.seg_weight * seg_loss + self.thick_weight * thick_loss
        return total, seg_loss.detach(), thick_loss.detach()


# -----------------------------------------------------------------------------
# Train / validate loops
# -----------------------------------------------------------------------------
def run_epoch(model, loader, criterion, device, optimizer: Optional[torch.optim.Optimizer] = None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    tot, tot_seg, tot_thick, n = 0.0, 0.0, 0.0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x, p, mask_gt, thick_gt in loader:
            x, p = x.to(device), p.to(device)
            mask_gt, thick_gt = mask_gt.to(device), thick_gt.to(device)

            seg_logits, thick_pred = model(x, p)
            loss, seg_loss, thick_loss = criterion(seg_logits, thick_pred, mask_gt, thick_gt)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            bs = x.size(0)
            tot_seg += seg_loss.item() * bs
            tot_thick += thick_loss.item() * bs
            n += bs

    return tot / n, tot_seg / n, tot_thick / n


def _save_history(history: dict, out_dir: str):
    """Writes loss_history.csv (raw numbers) and loss_history.png (a graph) into out_dir."""
    import csv

    csv_path = os.path.join(out_dir, "loss_history.csv")
    keys = list(history.keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for row in zip(*[history[k] for k in keys]):
            writer.writerow(row)

    epochs = history["epoch"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    axs[0].plot(epochs, history["train_loss"], label="train")
    axs[0].plot(epochs, history["val_loss"], label="val")
    axs[0].set_title("Total loss")
    axs[0].set_xlabel("epoch"); axs[0].set_ylabel("loss"); axs[0].legend()

    axs[1].plot(epochs, history["train_seg"], label="train")
    axs[1].plot(epochs, history["val_seg"], label="val")
    axs[1].set_title("Segmentation loss (BCE + Dice)")
    axs[1].set_xlabel("epoch"); axs[1].set_ylabel("loss"); axs[1].legend()

    axs[2].plot(epochs, history["train_thick"], label="train")
    axs[2].plot(epochs, history["val_thick"], label="val")
    axs[2].set_title("Thickness loss (L1)")
    axs[2].set_xlabel("epoch"); axs[2].set_ylabel("loss"); axs[2].legend()

    plt.tight_layout()
    png_path = os.path.join(out_dir, "loss_history.png")
    plt.savefig(png_path, dpi=150)
    plt.close(fig)


def train(
    train_dir: str,
    val_dir: str,
    out_dir: str = "weights",
    epochs: int = 1000,
    save_every: int = 10,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    in_channels: int = 8,
    film_params: int = 3,
    base_ch: int = 32,
    device_pref: str = "cuda",
    seg_weight: float = 1.0,
    thick_weight: float = 1.0,
    patience: int = 1000,  # effectively off by default since you asked for a fixed 1000-epoch run
    num_workers: int = 4,
    resume: Optional[str] = None,
):
    os.makedirs(out_dir, exist_ok=True)
    device = pick_device(device_pref)
    print(f"Using device: {device}")

    # Use your pre-made splits directly instead of re-splitting randomly.
    train_ds = LandslideDataset(train_dir, augment=True)
    val_ds = LandslideDataset(val_dir, augment=False)
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=(device == "cuda"))

    model = UNetFiLMPlus(in_channels=in_channels, film_params=film_params, base_ch=base_ch).to(device)
    criterion = LandslideLoss(seg_weight=seg_weight, thick_weight=thick_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    start_epoch = 1
    best_val = float("inf")
    epochs_no_improve = 0
    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "train_seg": [], "val_seg": [], "train_thick": [], "val_thick": []}

    if resume and os.path.exists(resume):
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("best_val", best_val)
        print(f"Resumed from {resume} at epoch {start_epoch}")

    history_csv = os.path.join(out_dir, "loss_history.csv")
    if resume and os.path.exists(history_csv):
        import csv
        with open(history_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for k in history:
                    history[k].append(float(row[k]) if k != "epoch" else int(row[k]))
        print(f"Loaded {len(history['epoch'])} rows of prior loss history from {history_csv}")

    training_start = time.time()
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        tr_loss, tr_seg, tr_thick = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_seg, val_thick = run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step(val_loss)

        cur_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start
        elapsed_total = time.time() - training_start
        n_done = epoch - start_epoch + 1
        avg_epoch_time = elapsed_total / n_done
        epochs_left = epochs - epoch
        eta_seconds = avg_epoch_time * epochs_left

        def _fmt(seconds: float) -> str:
            seconds = int(seconds)
            h, rem = divmod(seconds, 3600)
            m, s = divmod(rem, 60)
            if h:
                return f"{h}h {m}m {s}s"
            if m:
                return f"{m}m {s}s"
            return f"{s}s"

        print(
            f"epoch {epoch:03d} | train {tr_loss:.4f} (seg {tr_seg:.4f}, thick {tr_thick:.4f}) | "
            f"val {val_loss:.4f} (seg {val_seg:.4f}, thick {val_thick:.4f}) | lr {cur_lr:.2e} | "
            f"epoch time {_fmt(epoch_time)} | elapsed {_fmt(elapsed_total)} | ETA {_fmt(eta_seconds)}"
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_seg"].append(tr_seg)
        history["val_seg"].append(val_seg)
        history["train_thick"].append(tr_thick)
        history["val_thick"].append(val_thick)

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val": best_val,
                },
                os.path.join(out_dir, "best_model.pth"),
            )
        else:
            epochs_no_improve += 1

        # Always keep a rolling "last" checkpoint for resuming.
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
            },
            os.path.join(out_dir, "last_model.pth"),
        )

        # Periodic checkpoint every `save_every` epochs, as requested.
        if epoch % save_every == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val": best_val,
                },
                os.path.join(out_dir, f"checkpoint_epoch{epoch:04d}.pth"),
            )
            _save_history(history, out_dir)
            print(f"  💾 saved checkpoint_epoch{epoch:04d}.pth + updated loss_history.csv/png")

        if epochs_no_improve >= patience:
            print(f"Early stopping: no val improvement in {patience} epochs.")
            break

    _save_history(history, out_dir)
    total_time = time.time() - training_start
    h, rem = divmod(int(total_time), 3600)
    m, s = divmod(rem, 60)
    print(f"Done. Best val loss: {best_val:.4f}. Checkpoints + loss_history.png in {out_dir}/")
    print(f"Total training wall time: {h}h {m}m {s}s")
    return os.path.join(out_dir, "best_model.pth")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True, help="Folder with training .npz files (e.g. data_split/train)")
    ap.add_argument("--val_dir", required=True, help="Folder with validation .npz files (e.g. data_split/val)")
    ap.add_argument("--out_dir", default="weights")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--base_ch", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seg_weight", type=float, default=1.0)
    ap.add_argument("--thick_weight", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=1000)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    train(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        save_every=args.save_every,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        base_ch=args.base_ch,
        device_pref=args.device,
        seg_weight=args.seg_weight,
        thick_weight=args.thick_weight,
        patience=args.patience,
        resume=args.resume,
    )
