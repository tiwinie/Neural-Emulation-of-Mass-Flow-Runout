# Neural Emulation of Mass Flow Runout

This repository extends the work of Nava, Chen & Van Wyk de Vries (2025), which introduced a neural network-based emulator for landslide runout prediction as a fast surrogate for physics-based simulation. The original authors released their dataset and core architecture code but no training pipeline. This repository adds a complete, reproducible training pipeline built from scratch, along with bug fixes, evaluation results, and an applied case study.

## What's in this repo

```
├── training/          Training pipeline
├── evaluation/         Testing and validation scripts
├── visualization/       Data and channel visualization tools
├── results/           Training curves, metrics, and prediction samples
│   ├── predictions/     Ground truth vs. predicted outputs across regions
│   └── sample_inputs/    8-channel model input visualization
├── diagrams/          Architecture diagrams (PNG + SVG)
├── emulator/           (original authors' code)
├── runout/            (original authors' code)
└── utils/             (original authors' code)
```

## Background

Landslide runout modeling — predicting how far and where a landslide's debris will travel — is traditionally done with physics-based simulations that are computationally expensive. This project trains a neural network emulator to approximate those simulations in a fraction of the time, taking topographic data and flow parameters as input and predicting both the runout extent and deposit thickness.

## Model architecture: UNetFiLMPlus

The model is a modified U-Net with:
- **Residual blocks** with GroupNorm-8 and Dropout (0.2) at each stage
- **FiLM conditioning** (Feature-wise Linear Modulation) at four encoder stages, injecting three global flow parameters (cohesion, density, volume)
- **Attention-gated skip connections** between encoder and decoder
- **Dual output heads**: binary segmentation mask (runout extent) and continuous deposit thickness

Input is an 8-channel raster stack: DEM, slope, curvature, N–S coordinate, W–E coordinate, flow accumulation, distance-to-source, and initial thickness (h0).

See `diagrams/unetmod_final.png` (or `.svg`) for the full architecture diagram.

## Training pipeline

Since the original authors did not release training code, the full pipeline was built from scratch:

- **Loss function**: BCE + Dice (segmentation) + masked L1 (thickness)
- **Optimizer**: AdamW
- **Scheduler**: ReduceLROnPlateau
- **Dataset**: ~90,679 `.npz` samples, split into train/val/test
- **Training length**: ~460 epochs, run across multiple resumed sessions via `nohup` (interrupted repeatedly by server reboots)

Learning rate decayed to near-zero by epoch 400+, so later epochs contributed negligible additional improvement (see `results/5_lr_schedule.png`).

### Key scripts

| Script | Purpose |
|---|---|
| `training/train_unetfilmplus.py` | Main training loop |
| `training/split_dataset.py` | Train/val/test split |
| `evaluation/testing_script.py` | Evaluation on held-out test set (F1, IoU, MAE, RMSE) |
| `evaluation/sanity_check_original_pipeline.py` | Validates preprocessing against the original pipeline |
| `visualization/visualize_channels.py` | Visualizes the 8-channel model input |

## A critical bug fix

During evaluation, the original pretrained checkpoint produced near-zero scores. Investigation traced this to an inconsistency in how the slope and curvature channels were computed: one code path used raw elevation, another used normalized DEM values. Fixing this preprocessing inconsistency and retraining resolved the issue — see `evaluation/sanity_check_original_pipeline.py` for the validation approach.

## Results

Final training metrics (~epoch 460):

| Metric | Train | Val |
|---|---|---|
| Segmentation loss | ~0.077 | ~0.077 |
| Thickness loss | ~1.04 | ~1.25 |

Loss curves: `results/train_loss_every10.png`, `results/val_loss_every10.png`, `results/train_vs_val_loss_every10.png`

Per-sample test metrics (F1, IoU, precision, recall, MAE, RMSE) are in `results/metrics_summary.csv`, with visual predictions across five global regions (Alps, Alaska, Papua New Guinea, North America, Australia) in `results/predictions/`.

## Case study: Makunosawa, Japan

The pretrained emulator was applied to a real landslide case study at Makunosawa, Japan (36.910833°N, 138.140556°E, estimated volume ~78,164 m³), using a GeoTIFF DEM. Three bugs were identified and fixed in the inference pipeline during this process:

1. **Incorrect cell size** — hardcoded at 12.5m instead of the DEM's actual ~27.967m, causing scaling errors in predicted thickness
2. **Cross-run contamination** — Sobol sensitivity runs were silently mixing with baseline run outputs during probability map aggregation
3. **Zero-fill edge padding** — produced rectangular artifacts at patch boundaries; fixed with edge-replication padding

A corrected baseline inference script converts the source point from WGS84 to UTM and locates it at pixel (row=180, col=146) using the DEM's true resolution.

## Setup

This project uses a conda environment. Key dependencies: PyTorch, NumPy, GDAL/rasterio (for GeoTIFF handling), and standard scientific Python packages (matplotlib, scikit-learn).

```bash
conda create -n unet-env python=3.x
conda activate unet-env
# install dependencies (see environment.yml if provided)
```

## Acknowledgments

Built on the dataset and base architecture released by Nava, Chen & Van Wyk de Vries (2025). This repository extends their work with a full training pipeline, bug fixes, evaluation, and an applied case study, completed as part of a research internship.
