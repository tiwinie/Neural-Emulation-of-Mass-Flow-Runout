#!/usr/bin/env python3
"""
Sanity check: run the pretrained weights.pth through THEIR OWN
run_landslide_batch() function from emulator.py -- the same code path
runout_demo.py uses -- instead of our custom testing_epoch11wt.py pipeline.

This tells us whether the checkpoint itself is broken, or whether our
custom npz-reading script was feeding it data in the wrong format.

We take one .npz test file, pull its DEM out, un-normalize the
cohesion/density/volume back to real physical units (since the npz only
stores them pre-normalized 0-1), and hand everything to their own function
using their own baseline-parameter style, matching runout_demo.py.

Usage:
    python3 sanity_check_original_pipeline.py \
        --npz data_split/test/output_hh_Alaska_patch_15360_2048_sim0.npz \
        --model_path weights.pth \
        --case sanity_check_case \
        --device cuda
"""
import os
import math
import argparse
import numpy as np

from emulator import run_landslide_batch


def unnormalize_params(norm_coh: float, norm_rho: float, norm_logv: float):
    """Reverses normalize_params() in emulator.py, to recover real physical values
    from the 0-1 normalized numbers stored in our .npz files."""
    COH = (5000, 50000)
    RHO = (917, 2650)
    VLO = (1e4, 1e7)

    cohesion = norm_coh * (COH[1] - COH[0]) + COH[0]
    rho = norm_rho * (RHO[1] - RHO[0]) + RHO[0]
    log_lo, log_hi = math.log10(VLO[0]), math.log10(VLO[1])
    volume = 10 ** (norm_logv * (log_hi - log_lo) + log_lo)
    return cohesion, rho, volume


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="One .npz test file to pull the DEM + params from")
    ap.add_argument("--model_path", default="weights.pth")
    ap.add_argument("--case", default="sanity_check_case", help="Folder name their pipeline will create/use")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--use_baseline_params", action="store_true",
                     help="Ignore the npz's own params and use the same baseline "
                          "(volume=9e6, cohesion=25000, rho=1800) that runout_demo.py uses")
    args = ap.parse_args()

    d = np.load(args.npz)
    dem = d["dem"].astype(np.float32)  # NOTE: this is already 0-1 normalized in our npz files,
                                        # not real elevation in meters -- see caveat printed below

    os.makedirs(args.case, exist_ok=True)
    dem_npy_path = os.path.join(args.case, "dem.npy")
    np.save(dem_npy_path, dem)
    print(f"Saved DEM ({dem.shape}, range [{dem.min():.3f}, {dem.max():.3f}]) -> {dem_npy_path}")
    print("CAVEAT: this DEM is already 0-1 normalized (from our npz), not real elevation in "
          "meters. Their pipeline will re-normalize it again, and slope/curvature computed "
          "from a pre-normalized DEM will not match slope/curvature computed from the real "
          "elevation values their model may have actually been trained on. This is a known, "
          "separate possible mismatch -- flagging it, not hiding it.")

    if args.use_baseline_params:
        cohesion, rho, volume = 25000.0, 1800.0, 9e6
        print(f"Using runout_demo.py's baseline params: cohesion={cohesion}, rho={rho}, volume={volume}")
    else:
        cohesion, rho, volume = unnormalize_params(
            float(d["cohesion"]), float(d["density"]), float(d["volume"])
        )
        print(f"Recovered real params from npz: cohesion={cohesion:.1f}, rho={rho:.1f}, volume={volume:.2e}")

    print("\nCalling emulator.py's own run_landslide_batch()...\n")
    results = run_landslide_batch(
        landslide=args.case,
        image_size=(256, 256),
        cell_size=30.0,
        cohesions=[cohesion],
        rhos=[rho],
        volumes=[volume],
        model_path=args.model_path,
        device=args.device,
        plot=True,
        combination_mode="elementwise",
        reuse_dist=True,
        use_dem_npy_first=True,
    )

    print("\nDone. Results:")
    for r in results:
        print(r)
    print(f"\nCheck '{args.case}/output_nn/' for the saved thickness prediction .npy file, "
          f"and the plot window/output for the visual mask+thickness overlay.")


if __name__ == "__main__":
    main()
