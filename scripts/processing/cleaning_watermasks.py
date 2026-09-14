# =============================================================================
# clean_existing_masks.py
#
# Applies cut-and-smooth morphological cleaning to existing water mask
# mosaics without rerunning OmniWaterMask. Reads existing *_mosaic.tif
# files, applies erosion -> keep largest body -> dilation, and saves
# cleaned versions either in place or to a new output folder.
#
# Usage:
#   python clean_existing_masks.py <river> <outputs_folder>
#   e.g. python clean_existing_masks.py sacramento red_bluff_colusa_outputs
#
# For SLURM:
#   sbatch clean_masks.sh sacramento red_bluff_colusa_outputs
# =============================================================================

import argparse
import numpy as np
import rasterio
from pathlib import Path
from skimage.morphology import erosion, dilation, disk
from skimage.measure import label, regionprops


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("river",   help="River name e.g. sacramento")
parser.add_argument("outputs", help="Output folder name e.g. red_bluff_colusa_outputs")
args = parser.parse_args()

RIVER   = args.river
OUTPUTS = args.outputs


# =============================================================================
# CONFIG
# =============================================================================

DATA_ROOT  = Path("/home/geomorph/california_rivers/naip")
INPUT_DIR  = DATA_ROOT / "outputs" / "gee" / OUTPUTS

# Set SAVE_IN_PLACE = True to overwrite existing mosaics
# Set SAVE_IN_PLACE = False to save cleaned versions to a new folder
# (recommended first time so you can compare before/after)
SAVE_IN_PLACE = False
OUTPUT_DIR    = DATA_ROOT / "outputs" / "gee" / f"{OUTPUTS}_cleaned"

# =============================================================================
# CLEANING PARAMETERS
# Tune these based on how aggressively you want to clean
# =============================================================================

# How far to erode inward — severs thin connections between river
# and false positive blobs. Increase if blobs still connect after cleaning.
ERODE_RADIUS = 35

# How far to dilate back out after erosion — restores river width.
# Keep smaller than ERODE_RADIUS to permanently trim false positive edges.
# Difference (ERODE - DILATE) = pixels permanently trimmed from all edges.
DILATE_RADIUS = 20

# How many connected water bodies to keep after erosion.
# 1 = only the largest (strictest, good for single-channel rivers)
# 2-3 = keep a few largest (good for braided rivers or sections with islands)
KEEP_TOP_N = 1

# Set to True to reprocess files that already exist in output folder
FORCE_RERUN = True


# =============================================================================
# SETUP
# =============================================================================

if not SAVE_IN_PLACE:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"River:        {RIVER}")
print(f"Input dir:    {INPUT_DIR}")
print(f"Save in place: {SAVE_IN_PLACE}")
if not SAVE_IN_PLACE:
    print(f"Output dir:   {OUTPUT_DIR}")
print(f"Erode radius:  {ERODE_RADIUS}")
print(f"Dilate radius: {DILATE_RADIUS}")
print(f"Keep top N:    {KEEP_TOP_N}\n")

if not INPUT_DIR.exists():
    print(f"ERROR: Input directory not found: {INPUT_DIR}")
    raise SystemExit(1)


# =============================================================================
# CLEANING FUNCTION
# =============================================================================

def cut_and_smooth(mask,
                   erode_radius=ERODE_RADIUS,
                   dilate_radius=DILATE_RADIUS,
                   keep_top_n=KEEP_TOP_N):
    """
    Cut-and-smooth morphological cleaning:

    1. Erode inward aggressively — severs thin pixel bridges between
       the real river channel and false positive blobs (forested areas,
       fields, shadows that are connected by thin pixel bridges)

    2. Keep only the largest N connected bodies — now that connections
       are severed, blobs are separate components and can be discarded

    3. Dilate back outward to restore river width — using a smaller
       radius than erosion permanently trims false positive edges
       while restoring most of the real channel width

    The difference (erode_radius - dilate_radius) controls how many
    pixels are permanently trimmed from ALL edges — this also smooths
    rough tree-canopy boundaries.
    """
    if not mask.any():
        return mask

    # Step 1 — erode aggressively inward
    eroded = erosion(mask, footprint=disk(erode_radius))

    # Safety check — if erosion removed everything, try half the radius
    if not eroded.any():
        print(f"    WARNING: erosion radius {erode_radius} removed all pixels "
              f"— retrying with {erode_radius // 2}")
        eroded = erosion(mask, footprint=disk(erode_radius // 2))
        if not eroded.any():
            print(f"    WARNING: erosion removed everything — returning original")
            return mask

    # Step 2 — label connected components and keep largest N
    labeled = label(eroded, connectivity=2)
    if labeled.max() == 0:
        return mask

    props = regionprops(labeled)
    top   = sorted(props, key=lambda r: r.area, reverse=True)[:keep_top_n]
    kept  = np.isin(labeled, [r.label for r in top])

    # Step 3 — dilate back out to restore river width
    restored = dilation(kept, footprint=disk(dilate_radius))

    return restored.astype(bool)


def clean_mask_file(input_path, output_path):
    """
    Read a water mask GeoTIFF, apply cut_and_smooth cleaning,
    and save the result to output_path.
    """
    with rasterio.open(input_path) as src:
        data      = src.read(1).astype(bool)
        meta      = src.meta.copy()
        h, w      = data.shape

    print(f"  Input shape:  {h}x{w}")
    print(f"  Water pixels before: {data.sum():,}")

    cleaned = cut_and_smooth(data)

    print(f"  Water pixels after:  {cleaned.sum():,}")
    pct_removed = (data.sum() - cleaned.sum()) / max(data.sum(), 1) * 100
    print(f"  Removed: {pct_removed:.1f}% of water pixels")

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(cleaned.astype(np.uint8), 1)


# =============================================================================
# MAIN LOOP
# =============================================================================

# Find all mosaic files
all_mosaics = sorted(INPUT_DIR.glob("*_mosaic.tif"))
print(f"Found {len(all_mosaics)} mosaic files in {INPUT_DIR}\n")

if not all_mosaics:
    print("ERROR: No mosaic files found.")
    raise SystemExit(1)

processed = 0
skipped   = 0
errors    = 0

for mosaic_path in all_mosaics:
    print(f"{'='*50}")
    print(f"Processing: {mosaic_path.name}")

    # Determine output path
    if SAVE_IN_PLACE:
        output_path = mosaic_path
    else:
        output_path = OUTPUT_DIR / mosaic_path.name

    # Skip if already processed
    if not SAVE_IN_PLACE and output_path.exists() and not FORCE_RERUN:
        print(f"  Skipping — already exists in output folder")
        skipped += 1
        continue

    try:
        clean_mask_file(mosaic_path, output_path)
        print(f"  Saved -> {output_path.name}")
        processed += 1
    except Exception as e:
        import traceback
        print(f"  ERROR: {e}")
        print(traceback.format_exc())
        errors += 1

    print()

# Summary
print(f"{'='*50}")
print(f"SUMMARY")
print(f"{'='*50}")
print(f"  Processed: {processed}")
print(f"  Skipped:   {skipped}")
print(f"  Errors:    {errors}")
if not SAVE_IN_PLACE:
    print(f"\nCleaned files saved to: {OUTPUT_DIR}")
    print(f"Compare with originals in: {INPUT_DIR}")
    print(f"If results look good, set SAVE_IN_PLACE = True and rerun")
print("\nAll done!")
