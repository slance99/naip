# =============================================================================
# naip_first_last_overlay.py
#
# For each river section, finds the first and last year water mask mosaic,
# overlays them on the first available NAIP imagery tile with 50% transparency
# so the background landscape is visible underneath the water mask colors.
#
# Color scheme (at 50% transparency over NAIP imagery):
#   Blue  = water in first year only (water lost)
#   Pink  = water in last year only (water gained)
#   White = water in both years (persistent water)
#   Transparent = no water in either year (NAIP imagery shows through)
#
# Usage:
#   python naip_first_last_overlay.py <river_name>
#   e.g. python naip_first_last_overlay.py sacramento
#
# For SLURM:
#   sbatch first_last_overlay.sh sacramento
#
# Requirements:
#   conda install -c conda-forge rasterio numpy matplotlib
# =============================================================================

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from collections import defaultdict
import rasterio
from rasterio.enums import Resampling as RasterioResampling
from rasterio.warp import reproject, Resampling as WarpResampling


# =============================================================================
# ARGUMENT PARSING
# Allows river name and output folder to be passed from command line or SLURM
# =============================================================================

parser = argparse.ArgumentParser(
    description="Generate first vs last year water mask overlay images for NAIP data"
)
parser.add_argument("river", help="River name e.g. sacramento, mattole, eel")
parser.add_argument("outputs", help="Output folder name e.g. red_bluff_colusa_outputs")
args = parser.parse_args()

RIVER   = args.river
OUTPUTS = args.outputs


# =============================================================================
# CONFIG
# =============================================================================

DATA_ROOT = Path("/home/geomorph/california_rivers/naip")

# Directory containing flat mosaic .tif files named {river}_{n}_{year}_mosaic.tif
MASK_DIR   = DATA_ROOT / "outputs" / "gee" / OUTPUTS

# Directory containing per-section NAIP tiles from GEE named naip_{year}.tif
# Structure: gee_naip/{river}/{river}_{n}/naip_{year}.tif
NAIP_DIR   = DATA_ROOT / "gee_naip" / RIVER

# Directory where overlay images will be saved
OUTPUT_DIR = DATA_ROOT / "outputs" / "comparisons" / OUTPUTS

# Overlay transparency — 0.5 means 50% opaque, matching your movie script
OVERLAY_ALPHA = 0.5

# DPI for output images
DPI = 150

# Max pixels to load per NAIP tile before downsampling — keeps memory use low
MAX_PIXELS = 20_000_000


# =============================================================================
# SETUP
# =============================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"River:      {RIVER}")
print(f"Outputs:    {OUTPUTS}")
print(f"Masks from: {MASK_DIR}")
print(f"NAIP from:  {NAIP_DIR}")
print(f"Output to:  {OUTPUT_DIR}\n")

if not MASK_DIR.exists():
    print(f"ERROR: Mask directory does not exist: {MASK_DIR}")
    raise SystemExit(1)

if not NAIP_DIR.exists():
    print(f"ERROR: NAIP directory does not exist: {NAIP_DIR}")
    raise SystemExit(1)
# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_year_from_mask(path):
    """
    Extract the year from a mosaic filename.
    Expects format: {river}_{section}_{year}_mosaic.tif
    e.g. sacramento_1_2022_mosaic.tif -> 2022
    """
    match = re.search(r"_(\d{4})_mosaic", path.stem)
    return int(match.group(1)) if match else None


def parse_section_from_mask(path):
    """
    Extract the section name from a mosaic filename by stripping year and suffix.
    e.g. sacramento_1_2022_mosaic.tif -> sacramento_1
    """
    return re.sub(r"_\d{4}_mosaic$", "", path.stem)


def load_naip_rgb(naip_path):
    """
    Load a GEE NAIP tile as an RGB array normalized to 0-1.
    NAIP band order from GEE: R=1, G=2, B=3, N=4
    """
    with rasterio.open(naip_path) as src:
        r = src.read(1)
        g = src.read(2)
        b = src.read(3)
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
        out_shape = (src.height, src.width)

    rgb = np.stack([r, g, b], axis=-1).astype(float)
    rgb = np.clip(rgb / 255.0, 0, 1)
    return rgb, out_shape, transform, crs, bounds

def load_mask_aligned(mask_path, target_shape, naip_transform, naip_crs):
    """
    Load a water mask and reproject/resample it to exactly match the NAIP
    tile's shape, transform, and CRS — same approach as the movie script's
    align_mask_to_naip() function. This ensures pixel-perfect overlay.
    """
    with rasterio.open(mask_path) as src:
        aligned = np.zeros(target_shape, dtype=np.uint8)
        reproject(
            source=rasterio.band(src, 1),
            destination=aligned,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=naip_transform,
            dst_crs=naip_crs,
            resampling=WarpResampling.nearest,
        )
    return aligned > 0


def find_naip_for_year(naip_section_dir, year):
    """
    Look for a GEE NAIP tile for a specific year.
    GEE files are named naip_{year}.tif.
    Returns the path if found, None otherwise.
    """
    candidate = naip_section_dir / f"naip_{year}.tif"
    return candidate if candidate.exists() else None


def make_overlay_rgba(first_mask, last_mask, alpha=OVERLAY_ALPHA):
    """
    Build an RGBA overlay array from two binary water masks.
    Non-water pixels are fully transparent so the NAIP background shows through.

    Color scheme:
      Blue  [0.0, 0.5, 1.0] = first year only  (water lost)
      Pink  [1.0, 0.1, 0.7] = last year only   (water gained)
      White [1.0, 1.0, 1.0] = both years       (persistent water)
      Transparent            = no water

    Parameters
    ----------
    first_mask : bool array
    last_mask  : bool array
    alpha      : float — overlay opacity (0=invisible, 1=fully opaque)

    Returns
    -------
    overlay : float32 RGBA array
    stats   : dict of pixel counts per category
    """
    h, w = first_mask.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)

    # Non-water pixels stay fully transparent — NAIP shows through
    overlay[~first_mask & ~last_mask] = [0.0, 0.0, 0.0, 0.0]

    # First year only — blue
    overlay[first_mask & ~last_mask] = [0.0, 0.5, 1.0, alpha]

    # Last year only — pink
    overlay[~first_mask & last_mask] = [1.0, 0.1, 0.7, alpha]

    # Both years — white
    overlay[first_mask & last_mask] = [1.0, 1.0, 1.0, alpha]

    stats = {
        "lost":       int(np.sum(first_mask & ~last_mask)),
        "gained":     int(np.sum(~first_mask & last_mask)),
        "persistent": int(np.sum(first_mask & last_mask)),
    }

    return overlay, stats


# =============================================================================
# MAIN LOOP
# =============================================================================

all_mosaics = sorted(MASK_DIR.glob("*_mosaic.tif"))
print(f"Found {len(all_mosaics)} mosaic files\n")

if not all_mosaics:
    print("ERROR: No mosaic files found. Check MASK_DIR path and file naming.")
    raise SystemExit(1)

# Group mosaics by section
sections = defaultdict(dict)
for path in all_mosaics:
    year    = parse_year_from_mask(path)
    section = parse_section_from_mask(path)
    if year is not None and section:
        sections[section][year] = path

print(f"Found {len(sections)} sections\n")

for section_name, year_dict in sorted(sections.items()):
    print(f"{'='*50}")
    print(f"Processing: {section_name}")

    years = sorted(year_dict.keys())
    print(f"  Available years: {years}")

    if len(years) < 2:
        print(f"  Skipping — need at least 2 years, found {len(years)}")
        continue

    first_year = years[0]
    last_year  = years[-1]
    print(f"  Comparing: {first_year} vs {last_year}")

    output_path = OUTPUT_DIR / f"{section_name}_{first_year}_vs_{last_year}.png"
    if output_path.exists():
        print(f"  Skipping — output already exists")
        continue

    # Find the NAIP tile for the first year to use as background imagery
    naip_section_dir = NAIP_DIR / section_name
    naip_path = find_naip_for_year(naip_section_dir, first_year)

    if naip_path is None:
        print(f"  WARNING: no NAIP tile found for {first_year} in {naip_section_dir}")
        print(f"  Skipping — need NAIP background imagery")
        continue

    print(f"  Using NAIP background: naip_{first_year}.tif")

    try:
        # Load NAIP RGB background — this sets the spatial reference everything else aligns to
        naip_rgb, out_shape, naip_transform, naip_crs, bounds = load_naip_rgb(naip_path)

        # Load and align both water masks to match the NAIP tile exactly
        first_mask = load_mask_aligned(
            year_dict[first_year], out_shape, naip_transform, naip_crs
        )
        last_mask = load_mask_aligned(
            year_dict[last_year], out_shape, naip_transform, naip_crs
        )

        # Build the RGBA overlay
        overlay, stats = make_overlay_rgba(first_mask, last_mask)

    except Exception as e:
        print(f"  ERROR: {e}")
        continue

    # ==========================================================================
    # PLOTTING — same structure as movie script's make_frame()
    # ==========================================================================

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("black")
    fig.suptitle(
        f"{section_name.replace('_', ' ')}  —  {first_year} vs {last_year}",
        fontsize=18, fontweight="bold", color="white"
    )

    ax.set_facecolor("black")

    # Draw NAIP RGB as background — same as movie script
    ax.imshow(
        naip_rgb,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin="upper",
        aspect="auto",
        interpolation="bilinear",
        zorder=1
    )

    # Draw color overlay on top at 50% transparency — non-water is transparent
    ax.imshow(
        overlay,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin="upper",
        aspect="auto",
        interpolation="nearest",
        zorder=2
    )

    ax.set_xlim(bounds.left,   bounds.right)
    ax.set_ylim(bounds.bottom, bounds.top)
    ax.set_xlabel("Longitude", color="white", fontsize=9)
    ax.set_ylabel("Latitude",  color="white", fontsize=9)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")

    # Legend with pixel counts
    legend_elements = [
        Patch(facecolor=[0.0, 0.5, 1.0], alpha=OVERLAY_ALPHA,
              label=f"Water lost by {last_year} ({stats['lost']:,} px)"),
        Patch(facecolor=[1.0, 0.1, 0.7], alpha=OVERLAY_ALPHA,
              label=f"Water gained by {last_year} ({stats['gained']:,} px)"),
        Patch(facecolor=[1.0, 1.0, 1.0], alpha=OVERLAY_ALPHA,
              label=f"Persistent water ({stats['persistent']:,} px)"),
        Patch(facecolor=[0.05, 0.05, 0.05], edgecolor="white",
              label=f"No water either year (NAIP background visible)"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        facecolor="black",
        edgecolor="white",
        labelcolor="white",
        framealpha=0.9,
        fontsize=9
    )

    plt.tight_layout()
    plt.savefig(output_path, facecolor="black", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved -> {output_path.name}")

print("\nAll done!")
