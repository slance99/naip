# =============================================================================
# naip_first_last_overlay.py
#
# For each river section, finds the first and last year water mask mosaic,
# overlays them on BOTH the first and last year NAIP imagery, producing
# two output images per section:
#   {section}_{first_year}_vs_{last_year}_bg{first_year}.png  (first year background)
#   {section}_{first_year}_vs_{last_year}_bg{last_year}.png   (last year background)
#
# Color scheme (at 50% transparency over NAIP imagery):
#   Blue  = water in first year only (water lost)
#   Pink  = water in last year only (water gained)
#   White = water in both years (persistent water)
#   Transparent = no water in either year (NAIP imagery shows through)
#
# Usage:
#   python naip_first_last_overlay.py <river_name> <outputs_folder>
#   e.g. python naip_first_last_overlay.py sacramento red_bluff_colusa_outputs
#
# For SLURM:
#   sbatch first_last_overlay.sh sacramento red_bluff_colusa_outputs
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
from rasterio.warp import reproject, Resampling as WarpResampling
from rasterio.transform import from_bounds


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

parser = argparse.ArgumentParser(
    description="Generate first vs last year water mask overlay images for NAIP data"
)
parser.add_argument("river", help="River name e.g. sacramento, mattole, eel")
parser.add_argument(
    "outputs",
    nargs="?",
    default=None,
    help="Output folder name e.g. red_bluff_colusa_outputs. Defaults to {river}_outputs"
)
args = parser.parse_args()

RIVER   = args.river
OUTPUTS = args.outputs if args.outputs else f"{RIVER}_outputs"


# =============================================================================
# CONFIG
# =============================================================================

DATA_ROOT = Path("/home/geomorph/california_rivers/naip")

MASK_DIR   = DATA_ROOT / "outputs" / "gee" / OUTPUTS
NAIP_DIR   = DATA_ROOT / "gee_naip" / RIVER
OUTPUT_DIR = DATA_ROOT / "outputs" / "comparisons" / OUTPUTS

OVERLAY_ALPHA = 0.5
DPI           = 150


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
    """Extract year from mosaic filename e.g. sacramento_1_2022_mosaic.tif -> 2022"""
    match = re.search(r"_(\d{4})_mosaic", path.stem)
    return int(match.group(1)) if match else None


def parse_section_from_mask(path):
    """Extract section name e.g. sacramento_1_2022_mosaic.tif -> sacramento_1"""
    return re.sub(r"_\d{4}_mosaic$", "", path.stem)


def load_naip_rgb(naip_path):
    """
    Load a GEE NAIP tile as an RGB array normalized to 0-1.
    No downsampling — loads at full resolution to avoid alignment issues.
    NAIP band order from GEE: R=1, G=2, B=3, N=4
    Returns rgb array, shape, transform, crs, and bounds.
    """
    with rasterio.open(naip_path) as src:
        r = src.read(1)
        g = src.read(2)
        b = src.read(3)
        transform = src.transform
        crs       = src.crs
        bounds    = src.bounds
        out_shape = (src.height, src.width)

    rgb = np.stack([r, g, b], axis=-1).astype(float)
    rgb = np.clip(rgb / 255.0, 0, 1)
    return rgb, out_shape, transform, crs, bounds


def load_mask_aligned(mask_path, target_shape, naip_bounds, naip_crs):
    """
    Load a water mask and reproject/resample it to exactly match the
    target NAIP tile's shape and extent.
    Uses bounds-derived transform so alignment is always pixel-perfect
    regardless of resolution differences.
    """
    dst_transform = from_bounds(
        naip_bounds.left, naip_bounds.bottom,
        naip_bounds.right, naip_bounds.top,
        target_shape[1], target_shape[0]
    )

    with rasterio.open(mask_path) as src:
        aligned = np.zeros(target_shape, dtype=np.uint8)
        reproject(
            source=rasterio.band(src, 1),
            destination=aligned,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
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
    Build an RGBA overlay from two binary water masks.
    Non-water pixels are fully transparent so NAIP background shows through.

    Colors:
      Blue  [0.0, 0.5, 1.0] = first year only (water lost)
      Pink  [1.0, 0.1, 0.7] = last year only  (water gained)
      White [1.0, 1.0, 1.0] = both years      (persistent)
      Transparent            = no water
    """
    h, w = first_mask.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)

    overlay[first_mask & ~last_mask] = [0.0, 0.5, 1.0, alpha]
    overlay[~first_mask & last_mask] = [1.0, 0.1, 0.7, alpha]
    overlay[first_mask & last_mask]  = [1.0, 1.0, 1.0, alpha]

    stats = {
        "lost":       int(np.sum(first_mask & ~last_mask)),
        "gained":     int(np.sum(~first_mask & last_mask)),
        "persistent": int(np.sum(first_mask & last_mask)),
    }

    return overlay, stats


def save_overlay_image(naip_rgb, overlay, stats, bounds,
                       section_name, first_year, last_year,
                       bg_year, output_path):
    """
    Compose and save a single overlay image with the given NAIP background.

    Parameters
    ----------
    naip_rgb     : float RGB array — background imagery
    overlay      : float RGBA array — color overlay
    stats        : dict — pixel counts per category
    bounds       : rasterio BoundingBox — spatial extent for axis labels
    section_name : str
    first_year   : int — earliest year in comparison
    last_year    : int — most recent year in comparison
    bg_year      : int — which year's NAIP is used as background
    output_path  : Path — where to save the PNG
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("black")
    fig.suptitle(
        f"{section_name.replace('_', ' ')}  —  {first_year} vs {last_year}"
        f"  (background: {bg_year})",
        fontsize=16, fontweight="bold", color="white"
    )

    ax.set_facecolor("black")

    # NAIP RGB background
    ax.imshow(
        naip_rgb,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin="upper",
        aspect="auto",
        interpolation="bilinear",
        zorder=1
    )

    # Color overlay at OVERLAY_ALPHA transparency
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

    legend_elements = [
        Patch(facecolor=[0.0, 0.5, 1.0], alpha=OVERLAY_ALPHA,
              label=f"Water lost by {last_year} ({stats['lost']:,} px)"),
        Patch(facecolor=[1.0, 0.1, 0.7], alpha=OVERLAY_ALPHA,
              label=f"Water gained by {last_year} ({stats['gained']:,} px)"),
        Patch(facecolor=[1.0, 1.0, 1.0], alpha=OVERLAY_ALPHA,
              label=f"Persistent water ({stats['persistent']:,} px)"),
        Patch(facecolor=[0.05, 0.05, 0.05], edgecolor="white",
              label="No water either year (NAIP background visible)"),
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


# =============================================================================
# MAIN LOOP
# =============================================================================

all_mosaics = sorted(MASK_DIR.glob("*_mosaic.tif"))
print(f"Found {len(all_mosaics)} mosaic files\n")

if not all_mosaics:
    print("ERROR: No mosaic files found. Check MASK_DIR path and file naming.")
    raise SystemExit(1)

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

    # Output paths for both background versions
    out_first = OUTPUT_DIR / f"{section_name}_{first_year}_vs_{last_year}_bg{first_year}.png"
    out_last  = OUTPUT_DIR / f"{section_name}_{first_year}_vs_{last_year}_bg{last_year}.png"

    # Skip if both already exist
    if out_first.exists() and out_last.exists():
        print(f"  Skipping — both outputs already exist")
        continue

    naip_section_dir = NAIP_DIR / section_name

    # Find NAIP tiles for both years
    naip_first_path = find_naip_for_year(naip_section_dir, first_year)
    naip_last_path  = find_naip_for_year(naip_section_dir, last_year)

    if naip_first_path is None:
        print(f"  WARNING: no NAIP tile for {first_year}, skipping first background")
    if naip_last_path is None:
        print(f"  WARNING: no NAIP tile for {last_year}, skipping last background")
    if naip_first_path is None and naip_last_path is None:
        print(f"  Skipping — no NAIP tiles found for either year")
        continue

    try:
        # ── First year background ──────────────────────────────────────────
        if naip_first_path and not out_first.exists():
            print(f"  Loading NAIP background: naip_{first_year}.tif")
            naip_rgb, out_shape, naip_transform, naip_crs, bounds = \
                load_naip_rgb(naip_first_path)

            first_mask = load_mask_aligned(
                year_dict[first_year], out_shape, bounds, naip_crs
            )
            last_mask = load_mask_aligned(
                year_dict[last_year], out_shape, bounds, naip_crs
            )

            overlay, stats = make_overlay_rgba(first_mask, last_mask)

            save_overlay_image(
                naip_rgb, overlay, stats, bounds,
                section_name, first_year, last_year,
                bg_year=first_year, output_path=out_first
            )

            del naip_rgb, first_mask, last_mask, overlay

        # ── Last year background ───────────────────────────────────────────
        if naip_last_path and not out_last.exists():
            print(f"  Loading NAIP background: naip_{last_year}.tif")
            naip_rgb, out_shape, naip_transform, naip_crs, bounds = \
                load_naip_rgb(naip_last_path)

            first_mask = load_mask_aligned(
                year_dict[first_year], out_shape, bounds, naip_crs
            )
            last_mask = load_mask_aligned(
                year_dict[last_year], out_shape, bounds, naip_crs
            )

            overlay, stats = make_overlay_rgba(first_mask, last_mask)

            save_overlay_image(
                naip_rgb, overlay, stats, bounds,
                section_name, first_year, last_year,
                bg_year=last_year, output_path=out_last
            )

            del naip_rgb, first_mask, last_mask, overlay

    except Exception as e:
        print(f"  ERROR: {e}")
        continue

print("\nAll done!")
