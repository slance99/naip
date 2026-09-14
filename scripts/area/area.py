# =============================================================================
# water_area_change.py
#
# For each river section, calculates water area in square meters for:
#   - First year only (water lost by last year)
#   - Both years (persistent water)
#   - Last year only (water gained by last year)
#
# Produces:
#   - CSV with area statistics per section
#   - Bar chart showing area change per section
#   - Stacked area plot showing composition per section
#
# Usage:
#   python water_area_change.py <river> <outputs_folder>
#   e.g. python water_area_change.py sacramento red_bluff_colusa_outputs
# =============================================================================

import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict
import rasterio
from rasterio.warp import reproject, calculate_default_transform
from rasterio.enums import Resampling
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.coords import BoundingBox



# =============================================================================
# CONFIG
# =============================================================================

DATA_ROOT  = Path("/home/geomorph/california_rivers/naip")
MASK_DIR   = DATA_ROOT / "outputs" / "gee" / "red_bluff_colusa_utm"
OUTPUT_DIR = DATA_ROOT / "outputs" / "area_analysis" / "red_bluff_colusa_utm"

# CRS to use for area calculations — must be metric
# EPSG:26910 - what everything should be in, UTM10 
ANALYSIS_CRS = "EPSG:26910"

# Which years to compare — if None uses first and second-to-last available
FIRST_YEAR = 2009
LAST_YEAR  = 2024

# =============================================================================
# SETUP
# =============================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Masks from:   {MASK_DIR}")
print(f"Output to:    {OUTPUT_DIR}")
print(f"Analysis CRS: {ANALYSIS_CRS}\n")

if not MASK_DIR.exists():
    print(f"ERROR: Mask directory not found: {MASK_DIR}")
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


def parse_section_num(section_name):
    """Extract section number e.g. sacramento_25 -> 25"""
    match = re.search(r"_(\d+)$", section_name)
    return int(match.group(1)) if match else 0


def load_mask_in_crs(mask_path, target_crs=ANALYSIS_CRS):
    """
    Load a water mask and reproject to target CRS for area calculation.
    Returns the boolean mask array and pixel area in square meters.
    """
    with rasterio.open(mask_path) as src:
        src_crs = src.crs

        # Check if reprojection needed
        if src_crs.to_epsg() != int(target_crs.split(":")[1]):
            transform, width, height = calculate_default_transform(
                src_crs, target_crs, src.width, src.height, *src.bounds
            )
            dest = np.zeros((height, width), dtype=np.uint8)
            reproject(
                source=rasterio.band(src, 1),
                destination=dest,
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest
            )
            pixel_area_m2 = abs(transform.a * transform.e)
            return dest.astype(bool), pixel_area_m2

        else:
            data = src.read(1).astype(bool)
            pixel_area_m2 = abs(src.transform.a * src.transform.e)
            return data, pixel_area_m2


def align_mask(mask_path, reference_path, target_crs=ANALYSIS_CRS):
    """
    Load and reproject a mask to match the reference mask's shape and extent.
    Both are reprojected to target_crs for area calculations.
    """
    with rasterio.open(reference_path) as ref:
        ref_transform, ref_width, ref_height = calculate_default_transform(
            ref.crs, target_crs, ref.width, ref.height, *ref.bounds
        )
        ref_bounds = ref.bounds

    # Compute destination transform from reference bounds
    dst_transform = calculate_default_transform(
        rasterio.open(reference_path).crs,
        target_crs,
        ref_width,
        ref_height,
        *ref_bounds
    )[0]

    with rasterio.open(mask_path) as src:
        dest = np.zeros((ref_height, ref_width), dtype=np.uint8)
        reproject(
            source=rasterio.band(src, 1),
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest
        )

    pixel_area_m2 = abs(dst_transform.a * dst_transform.e)
    return dest.astype(bool), pixel_area_m2


def calculate_areas(first_mask, last_mask, pixel_area_m2):
    """
    Calculate water areas for each category in square meters and hectares.

    Categories:
      first_only  = water in first year only  (water lost by last year)
      both        = water in both years        (persistent water)
      last_only   = water in last year only    (water gained by last year)
      total_first = total water in first year  (first_only + both)
      total_last  = total water in last year   (last_only + both)
    """
    first_only = first_mask & ~last_mask
    both       = first_mask & last_mask
    last_only  = ~first_mask & last_mask

    def px_to_m2(mask):
        return int(mask.sum()) * pixel_area_m2

    def px_to_ha(mask):
        return px_to_m2(mask) / 10_000

    return {
        "first_only_m2":  px_to_m2(first_only),
        "both_m2":        px_to_m2(both),
        "last_only_m2":   px_to_m2(last_only),
        "total_first_m2": px_to_m2(first_mask),
        "total_last_m2":  px_to_m2(last_mask),
        "net_change_m2":  px_to_m2(last_mask) - px_to_m2(first_mask),
        "pixel_area_m2":  pixel_area_m2,
        "first_px":       int(first_mask.sum()),
        "last_px":        int(last_mask.sum()),
    }


# =============================================================================
# MAIN — scan mosaics and calculate areas
# =============================================================================

all_mosaics = sorted(MASK_DIR.glob("*_mosaic.tif"))
print(f"Found {len(all_mosaics)} mosaic files\n")

if not all_mosaics:
    print("ERROR: No mosaic files found.")
    raise SystemExit(1)

# Group by section
sections = defaultdict(dict)
for path in all_mosaics:
    year    = parse_year_from_mask(path)
    section = parse_section_from_mask(path)
    if year is not None and section:
        sections[section][year] = path

print(f"Found {len(sections)} sections\n")

# Calculate areas for each section
rows = []

for section_name, year_dict in sorted(sections.items(),
                                       key=lambda x: parse_section_num(x[0])):
    years = sorted(year_dict.keys())

    if len(years) < 2:
        print(f"  Skipping {section_name} — need at least 2 years")
        continue

    # Select comparison years
    first_year = FIRST_YEAR if FIRST_YEAR in years else years[0]
    last_year  = LAST_YEAR  if LAST_YEAR  in years else years[-2]

    print(f"Processing: {section_name}  ({first_year} vs {last_year})")

    try:
        # Load first year mask in analysis CRS
        first_mask, pixel_area_m2 = load_mask_in_crs(year_dict[first_year])

        # Load last year mask aligned to first year extent
        last_mask, _ = align_mask(
            year_dict[last_year],
            year_dict[first_year]
        )

        # Handle shape mismatch just in case
        if first_mask.shape != last_mask.shape:
            min_h = min(first_mask.shape[0], last_mask.shape[0])
            min_w = min(first_mask.shape[1], last_mask.shape[1])
            first_mask = first_mask[:min_h, :min_w]
            last_mask  = last_mask[:min_h,  :min_w]

        areas = calculate_areas(first_mask, last_mask, pixel_area_m2)

        row = {
            "section":     section_name,
            "section_num": parse_section_num(section_name),
            "first_year":  first_year,
            "last_year":   last_year,
            **areas
        }
        rows.append(row)

        print(f"  First year total:  {areas['total_first_m2']:.2f} m2")
        print(f"  Last year total:   {areas['total_last_m2']:.2f} m2")
        print(f"  Net change:        {areas['net_change_m2']:+.2f} m2")
        print(f"  Water lost:        {areas['first_only_m2']:.2f} m2")
        print(f"  Water gained:      {areas['last_only_m2']:.2f} m2")
        print(f"  Persistent:        {areas['both_m2']:.2f} m2")
        print()

    except Exception as e:
        import traceback
        print(f"  ERROR: {e}")
        print(traceback.format_exc())


if not rows:
    print("ERROR: No results computed.")
    raise SystemExit(1)

df = pd.DataFrame(rows).sort_values("section_num")

# Save CSV
csv_out = OUTPUT_DIR / f"red_bluff_colusa_area_change.csv"
df.to_csv(csv_out, index=False)
print(f"Saved CSV -> {csv_out}\n")


# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n{'='*50}")
print(f"SUMMARY — {red_bluff_colusa.upper()} {red_bluff_colusa}")
print(f"{'='*50}")
print(f"Sections analyzed:     {len(df)}")
print(f"First year:            {first_year_label}")
print(f"Last year:             {last_year_label}")
print(f"\nTotal water area:")
print(f"  {first_year_label}: {df['total_first_ha'].sum():.2f} ha")
print(f"  {last_year_label}:  {df['total_last_ha'].sum():.2f} ha")
print(f"  Net change:  {df['net_change_ha'].sum():+.2f} ha")
print(f"\nBreakdown:")
print(f"  Lost (first year only):  {df['first_only_ha'].sum():.2f} ha")
print(f"  Gained (last year only): {df['last_only_ha'].sum():.2f} ha")
print(f"  Persistent (both years): {df['both_ha'].sum():.2f} ha")
print("\nAll done!")
