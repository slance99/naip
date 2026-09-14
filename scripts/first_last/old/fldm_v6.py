# =============================================================================
# first_last_doubles.py
#
# For each river section, finds the first and last year water mask mosaic,
# overlays them on BOTH the first and last year NAIP imagery producing
# two output images per section.
#
# All NAIP tiles displayed in EPSG:26910 (UTM Zone 10N) — no reprojection
# needed since all tiles are already in this CRS, preserving full quality.
#
# Usage:
#   python first_last_doubles.py
#   (all paths hardcoded in CONFIG section below)
#
# For SLURM:
#   sbatch first_last_doubles.sh
# =============================================================================

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict
import rasterio
from rasterio.enums import Resampling as RasterioResampling
from rasterio.warp import reproject, Resampling as WarpResampling
from rasterio.warp import calculate_default_transform
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.crs import CRS
from rasterio.coords import BoundingBox
from rasterio.transform import array_bounds
from shapely.geometry import box as shapely_box
import geopandas as gpd
from multiprocessing import Pool


# =============================================================================
# CONFIG — set all paths here
# =============================================================================

# Directory containing water mask mosaics (*_mosaic.tif)
MASK_DIR = Path("/home/geomorph/california_rivers/naip/outputs/gee/red_bluff_colusa_utm")

# Directory containing per-section NAIP tiles
# Structure: NAIP_DIR/{section_name}/naip_{year}.tif
NAIP_DIR = Path("/home/geomorph/california_rivers/naip/gee_naip/red_bluff_colusa_utm")

# Where to save output PNG images
OUTPUT_DIR = Path("/home/geomorph/california_rivers/naip/outputs/comparisons/red_bluff_colusa_unstretch")

# River miles shapefile
RIVER_MILES_SHP = Path("/home/geomorph/california_rivers/naip/river_miles/river_mile_markers_sacriver_2012.shp")

# Column name for river mile labels in the shapefile
MARKER_COL = "MARKER"

# Overlay transparency (0=invisible, 1=fully opaque)
OVERLAY_ALPHA = 0.5

# Output image DPI
DPI = 300

# Max pixels per NAIP tile before downsampling
MAX_PIXELS = 20_000_000

# Number of parallel workers
N_WORKERS = 4

# Display CRS — EPSG:26910 (UTM Zone 10N)
# All tiles already in this CRS so no reprojection needed — full quality
TARGET_CRS = "EPSG:26910"

# Font sizes (bumped up for readability)
TITLE_FONTSIZE  = 26
AXIS_FONTSIZE   = 22
TICK_FONTSIZE   = 17
LEGEND_FONTSIZE = 18
MARKER_FONTSIZE = 19


# =============================================================================
# SETUP
# =============================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Masks from:    {MASK_DIR}")
print(f"NAIP from:     {NAIP_DIR}")
print(f"Output to:     {OUTPUT_DIR}")
print(f"River miles:   {RIVER_MILES_SHP}")
print(f"Display CRS:   {TARGET_CRS}")
print(f"Workers:       {N_WORKERS}\n")

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


def load_naip_rgb(naip_path, target_crs=TARGET_CRS):
    """
    Load a NAIP tile as RGB normalized to 0-1.
    Since all tiles are already in EPSG:26910, no reprojection is needed
    and images are loaded at full native resolution with no quality loss.
    Reprojection is still handled if a tile is in a different CRS.
    """
    with rasterio.open(naip_path) as src:
        src_crs  = src.crs
        src_epsg = src_crs.to_epsg()
        tgt_epsg = int(target_crs.split(":")[1])
        needs_reproject = src_epsg != tgt_epsg

        if needs_reproject:
            print(f"    Reprojecting from EPSG:{src_epsg} to {target_crs}...")
            transform, width, height = calculate_default_transform(
                src_crs, target_crs, src.width, src.height, *src.bounds
            )

            if height * width > MAX_PIXELS:
                scale  = (MAX_PIXELS / (height * width)) ** 0.5
                height = max(1, int(height * scale))
                width  = max(1, int(width  * scale))
                transform, _, _ = calculate_default_transform(
                    src_crs, target_crs, width, height, *src.bounds
                )
                print(f"    Downsampling to {height}x{width}")

            r = np.zeros((height, width), dtype=np.uint8)
            g = np.zeros((height, width), dtype=np.uint8)
            b = np.zeros((height, width), dtype=np.uint8)

            for band_idx, arr in enumerate([r, g, b], start=1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=arr,
                    src_transform=src.transform,
                    src_crs=src_crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=RasterioResampling.bilinear
                )

            left, bottom, right, top = array_bounds(height, width, transform)
            bounds    = BoundingBox(left, bottom, right, top)
            out_crs   = CRS.from_string(target_crs)
            out_shape = (height, width)

        else:
            # Already in target CRS — load at full resolution, no resampling
            h, w = src.height, src.width
            if h * w > MAX_PIXELS:
                scale = (MAX_PIXELS / (h * w)) ** 0.5
                h = max(1, int(h * scale))
                w = max(1, int(w  * scale))
                print(f"    Downsampling to {h}x{w}")

            r = src.read(1, out_shape=(h, w),
                         resampling=RasterioResampling.bilinear)
            g = src.read(2, out_shape=(h, w),
                         resampling=RasterioResampling.bilinear)
            b = src.read(3, out_shape=(h, w),
                         resampling=RasterioResampling.bilinear)
            transform = src.transform
            bounds    = src.bounds
            out_crs   = src_crs
            out_shape = (h, w)

    rgb = np.stack([r, g, b], axis=-1).astype(float)
    rgb = np.clip(rgb / 255.0, 0, 1)
    return rgb, out_shape, transform, out_crs, bounds


def load_mask_aligned(mask_path, target_shape, naip_bounds, naip_crs):
    """
    Load a water mask and reproject/resample it to exactly match the
    target NAIP tile's shape and extent.
    Uses bounds-derived transform for pixel-perfect alignment.
    """
    dst_transform = transform_from_bounds(
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
      Orange [1.0, 0.5, 0.0] = first year only (water lost by last year)
      Pink   [1.0, 0.1, 0.7] = last year only  (water gained by last year)
      Grey   [0.7, 0.7, 0.7] = both years      (persistent water)
      Transparent             = no water either year
    """
    h, w = first_mask.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)

    overlay[first_mask & ~last_mask] = [1.0, 0.5, 0.0, alpha]
    overlay[~first_mask & last_mask] = [1.0, 0.1, 0.7, alpha]
    overlay[first_mask & last_mask]  = [0.7, 0.7, 0.7, alpha]

    stats = {
        "lost":       int(np.sum(first_mask & ~last_mask)),
        "gained":     int(np.sum(~first_mask & last_mask)),
        "persistent": int(np.sum(first_mask & last_mask)),
    }
    return overlay, stats


def get_river_mile_points(river_miles_gdf, naip_crs, bounds, pad_frac=0.02):
    """
    Reproject river mile markers to match NAIP CRS (EPSG:26910),
    clip to tile bounds with small padding, return (x, y, label) tuples.
    Handles both Point and MultiPoint geometries.
    """
    if river_miles_gdf is None or river_miles_gdf.empty:
        return []

    # Reproject to EPSG:26910 to match NAIP tiles
    rm_reprojected = river_miles_gdf.to_crs(naip_crs)

    pad_x    = (bounds.right  - bounds.left)   * pad_frac
    pad_y    = (bounds.top    - bounds.bottom)  * pad_frac
    tile_box = shapely_box(
        bounds.left   - pad_x, bounds.bottom - pad_y,
        bounds.right  + pad_x, bounds.top    + pad_y
    )
    rm_clipped = rm_reprojected[rm_reprojected.geometry.intersects(tile_box)]

    if rm_clipped.empty:
        return []

    def _as_point(geom):
        """Unwrap MultiPoint to first Point — ArcGIS export quirk."""
        if geom.geom_type == "MultiPoint":
            pts = list(geom.geoms)
            return pts[0] if pts else None
        return geom

    result     = []
    marker_col = MARKER_COL if MARKER_COL in rm_clipped.columns else \
                 next((c for c in rm_clipped.columns
                       if c.lower() == MARKER_COL.lower()), None)

    for _, row in rm_clipped.iterrows():
        pt        = _as_point(row.geometry)
        label_val = row[marker_col] if marker_col else ""
        if pt is None:
            continue
        result.append((pt.x, pt.y, str(label_val)))

    return result


def save_overlay_image(naip_rgb, overlay, stats, bounds, naip_crs,
                       section_name, first_year, last_year,
                       bg_year, output_path,
                       display_bounds=None,
                       river_mile_points=None):
    """
    Compose and save a single overlay image.
    Axes show UTM coordinates in meters since all data is in EPSG:26910.
    display_bounds enforces consistent zoom between first/last year images.
    """
    fig, ax = plt.subplots()
    fig.patch.set_facecolor("white")

    # Concise, descriptive title: what changed, where, and over what span
    section_label = section_name.replace('_', ' ').title()
    fig.suptitle(
        f"{section_label} Water Extent, {first_year}\u2013{last_year} "
        f"(NAIP {bg_year} imagery)",
        fontsize=TITLE_FONTSIZE, fontweight="bold", color="black"
    )

    ax.set_facecolor("black")

    # NAIP RGB background
    ax.imshow(
        naip_rgb,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin="upper", aspect="equal", interpolation="bilinear", zorder=1
    )

    # Color overlay
    ax.imshow(
        overlay,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin="upper", aspect="equal", interpolation="nearest", zorder=2
    )

    # River mile markers
    if river_mile_points:
        for x, y, label_val in river_mile_points:
            ax.plot(x, y, "o", color="white", markersize=14,
                    markeredgecolor="black", markeredgewidth=1, zorder=5)
            ax.annotate(
                label_val, (x, y),
                textcoords="offset points", xytext=(6, 3),
                fontsize=MARKER_FONTSIZE, color="white", fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black",
                          alpha=0.4, edgecolor="none")
            )

    # Axis limits — use display_bounds for consistent zoom across both images
    plot_bounds = display_bounds if display_bounds is not None else bounds
    ax.set_xlim(plot_bounds.left,   plot_bounds.right)
    ax.set_ylim(plot_bounds.bottom, plot_bounds.top)

    data_width  = plot_bounds.right  - plot_bounds.left
    data_height = plot_bounds.top    - plot_bounds.bottom
    aspect_ratio = data_height / data_width

    # Set a maximum for either dimension to keep files manageable
    MAX_DIM = 16  # inches

    if aspect_ratio >= 1:
    # Taller than wide — height is the limiting dimension
        fig_height = MAX_DIM
        fig_width  = MAX_DIM / aspect_ratio
    else:
    # Wider than tall — width is the limiting dimension
        fig_width  = MAX_DIM
        fig_height = MAX_DIM * aspect_ratio

# Add a little extra height for title and legend
    fig_height += 1.5
    fig.set_size_inches(fig_width, fig_height)
    # Axis labels in meters since CRS is UTM
    ax.set_xlabel("Easting (m)",  color="black", fontsize=AXIS_FONTSIZE)
    ax.set_ylabel("Northing (m)", color="black", fontsize=AXIS_FONTSIZE)
    ax.tick_params(colors="black", labelsize=TICK_FONTSIZE)

    # No border/spine outline around the plot
    for spine in ax.spines.values():
        spine.set_visible(False)

    legend_elements = [
        mpatches.Patch(facecolor=[1.0, 0.5, 0.0], alpha=OVERLAY_ALPHA,
                       label=f"Water Present 2009 ({stats['lost']:,} px)"),
        mpatches.Patch(facecolor=[1.0, 0.1, 0.7], alpha=OVERLAY_ALPHA,
                       label=f"New Water Present in 2024 ({stats['gained']:,} px)"),
        mpatches.Patch(facecolor=[0.7, 0.7, 0.7], alpha=OVERLAY_ALPHA,
                       label=f"Persistent Water ({stats['persistent']:,} px)"),
        mpatches.Patch(facecolor=[0.05, 0.05, 0.05], edgecolor="none",
                       label="NAIP Background "),
        plt.Line2D([0], [0], color="white", linewidth=1,
                   marker="o", markersize=10, markerfacecolor="white",
                   label="River mile marker"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", facecolor="white",
              edgecolor="none", labelcolor="black", framealpha=0.9,
              fontsize=LEGEND_FONTSIZE)

    plt.tight_layout()
    plt.savefig(output_path, facecolor="white", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {output_path.name}")


# =============================================================================
# PROCESS_SECTION — called by each parallel worker
# =============================================================================

def process_section(args):
    """
    Process a single section — load NAIP, load masks, build overlay, save PNGs.
    All arguments passed explicitly since multiprocessing workers have
    separate memory spaces and cannot access main-scope variables.
    """
    (section_name, year_dict, naip_section_dir,
     out_first, out_last, first_year, last_year,
     river_miles_gdf, marker_col) = args

    messages = []

    try:
        naip_first_path = find_naip_for_year(naip_section_dir, first_year)
        naip_last_path  = find_naip_for_year(naip_section_dir, last_year)

        if naip_first_path is None:
            return f"  WARNING: no NAIP tile for {first_year} in {section_name}"
        if naip_last_path is None:
            return f"  WARNING: no NAIP tile for {last_year} in {section_name}"

        # Load first year — its bounds are the reference for consistent zoom
        naip_rgb_first, out_shape_first, _, naip_crs_first, bounds_first = \
            load_naip_rgb(naip_first_path)

        # River mile points clipped to first year extent
        river_mile_points = get_river_mile_points(
            river_miles_gdf, naip_crs_first, bounds_first
        ) if river_miles_gdf is not None else []

        # ── First year background ─────────────────────────────────────────
        if not out_first.exists():
            first_mask = load_mask_aligned(
                year_dict[first_year], out_shape_first,
                bounds_first, naip_crs_first
            )
            last_mask = load_mask_aligned(
                year_dict[last_year], out_shape_first,
                bounds_first, naip_crs_first
            )
            overlay, stats = make_overlay_rgba(first_mask, last_mask)

            save_overlay_image(
                naip_rgb_first, overlay, stats,
                bounds_first, naip_crs_first,
                section_name, first_year, last_year,
                bg_year=first_year, output_path=out_first,
                display_bounds=bounds_first,
                river_mile_points=river_mile_points
            )
            del first_mask, last_mask, overlay
            messages.append(f"  Saved {out_first.name}")

        # ── Last year background ──────────────────────────────────────────
        if not out_last.exists():
            naip_rgb_last, out_shape_last, _, naip_crs_last, bounds_last = \
                load_naip_rgb(naip_last_path)

            first_mask = load_mask_aligned(
                year_dict[first_year], out_shape_last,
                bounds_last, naip_crs_last
            )
            last_mask = load_mask_aligned(
                year_dict[last_year], out_shape_last,
                bounds_last, naip_crs_last
            )
            overlay, stats = make_overlay_rgba(first_mask, last_mask)

            save_overlay_image(
                naip_rgb_last, overlay, stats,
                bounds_last, naip_crs_last,
                section_name, first_year, last_year,
                bg_year=last_year, output_path=out_last,
                display_bounds=bounds_first,
                river_mile_points=river_mile_points
            )
            del naip_rgb_last, first_mask, last_mask, overlay
            messages.append(f"  Saved {out_last.name}")

        del naip_rgb_first

        if not messages:
            return f"  Skipped {section_name} — both outputs already exist"
        return "\n".join(messages)

    except Exception as e:
        import traceback
        return f"  ERROR on {section_name}: {e}\n{traceback.format_exc()}"


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # Load river miles
    river_miles_gdf = None
    if RIVER_MILES_SHP.exists():
        print(f"Loading river miles from {RIVER_MILES_SHP}...")
        river_miles_gdf = gpd.read_file(RIVER_MILES_SHP)

        if river_miles_gdf.geometry.geom_type.eq("MultiPoint").any():
            print("  Converting MultiPoint to Point geometries...")
            river_miles_gdf = river_miles_gdf.explode(index_parts=False)
            river_miles_gdf = river_miles_gdf.reset_index(drop=True)

        print(f"  Loaded {len(river_miles_gdf)} river mile markers")
        print(f"  CRS: {river_miles_gdf.crs}")
        print(f"  Columns: {list(river_miles_gdf.columns)}\n")
    else:
        print(f"WARNING: No river miles shapefile found at {RIVER_MILES_SHP}\n")

    # Find all mosaic files and group by section
    all_mosaics = sorted(MASK_DIR.glob("*_mosaic.tif"))
    print(f"Found {len(all_mosaics)} mosaic files\n")

    if not all_mosaics:
        print("ERROR: No mosaic files found.")
        raise SystemExit(1)

    sections = defaultdict(dict)
    for path in all_mosaics:
        year    = parse_year_from_mask(path)
        section = parse_section_from_mask(path)
        if year is not None and section:
            sections[section][year] = path

    print(f"Found {len(sections)} sections\n")

    section_args = []
    for section_name, year_dict in sorted(sections.items()):
        years = sorted(year_dict.keys())

        if len(years) < 2:
            print(f"  Skipping {section_name} — need at least 2 years")
            continue

        first_year = 2009
        last_year  = 2024

        # Simplified output filenames
        out_first = OUTPUT_DIR / f"{section_name}_bg{first_year}.png"
        out_last  = OUTPUT_DIR / f"{section_name}_bg{last_year}.png"

        if out_first.exists() and out_last.exists():
            print(f"  Skipping {section_name} — both outputs already exist")
            continue

        naip_section_dir = NAIP_DIR / section_name

        section_args.append((
            section_name, year_dict, naip_section_dir,
            out_first, out_last, first_year, last_year,
            river_miles_gdf, MARKER_COL
        ))

    print(f"Processing {len(section_args)} sections with {N_WORKERS} workers\n")

    if section_args:
        with Pool(processes=N_WORKERS) as pool:
            results = pool.map(process_section, section_args)

        for result in results:
            if result:
                print(result)

    print("\nAll done!")
