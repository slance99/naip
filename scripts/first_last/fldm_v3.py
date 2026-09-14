# =============================================================================
# first_last_doubles.py
#
# For each river section, finds the first and last year water mask mosaic,
# overlays them on BOTH the first and last year NAIP imagery producing
# two output images per section.
#
# Key features:
#   - All NAIP tiles reprojected to EPSG:4326 for consistent coordinates
#   - Both images zoomed to first year extent for consistent zoom levels
#   - River mile markers with labels shown on all images
#   - MultiPoint geometry handled correctly (ArcGIS export quirk)
#   - Parallelized across N_WORKERS CPU cores
#   - river_miles_gdf passed explicitly through multiprocessing args
#   - Pool wrapped in __name__ guard (required for spawn method)
#
# Usage:
#   python first_last_doubles.py <river> <outputs_folder>
#   e.g. python first_last_doubles.py sacramento red_bluff_colusa_outputs
#
# For SLURM:
#   sbatch first_last_doubles.sh sacramento red_bluff_colusa_outputs
# =============================================================================

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import os
import re
import argparse
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
parser.add_argument(
    "--river-miles-shp",
    default=None,
    help="Explicit path to the river-mile marker point shapefile."
)
parser.add_argument(
    "--marker-col",
    default="MARKER",
    help="Name of the marker/label column in the shapefile (default: MARKER)"
)
args = parser.parse_args()

RIVER      = args.river
OUTPUTS    = args.outputs if args.outputs else f"{RIVER}_outputs"
MARKER_COL = args.marker_col


# =============================================================================
# CONFIG
# =============================================================================

DATA_ROOT       = Path("/home/geomorph/california_rivers/naip")
MASK_DIR        = DATA_ROOT / "outputs" / "gee" / OUTPUTS
NAIP_DIR        = DATA_ROOT / "gee_naip" / RIVER
OUTPUT_DIR      = DATA_ROOT / "outputs" /"2004_vs_2009"
RIVER_MILES_DIR = DATA_ROOT / "river_miles"

# River miles shapefile — tries explicit arg first, then standard naming,
# then falls back to Sacramento default
if args.river_miles_shp:
    RIVER_MILES_SHP = Path(args.river_miles_shp)
elif (RIVER_MILES_DIR / f"river_mile_markers_{RIVER}.shp").exists():
    RIVER_MILES_SHP = RIVER_MILES_DIR / f"river_mile_markers_{RIVER}.shp"
elif (RIVER_MILES_DIR / f"river_mile_markers_{RIVER}river_2012.shp").exists():
    RIVER_MILES_SHP = RIVER_MILES_DIR / \
        f"river_mile_markers_{RIVER}river_2012.shp"
else:
    RIVER_MILES_SHP = RIVER_MILES_DIR / "river_mile_markers_sacriver_2012.shp"

OVERLAY_ALPHA = 0.5
DPI           = 150
N_WORKERS     = 4
MAX_PIXELS    = 20_000_000

# All tiles reprojected to this CRS for consistent zoom/coordinates
TARGET_CRS = "EPSG:4326"


# =============================================================================
# SETUP
# =============================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"River:           {RIVER}")
print(f"Outputs:         {OUTPUTS}")
print(f"Masks from:      {MASK_DIR}")
print(f"NAIP from:       {NAIP_DIR}")
print(f"Output to:       {OUTPUT_DIR}")
print(f"River miles:     {RIVER_MILES_SHP}")
print(f"Marker column:   {MARKER_COL}")
print(f"Workers:         {N_WORKERS}\n")

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
    Always reprojects to TARGET_CRS (EPSG:4326) so all tiles —
    both GEE (already WGS84) and EarthExplorer (UTM) — are in
    the same coordinate space and produce consistent zoom levels.
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

            # Downsample if too large
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
    Uses bounds-derived transform so alignment is pixel-perfect
    regardless of resolution differences between years.
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
    Reproject river mile markers to match NAIP CRS, clip to tile bounds
    with a small padding buffer, and return list of (x, y, label) tuples.

    Handles both Point and MultiPoint geometries — ArcGIS commonly
    exports single point markers as MultiPoint wrappers.
    """
    if river_miles_gdf is None or river_miles_gdf.empty:
        return []

    # Reproject to match NAIP tile CRS (always EPSG:4326 after load_naip_rgb)
    rm_reprojected = river_miles_gdf.to_crs(naip_crs)

    # Clip to tile bounds with small padding so edge markers still show
    pad_x     = (bounds.right  - bounds.left)   * pad_frac
    pad_y     = (bounds.top    - bounds.bottom)  * pad_frac
    tile_box  = shapely_box(
        bounds.left   - pad_x, bounds.bottom - pad_y,
        bounds.right  + pad_x, bounds.top    + pad_y
    )
    rm_clipped = rm_reprojected[rm_reprojected.geometry.intersects(tile_box)]

    if rm_clipped.empty:
        return []

    def _as_point(geom):
        """
        Unwrap MultiPoint to its first Point.
        ArcGIS often exports single-point markers as MultiPoint wrappers.
        Plain Point geometries are returned unchanged.
        """
        if geom.geom_type == "MultiPoint":
            pts = list(geom.geoms)
            return pts[0] if pts else None
        return geom

    result = []
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

    Parameters
    ----------
    naip_rgb        : float RGB array — background imagery
    overlay         : float RGBA array — color overlay
    stats           : dict — pixel counts per category
    bounds          : BoundingBox — actual extent of naip_rgb
    naip_crs        : CRS — coordinate system (always EPSG:4326)
    section_name    : str
    first_year      : int
    last_year       : int
    bg_year         : int — which year's NAIP is the background
    output_path     : Path
    display_bounds  : BoundingBox or None — if provided, axis limits are set
                      to this extent instead of the tile extent. Used to
                      enforce consistent zoom between first/last year images
                      by passing first year bounds for both.
    river_mile_points : list of (x, y, label) tuples in naip_crs coordinates
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("black")
    fig.suptitle(
        f"{section_name.replace('_', ' ')}  —  {first_year} vs {last_year}"
        f"  (background: {bg_year})",
        fontsize=16, fontweight="bold", color="white"
    )

    ax.set_facecolor("black")

    # NAIP RGB background — drawn at its full native extent
    ax.imshow(
        naip_rgb,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin="upper",
        aspect="auto",
        interpolation="bilinear",
        zorder=1
    )

    # Color overlay — drawn at same extent as background
    ax.imshow(
        overlay,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin="upper",
        aspect="auto",
        interpolation="nearest",
        zorder=2
    )

    # River mile markers with labels
    if river_mile_points:
        for x, y, label_val in river_mile_points:
            ax.plot(x, y, "o",
                    color="white", markersize=8,
                    markeredgecolor="black", markeredgewidth=1,
                    zorder=5)
            ax.annotate(
                label_val, (x, y),
                textcoords="offset points", xytext=(6, 3),
                fontsize=9, color="white", fontweight="bold",
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="black", alpha=0.4,
                          edgecolor="none")
            )

    # Axis limits — use display_bounds if provided so both images
    # show the same geographic extent (first year's bounds used for both)
    plot_bounds = display_bounds if display_bounds is not None else bounds
    ax.set_xlim(plot_bounds.left,   plot_bounds.right)
    ax.set_ylim(plot_bounds.bottom, plot_bounds.top)
    ax.set_xlabel("Longitude", color="white", fontsize=9)
    ax.set_ylabel("Latitude",  color="white", fontsize=9)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")

    legend_elements = [
        mpatches.Patch(facecolor=[1.0, 0.5, 0.0], alpha=OVERLAY_ALPHA,
                       label=f"Water lost by {last_year} "
                             f"({stats['lost']:,} px)"),
        mpatches.Patch(facecolor=[1.0, 0.1, 0.7], alpha=OVERLAY_ALPHA,
                       label=f"Water gained by {last_year} "
                             f"({stats['gained']:,} px)"),
        mpatches.Patch(facecolor=[0.7, 0.7, 0.7], alpha=OVERLAY_ALPHA,
                       label=f"Persistent water "
                             f"({stats['persistent']:,} px)"),
        mpatches.Patch(facecolor=[0.05, 0.05, 0.05], edgecolor="white",
                       label="No water either year (NAIP background visible)"),
        plt.Line2D([0], [0], color="white", linewidth=1,
                   marker="o", markersize=6, markerfacecolor="white",
                   label="River mile marker"),
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

        # Load first year tile first — its bounds become the reference
        # extent used for both output images so zoom is always consistent
        naip_rgb_first, out_shape_first, _, naip_crs_first, bounds_first = \
            load_naip_rgb(naip_first_path)

        # Get river mile points clipped to first year extent —
        # same points used for both images since extent is the same
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

            # Load masks aligned to last year tile dimensions
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
                display_bounds=bounds_first,   # ← first year bounds for consistent zoom
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
# MAIN — wrapped in __name__ guard required for multiprocessing spawn method
# =============================================================================

if __name__ == "__main__":

    # Load river miles shapefile
    river_miles_gdf = None
    if RIVER_MILES_SHP.exists():
        print(f"Loading river miles from {RIVER_MILES_SHP}...")
        river_miles_gdf = gpd.read_file(RIVER_MILES_SHP)

        # Convert MultiPoint to single Points if needed —
        # ArcGIS commonly exports point shapefiles as MultiPoint
        if river_miles_gdf.geometry.geom_type.eq("MultiPoint").any():
            print("  Converting MultiPoint to Point geometries...")
            river_miles_gdf = river_miles_gdf.explode(index_parts=False)
            river_miles_gdf = river_miles_gdf.reset_index(drop=True)

        print(f"  Loaded {len(river_miles_gdf)} river mile markers")
        print(f"  CRS: {river_miles_gdf.crs}")
        print(f"  Columns: {list(river_miles_gdf.columns)}\n")
    else:
        print(f"WARNING: No river miles shapefile found at {RIVER_MILES_SHP}")
        print(f"  Continuing without river mile markers\n")

    # Find all mosaic files and group by section
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

    # Build argument list for parallel workers
    section_args = []
    for section_name, year_dict in sorted(sections.items()):
        years = sorted(year_dict.keys())

        if len(years) < 2:
            print(f"  Skipping {section_name} — need at least 2 years, "
                  f"found {len(years)}")
            continue

        first_year = 2009
        last_year  = 2024   # second to last year

        out_first = OUTPUT_DIR / \
            f"{section_name}_{first_year}_vs_{last_year}_bg{first_year}.png"
        out_last  = OUTPUT_DIR / \
            f"{section_name}_{first_year}_vs_{last_year}_bg{last_year}.png"

        if out_first.exists() and out_last.exists():
            print(f"  Skipping {section_name} — both outputs already exist")
            continue

        naip_section_dir = NAIP_DIR / section_name

        # river_miles_gdf and marker_col passed explicitly through args —
        # multiprocessing workers have separate memory and cannot access
        # variables defined in the main scope
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
