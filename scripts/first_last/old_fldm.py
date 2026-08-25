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
# River mile markers (from a point shapefile in NAD83 / EPSG:4269) are
# reprojected into each NAIP tile's native CRS and drawn as big white
# circles with the marker number labeled next to each point.
#
# Usage:
#   python naip_first_last_overlay.py <river_name> <outputs_folder>
#   e.g. python naip_first_last_overlay.py sacramento red_bluff_colusa_outputs
#
# For SLURM:
#   sbatch first_last_overlay.sh sacramento red_bluff_colusa_outputs
#
# Requirements:
#   conda install -c conda-forge rasterio numpy matplotlib geopandas shapely pyproj
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
import geopandas as gpd


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
    help="Explicit path to the river-mile marker point shapefile. "
         "If omitted, the script searches river_miles/ for a shapefile matching the river name."
)
parser.add_argument(
    "--marker-col",
    default="marker",
    help="Name of the marker/label column in the shapefile attribute table (default: 'marker')"
)
args = parser.parse_args()

RIVER      = args.river
OUTPUTS    = args.outputs if args.outputs else f"{RIVER}_outputs"
MARKER_COL = args.marker_col


# =============================================================================
# CONFIG
# =============================================================================

DATA_ROOT = Path("/home/geomorph/california_rivers/naip")

MASK_DIR         = DATA_ROOT / "outputs" / "gee" / OUTPUTS
NAIP_DIR         = DATA_ROOT / "gee_naip" / RIVER
OUTPUT_DIR       = DATA_ROOT / "outputs" / "comparisons" / f"{OUTPUTS}_miles"
RIVER_MILES_DIR  = DATA_ROOT / "river_miles"

OVERLAY_ALPHA = 0.3
DPI           = 150

# Marker styling
MARKER_SIZE       = 220      # scatter point size (points^2)
MARKER_EDGECOLOR  = "black"
MARKER_FACECOLOR  = "white"
MARKER_LINEWIDTH  = 1.5
LABEL_FONTSIZE    = 10
LABEL_COLOR       = "white"
LABEL_OUTLINE     = "black"  # text outline so labels stay legible over any background


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


def find_river_miles_shapefile(explicit_path, river_name):
    """
    Locate the river-mile marker shapefile.
    Priority:
      1. --river-miles-shp if explicitly given
      2. river_miles/{river_name}*.shp (case-insensitive match)
      3. the single .shp in river_miles/ if there's exactly one
    Returns a Path, or None if nothing usable was found.
    """
    if explicit_path:
        p = Path(explicit_path)
        return p if p.exists() else None

    if not RIVER_MILES_DIR.exists():
        return None

    matches = sorted(RIVER_MILES_DIR.glob(f"*{river_name}*.shp"))
    if matches:
        return matches[0]

    all_shp = sorted(RIVER_MILES_DIR.glob("*.shp"))
    if len(all_shp) == 1:
        return all_shp[0]

    return None


def load_river_miles(shp_path, marker_col):
    """
    Load the river-mile marker point shapefile.

    Assumes source CRS is NAD83 / EPSG:4269 (as stated), but reads
    whatever CRS is actually declared in the .prj sidecar — falling
    back to EPSG:4269 only if no CRS is set at all. This avoids
    silently mis-projecting the points if the .prj disagrees.

    If the geometry column is missing/empty but lat/lon attribute
    columns exist, builds point geometry from those instead.

    Returns a GeoDataFrame in its native (lat/lon) CRS — reprojection
    to each NAIP tile's CRS happens later, per-tile, in
    points_for_tile().
    """
    gdf = gpd.read_file(shp_path)

    if gdf.geometry.isna().all() or gdf.geometry.is_empty.all():
        lat_col = next((c for c in gdf.columns if c.lower() in ("lat", "latitude")), None)
        lon_col = next((c for c in gdf.columns if c.lower() in ("lon", "long", "longitude")), None)
        if lat_col and lon_col:
            gdf = gpd.GeoDataFrame(
                gdf,
                geometry=gpd.points_from_xy(gdf[lon_col], gdf[lat_col]),
                crs=gdf.crs if gdf.crs else "EPSG:4269",
            )
        else:
            raise ValueError(
                f"{shp_path} has no usable point geometry and no lat/lon columns were found."
            )

    if gdf.crs is None:
        print(f"  WARNING: {shp_path.name} has no CRS defined — assuming EPSG:4269 (NAD83)")
        gdf = gdf.set_crs("EPSG:4269")

    if marker_col not in gdf.columns:
        candidates = [c for c in gdf.columns if c.lower() == marker_col.lower()]
        if candidates:
            marker_col_actual = candidates[0]
        else:
            raise ValueError(
                f"Column '{marker_col}' not found in {shp_path.name}. "
                f"Available columns: {list(gdf.columns)}"
            )
    else:
        marker_col_actual = marker_col

    gdf = gdf.rename(columns={marker_col_actual: "marker"})
    return gdf[["marker", "geometry"]]


def points_for_tile(river_miles_gdf, target_crs, bounds, pad_frac=0.02):
    """
    Reproject river-mile points into the NAIP tile's CRS and clip to
    the tile's extent (with a small padding fraction so points right
    at the edge still show).

    Returns a list of (x, y, marker) tuples in target_crs coordinates.
    """
    if river_miles_gdf is None or river_miles_gdf.empty:
        return []

    reprojected = river_miles_gdf.to_crs(target_crs)

    pad_x = (bounds.right - bounds.left) * pad_frac
    pad_y = (bounds.top - bounds.bottom) * pad_frac
    minx, maxx = bounds.left - pad_x, bounds.right + pad_x
    miny, maxy = bounds.bottom - pad_y, bounds.top + pad_y

    clipped = reprojected.cx[minx:maxx, miny:maxy]

    return [
        (geom.x, geom.y, marker)
        for geom, marker in zip(clipped.geometry, clipped["marker"])
        if geom is not None and not geom.is_empty
    ]


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

    overlay[first_mask & ~last_mask] = [1.0, 0.5, 0.0, alpha]
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
                       bg_year, output_path, river_mile_points=None):
    """
    Compose and save a single overlay image with the given NAIP background.

    Parameters
    ----------
    naip_rgb          : float RGB array — background imagery
    overlay            : float RGBA array — color overlay
    stats              : dict — pixel counts per category
    bounds              : rasterio BoundingBox — spatial extent for axis labels
    section_name        : str
    first_year           : int — earliest year in comparison
    last_year            : int — most recent year in comparison
    bg_year               : int — which year's NAIP is used as background
    output_path            : Path — where to save the PNG
    river_mile_points : list of (x, y, marker) in the same CRS as `bounds`
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

    # River mile marker points — big white circles with number labels
    if river_mile_points:
        xs = [p[0] for p in river_mile_points]
        ys = [p[1] for p in river_mile_points]
        ax.scatter(
            xs, ys,
            s=MARKER_SIZE,
            facecolor=MARKER_FACECOLOR,
            edgecolor=MARKER_EDGECOLOR,
            linewidth=MARKER_LINEWIDTH,
            zorder=3,
        )
        # small offset (in data units) so the label sits just right of the dot
        x_offset = (bounds.right - bounds.left) * 0.01
        for x, y, marker in river_mile_points:
            ax.annotate(
                str(marker),
                xy=(x, y),
                xytext=(x + x_offset, y),
                fontsize=LABEL_FONTSIZE,
                fontweight="bold",
                color=LABEL_COLOR,
                zorder=4,
                va="center",
                path_effects=[
                    __import__("matplotlib.patheffects", fromlist=["withStroke"])
                    .withStroke(linewidth=2.5, foreground=LABEL_OUTLINE)
                ],
            )

    ax.set_xlim(bounds.left,   bounds.right)
    ax.set_ylim(bounds.bottom, bounds.top)
    ax.set_xlabel("Longitude", color="white", fontsize=9)
    ax.set_ylabel("Latitude",  color="white", fontsize=9)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")

    legend_elements = [
        Patch(facecolor=[1.0, 0.5, 0.0], alpha=OVERLAY_ALPHA,
              label=f"Water lost by {last_year} ({stats['lost']:,} px)"),
        Patch(facecolor=[1.0, 0.1, 0.7], alpha=OVERLAY_ALPHA,
              label=f"Water gained by {last_year} ({stats['gained']:,} px)"),
        Patch(facecolor=[1.0, 1.0, 1.0], alpha=OVERLAY_ALPHA,
              label=f"Persistent water ({stats['persistent']:,} px)"),
        Patch(facecolor=[0.05, 0.05, 0.05], edgecolor="white",
              label="No water either year (NAIP background visible)"),
    ]
    if river_mile_points:
        legend_elements.append(
            Patch(facecolor="white", edgecolor="black", label="River mile marker")
        )
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
# LOAD RIVER MILE MARKERS (once, in native lat/lon CRS — reprojected per tile)
# =============================================================================

river_miles_gdf = None
shp_path = find_river_miles_shapefile(args.river_miles_shp, RIVER)

if shp_path is None:
    print(f"NOTE: no river-mile shapefile found for '{RIVER}' in {RIVER_MILES_DIR} "
          f"— proceeding without markers.\n")
else:
    print(f"River miles: {shp_path}")
    river_miles_gdf = load_river_miles(shp_path, MARKER_COL)
    print(f"  Loaded {len(river_miles_gdf)} markers, source CRS: {river_miles_gdf.crs}\n")


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

    first_year = years[0] #pulling first year
    last_year  = years[-2] #pulling second to last year (aka 2020 because better discharge alignment for sacramento)

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

            tile_points = points_for_tile(river_miles_gdf, naip_crs, bounds)

            save_overlay_image(
                naip_rgb, overlay, stats, bounds,
                section_name, first_year, last_year,
                bg_year=first_year, output_path=out_first,
                river_mile_points=tile_points
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

            tile_points = points_for_tile(river_miles_gdf, naip_crs, bounds)

            save_overlay_image(
                naip_rgb, overlay, stats, bounds,
                section_name, first_year, last_year,
                bg_year=last_year, output_path=out_last,
                river_mile_points=tile_points
            )

            del naip_rgb, first_mask, last_mask, overlay

    except Exception as e:
        print(f"  ERROR: {e}")
        continue

print("\nAll done!")
