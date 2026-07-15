# =============================================================================
# naip_watermask_gee_nomask.py
#
# Runs OmniWaterMask and mosaicking on already-downloaded GEE NAIP tiles.
# Skips the GEE download step entirely.
#
# Expects NAIP tiles already downloaded at:
#   {NAIP_DIR}/{section}/naip_{year}.tif
#
# Usage:
#   python naip_watermask_nomask.py <river>
#   e.g. python naip_watermask_nomask.py sacramento
#
# For SLURM:
#   sbatch watermask_nomosaic.sh sacramento
# =============================================================================

from pathlib import Path
from collections import defaultdict
import time
import geopandas as gpd
import fiona
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import mapping, box
import subprocess
import numpy as np
from omniwatermask import make_water_mask
from scipy.ndimage import binary_fill_holes
from skimage.morphology import closing, opening, disk, remove_small_objects
from skimage.measure import label, regionprops
import builtins
import argparse


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("river", help="River name e.g. mattole, smith, eel")
args = parser.parse_args()

RIVER = args.river


# =============================================================================
# CONFIG
# =============================================================================

GPKG_DIR   = Path(f"/home/geomorph/california_rivers/naip/gpkgs/all/{RIVER}_gpkgs/")
NAIP_DIR   = Path(f"/home/geomorph/california_rivers/naip/gee_naip/{RIVER}")
OUTPUT_DIR = Path(f"/home/geomorph/california_rivers/naip/outputs/gee/{RIVER}_outputs/")

START_YEAR = 2009
END_YEAR   = 2025

# NAIP band order for OmniWaterMask: R=1, G=2, B=3, NIR=4 (1-based)
BAND_ORDER = [1, 2, 3, 4]

# Set to "cpu" or "cuda"
MOSAIC_DEVICE = "cuda"

# Buffer in meters if your gpkgs are line features, None if already polygons
BUFFER_METERS = None

# Set to True to re-run masking even if mosaic already exists
FORCE_RERUN = True #switching to true 7/13/26 because want to overwrite masks for sac 

# =============================================================================
# CLEANING PARAMETERS
# =============================================================================

CLOSING_RADIUS = 4
OPENING_RADIUS = 2
MIN_BLOB_SIZE  = 500
MAX_HOLE_SIZE  = 1000
KEEP_TOP_N     = 3


# =============================================================================
# SETUP
# =============================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# SAFE PRINT
# =============================================================================

_original_print = builtins.print

def safe_print(*args, **kwargs):
    try:
        _original_print(*args, **kwargs, flush=True)
    except OSError:
        pass

builtins.print = safe_print


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_aoi(gpkg_path):
    """Load AOI from GeoPackage and return a single unified WGS84 geometry."""
    layers = fiona.listlayers(str(gpkg_path))
    gdf = gpd.read_file(gpkg_path, layer=layers[0])
    if BUFFER_METERS:
        gdf = gdf.to_crs("EPSG:3310")
        gdf["geometry"] = gdf.buffer(BUFFER_METERS)
    gdf = gdf.to_crs("EPSG:4326")
    return gdf.union_all()


def clip_tile_to_aoi(tif_path, aoi):
    """
    Clip a NAIP tile to the AOI and save a temporary clipped version.
    Returns the path to the clipped file, or None if there is no overlap.
    """
    with rasterio.open(tif_path) as src:
        tile_crs = src.crs
        aoi_gdf = gpd.GeoDataFrame(geometry=[aoi], crs="EPSG:4326")
        aoi_reprojected = aoi_gdf.to_crs(tile_crs).union_all()

        tile_bounds = box(*src.bounds)
        if not tile_bounds.intersects(aoi_reprojected):
            return None

        aoi_clipped = aoi_reprojected.intersection(tile_bounds)
        print(f"    Intersection area: {aoi_clipped.area:.2f} sq meters")

        try:
            clipped, transform = rio_mask(
                src, [mapping(aoi_clipped)], crop=True, nodata=0, all_touched=True
            )
        except ValueError:
            return None

        clipped_path = tif_path.parent / f"{tif_path.stem}_clipped.tif"
        meta = src.meta.copy()
        meta.update({
            "height": clipped.shape[1],
            "width": clipped.shape[2],
            "transform": transform,
        })
        with rasterio.open(clipped_path, "w", **meta) as dst:
            dst.write(clipped)

    return clipped_path


def clean_water_mask(mask,
                     closing_radius=CLOSING_RADIUS,
                     opening_radius=OPENING_RADIUS,
                     min_size=MIN_BLOB_SIZE,
                     max_hole_size=MAX_HOLE_SIZE,
                     keep_top_n=KEEP_TOP_N):
    """Spatially clean raw OmniWaterMask output."""
    cleaned = mask.copy()
    cleaned = closing(cleaned, footprint=disk(closing_radius))
    cleaned = remove_small_objects(cleaned, min_size=min_size)

    filled    = binary_fill_holes(cleaned)
    holes     = filled & ~cleaned
    big_holes = remove_small_objects(holes, min_size=max_hole_size)
    cleaned[holes & ~big_holes] = True

    cleaned = opening(cleaned, footprint=disk(opening_radius))

    labeled = label(cleaned, connectivity=2)
    if labeled.max() == 0:
        print("    WARNING: cleaning removed all water pixels, returning raw mask")
        return mask

    props = regionprops(labeled)
    top_components = sorted(props, key=lambda r: r.area, reverse=True)[:keep_top_n]
    top_labels = [r.label for r in top_components]

    return np.isin(labeled, top_labels)


def apply_cleaning_to_mask_file(mask_path):
    """Read an OmniWaterMask output GeoTIFF, clean it, write back to same file."""
    with rasterio.open(mask_path) as src:
        data = src.read(1).astype(bool)
        meta = src.meta.copy()

    cleaned = clean_water_mask(data)

    with rasterio.open(mask_path, "w", **meta) as dst:
        dst.write(cleaned.astype(np.uint8), 1)


def mosaic_masks(mask_paths, out_path):
    """Mosaic per-tile water masks into a single GeoTIFF using GDAL."""
    vrt = str(out_path).replace(".tif", ".vrt")
    subprocess.run(["gdalbuildvrt", vrt, *[str(p) for p in mask_paths]], check=True)
    subprocess.run([
        "gdal_translate", "-of", "GTiff",
        "-co", "COMPRESS=LZW",
        "-co", "TILED=YES",
        vrt, str(out_path)
    ], check=True)
    Path(vrt).unlink(missing_ok=True)


# =============================================================================
# MAIN LOOP
# =============================================================================

gpkg_files = sorted(GPKG_DIR.glob("*.gpkg"))
print(f"Found {len(gpkg_files)} GeoPackages in {GPKG_DIR}\n")

for gpkg in gpkg_files:
    name = gpkg.stem
    print(f"{'='*50}")
    print(f"Processing: {name}")

    gpkg_naip_dir = NAIP_DIR / name

    # Check the section's NAIP directory exists and has tiles
    if not gpkg_naip_dir.exists():
        print(f"  No NAIP directory found at {gpkg_naip_dir}, skipping")
        continue

    # Find all naip_{year}.tif files for this section
    naip_tiles = sorted(gpkg_naip_dir.glob("naip_*.tif"))
    if not naip_tiles:
        print(f"  No NAIP tiles found in {gpkg_naip_dir}, skipping")
        continue

    # Group tiles by year
    tiles_by_year = defaultdict(list)
    for tile in naip_tiles:
        year = tile.stem.replace("naip_", "")
        if year.isdigit() and START_YEAR <= int(year) <= END_YEAR:
            tiles_by_year[year].append(tile)

    print(f"  Found tiles for years: {sorted(tiles_by_year.keys())}")

    # Load AOI for clipping
    aoi = load_aoi(gpkg)

    for year, year_tiles in sorted(tiles_by_year.items()):

        mosaic_out = OUTPUT_DIR / f"{name}_{year}_mosaic.tif"
        if mosaic_out.exists() and not FORCE_RERUN:
            print(f"  Skipping {name} {year} — mosaic already exists")
            continue

        print(f"  Processing year {year} ({len(year_tiles)} tiles)")

        clipped_paths = []
        for tile in year_tiles:
            clipped = clip_tile_to_aoi(tile, aoi)
            if clipped:
                clipped_paths.append(clipped)

        if not clipped_paths:
            print(f"  No valid tiles for {name} {year}")
            continue

        mask_paths = []
        for clipped in clipped_paths:
            try:
                result = make_water_mask(
                    scene_paths=[clipped],
                    band_order=BAND_ORDER,
                    output_dir=OUTPUT_DIR,
                    mosaic_device=MOSAIC_DEVICE,
                )
                if result:
                    for mask_path in result:
                        print(f"    Cleaning {mask_path.name}...")
                        apply_cleaning_to_mask_file(mask_path)
                    mask_paths.extend(result)
                    print(f"    Masked and cleaned {clipped.name}")
            except Exception as e:
                print(f"    ERROR on {clipped.name}: {e}")

        if mask_paths:
            mosaic_masks(mask_paths, mosaic_out)
            print(f"  Mosaic -> {mosaic_out}")

            for mp in mask_paths:
                mp.unlink(missing_ok=True)
        else:
            print(f"  No valid masks for {name} {year}")

        for f in clipped_paths:
            f.unlink(missing_ok=True)

    print()

print("All done!")
