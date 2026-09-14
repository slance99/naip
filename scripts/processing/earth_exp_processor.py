# =============================================================================
# naip_2024_processor.py
#
# Processes already-downloaded EarthExplorer NAIP tiles for 2024,
# renames them to match GEE naming convention (naip_2024.tif),
# clips to AOI, runs OmniWaterMask, and saves mosaics alongside
# existing GEE outputs so everything stays in one place.
#
# Expects EarthExplorer tiles already downloaded and placed in:
#   {EARTHEXPLORER_DIR}/*.tif
#
# Outputs match existing GEE structure:
#   NAIP tile:  gee_naip/{river}/{section}/naip_2024.tif
#   Mosaic:     outputs/gee/{outputs}/{section}_2024_mosaic.tif
#
# Usage:
#   python naip_2024_processor.py <river> <gpkgs>
#   e.g. python naip_2024_processor.py sacramento red_bluff_colusa_gpkgs
#
# Requirements:
#   conda install -c conda-forge gdal rasterio geopandas fiona shapely numpy scikit-image
#   pip install omniwatermask
# =============================================================================

from pathlib import Path
from collections import defaultdict
import shutil
import time
import geopandas as gpd
import fiona
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import transform_bounds
from shapely.geometry import mapping, box
import subprocess
import numpy as np
from omniwatermask import make_water_mask
from scipy.ndimage import binary_fill_holes
from skimage.morphology import closing, opening, disk, remove_small_objects
from skimage.measure import label, regionprops
import builtins
import argparse
import traceback


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("river", help="River name e.g. sacramento")
parser.add_argument("gpkgs", help="GPKG folder name e.g. red_bluff_colusa_gpkgs")
args = parser.parse_args()

RIVER  = args.river
GPKGS  = args.gpkgs


# =============================================================================
# CONFIG
# =============================================================================

# Where your EarthExplorer downloaded tiles are sitting
EARTHEXPLORER_DIR = Path(f"/home/geomorph/california_rivers/naip/earthexplorer/2024/")

# These match your existing GEE structure exactly
GPKG_DIR   = Path(f"/home/geomorph/california_rivers/naip/gpkgs/all/{GPKGS}/")
NAIP_DIR   = Path(f"/home/geomorph/california_rivers/naip/gee_naip/{RIVER}")
OUTPUT_DIR = Path(f"/home/geomorph/california_rivers/naip/outputs/gee/red_bluff_colusa_utm/")

# Year being processed
YEAR = 2024

# NAIP band order for OmniWaterMask: R=1, G=2, B=3, NIR=4 (1-based)
BAND_ORDER = [1, 2, 3, 4]

# Set to "cpu" or "cuda"
MOSAIC_DEVICE = "cuda"

# Buffer in meters if your gpkgs are line features, None if already polygons
BUFFER_METERS = None

# Set to True to reprocess even if outputs already exist
FORCE_RERUN = False

# Chunk size in pixels for large tiles — 5000x5000 fits in L40S VRAM
CHUNK_SIZE = 5000

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

# Safe print to handle NFS stale file handle errors
_original_print = builtins.print

def safe_print(*args, **kwargs):
    try:
        _original_print(*args, **kwargs, flush=True)
    except OSError:
        pass

builtins.print = safe_print

print(f"River:             {RIVER}")
print(f"GPKGs:             {GPKGS}")
print(f"EarthExplorer dir: {EARTHEXPLORER_DIR}")
print(f"NAIP output dir:   {NAIP_DIR}")
print(f"Mask output dir:   {OUTPUT_DIR}")
print(f"Year:              {YEAR}\n")

for check_path, check_label in [
    (GPKG_DIR,          "GPKG directory"),
    (EARTHEXPLORER_DIR, "EarthExplorer tile directory"),
]:
    if not check_path.exists():
        print(f"ERROR: {check_label} not found: {check_path}")
        raise SystemExit(1)

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


def verify_tile(fname):
    """Verify a tile is readable by sampling three locations."""
    try:
        with rasterio.open(fname) as src:
            h, w = src.height, src.width
            src.read(1, window=rasterio.windows.Window(0, 0, 256, 256))
            src.read(1, window=rasterio.windows.Window(w // 2, h // 2, 256, 256))
            src.read(1, window=rasterio.windows.Window(
                max(0, w - 256), max(0, h - 256), 256, 256))
        return True
    except Exception:
        return False


def find_earthexplorer_tiles_for_aoi(aoi, tiles_dir):
    """
    Find all EarthExplorer NAIP tiles that intersect the AOI.
    Checks each tile's spatial extent against the AOI bounding box.
    Returns list of matching tile paths.
    """
    all_tiles = [t for t in tiles_dir.glob("*.tif")
                 if "_clipped" not in t.stem and "naip_" not in t.stem]

    if not all_tiles:
        return []

    matching = []
    aoi_bounds = box(*aoi.bounds)

    for tile in all_tiles:
        try:
            with rasterio.open(tile) as src:
                tile_bounds_wgs84 = transform_bounds(
                    src.crs, "EPSG:4326", *src.bounds
                )
                tile_box = box(*tile_bounds_wgs84)
                if aoi_bounds.intersects(tile_box):
                    matching.append(tile)
        except Exception:
            continue

    return matching


def mosaic_tiles_to_naip(tile_paths, out_path):
    """
    Mosaic multiple EarthExplorer tiles into a single GeoTIFF named
    naip_{year}.tif to match the GEE naming convention.
    """
    if len(tile_paths) == 1:
        shutil.copy2(tile_paths[0], out_path)
        print(f"    Copied single tile -> {out_path.name}")
    else:
        vrt_path = out_path.parent / f"_temp_{out_path.stem}.vrt"
        subprocess.run(
            ["gdalbuildvrt", str(vrt_path), *[str(t) for t in tile_paths]],
            check=True, capture_output=True
        )
        subprocess.run([
            "gdal_translate", "-of", "GTiff",
            "-co", "COMPRESS=LZW",
            "-co", "TILED=YES",
            str(vrt_path), str(out_path)
        ], check=True, capture_output=True)
        vrt_path.unlink(missing_ok=True)
        print(f"    Mosaicked {len(tile_paths)} tiles -> {out_path.name}")


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
            "width":  clipped.shape[2],
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
    mask_path = Path(mask_path)  # ensure Path object
    with rasterio.open(mask_path) as src:
        data = src.read(1).astype(bool)
        meta = src.meta.copy()

    cleaned = clean_water_mask(data)

    with rasterio.open(mask_path, "w", **meta) as dst:
        dst.write(cleaned.astype(np.uint8), 1)


def process_large_tile_in_chunks(clipped_path, band_order,
                                  mosaic_device, chunk_size=CHUNK_SIZE):
    """
    Split a large clipped tile into smaller chunks, run OmniWaterMask
    on each chunk separately, then return list of mask paths to mosaic.
    chunk_size is in pixels — 5000x5000 should fit in L40S GPU VRAM.
    """
    with rasterio.open(clipped_path) as src:
        h, w   = src.height, src.width
        meta   = src.meta.copy()
        transform = src.transform

    print(f"    Tile size: {h}x{w} pixels — splitting into chunks of {chunk_size}")

    chunk_mask_paths = []
    chunk_idx = 0

    for row_start in range(0, h, chunk_size):
        for col_start in range(0, w, chunk_size):
            row_end = min(row_start + chunk_size, h)
            col_end = min(col_start + chunk_size, w)

            chunk_path = clipped_path.parent / \
                         f"{clipped_path.stem}_chunk_{chunk_idx}.tif"

            # Write chunk to disk
            window = rasterio.windows.Window(
                col_start, row_start,
                col_end - col_start,
                row_end - row_start
            )
            chunk_transform = rasterio.windows.transform(window, transform)

            with rasterio.open(clipped_path) as src:
                chunk_data = src.read(window=window)

            chunk_meta = meta.copy()
            chunk_meta.update({
                "height":    chunk_data.shape[1],
                "width":     chunk_data.shape[2],
                "transform": chunk_transform,
            })

            with rasterio.open(chunk_path, "w", **chunk_meta) as dst:
                dst.write(chunk_data)

            print(f"    Processing chunk {chunk_idx + 1} "
                  f"(rows {row_start}-{row_end}, cols {col_start}-{col_end})...")

            try:
                # Pass paths as strings — OmniWaterMask expects strings not Path objects
                result = make_water_mask(
                    scene_paths=[str(chunk_path)],
                    band_order=band_order,
                    output_dir=OUTPUT_DIR,
                    mosaic_device=mosaic_device,
                )
                if result:
                    for mask_path in result:
                        apply_cleaning_to_mask_file(Path(mask_path))
                    chunk_mask_paths.extend([Path(p) for p in result])
                    print(f"    Chunk {chunk_idx + 1} complete")
            except Exception as e:
                print(f"    ERROR on chunk {chunk_idx}: {e}")
                print(traceback.format_exc())
            finally:
                chunk_path.unlink(missing_ok=True)

            chunk_idx += 1

    return chunk_mask_paths


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

ee_tiles = [t for t in EARTHEXPLORER_DIR.glob("*.tif")
            if "_clipped" not in t.stem]
print(f"Found {len(ee_tiles)} EarthExplorer tiles in {EARTHEXPLORER_DIR}\n")

for gpkg in gpkg_files:
    name = gpkg.stem
    print(f"{'='*50}")
    print(f"Processing: {name}")

    mosaic_out = OUTPUT_DIR / f"{name}_{YEAR}_mosaic.tif"
    if mosaic_out.exists() and not FORCE_RERUN:
        print(f"  Skipping {name} {YEAR} — mosaic already exists")
        print()
        continue

    aoi = load_aoi(gpkg)

    naip_section_dir = NAIP_DIR / name
    naip_section_dir.mkdir(parents=True, exist_ok=True)

    naip_out = naip_section_dir / f"naip_{YEAR}.tif"

    if naip_out.exists() and not FORCE_RERUN:
        print(f"  Using existing {naip_out.name}")
        naip_tile = naip_out
    else:
        print(f"  Finding EarthExplorer tiles for {name}...")
        matching_tiles = find_earthexplorer_tiles_for_aoi(aoi, EARTHEXPLORER_DIR)

        if not matching_tiles:
            print(f"  No EarthExplorer tiles found for {name}, skipping")
            print()
            continue

        print(f"  Found {len(matching_tiles)} matching tiles")
        mosaic_tiles_to_naip(matching_tiles, naip_out)

        if not verify_tile(naip_out):
            print(f"  ERROR: {naip_out.name} failed verification, skipping")
            naip_out.unlink(missing_ok=True)
            print()
            continue

        naip_tile = naip_out

    # Clip to AOI
    print(f"  Clipping to AOI...")
    clipped = clip_tile_to_aoi(naip_tile, aoi)

    if clipped is None:
        print(f"  No overlap between NAIP tile and AOI for {name}, skipping")
        print()
        continue

    # Run OmniWaterMask — split into chunks if tile is too large for GPU VRAM
    print(f"  Running OmniWaterMask...")
    mask_paths = []

    try:
        with rasterio.open(clipped) as src:
            h, w = src.height, src.width

        if h * w > CHUNK_SIZE * CHUNK_SIZE:
            print(f"  Tile is large ({h}x{w}) — processing in chunks")
            mask_paths = process_large_tile_in_chunks(
                clipped_path=clipped,
                band_order=BAND_ORDER,
                mosaic_device=MOSAIC_DEVICE,
                chunk_size=CHUNK_SIZE
            )
        else:
            # Small enough to process in one go
            result = make_water_mask(
                scene_paths=[str(clipped)],
                band_order=BAND_ORDER,
                output_dir=OUTPUT_DIR,
                mosaic_device=MOSAIC_DEVICE,
            )
            if result:
                for mask_path in result:
                    print(f"    Cleaning {Path(mask_path).name}...")
                    apply_cleaning_to_mask_file(Path(mask_path))
                mask_paths.extend([Path(p) for p in result])
                print(f"    Masked and cleaned {clipped.name}")

    except Exception as e:
        print(f"    ERROR on {clipped.name}: {e}")
        print(traceback.format_exc())

    # Clean up clipped temp file
    clipped.unlink(missing_ok=True)

    # Mosaic masks into final output
    if mask_paths:
        mosaic_masks(mask_paths, mosaic_out)
        print(f"  Mosaic -> {mosaic_out}")

        # Clean up intermediate chunk mask files
        for mp in mask_paths:
            mp.unlink(missing_ok=True)
    else:
        print(f"  No valid masks for {name} {YEAR}")

    print()

print("All done!")
