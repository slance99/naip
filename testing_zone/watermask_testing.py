# =============================================================================
# test_watermask_cleaning.py
#
# Testing script: runs OmniWaterMask + 4 different cleaning approaches on a
# SINGLE gpkg section (sacramento_22) so cleaning strategies can be compared
# side by side without re-running GPU inference 4x or touching production
# outputs.
#
# Cleaning variants produced (one mosaic per year per variant):
#   baseline                    - current production clean_water_mask(),
#                                  unchanged
#   shape_filter                - baseline, but "keep top-N by area" is
#                                  replaced with a shape-based filter
#                                  (eccentricity/solidity/extent), which
#                                  should drop compact/rectangular blobs
#                                  (e.g. misclassified fields) while keeping
#                                  elongated, sinuous river shapes. Top-N by
#                                  area is still applied among survivors as
#                                  a safety net.
#   skeleton_trim                - baseline, then: skeletonize the single
#                                  largest connected component, buffer that
#                                  skeleton by SKELETON_BUFFER_RADIUS pixels,
#                                  and trim the mask down to that corridor.
#                                  Kills anything not connected to / near
#                                  the main channel.
#   shape_filter_skeleton_trim   - shape_filter cleaning, then the
#                                  skeleton_trim skeleton/buffer trim.
#
# INPUT paths are UNCHANGED from the production script (same GPKG_DIR /
# NAIP_DIR). ALL outputs (clipped tiles, raw masks, cleaned mosaics) are
# written under testing_zone — nothing in production dirs is read-from-only
# or ever overwritten.
#
# Usage:
#   python test_watermask_cleaning.py
# =============================================================================

from pathlib import Path
from collections import defaultdict
import subprocess
import builtins
import numpy as np
import geopandas as gpd
import fiona
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import mapping, box
from scipy.ndimage import binary_fill_holes
from skimage.morphology import (
    closing, opening, disk, remove_small_objects, skeletonize, binary_dilation
)
from skimage.measure import label, regionprops
from omniwatermask import make_water_mask

# =============================================================================
# CONFIG — INPUT PATHS UNCHANGED FROM PRODUCTION SCRIPT
# =============================================================================

RIVER   = "sacramento"
SECTION = "sacramento_22"   # single gpkg/section being tested

GPKG_DIR = Path(f"/home/geomorph/california_rivers/naip/gpkgs/all/{RIVER}_gpkgs/")
NAIP_DIR = Path(f"/home/geomorph/california_rivers/naip/gee_naip/{RIVER}")

# All testing outputs live here — production dirs are never written to
TEST_DIR        = Path("/home/geomorph/california_rivers/naip/testing_zone")
CLIPPED_DIR     = TEST_DIR / "clipped"
RAW_MASK_DIR    = TEST_DIR / "raw_masks"
BASELINE_DIR    = TEST_DIR / "baseline"
SHAPE_FILTER_DIR       = TEST_DIR / "shape_filter"
SKELETON_TRIM_DIR       = TEST_DIR / "skeleton_trim"
SHAPE_FILTER_SKELETON_TRIM_DIR = TEST_DIR / "shape_filter_skeleton_trim"

START_YEAR = 2009
END_YEAR   = 2025

BAND_ORDER    = [1, 2, 3, 4]
MOSAIC_DEVICE = "cuda"
BUFFER_METERS = None
FORCE_RERUN   = True

# Set False once you're happy with the pipeline — True keeps every
# intermediate (clipped tiles, raw masks, per-tile cleaned masks) around
# in testing_zone so you can inspect individual tiles, not just mosaics.
KEEP_INTERMEDIATES = True

# =============================================================================
# BASELINE CLEANING PARAMETERS (unchanged from production)
# =============================================================================

CLOSING_RADIUS = 4
OPENING_RADIUS = 2
MIN_BLOB_SIZE  = 500
MAX_HOLE_SIZE  = 1000
KEEP_TOP_N     = 3

# =============================================================================
# STEP 1 — SHAPE-FILTER PARAMETERS (tune these against your results)
# =============================================================================

# A component is kept if it looks "river-like": elongated (high eccentricity)
# OR non-compact (low solidity / low extent). Field blobs tend to be compact
# rectangles: low eccentricity, high solidity, high extent — these get
# dropped. Start here, then loosen/tighten based on what shape_filter/ mosaics show.
SHAPE_MIN_ECCENTRICITY = 0.90
SHAPE_MAX_SOLIDITY     = 0.55
SHAPE_MAX_EXTENT       = 0.55

# =============================================================================
# STEP 2 — SKELETON/BUFFER TRIM PARAMETERS (tune these against your results)
# =============================================================================

# Buffer distance in PIXELS around the skeleton of the largest component.
# At 0.6m NAIP resolution, 15px ~= 9m each side of centerline (~18m corridor
# total). Widen this if real channel width / point bars / braids are getting
# trimmed off; narrow it if isolated blobs are still surviving.
SKELETON_BUFFER_RADIUS = 15

# =============================================================================
# SETUP
# =============================================================================

for d in [CLIPPED_DIR, RAW_MASK_DIR, BASELINE_DIR, SHAPE_FILTER_DIR, SKELETON_TRIM_DIR, SHAPE_FILTER_SKELETON_TRIM_DIR]:
    d.mkdir(parents=True, exist_ok=True)

_original_print = builtins.print
def safe_print(*args, **kwargs):
    try:
        _original_print(*args, **kwargs, flush=True)
    except OSError:
        pass
builtins.print = safe_print

# =============================================================================
# HELPERS — AOI / CLIPPING (same logic as production; clipped output goes to
# testing_zone instead of the production NAIP directory)
# =============================================================================

def load_aoi(gpkg_path):
    layers = fiona.listlayers(str(gpkg_path))
    gdf = gpd.read_file(gpkg_path, layer=layers[0])
    if BUFFER_METERS:
        gdf = gdf.to_crs("EPSG:3310")
        gdf["geometry"] = gdf.buffer(BUFFER_METERS)
    gdf = gdf.to_crs("EPSG:4326")
    return gdf.union_all()


def clip_tile_to_aoi(tif_path, aoi, out_dir):
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

        clipped_path = out_dir / f"{tif_path.stem}_clipped.tif"
        meta = src.meta.copy()
        meta.update({
            "height": clipped.shape[1],
            "width": clipped.shape[2],
            "transform": transform,
        })
        with rasterio.open(clipped_path, "w", **meta) as dst:
            dst.write(clipped)

    return clipped_path

# =============================================================================
# BASELINE CLEANING (identical to current production clean_water_mask)
# =============================================================================

def clean_baseline(mask):
    cleaned = mask.copy()
    cleaned = closing(cleaned, footprint=disk(CLOSING_RADIUS))
    cleaned = remove_small_objects(cleaned, min_size=MIN_BLOB_SIZE)

    filled    = binary_fill_holes(cleaned)
    holes     = filled & ~cleaned
    big_holes = remove_small_objects(holes, min_size=MAX_HOLE_SIZE)
    cleaned[holes & ~big_holes] = True

    cleaned = opening(cleaned, footprint=disk(OPENING_RADIUS))

    labeled = label(cleaned, connectivity=2)
    if labeled.max() == 0:
        print("    WARNING: baseline cleaning removed all water pixels, returning raw mask")
        return mask

    props = regionprops(labeled)
    top_components = sorted(props, key=lambda r: r.area, reverse=True)[:KEEP_TOP_N]
    top_labels = [r.label for r in top_components]
    return np.isin(labeled, top_labels)

# =============================================================================
# STEP 1 — same pipeline as baseline, but the final "keep top-N by area" step
# is replaced with a shape-based filter (top-N by area is still applied
# among the shape-filter survivors, as a safety net).
# =============================================================================

def clean_shape_filter(mask):
    cleaned = mask.copy()
    cleaned = closing(cleaned, footprint=disk(CLOSING_RADIUS))
    cleaned = remove_small_objects(cleaned, min_size=MIN_BLOB_SIZE)

    filled    = binary_fill_holes(cleaned)
    holes     = filled & ~cleaned
    big_holes = remove_small_objects(holes, min_size=MAX_HOLE_SIZE)
    cleaned[holes & ~big_holes] = True

    cleaned = opening(cleaned, footprint=disk(OPENING_RADIUS))

    labeled = label(cleaned, connectivity=2)
    if labeled.max() == 0:
        print("    WARNING: shape_filter cleaning removed all water pixels, returning raw mask")
        return mask

    props = regionprops(labeled)

    shape_ok = [
        r for r in props
        if r.eccentricity >= SHAPE_MIN_ECCENTRICITY
        or r.solidity <= SHAPE_MAX_SOLIDITY
        or r.extent <= SHAPE_MAX_EXTENT
    ]

    if not shape_ok:
        print("    WARNING: shape filter removed every component, falling back to top-N by area")
        shape_ok = props

    top_components = sorted(shape_ok, key=lambda r: r.area, reverse=True)[:KEEP_TOP_N]
    top_labels = [r.label for r in top_components]
    return np.isin(labeled, top_labels)

# =============================================================================
# STEP 2 — skeletonize the single largest connected component in an
# already-cleaned mask, buffer it, and trim the FULL mask down to that
# corridor. Applied on top of either clean_baseline() or clean_shape_filter().
# =============================================================================

def skeleton_buffer_trim(cleaned_mask, buffer_radius=SKELETON_BUFFER_RADIUS):
    labeled = label(cleaned_mask, connectivity=2)
    if labeled.max() == 0:
        return cleaned_mask

    props = regionprops(labeled)
    largest = max(props, key=lambda r: r.area)
    largest_mask = labeled == largest.label

    skel = skeletonize(largest_mask)
    buffered = binary_dilation(skel, footprint=disk(buffer_radius))

    return cleaned_mask & buffered

# =============================================================================
# WRITE / MOSAIC HELPERS
# =============================================================================

def write_mask(path, array, meta):
    meta = meta.copy()
    meta.update(dtype="uint8", count=1)
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(array.astype(np.uint8), 1)


def mosaic_masks(mask_paths, out_path):
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
# MAIN
# =============================================================================

gpkg_matches = sorted(GPKG_DIR.glob(f"{SECTION}.gpkg"))
if not gpkg_matches:
    raise FileNotFoundError(f"No gpkg found matching {SECTION}.gpkg in {GPKG_DIR}")
gpkg = gpkg_matches[0]

print(f"Testing cleaning variants on: {SECTION}")
print(f"gpkg: {gpkg}")

gpkg_naip_dir = NAIP_DIR / SECTION
if not gpkg_naip_dir.exists():
    raise FileNotFoundError(f"No NAIP directory found at {gpkg_naip_dir}")

naip_tiles = sorted(gpkg_naip_dir.glob("naip_*.tif"))
if not naip_tiles:
    raise FileNotFoundError(f"No NAIP tiles found in {gpkg_naip_dir}")

tiles_by_year = defaultdict(list)
for tile in naip_tiles:
    year = tile.stem.replace("naip_", "")
    if year.isdigit() and START_YEAR <= int(year) <= END_YEAR:
        tiles_by_year[year].append(tile)

print(f"Found tiles for years: {sorted(tiles_by_year.keys())}\n")

aoi = load_aoi(gpkg)

for year, year_tiles in sorted(tiles_by_year.items()):
    print(f"{'='*50}")
    print(f"Year {year} ({len(year_tiles)} tiles)")

    baseline_out    = BASELINE_DIR    / f"{SECTION}_{year}_mosaic.tif"
    shape_filter_out       = SHAPE_FILTER_DIR       / f"{SECTION}_{year}_mosaic.tif"
    skeleton_trim_out       = SKELETON_TRIM_DIR       / f"{SECTION}_{year}_mosaic.tif"
    shape_filter_skeleton_trim_out = SHAPE_FILTER_SKELETON_TRIM_DIR / f"{SECTION}_{year}_mosaic.tif"

    if not FORCE_RERUN and all(p.exists() for p in
                                [baseline_out, shape_filter_out, skeleton_trim_out, shape_filter_skeleton_trim_out]):
        print(f"  Skipping {year} — all 4 mosaics already exist")
        continue

    # --- clip tiles once, shared across all 4 variants ---
    clipped_paths = []
    for tile in year_tiles:
        clipped = clip_tile_to_aoi(tile, aoi, CLIPPED_DIR)
        if clipped:
            clipped_paths.append(clipped)

    if not clipped_paths:
        print(f"  No valid tiles for {year}")
        continue

    # --- run raw water masking ONCE per tile (expensive GPU step) ---
    raw_mask_paths = []
    for clipped in clipped_paths:
        try:
            result = make_water_mask(
                scene_paths=[clipped],
                band_order=BAND_ORDER,
                output_dir=RAW_MASK_DIR,
                mosaic_device=MOSAIC_DEVICE,
            )
            if result:
                raw_mask_paths.extend(result)
                print(f"    Raw mask -> {[p.name for p in result]}")
        except Exception as e:
            print(f"    ERROR on {clipped.name}: {e}")

    if not raw_mask_paths:
        print(f"  No valid raw masks for {year}")
        if not KEEP_INTERMEDIATES:
            for f in clipped_paths:
                f.unlink(missing_ok=True)
        continue

    # --- apply all 4 cleaning variants to each raw mask ---
    baseline_paths, shape_filter_paths, skeleton_trim_paths, shape_filter_skeleton_trim_paths = [], [], [], []

    for raw_path in raw_mask_paths:
        with rasterio.open(raw_path) as src:
            raw = src.read(1).astype(bool)
            meta = src.meta.copy()

        base_cleaned  = clean_baseline(raw)
        shape_filter_cleaned = clean_shape_filter(raw)
        skeleton_trim_cleaned = skeleton_buffer_trim(base_cleaned)
        both_cleaned  = skeleton_buffer_trim(shape_filter_cleaned)

        base_p = BASELINE_DIR    / f"{raw_path.stem}_baseline.tif"
        shape_filter_p   = SHAPE_FILTER_DIR       / f"{raw_path.stem}_shape_filter.tif"
        skeleton_trim_p   = SKELETON_TRIM_DIR       / f"{raw_path.stem}_skeleton_trim.tif"
        shape_filter_skeleton_trim_p  = SHAPE_FILTER_SKELETON_TRIM_DIR / f"{raw_path.stem}_shape_filter_skeleton_trim.tif"

        write_mask(base_p, base_cleaned, meta)
        write_mask(shape_filter_p,   shape_filter_cleaned, meta)
        write_mask(skeleton_trim_p,   skeleton_trim_cleaned, meta)
        write_mask(shape_filter_skeleton_trim_p,  both_cleaned,  meta)

        baseline_paths.append(base_p)
        shape_filter_paths.append(shape_filter_p)
        skeleton_trim_paths.append(skeleton_trim_p)
        shape_filter_skeleton_trim_paths.append(shape_filter_skeleton_trim_p)

        print(f"    Cleaned {raw_path.name} -> 4 variants")

    # --- mosaic each variant separately ---
    mosaic_masks(baseline_paths,    baseline_out)
    mosaic_masks(shape_filter_paths,       shape_filter_out)
    mosaic_masks(skeleton_trim_paths,       skeleton_trim_out)
    mosaic_masks(shape_filter_skeleton_trim_paths, shape_filter_skeleton_trim_out)

    print("  Mosaics written:")
    print(f"    baseline    -> {baseline_out}")
    print(f"    shape_filter       -> {shape_filter_out}")
    print(f"    skeleton_trim       -> {skeleton_trim_out}")
    print(f"    shape_filter_skeleton_trim -> {shape_filter_skeleton_trim_out}")

    # --- cleanup per-tile intermediates (skip if KEEP_INTERMEDIATES) ---
    if not KEEP_INTERMEDIATES:
        for p in (baseline_paths + shape_filter_paths + skeleton_trim_paths + shape_filter_skeleton_trim_paths
                  + raw_mask_paths + clipped_paths):
            Path(p).unlink(missing_ok=True)

    print()

print("Testing run complete. Compare mosaics in:")
print(f"  {BASELINE_DIR}")
print(f"  {SHAPE_FILTER_DIR}")
print(f"  {SKELETON_TRIM_DIR}")
print(f"  {SHAPE_FILTER_SKELETON_TRIM_DIR}")
