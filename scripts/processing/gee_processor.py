# =============================================================================
# naip_watermask_gee.py
#
# Downloads NAIP imagery from Google Earth Engine for a set of GeoPackage AOIs
# and generates water masks using OmniWaterMask.
#
# Large AOIs are automatically split into a grid of sub-tiles to stay under
# GEE's direct download size limit, then mosaicked back together locally.
# No Google Drive or Cloud Storage needed.
#
# Install:
#   conda activate omni_env
#   pip install earthengine-api
#   earthengine authenticate   # only needed once
# =============================================================================

from pathlib import Path
from collections import defaultdict
import time
import geopandas as gpd
import fiona
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import mapping, box
import requests
import subprocess
import numpy as np
import ee
from omniwatermask import make_water_mask
from scipy.ndimage import binary_fill_holes
from skimage.morphology import closing, opening, disk, remove_small_objects
from skimage.measure import label, regionprops


# =============================================================================
# Setup Conditions - set by user when needed
# =============================================================================

GPKG_DIR   = Path("/home/geomorph/california_rivers/naip/gpkgs/all/smith_gpkgs/")
NAIP_DIR   = Path("/home/geomorph/california_rivers/naip/naip_omni_tiles/smith")
OUTPUT_DIR = Path("/home/geomorph/california_rivers/naip/outputs/smith_outputs/")

GEE_PROJECT = "california-rivers-492000"   # set this to your actual GEE project ID

START_YEAR = 2003
END_YEAR   = 2025

# Grid size for splitting large AOIs before download — 2 means a 2x2 grid
# (4 sub-tiles). Increase if you still hit size limit errors.
GRID_SPLIT = 4

# NAIP band order for OmniWaterMask: R=1, G=2, B=3, NIR=4 (1-based)
BAND_ORDER = [1, 2, 3, 4]

# Set to "cpu" or "cuda"
MOSAIC_DEVICE = "cuda"

# Buffer in meters if your gpkgs are line features, None if already polygons
BUFFER_METERS = None

# Set to True to re-download tiles even if they already exist on disk
FORCE_REDOWNLOAD = False

# =============================================================================
# CLEANING PARAMETERS
# =============================================================================

CLOSING_RADIUS = 8
OPENING_RADIUS = 2
MIN_BLOB_SIZE  = 500
MAX_HOLE_SIZE  = 1000
KEEP_TOP_N     = 3


# =============================================================================
# SETUP
# =============================================================================

NAIP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ee.Initialize(project=GEE_PROJECT)


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
    """
    Verify a tile is readable by sampling three locations in the file.
    Catches truncated downloads that open fine but fail on read.
    """
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


def split_aoi_into_grid(aoi_ee, n_splits=GRID_SPLIT):
    """
    Split an AOI into an n_splits x n_splits grid of sub-tiles, each
    intersected with the original AOI shape. Keeps each sub-download
    under GEE's direct download size limit.
    """
    bounds = aoi_ee.bounds().getInfo()["coordinates"][0]
    lons = [pt[0] for pt in bounds]
    lats = [pt[1] for pt in bounds]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    lon_step = (max_lon - min_lon) / n_splits
    lat_step = (max_lat - min_lat) / n_splits

    sub_tiles = []
    for i in range(n_splits):
        for j in range(n_splits):
            rect = ee.Geometry.Rectangle([
                min_lon + i * lon_step,
                min_lat + j * lat_step,
                min_lon + (i + 1) * lon_step,
                min_lat + (j + 1) * lat_step,
            ])
            sub_tiles.append(rect.intersection(aoi_ee, ee.ErrorMargin(1)))
    return sub_tiles


def get_naip_for_year(aoi_ee, year):
    collection = (
        ee.ImageCollection("USDA/NAIP/DOQQ")
        .filterBounds(aoi_ee)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
    )
    count = collection.size().getInfo()
    if count == 0:
        return None, 0

    image = collection.mosaic().clip(aoi_ee)
    
    # Check if NIR band exists — skip year if not, since OmniWaterMask needs it
    available_bands = image.bandNames().getInfo()
    if "N" not in available_bands:
        print(f"  {year}: no NIR band available, skipping (bands found: {available_bands})")
        return None, 0
    
    return image, count

def download_single_image(image, region_ee, out_path, scale=0.6, retries=3, wait=5):
    for attempt in range(1, retries + 1):
        try:
            url = image.getDownloadURL({
                "scale": scale,
                "crs": "EPSG:4326",
                "region": region_ee,
                "format": "GEO_TIFF",
                "bands": ["R", "G", "B", "N"],
            })
            r = requests.get(url, stream=True, timeout=300)
            r.raise_for_status()

            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

            if verify_tile(out_path):
                return True
            else:
                out_path.unlink(missing_ok=True)

        except Exception as e:
            error_str = str(e).lower()
            print(f"    Attempt {attempt}: {e}")
            if out_path.exists():
                out_path.unlink()
            # Recognize size-limit errors and bail immediately — no point retrying
            if "must be less than" in error_str or "too large" in error_str or "limit" in error_str:
                return False
        time.sleep(wait)

    return False

def download_naip_image_tiled(image, aoi_ee, out_path, year, temp_dir,
                                scale=0.6, n_splits=GRID_SPLIT):
    """
    Download a NAIP image for one year, automatically splitting into a
    grid of sub-tiles if the AOI is too large for a single direct download.
    Sub-tiles are mosaicked back together into out_path using GDAL.
    """
    # Try the whole AOI as a single direct download first
    print(f"    Trying direct download...")
    if download_single_image(image, aoi_ee, out_path, scale=scale):
        return True

    # Fall back to grid splitting
    print(f"    Direct download too large, splitting into {n_splits}x{n_splits} grid...")
    sub_tiles = split_aoi_into_grid(aoi_ee, n_splits=n_splits)

    sub_paths = []
    for idx, sub_tile in enumerate(sub_tiles):
        sub_path = temp_dir / f"_temp_{year}_sub{idx}.tif"
        print(f"    Downloading sub-tile {idx + 1}/{len(sub_tiles)}...")
        success = download_single_image(image, sub_tile, sub_path, scale=scale)
        if success:
            sub_paths.append(sub_path)
        else:
            print(f"    WARNING: sub-tile {idx} failed, skipping")

    if not sub_paths:
        print(f"    WARNING: all sub-tiles failed for {year}")
        return False

    # Mosaic sub-tiles back together into the final output
    vrt_path = temp_dir / f"_temp_{year}.vrt"
    subprocess.run(
        ["gdalbuildvrt", str(vrt_path), *[str(p) for p in sub_paths]],
        check=True, capture_output=True
    )
    subprocess.run([
        "gdal_translate", "-of", "GTiff",
        "-co", "COMPRESS=LZW",
        "-co", "TILED=YES",
        str(vrt_path), str(out_path)
    ], check=True, capture_output=True)

    # Clean up sub-tiles and VRT
    vrt_path.unlink(missing_ok=True)
    for sp in sub_paths:
        sp.unlink(missing_ok=True)

    return verify_tile(out_path)


def fetch_naip_tiles_gee(aoi, aoi_ee, out_dir):
    """
    Search GEE for NAIP imagery year by year and download each year's
    mosaic as a single GeoTIFF per year, splitting into a grid if needed.
    Returns list of downloaded paths.
    """
    tile_paths = []

    for year in range(START_YEAR, END_YEAR + 1):
        fname = out_dir / f"naip_{year}.tif"

        if fname.exists() and not FORCE_REDOWNLOAD:
            if verify_tile(fname):
                print(f"  Skipping {fname.name} (already exists)")
                tile_paths.append(fname)
                continue
            else:
                print(f"  {fname.name} is corrupted, re-downloading...")
                fname.unlink()
        elif fname.exists() and FORCE_REDOWNLOAD:
            print(f"  Force re-downloading {fname.name}...")
            fname.unlink()

        try:
            image, count = get_naip_for_year(aoi_ee, year)
        except Exception as e:
            print(f"  {year}: GEE search error: {e}")
            continue

        if image is None:
            continue

        print(f"  {year}: found {count} tiles, downloading...")
        success = download_naip_image_tiled(image, aoi_ee, fname, year, out_dir)

        if success:
            tile_paths.append(fname)
            print(f"  Downloaded {fname.name}")
        else:
            print(f"  WARNING: {fname.name} failed, skipping.")

    return tile_paths


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
    gpkg_naip_dir.mkdir(parents=True, exist_ok=True)

    aoi = load_aoi(gpkg)
    aoi_ee = ee.Geometry(aoi.__geo_interface__)

    tiles = fetch_naip_tiles_gee(aoi, aoi_ee, gpkg_naip_dir)

    tiles_by_year = defaultdict(list)
    for tile in tiles:
        year = tile.stem.replace("naip_", "")
        tiles_by_year[year].append(tile)

    print(f"  Found tiles for years: {sorted(tiles_by_year.keys())}")

    for year, year_tiles in sorted(tiles_by_year.items()):

        mosaic_out = OUTPUT_DIR / f"{name}_{year}_mosaic.tif"
        if mosaic_out.exists() and not FORCE_REDOWNLOAD:
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
