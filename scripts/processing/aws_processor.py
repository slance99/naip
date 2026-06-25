# =============================================================================
# naip_watermask_owm.py
#
# Downloads NAIP imagery from AWS Earth Search for a set of GeoPackage AOIs
# and generates water masks using OmniWaterMask.
#
# Install:
#   conda create -n owm python=3.12
#   conda activate owm
#   conda install -c conda-forge gdal rasterio geopandas fiona shapely numpy requests scikit-image
#   pip install pystac-client omniwatermask
#
# Run with: python -u naip_watermask_owm.py   (the -u flushes print output)
# =============================================================================

from pathlib import Path
from collections import defaultdict
import time
import geopandas as gpd
import fiona
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import mapping, box
from pystac_client import Client
import requests
import subprocess
import numpy as np
from omniwatermask import make_water_mask
from scipy.ndimage import binary_fill_holes
from skimage.morphology import closing, opening, disk, remove_small_objects
from skimage.measure import label, regionprops


# =============================================================================
# Setup Conditions - set by user when needed
# =============================================================================

GPKG_DIR   = Path("/home/geomorph/california_rivers/naip/scripts/gpkg_maker/eel_gpkgs/")
NAIP_DIR   = Path("/home/geomorph/california_rivers/naip/naip_omni_tiles/eel_aws")
OUTPUT_DIR = Path("/home/geomorph/california_rivers/naip/outputs/eel_outputs_aws/")

START_YEAR = 2010
END_YEAR   = 2025

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
# Tune these to control how aggressively the mask is cleaned after OmniWaterMask.
# =============================================================================

# Morphological closing radius (pixels) — joins nearby water patches and fills
# thin gaps, including bridges over the channel. Higher = more aggressive joining.
CLOSING_RADIUS = 8

# Morphological opening radius (pixels) — removes thin protrusions and speckle
# at patch edges, including bank sediment fringe. Try 2-4.
OPENING_RADIUS = 2

# Minimum water patch size in pixels — patches smaller than this are removed.
# At 0.6m resolution: 500px = 180 sq meters. At 1m: 500px = 500 sq meters.
MIN_BLOB_SIZE = 500

# Holes smaller than this (pixels) are filled. Larger holes are left open.
MAX_HOLE_SIZE = 1000

# Number of largest connected water bodies to keep. Use >1 so a bridge that
# splits the channel into two large pieces doesn't cause one piece to be
# discarded entirely — both segments will likely be in the top N.
KEEP_TOP_N = 3


# =============================================================================
# SETUP
# =============================================================================

NAIP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# AWS Earth Search — no authentication required
catalog = Client.open("https://earth-search.aws.element84.com/v1")


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


def s3_to_https(url):
    """
    Convert an s3:// URL to the public HTTPS endpoint for AWS Open Data buckets.
    e.g. s3://naip-analytic/ca/2012/... -> https://naip-analytic.s3.amazonaws.com/ca/2012/...
    """
    if url.startswith("s3://"):
        path = url[len("s3://"):]
        bucket, key = path.split("/", 1)
        return f"https://{bucket}.s3.amazonaws.com/{key}"
    return url


def download_tile(href, fname, retries=3, wait=5):
    """Download a tile with retry logic, verifying integrity after each attempt."""
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(href, stream=True, timeout=60)
            r.raise_for_status()
            with open(fname, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            if verify_tile(fname):
                return True
            else:
                print(f"  Attempt {attempt}: {fname.name} failed verification, retrying...")
                fname.unlink()
        except Exception as e:
            print(f"  Attempt {attempt}: {fname.name} error: {e}")
            if fname.exists():
                fname.unlink()
        time.sleep(wait)
    return False


def fetch_naip_tiles(aoi, out_dir):
    """
    Search AWS Earth Search for NAIP tiles intersecting the AOI and download them.
    Searches year by year. No authentication required.
    """
    all_items = []
    for year in range(START_YEAR, END_YEAR + 1):
        for attempt in range(1, 4):
            try:
                search = catalog.search(
                    collections=["naip"],
                    intersects=aoi.__geo_interface__,
                    datetime=f"{year}-01-01/{year}-12-31",
                )
                items = list(search.items())
                all_items.extend(items)
                if items:
                    print(f"  {year}: found {len(items)} tiles")
                break
            except Exception as e:
                print(f"  {year} attempt {attempt} failed: {e}")
                if attempt == 3:
                    print(f"  Skipping {year}")
                time.sleep(30)

    print(f"  Found {len(all_items)} NAIP tiles total")

    tile_paths = []
    for item in all_items:
        fname = out_dir / f"{item.id}.tif"

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

        # Try common asset keys in order of preference
        href = None
        for asset_key in ["image", "visual", "cog", "data"]:
            if asset_key in item.assets:
                href = item.assets[asset_key].href
                break

        if href is None:
            print(f"  WARNING: no downloadable asset found for {item.id}")
            print(f"  Available assets: {list(item.assets.keys())}")
            continue

        # Convert s3:// URLs to https:// since requests can't handle s3://
        href = s3_to_https(href)

        print(f"  Downloading {fname.name}...")
        if download_tile(href, fname):
            tile_paths.append(fname)
        else:
            print(f"  WARNING: {fname.name} failed after all retries, skipping.")

    return tile_paths


def clip_tile_to_aoi(tif_path, aoi):
    """
    Clip a NAIP tile to the AOI and save a temporary clipped version.
    OmniWaterMask processes whole files, so we clip first to keep things
    focused on the river corridor and reduce processing time.
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
    """
    Spatially clean raw OmniWaterMask output.

    Steps:
      1. Morphological closing  — joins nearby patches, fills thin gaps,
                                   and helps reconnect channel segments split
                                   by bridges
      2. Remove small objects   — kills isolated speckle and small false positives
      3. Fill small holes       — fills gaps inside the water mask
      4. Morphological opening  — trims edge protrusions (bank sediment fringe)
      5. Keep top N components  — removes small off-channel water bodies while
                                   preserving large river segments that got
                                   disconnected (e.g. by a bridge deck)
    """
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
    """
    Read an OmniWaterMask output GeoTIFF, apply clean_water_mask(),
    and write the cleaned result back to the same file.
    """
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
    tiles = fetch_naip_tiles(aoi, gpkg_naip_dir)

    tiles_by_year = defaultdict(list)
    for tile in tiles:
        parts = tile.stem.split("_")
        year = None
        for part in parts:
            if len(part) == 8 and part.isdigit():
                year = part[:4]
                break
        if year:
            tiles_by_year[year].append(tile)
        else:
            print(f"  WARNING: could not parse year from {tile.name}, skipping")

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
        else:
            print(f"  No valid masks for {name} {year}")

        for f in clipped_paths:
            f.unlink(missing_ok=True)

    print()

print("All done!")
