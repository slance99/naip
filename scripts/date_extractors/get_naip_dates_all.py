# =============================================================================
# get_naip_dates_all.py
#
# Gets acquisition dates for ALL NAIP data across both sources:
#   1. GEE downloads (naip_{year}.tif) — queries GEE API for exact dates
#   2. EarthExplorer downloads — parses dates from filenames
#
# Outputs a single combined CSV with one row per tile/section/year
#
# Usage:
#   python get_naip_dates_all.py <river>
#   e.g. python get_naip_dates_all.py sacramento
# =============================================================================

from pathlib import Path
import argparse
import re
import csv
from datetime import datetime
import geopandas as gpd
import fiona
import ee


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("river", help="River name e.g. sacramento")
args = parser.parse_args()

RIVER = args.river


# =============================================================================
# CONFIG
# =============================================================================

GEE_PROJECT       = "california-rivers-492000"
GPKG_DIR          = Path(f"/home/geomorph/california_rivers/naip/gpkgs/all/{RIVER}_gpkgs/")
NAIP_DIR          = Path(f"/home/geomorph/california_rivers/naip/gee_naip/{RIVER}")
EARTHEXPLORER_DIR = Path(f"/home/geomorph/california_rivers/naip/earthexplorer/2024/")
OUTPUT_CSV        = Path(f"/home/geomorph/california_rivers/naip/gee_naip/{RIVER}_all_acquisition_dates.csv")

# Years downloaded from GEE
GEE_YEARS = list(range(2009, 2024))

# Years downloaded from EarthExplorer
EE_YEARS = [2024]


# =============================================================================
# SETUP
# =============================================================================

ee.Initialize(project=GEE_PROJECT)

print(f"River:            {RIVER}")
print(f"GEE years:        {GEE_YEARS}")
print(f"EarthExplorer:    {EE_YEARS}")
print(f"Output:           {OUTPUT_CSV}\n")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_aoi(gpkg_path):
    """Load AOI from GeoPackage and return a GEE geometry."""
    layers = fiona.listlayers(str(gpkg_path))
    gdf = gpd.read_file(gpkg_path, layer=layers[0]).to_crs("EPSG:4326")
    aoi = gdf.union_all()
    return ee.Geometry(aoi.__geo_interface__)


def get_gee_dates_for_year(aoi_ee, year):
    """
    Query GEE NAIP collection for acquisition dates intersecting AOI
    in a given year. Returns list of date strings.
    """
    collection = (
        ee.ImageCollection("USDA/NAIP/DOQQ")
        .filterBounds(aoi_ee)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
    )
    count = collection.size().getInfo()
    if count == 0:
        return []

    timestamps = collection.aggregate_array("system:time_start").getInfo()
    dates = sorted(set(
        datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d")
        for ts in timestamps
    ))
    return dates


def get_ee_dates_for_section(aoi, tiles_dir):
    """
    Find EarthExplorer tiles that intersect the AOI and extract
    acquisition dates from their filenames.
    Returns list of date strings.
    """
    from rasterio.warp import transform_bounds
    from shapely.geometry import box
    import rasterio

    all_tiles = [t for t in tiles_dir.glob("*.tif")
                 if "_clipped" not in t.stem]

    if not all_tiles:
        return []

    aoi_bounds = box(*aoi.bounds)
    dates = []

    for tile in all_tiles:
        try:
            with rasterio.open(tile) as src:
                tile_bounds_wgs84 = transform_bounds(
                    src.crs, "EPSG:4326", *src.bounds
                )
                tile_box = box(*tile_bounds_wgs84)

                # Check if tile intersects AOI
                from shapely.geometry import shape
                aoi_shape = shape(aoi.__geo_interface__) if hasattr(aoi, '__geo_interface__') else aoi
                if not tile_box.intersects(aoi_shape):
                    continue

                # Extract date from filename
                match = re.search(r"_(\d{8})\.tif$", tile.name)
                if match:
                    date = datetime.strptime(
                        match.group(1), "%Y%m%d"
                    ).strftime("%Y-%m-%d")
                    dates.append(date)
        except Exception:
            continue

    return sorted(set(dates))


# =============================================================================
# MAIN
# =============================================================================

gpkg_files = sorted(GPKG_DIR.glob("*.gpkg"))
print(f"Found {len(gpkg_files)} GeoPackages\n")

rows = []

for gpkg in gpkg_files:
    section = gpkg.stem
    print(f"{'='*40}")
    print(f"Section: {section}")

    # Check if this section has any downloaded NAIP tiles
    naip_section_dir = NAIP_DIR / section
    if not naip_section_dir.exists():
        print(f"  No NAIP directory found, skipping")
        continue

    # Load AOI for spatial filtering of EarthExplorer tiles
    try:
        layers = fiona.listlayers(str(gpkg))
        gdf = gpd.read_file(gpkg, layer=layers[0]).to_crs("EPSG:4326")
        aoi_shapely = gdf.union_all()
        aoi_ee = ee.Geometry(aoi_shapely.__geo_interface__)
    except Exception as e:
        print(f"  ERROR loading AOI: {e}")
        continue

    # ── GEE years ────────────────────────────────────────────────────────
    for year in GEE_YEARS:
        naip_file = naip_section_dir / f"naip_{year}.tif"
        if not naip_file.exists():
            continue

        try:
            dates = get_gee_dates_for_year(aoi_ee, year)
        except Exception as e:
            print(f"  {year}: GEE query error: {e}")
            dates = []

        date_str = ", ".join(dates) if dates else "not found"
        print(f"  {year} (GEE): {date_str}")

        rows.append({
            "section":           section,
            "year":              year,
            "source":            "GEE",
            "acquisition_dates": date_str,
            "n_scenes":          len(dates),
            "first_date":        dates[0] if dates else "",
            "last_date":         dates[-1] if dates else "",
        })

    # ── EarthExplorer years ──────────────────────────────────────────────
    for year in EE_YEARS:
        naip_file = naip_section_dir / f"naip_{year}.tif"
        if not naip_file.exists():
            continue

        if not EARTHEXPLORER_DIR.exists():
            print(f"  {year} (EarthExplorer): directory not found, skipping")
            continue

        dates = get_ee_dates_for_section(aoi_shapely, EARTHEXPLORER_DIR)
        date_str = ", ".join(dates) if dates else "not found"
        print(f"  {year} (EarthExplorer): {date_str}")

        rows.append({
            "section":           section,
            "year":              year,
            "source":            "EarthExplorer",
            "acquisition_dates": date_str,
            "n_scenes":          len(dates),
            "first_date":        dates[0] if dates else "",
            "last_date":         dates[-1] if dates else "",
        })

# ── Save CSV ─────────────────────────────────────────────────────────────────
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "section", "year", "source", "acquisition_dates",
        "n_scenes", "first_date", "last_date"
    ])
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved {len(rows)} entries -> {OUTPUT_CSV}")
print("All done!")
