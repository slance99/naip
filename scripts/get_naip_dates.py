# =============================================================================
# get_naip_dates_gee.py
#
# Queries Google Earth Engine to get actual NAIP acquisition dates
# for each section and year, and saves a summary CSV.
#
# Usage:
#   python get_naip_dates_gee.py <river>
#   e.g. python get_naip_dates_gee.py sacramento
# =============================================================================

from pathlib import Path
import argparse
import csv
import datetime
import fiona
import geopandas as gpd
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

GPKG_DIR   = Path(f"/home/geomorph/california_rivers/naip/gpkgs/all/{RIVER}_gpkgs/")
NAIP_DIR   = Path(f"/home/geomorph/california_rivers/naip/gee_naip/{RIVER}")
OUTPUT_CSV = Path(f"/home/geomorph/california_rivers/naip/gee_naip/{RIVER}_acquisition_dates.csv")

GEE_PROJECT = "california-rivers-492000"

START_YEAR = 2009
END_YEAR   = 2025

# =============================================================================
# SETUP
# =============================================================================

ee.Initialize(project=GEE_PROJECT)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_aoi(gpkg_path):
    """Load AOI from GeoPackage and return a GEE geometry."""
    layers = fiona.listlayers(str(gpkg_path))
    gdf = gpd.read_file(gpkg_path, layer=layers[0]).to_crs("EPSG:4326")
    aoi = gdf.union_all()
    return ee.Geometry(aoi.__geo_interface__)


def get_naip_dates_for_year(aoi_ee, year):
    """
    Query GEE for all NAIP scenes intersecting the AOI in a given year.
    Returns a list of acquisition date strings.
    """
    collection = (
        ee.ImageCollection("USDA/NAIP/DOQQ")
        .filterBounds(aoi_ee)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
    )

    count = collection.size().getInfo()
    if count == 0:
        return []

    # Get acquisition dates from image metadata
    timestamps = collection.aggregate_array("system:time_start").getInfo()
    dates = [
        datetime.datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d")
        for ts in timestamps
    ]
    return sorted(set(dates))  # unique dates only


# =============================================================================
# MAIN
# =============================================================================

print(f"Querying GEE for NAIP acquisition dates...\n")
print(f"River:    {RIVER}")
print(f"GPKGs:    {GPKG_DIR}")
print(f"Output:   {OUTPUT_CSV}\n")

if not GPKG_DIR.exists():
    print(f"ERROR: GPKG directory does not exist: {GPKG_DIR}")
    raise SystemExit(1)

rows = []

gpkg_files = sorted(GPKG_DIR.glob("*.gpkg"))
print(f"Found {len(gpkg_files)} GeoPackages\n")

for gpkg in gpkg_files:
    section = gpkg.stem
    print(f"{'='*40}")
    print(f"Section: {section}")

    # Check if this section has any downloaded NAIP tiles
    naip_section_dir = NAIP_DIR / section
    if not naip_section_dir.exists():
        print(f"  No NAIP directory found, skipping")
        continue

    downloaded_years = [
        f.stem.replace("naip_", "")
        for f in sorted(naip_section_dir.glob("naip_*.tif"))
    ]

    if not downloaded_years:
        print(f"  No downloaded tiles found, skipping")
        continue

    print(f"  Downloaded years: {downloaded_years}")

    # Load AOI and query GEE
    try:
        aoi_ee = load_aoi(gpkg)
    except Exception as e:
        print(f"  ERROR loading AOI: {e}")
        continue

    for year in downloaded_years:
        try:
            dates = get_naip_dates_for_year(aoi_ee, int(year))
        except Exception as e:
            print(f"  {year}: GEE query error: {e}")
            dates = []

        date_str = ", ".join(dates) if dates else "not found"
        print(f"  {year}: {date_str}")

        rows.append({
            "section":           section,
            "year":              year,
            "acquisition_dates": date_str,
            "n_scenes":          len(dates),
        })

# Save to CSV
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["section", "year", "acquisition_dates", "n_scenes"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved -> {OUTPUT_CSV}")
print(f"Total entries: {len(rows)}")
print("All done!")
