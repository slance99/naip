import geopandas as gpd
from shapely.ops import substring
from shapely.geometry import LineString, MultiLineString
from pathlib import Path
import os
import re

# ============================================================
# UPDATED CONFIG
# ============================================================

# Folder where your river line shapefiles live
SHAPEFILE_DIR = "/home/geomorph/california_rivers/naip/shapefiles/lines/"

# Just list the river names — the script builds the path as
# {SHAPEFILE_DIR}/{river_name}_line.shp automatically
RIVER_NAMES = [
    "smith",
    "klamath",
    "trinity",
    "mad",
    "west",
    "mattole",
    "russian",
    "salinas",
    "santa_maria",
    "ventura",
    "santa_clara",
    "american",
    "sacramento",
    "feather",
    "yuba",
    "cosumnes",
    "mokelumne",
    "san_joaquin",
    "merced",
    "stanislaus",
    "tuolumne"
]

# Where the {river}_gpkgs folders will be created
OUTPUT_ROOT = "/home/geomorph/california_rivers/naip/gpkgs/all/"

# Where the reprojected centerlines are saved (kept separate from the
# segment folders so they don't get picked up by any *.gpkg globbing)
LINES_OUTPUT_DIR = "/home/geomorph/california_rivers/naip/gpkgs/lines_utm/"
SAVE_REPROJECTED_LINES = True

SEGMENT_LENGTH_M = 5000   # length of each slice, in meters
BUFFER_M = 1000           # buffer distance, in meters

# Projected CRS used for slicing/buffering, and the CRS of every output file.
# EPSG:26910 = NAD83 / UTM zone 10N. Confirm it matches your NAIP with:
#   gdalsrsinfo -o epsg some_naip_tile.tif
WORKING_CRS = "EPSG:26910"

# Optional per-river CRS overrides, e.g. if a river's NAIP is in UTM 11N:
#   CRS_OVERRIDES = {"ventura": "EPSG:26911", "santa_clara": "EPSG:26911"}
CRS_OVERRIDES = {}

# UTM zone 10 spans 126°W to 120°W
UTM10_EAST_EDGE_LON = -120.0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_name(name: str):
    return re.sub(r'[^A-Za-z0-9_-]+', '_', str(name)).strip('_')


def slice_line(line: LineString, segment_length: float):
    total_length = line.length
    pieces = []
    start = 0
    while start < total_length:
        end = min(start + segment_length, total_length)
        piece = substring(line, start, end)
        if piece.length > 0:
            pieces.append(piece)
        start += segment_length
    return pieces


def warn_if_outside_utm10(gdf, river_name):
    """Print a note if the river extends east of UTM zone 10."""
    min_lon, _, max_lon, _ = gdf.to_crs("EPSG:4326").total_bounds
    if max_lon > UTM10_EAST_EDGE_LON:
        print(f"  NOTE: {river_name} extends to {max_lon:.2f}° longitude, east of "
              f"UTM zone 10 ({UTM10_EAST_EDGE_LON}°). Check whether its NAIP is in "
              f"UTM 11N (EPSG:26911) and add a CRS_OVERRIDES entry if so.", flush=True)


def process_river(input_path, river_name, output_root,
                  segment_length_m, buffer_m, working_crs):
    print(f"Reading {input_path} ...", flush=True)
    gdf = gpd.read_file(input_path)
    print(f"Read {len(gdf)} feature(s). Original CRS: {gdf.crs}", flush=True)

    river_name = safe_name(river_name)

    if working_crs.upper() == "EPSG:26910":
        warn_if_outside_utm10(gdf, river_name)

    print(f"Reprojecting to {working_crs} ...", flush=True)
    gdf = gdf.to_crs(working_crs)

    if SAVE_REPROJECTED_LINES:
        os.makedirs(LINES_OUTPUT_DIR, exist_ok=True)
        epsg = gdf.crs.to_epsg()
        line_out = os.path.join(LINES_OUTPUT_DIR, f"{river_name}_line_{epsg}.gpkg")
        gdf.to_file(line_out, driver="GPKG")
        print(f"Saved reprojected centerline to {line_out}", flush=True)

    lines = []
    for geom in gdf.geometry.values:
        if isinstance(geom, MultiLineString):
            lines.extend(list(geom.geoms))
        elif isinstance(geom, LineString):
            lines.append(geom)

    total_km = sum(l.length for l in lines) / 1000
    print(f"Found {len(lines)} line part(s). Total length: {total_km:.2f} km", flush=True)

    river_folder = os.path.join(output_root, f"{river_name}_gpkgs")
    os.makedirs(river_folder, exist_ok=True)

    # Pre-build all pieces first so we know the total count up front
    all_pieces = []
    for line in lines:
        all_pieces.extend(slice_line(line, segment_length_m))

    print(f"Sliced into {len(all_pieces)} segment(s) of ~{segment_length_m}m each. Writing files...", flush=True)

    counter = 1
    for piece in all_pieces:
        # Flat caps: each buffer ends square at the cut point instead of
        # bulging 1 km past it, so neighboring segments meet edge to edge.
        # (Requires Shapely >= 2.0; on older Shapely use cap_style=2.)
        buffered = piece.buffer(buffer_m, cap_style="flat")
        out_gdf = gpd.GeoDataFrame(
            {"river": [river_name], "segment": [counter]},
            geometry=[buffered],
            crs=working_crs
        )
        out_path = os.path.join(river_folder, f"{river_name}_{counter}.gpkg")
        out_gdf.to_file(out_path, driver="GPKG")

        if counter % 10 == 0 or counter == len(all_pieces):
            print(f"  ...wrote {counter}/{len(all_pieces)}", flush=True)
        counter += 1

    print(f"Done. Wrote {counter - 1} segments to {river_folder}\n")


# ============================================================
# MAIN LOOP — process every river in RIVER_NAMES
# ============================================================

if __name__ == "__main__":
    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(RIVER_NAMES)} river(s)\n")

    for river_name in RIVER_NAMES:
        input_path = Path(SHAPEFILE_DIR) / f"{river_name}_line.shp"

        print(f"{'='*50}")
        print(f"River: {river_name}")

        if not input_path.exists():
            print(f"  WARNING: file not found, skipping: {input_path}\n")
            continue

        process_river(
            str(input_path),
            river_name,
            output_root=OUTPUT_ROOT,
            segment_length_m=SEGMENT_LENGTH_M,
            buffer_m=BUFFER_M,
            working_crs=CRS_OVERRIDES.get(river_name, WORKING_CRS)
        )

    print("All rivers processed!")