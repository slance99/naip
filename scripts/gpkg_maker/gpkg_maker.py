import geopandas as gpd
from shapely.ops import substring
from shapely.geometry import LineString, MultiLineString
from pathlib import Path
import os
import re

# ============================================================
# CONFIG — edit these values for each run
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

SEGMENT_LENGTH_M = 5000   # length of each slice, in meters
BUFFER_M = 1000            # buffer distance, in meters
WORKING_CRS = "EPSG:3310" # projected CRS used for slicing/buffering math


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


def process_river(input_path, river_name, output_root=".",
                   segment_length_m=3000, buffer_m=5000,
                   working_crs="EPSG:3310"):
    print(f"Reading {input_path} ...", flush=True)
    gdf = gpd.read_file(input_path)
    print(f"Read {len(gdf)} feature(s). Original CRS: {gdf.crs}", flush=True)

    print(f"Reprojecting to {working_crs} ...", flush=True)
    gdf = gdf.to_crs(working_crs)

    geoms = list(gdf.geometry.values)
    lines = []
    for geom in geoms:
        if isinstance(geom, MultiLineString):
            lines.extend(list(geom.geoms))
        elif isinstance(geom, LineString):
            lines.append(geom)

    total_km = sum(l.length for l in lines) / 1000
    print(f"Found {len(lines)} line part(s). Total length: {total_km:.2f} km", flush=True)

    river_name = safe_name(river_name)
    river_folder = os.path.join(output_root, f"{river_name}_gpkgs")
    os.makedirs(river_folder, exist_ok=True)

    # Pre-build all pieces first so we know the total count up front
    all_pieces = []
    for line in lines:
        all_pieces.extend(slice_line(line, segment_length_m))

    print(f"Sliced into {len(all_pieces)} segment(s) of ~{segment_length_m}m each. Writing files...", flush=True)

    counter = 1
    for piece in all_pieces:
        buffered = piece.buffer(buffer_m)
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
            working_crs=WORKING_CRS
        )

    print("All rivers processed!")
