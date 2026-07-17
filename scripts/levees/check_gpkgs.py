# =============================================================================
# check_gpkg.py
# Quick diagnostic to check geometry issues in a single GPKG file
#
# Usage:
#   python check_gpkg.py /path/to/sacramento_33.gpkg
# =============================================================================

import sys
import geopandas as gpd
from shapely.ops import split, transform
from shapely.geometry import LineString

if len(sys.argv) < 2:
    print("Usage: python check_gpkg.py /path/to/section.gpkg")
    sys.exit(1)

gpkg_path = sys.argv[1]

print(f"Checking: {gpkg_path}\n")

gpkg = gpd.read_file(gpkg_path)
gpkg = gpkg.to_crs("EPSG:3310")
polygon = gpkg.geometry.union_all()

print(f"Geometry type: {polygon.geom_type}")
print(f"Has Z coords:  {polygon.has_z}")
print(f"Is valid:      {polygon.is_valid}")
print(f"Is empty:      {polygon.is_empty}")
print(f"Bounds:        {polygon.bounds}")
print(f"Area (m²):     {polygon.area:.0f}")
print(f"Perimeter (m): {polygon.length:.0f}")

# Try dropping Z if present
if polygon.has_z:
    print("\nDropping Z coordinates...")
    polygon = transform(lambda x, y, z=None: (x, y), polygon)
    print(f"Is valid after Z drop: {polygon.is_valid}")

# Try fixing invalid geometry
if not polygon.is_valid:
    print("\nFixing invalid geometry with buffer(0)...")
    polygon = polygon.buffer(0)
    print(f"Is valid after fix: {polygon.is_valid}")

# Try splitting
print("\nAttempting split...")
rect = polygon.minimum_rotated_rectangle
coords = list(rect.exterior.coords)[:-1]

max_len = 0
long_axis = None
for i in range(len(coords)):
    p1 = coords[i]
    p2 = coords[(i + 1) % len(coords)]
    length = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2) ** 0.5
    if length > max_len:
        max_len = length
        long_axis = (p1, p2)

if long_axis:
    p1, p2 = long_axis
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    scale = 10
    extended_p1 = (p1[0] - dx * scale, p1[1] - dy * scale)
    extended_p2 = (p2[0] + dx * scale, p2[1] + dy * scale)
    axis = LineString([extended_p1, extended_p2])

    try:
        parts = split(polygon, axis)
        geoms = list(parts.geoms)
        print(f"Split succeeded — {len(geoms)} parts produced")
        for i, g in enumerate(geoms):
            print(f"  Part {i+1}: {g.geom_type}, area={g.area:.0f} m²")
    except Exception as e:
        print(f"Split failed: {e}")
else:
    print("Could not find long axis")
