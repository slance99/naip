# =============================================================================
# levee_coverage.py
#
# Analyzes levee coverage for river segments using NAIP water mask GPKGs
# and a levee centerline shapefile.
#
# Uses the same centerline slicing logic as gpkg_maker.py so left/right
# splits align perfectly with the GPKG segment boundaries.
#
# For each river section GPKG:
#   1. Gets the matching centerline slice for this section
#   2. Splits the section polygon into left and right halves along the centerline
#   3. Clips levee centerlines to each half + buffer distance
#   4. Measures leveed length per side
#   5. Calculates percent coverage and classifies as Both/One/Unleveed
#
# Outputs:
#   - CSV with levee coverage statistics per section
#   - GeoPackage with classifications joined to section geometries
#   - Diagnostic PNG per section showing the split and levee overlay
#
# Usage:
#   python levee_coverage.py <river>
#   e.g. python levee_coverage.py sacramento
#
# Requirements:
#   conda install -c conda-forge geopandas shapely pandas fiona matplotlib
# =============================================================================

import argparse
import re
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from shapely.ops import nearest_points, split, unary_union, substring, transform, polygonize
from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon, Point


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

GPKG_DIR        = Path(f"/home/geomorph/california_rivers/naip/gpkgs/all/{RIVER}_gpkgs/")
OUTPUT_DIR      = Path(f"/home/geomorph/california_rivers/naip/outputs/levee_analysis/")
VIZ_DIR         = OUTPUT_DIR / "diagnostics"

# Path to levee centerline shapefile
LEVEE_SHP       = Path("/home/geomorph/california_rivers/naip/levees/california_levees/california_levees.shp")

# Path to river centerline shapefile — same one used in gpkg_maker.py
CENTERLINE_SHP  = Path(f"/home/geomorph/california_rivers/naip/shapefiles/lines/{RIVER}_line.shp")

# Segment length — must match what was used in gpkg_maker.py
SEGMENT_LENGTH_M = 5000

# Working CRS — meters required for buffer/length calculations
WORKING_CRS = "EPSG:3310"

# Buffer distance for levee proximity
BUFFER_M = 100

# Threshold for classifying a side as leveed (percent)
LEVEE_THRESHOLD_PCT = 50

# Set to True to generate diagnostic plots for every section
SAVE_DIAGNOSTICS = True


# =============================================================================
# SETUP
# =============================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

print(f"River:       {RIVER}")
print(f"GPKGs:       {GPKG_DIR}")
print(f"Levees:      {LEVEE_SHP}")
print(f"Centerline:  {CENTERLINE_SHP}")
print(f"Segment len: {SEGMENT_LENGTH_M}m")
print(f"Buffer:      {BUFFER_M}m")
print(f"Output:      {OUTPUT_DIR}\n")

for path, label in [
    (GPKG_DIR,       "GPKG directory"),
    (LEVEE_SHP,      "Levee shapefile"),
    (CENTERLINE_SHP, "Centerline shapefile"),
]:
    if not Path(path).exists():
        print(f"ERROR: {label} not found: {path}")
        raise SystemExit(1)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def slice_line(line, segment_length):
    """
    Slice a LineString into pieces of segment_length.
    Identical to gpkg_maker.py so section numbering matches exactly.
    """
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


def build_all_centerline_pieces(centerline_gdf, segment_length_m):
    """
    Build all centerline slices from the river centerline in the same
    order as gpkg_maker.py so indices match section numbers.
    """
    lines = []
    for geom in centerline_gdf.geometry:
        if isinstance(geom, MultiLineString):
            lines.extend(list(geom.geoms))
        elif isinstance(geom, LineString):
            lines.append(geom)

    all_pieces = []
    for line in lines:
        all_pieces.extend(slice_line(line, segment_length_m))

    return all_pieces


def get_section_centerline(all_pieces, section_index):
    """
    Return the centerline slice for a given 1-based section index.
    Returns None if index is out of range.
    """
    idx = section_index - 1
    if 0 <= idx < len(all_pieces):
        return all_pieces[idx]
    return None

def split_polygon_with_centerline(polygon, centerline_piece):
    """
    Split corridor polygon into left and right halves using nearest-point
    assignment — each part of the polygon is assigned to left or right
    based on which side of the centerline it's closest to.

    Approach:
    1. Densify the centerline into points every ~10m so we have good
       local direction information at every location along the river
    2. Create a grid of test points covering the corridor polygon
    3. For each grid point inside the corridor, use a cross product
       against the nearest centerline segment to assign it left or right
    4. Buffer each grid point into a small circle so adjacent points
       merge together when unioned
    5. Smooth the jagged circle-boundary edges using morphological
       closing (buffer out then back in) and simplify
    6. Use the centerline itself as a cookie-cutter to give a clean
       internal edge right along the river centerline
    """
    if centerline_piece is None:
        return polygon, None

    try:
        minx, miny, maxx, maxy = polygon.bounds

        # ── Step 1: Densify centerline ────────────────────────────────────
        # Place a point every ~10m along the centerline so we can look up
        # the local river direction at any location
        total_length = centerline_piece.length
        n_points     = max(100, int(total_length / 10))

        centerline_points = [
            centerline_piece.interpolate(i / n_points, normalized=True)
            for i in range(n_points + 1)
        ]

        # ── Step 2: Side-detection function ──────────────────────────────
        def get_side(test_point):
            """
            Determine whether a point is to the left or right of the
            centerline by finding the nearest centerline segment and
            computing the cross product of the local direction vector
            against the vector to the test point.
            Negative cross product = left (after sign flip for desired orientation)
            Positive cross product = right
            """
            # Find the nearest densified centerline point
            min_dist    = float('inf')
            nearest_idx = 0
            for i, cp in enumerate(centerline_points[:-1]):
                d = test_point.distance(cp)
                if d < min_dist:
                    min_dist    = d
                    nearest_idx = i

            # Local direction vector at that centerline segment
            p1 = centerline_points[nearest_idx]
            p2 = centerline_points[min(nearest_idx + 1, len(centerline_points) - 1)]
            dx = p2.x - p1.x
            dy = p2.y - p1.y

            # Cross product: positive = left of direction, negative = right
            # Sign is flipped here to match desired left/right orientation
            cross = dx * (test_point.y - p1.y) - dy * (test_point.x - p1.x)
            return 'left' if cross < 0 else 'right'

        # ── Step 3: Create grid of test points ───────────────────────────
        # Larger grid spacing = faster but coarser boundary
        # Smaller grid spacing = slower but smoother boundary
        # Clamped between 100m and 500m based on corridor width
        grid_spacing = max(100, min(500, (maxx - minx) / 30))

        xs = list(np.arange(minx, maxx, grid_spacing))
        ys = list(np.arange(miny, maxy, grid_spacing))

        left_points  = []
        right_points = []

        # ── Step 4: Assign each grid point to left or right ──────────────
        for x in xs:
            for y in ys:
                pt = Point(x, y)
                if polygon.contains(pt):
                    side = get_side(pt)
                    # Buffer each point into a circle so adjacent points
                    # overlap and merge when unioned — 75% of grid spacing
                    # ensures full coverage with no gaps
                    circle = pt.buffer(grid_spacing * 0.75)
                    if side == 'left':
                        left_points.append(circle)
                    else:
                        right_points.append(circle)

        if not left_points or not right_points:
            print(f"    WARNING: all points assigned to one side")
            return polygon, None

        # ── Step 5: Union and smooth ──────────────────────────────────────
        # Merge all circles on each side into one blob
        left_half  = polygon.intersection(unary_union(left_points))
        right_half = polygon.intersection(unary_union(right_points))

        # Morphological closing: buffer out by 2x grid spacing then back in
        # This fills the gaps between circles and removes the bumpy boundary
        smooth_dist = grid_spacing * 2.0

        left_half  = (left_half
                      .buffer(smooth_dist)
                      .buffer(-smooth_dist)
                      .simplify(grid_spacing, preserve_topology=True))
        right_half = (right_half
                      .buffer(smooth_dist)
                      .buffer(-smooth_dist)
                      .simplify(grid_spacing, preserve_topology=True))

        # Re-clip to corridor bounds after smoothing in case we expanded outside
        left_half  = polygon.intersection(left_half)
        right_half = polygon.intersection(right_half)

        # ── Step 6: Clean internal edge with centerline cookie-cutter ────
        # Remove a 1m strip along the centerline from both halves so the
        # internal boundary is clean and sharp rather than a bumpy circle edge
        centerline_buffer = centerline_piece.buffer(1.0)
        left_half         = left_half.difference(centerline_buffer)
        right_half        = right_half.difference(centerline_buffer)

        if left_half.is_empty or right_half.is_empty:
            print(f"    WARNING: one half is empty after split")
            return polygon, None

        return left_half, right_half

    except Exception as e:
        print(f"    Split error: {e}")
        return polygon, None

def measure_levee_length_in_area(levees_gdf, area_polygon):
    """
    Measure total length of levee centerlines within a given polygon area.
    Clips levees to the area and sums their lengths in meters.
    """
    if area_polygon is None or area_polygon.is_empty:
        return 0.0

    area_gdf = gpd.GeoDataFrame(geometry=[area_polygon], crs=WORKING_CRS)

    try:
        clipped = gpd.clip(levees_gdf, area_gdf)
        if clipped.empty:
            return 0.0
        return float(clipped.geometry.length.sum())
    except Exception:
        return 0.0


def classify_levee(left_pct, right_pct, threshold=LEVEE_THRESHOLD_PCT):
    """
    Classify a segment based on percent levee coverage on each side.
      Both_Sides: both sides exceed threshold
      One_Side:   only one side exceeds threshold
      Unleveed:   neither side exceeds threshold
    """
    left_leveed  = left_pct  >= threshold
    right_leveed = right_pct >= threshold

    if left_leveed and right_leveed:
        return "Both_Sides"
    elif left_leveed or right_leveed:
        return "One_Side"
    else:
        return "Unleveed"


def draw_polygon(ax, polygon, color, alpha, linewidth=1.5):
    """Helper to draw a Polygon or MultiPolygon on a matplotlib axis."""
    if polygon is None or polygon.is_empty:
        return
    if polygon.geom_type == "Polygon":
        x, y = polygon.exterior.xy
        ax.fill(x, y, alpha=alpha, color=color)
        ax.plot(x, y, color=color, linewidth=linewidth)
    elif polygon.geom_type == "MultiPolygon":
        for geom in polygon.geoms:
            x, y = geom.exterior.xy
            ax.fill(x, y, alpha=alpha, color=color)
            ax.plot(x, y, color=color, linewidth=linewidth)


def save_diagnostic_plot(section_name, section_polygon, left_half, right_half,
                         centerline_piece, levees_local, levees_buffered_local,
                         left_pct, right_pct, classification):
    """
    Save a diagnostic PNG showing:
      - Section polygon split into left (blue) and right (orange) halves
      - River centerline slice (green) used as the splitting axis
      - Levee centerlines (red)
      - Levee buffer zones (pink, transparent)
      - Classification result and percentages as title
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(
        f"{section_name}  —  {classification}\n"
        f"Left: {left_pct:.1f}%  |  Right: {right_pct:.1f}%",
        fontsize=14, fontweight="bold"
    )

    for ax, (title, half, color) in zip(axes, [
        ("Left Half",  left_half,  "#4A90D9"),
        ("Right Half", right_half, "#E8A838")
    ]):
        ax.set_title(title, fontsize=12)
        ax.set_aspect("equal")
        ax.set_facecolor("#f5f5f5")

        # Full section outline in grey
        draw_polygon(ax, section_polygon, color="grey", alpha=0.1, linewidth=0.5)

        # This half filled
        draw_polygon(ax, half, color=color, alpha=0.4)

        # River centerline slice in green
        if centerline_piece is not None:
            cx, cy = centerline_piece.xy
            ax.plot(cx, cy, color="green", linewidth=2.5,
                    zorder=4, label="River centerline")

        # Levee buffer zones (pink transparent)
        if not levees_buffered_local.empty:
            levees_buffered_local.plot(
                ax=ax, color="#FF69B4", alpha=0.2, zorder=2
            )

        # Levee centerlines (red)
        if not levees_local.empty:
            levees_local.plot(
                ax=ax, color="red", linewidth=1.5, zorder=3
            )

        legend_elements = [
            mpatches.Patch(facecolor=color, alpha=0.4,
                           label=f"{title} corridor"),
            plt.Line2D([0], [0], color="green", linewidth=2.5,
                       label="River centerline"),
            mpatches.Patch(facecolor="#FF69B4", alpha=0.3,
                           label=f"Levee buffer ({BUFFER_M}m)"),
            plt.Line2D([0], [0], color="red", linewidth=2,
                       label="Levee centerline"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("Easting (m)", fontsize=8)
        ax.set_ylabel("Northing (m)", fontsize=8)

    plt.tight_layout()
    out_path = VIZ_DIR / f"{section_name}_diagnostic.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Diagnostic -> {out_path.name}")


# =============================================================================
# LOAD DATA
# =============================================================================

print("Loading levee data...")
levees = gpd.read_file(LEVEE_SHP).to_crs(WORKING_CRS)
print(f"  Loaded {len(levees)} levee features")

print(f"  Buffering levees by {BUFFER_M}m...")
levees_buffered = levees.copy()
levees_buffered["geometry"] = levees.geometry.buffer(BUFFER_M)

print("\nLoading river centerline...")
centerline_gdf = gpd.read_file(CENTERLINE_SHP).to_crs(WORKING_CRS)

# Drop Z coordinates if present — shapely split requires 2D geometry
if centerline_gdf.geometry.has_z.any():
    print("  Dropping Z coordinates from centerline...")
    centerline_gdf["geometry"] = centerline_gdf.geometry.apply(
        lambda g: transform(lambda x, y, z=None: (x, y), g)
    )

print(f"  Loaded centerline, building {SEGMENT_LENGTH_M}m slices...")
all_centerline_pieces = build_all_centerline_pieces(centerline_gdf, SEGMENT_LENGTH_M)
print(f"  Built {len(all_centerline_pieces)} centerline slices")

print(f"\nLoading section GPKGs from {GPKG_DIR}...")
gpkg_files = sorted(GPKG_DIR.glob("*.gpkg"))
print(f"  Found {len(gpkg_files)} sections\n")


# =============================================================================
# MAIN ANALYSIS LOOP
# =============================================================================

results = []
section_gdfs = []

for gpkg_path in gpkg_files:
    section_name = gpkg_path.stem
    print(f"Processing: {section_name}")

    # Parse section number from name e.g. sacramento_33 -> 33
    match = re.search(r"_(\d+)$", section_name)
    if not match:
        print(f"  WARNING: could not parse section number from {section_name}, skipping")
        continue
    section_num = int(match.group(1))

    try:
        section_gdf     = gpd.read_file(gpkg_path)
        section_gdf     = section_gdf.to_crs(WORKING_CRS)
        section_polygon = section_gdf.geometry.union_all()
    except Exception as e:
        print(f"  ERROR loading {section_name}: {e}")
        continue

    # Drop Z if present
    if section_polygon.has_z:
        section_polygon = transform(lambda x, y, z=None: (x, y), section_polygon)

    # Fix invalid geometry
    if not section_polygon.is_valid:
        section_polygon = section_polygon.buffer(0)

    # Get matching centerline slice for this section
    centerline_piece = get_section_centerline(all_centerline_pieces, section_num)
    if centerline_piece is None:
        print(f"  WARNING: no centerline slice found for section {section_num}")

    # Use centerline length as the reference river length for this section
    section_length_m = centerline_piece.length if centerline_piece else section_polygon.length / 2

    # Split polygon into left and right halves along centerline
    left_half, right_half = split_polygon_with_centerline(section_polygon, centerline_piece)

    if right_half is None:
        print(f"  WARNING: split failed, using whole polygon for both sides")
        left_half  = section_polygon
        right_half = section_polygon

    # Clip levees to local section area for analysis and visualization
    section_bounds     = section_gdf.buffer(BUFFER_M * 2).union_all()
    section_bounds_gdf = gpd.GeoDataFrame(geometry=[section_bounds], crs=WORKING_CRS)

    try:
        levees_local          = gpd.clip(levees,          section_bounds_gdf)
        levees_buffered_local = gpd.clip(levees_buffered, section_bounds_gdf)
    except Exception:
        levees_local          = levees.iloc[0:0]
        levees_buffered_local = levees_buffered.iloc[0:0]

    # Measure leveed length on each side
    left_leveed_m  = measure_levee_length_in_area(levees_buffered, left_half)
    right_leveed_m = measure_levee_length_in_area(levees_buffered, right_half)

    # Calculate percent coverage
    left_pct  = min((left_leveed_m  / section_length_m * 100) if section_length_m > 0 else 0, 100.0)
    right_pct = min((right_leveed_m / section_length_m * 100) if section_length_m > 0 else 0, 100.0)

    # Classify
    classification = classify_levee(left_pct, right_pct)

    print(f"  Section #:  {section_num}")
    print(f"  Length:     {section_length_m:.0f}m")
    print(f"  Left:       {left_leveed_m:.0f}m leveed ({left_pct:.1f}%)")
    print(f"  Right:      {right_leveed_m:.0f}m leveed ({right_pct:.1f}%)")
    print(f"  Class:      {classification}")

    # Save diagnostic plot
    if SAVE_DIAGNOSTICS:
        save_diagnostic_plot(
            section_name          = section_name,
            section_polygon       = section_polygon,
            left_half             = left_half,
            right_half            = right_half,
            centerline_piece      = centerline_piece,
            levees_local          = levees_local,
            levees_buffered_local = levees_buffered_local,
            left_pct              = left_pct,
            right_pct             = right_pct,
            classification        = classification
        )

    results.append({
        "section":          section_name,
        "section_num":      section_num,
        "section_length_m": round(section_length_m, 1),
        "left_leveed_m":    round(left_leveed_m, 1),
        "right_leveed_m":   round(right_leveed_m, 1),
        "left_pct":         round(left_pct, 1),
        "right_pct":        round(right_pct, 1),
        "classification":   classification,
    })

    section_gdf["section"]          = section_name
    section_gdf["section_num"]      = section_num
    section_gdf["section_length_m"] = round(section_length_m, 1)
    section_gdf["left_leveed_m"]    = round(left_leveed_m, 1)
    section_gdf["right_leveed_m"]   = round(right_leveed_m, 1)
    section_gdf["left_pct"]         = round(left_pct, 1)
    section_gdf["right_pct"]        = round(right_pct, 1)
    section_gdf["classification"]   = classification
    section_gdfs.append(section_gdf)

    print()


# =============================================================================
# OUTPUTS
# =============================================================================

# Save CSV
csv_out    = OUTPUT_DIR / f"{RIVER}_levee_coverage.csv"
results_df = pd.DataFrame(results)
results_df.to_csv(csv_out, index=False)
print(f"Saved CSV -> {csv_out}")

# Save GeoPackage
if section_gdfs:
    combined_gdf = pd.concat(section_gdfs, ignore_index=True)
    gpkg_out     = OUTPUT_DIR / f"{RIVER}_levee_coverage.gpkg"
    combined_gdf.to_file(gpkg_out, driver="GPKG")
    print(f"Saved GeoPackage -> {gpkg_out}")

# Print summary
print(f"\n{'='*50}")
print(f"SUMMARY — {RIVER.upper()}")
print(f"{'='*50}")
print(f"Total sections: {len(results_df)}")
print(f"\nClassification breakdown:")
for cls, count in results_df["classification"].value_counts().items():
    pct = count / len(results_df) * 100
    print(f"  {cls}: {count} sections ({pct:.1f}%)")

print(f"\nAverage levee coverage:")
print(f"  Left side:  {results_df['left_pct'].mean():.1f}%")
print(f"  Right side: {results_df['right_pct'].mean():.1f}%")

print("\nAll done!")
