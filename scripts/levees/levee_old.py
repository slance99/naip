# =============================================================================
# levee_coverage.py
#
# Analyzes levee coverage for river segments using NAIP water mask GPKGs
# and a levee centerline shapefile.
#
# For each river section GPKG:
#   1. Splits the section polygon into left and right halves along its long axis
#   2. Clips levee centerlines to each half + 500m buffer
#   3. Measures leveed length per side
#   4. Calculates percent coverage and classifies as Both/One/Unleveed
#
# Outputs:
#   - CSV with levee coverage statistics per section
#   - GeoPackage with classifications joined to section geometries
#   - Diagnostic PNG per section showing the split and levee overlay
#
# Usage:
#   python levee_coverage.py <river> <levee_shapefile>
#   e.g. python levee_coverage.py sacramento /path/to/levees.shp
#
# Requirements:
#   conda install -c conda-forge geopandas shapely pandas fiona matplotlib
# =============================================================================

import argparse
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon
from shapely.ops import split, unary_union
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("river", help="River name e.g. sacramento")
args = parser.parse_args()

RIVER     = args.river

# =============================================================================
# CONFIG
# =============================================================================

GPKG_DIR   = Path(f"/home/geomorph/california_rivers/naip/gpkgs/all/{RIVER}_gpkgs/")
OUTPUT_DIR = Path(f"/home/geomorph/california_rivers/naip/outputs/levee_analysis/")
VIZ_DIR    = OUTPUT_DIR / "diagnostics"
LEVEE_SHP  = Path("/home/geomorph/california_rivers/naip/levees/california_levees/california_levees.shp")

# Working CRS — meters required for buffer/length calculations
WORKING_CRS = "EPSG:3310"

# Buffer distance for levee proximity
BUFFER_M = 500

# Threshold for classifying a side as "leveed"
LEVEE_THRESHOLD_PCT = 50

# Set to True to generate diagnostic plots for every section
SAVE_DIAGNOSTICS = True


# =============================================================================
# SETUP
# =============================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

print(f"River:      {RIVER}")
print(f"GPKGs:      {GPKG_DIR}")
print(f"Levees:     {LEVEE_SHP}")
print(f"Buffer:     {BUFFER_M}m")
print(f"Output:     {OUTPUT_DIR}\n")

if not GPKG_DIR.exists():
    print(f"ERROR: GPKG directory not found: {GPKG_DIR}")
    raise SystemExit(1)

if not LEVEE_SHP.exists():
    print(f"ERROR: Levee shapefile not found: {LEVEE_SHP}")
    raise SystemExit(1)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_long_axis(polygon):
    """
    Get the long axis of a polygon as a LineString by finding the longest
    edge of the minimum rotated rectangle and extending it well beyond
    the polygon bounds so it fully bisects the polygon when used as a splitter.
    """
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
        return LineString([extended_p1, extended_p2])

    return None


def split_polygon_into_halves(polygon):
    """
    Split a polygon into two halves along its long axis.
    Returns (half1, half2) or (polygon, None) if split fails.
    """
    axis = get_long_axis(polygon)
    if axis is None:
        return polygon, None

    try:
        parts = split(polygon, axis)
        geoms = list(parts.geoms)
        if len(geoms) >= 2:
            return geoms[0], geoms[1]
    except Exception:
        pass

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
    """Helper to draw a Polygon or MultiPolygon on an axis."""
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
                         levees_local, levees_buffered_local,
                         left_pct, right_pct, classification):
    """
    Save a diagnostic image showing:
      - Section polygon split into left (blue) and right (orange) halves
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

        # Draw full section outline in grey
        draw_polygon(ax, section_polygon, color="grey", alpha=0.1, linewidth=0.5)

        # Draw this half filled
        draw_polygon(ax, half, color=color, alpha=0.4)

        # Draw levee buffer zones (pink transparent)
        if not levees_buffered_local.empty:
            levees_buffered_local.plot(
                ax=ax, color="#FF69B4", alpha=0.2, zorder=2
            )

        # Draw levee centerlines (red)
        if not levees_local.empty:
            levees_local.plot(
                ax=ax, color="red", linewidth=1.5, zorder=3
            )

        # Legend
        legend_elements = [
            mpatches.Patch(facecolor=color, alpha=0.4,
                           label=f"{title} corridor"),
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
levees = gpd.read_file(LEVEE_SHP)
print(f"  Loaded {len(levees)} levee features, CRS: {levees.crs}")

levees = levees.to_crs(WORKING_CRS)
print(f"  Reprojected to {WORKING_CRS}")

print(f"  Buffering levees by {BUFFER_M}m...")
levees_buffered = levees.copy()
levees_buffered["geometry"] = levees.geometry.buffer(BUFFER_M)

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

    try:
        section_gdf = gpd.read_file(gpkg_path)
        section_gdf = section_gdf.to_crs(WORKING_CRS)
        section_polygon = section_gdf.geometry.union_all()
    except Exception as e:
        print(f"  ERROR loading {section_name}: {e}")
        continue

    # Estimate river length within section from long axis
    axis = get_long_axis(section_polygon)
    if axis:
        section_length_m = axis.length / 10
    else:
        section_length_m = section_polygon.length / 2

    # Split into left and right halves
    left_half, right_half = split_polygon_into_halves(section_polygon)

    if right_half is None:
        print(f"  WARNING: could not split {section_name}, using whole polygon for both sides")
        left_half  = section_polygon
        right_half = section_polygon

    # Clip levees to local section area for analysis and visualization
    section_bounds = section_gdf.buffer(BUFFER_M * 2).union_all()
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

    print(f"  Length: {section_length_m:.0f}m")
    print(f"  Left:   {left_leveed_m:.0f}m leveed ({left_pct:.1f}%)")
    print(f"  Right:  {right_leveed_m:.0f}m leveed ({right_pct:.1f}%)")
    print(f"  Class:  {classification}")

    # Save diagnostic plot
    if SAVE_DIAGNOSTICS:
        save_diagnostic_plot(
            section_name          = section_name,
            section_polygon       = section_polygon,
            left_half             = left_half,
            right_half            = right_half,
            levees_local          = levees_local,
            levees_buffered_local = levees_buffered_local,
            left_pct              = left_pct,
            right_pct             = right_pct,
            classification        = classification
        )

    results.append({
        "section":          section_name,
        "section_length_m": round(section_length_m, 1),
        "left_leveed_m":    round(left_leveed_m, 1),
        "right_leveed_m":   round(right_leveed_m, 1),
        "left_pct":         round(left_pct, 1),
        "right_pct":        round(right_pct, 1),
        "classification":   classification,
    })

    section_gdf["section"]          = section_name
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
csv_out = OUTPUT_DIR / f"{RIVER}_levee_coverage.csv"
results_df = pd.DataFrame(results)
results_df.to_csv(csv_out, index=False)
print(f"Saved CSV -> {csv_out}")

# Save GeoPackage
if section_gdfs:
    combined_gdf = pd.concat(section_gdfs, ignore_index=True)
    gpkg_out = OUTPUT_DIR / f"{RIVER}_levee_coverage.gpkg"
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
