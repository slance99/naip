# =============================================================================
# first_last_doubles.py
# For each river section, finds the first and last year water mask mosaic,
# overlays them on BOTH the first and last year NAIP imagery producing
# two output images per section — one original orientation and one rotated
# so the river flows left to right with upstream on the left.
# =============================================================================

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict
import rasterio
from rasterio.enums import Resampling as RasterioResampling
from rasterio.warp import reproject, Resampling as WarpResampling
from rasterio.warp import calculate_default_transform
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.crs import CRS
from rasterio.coords import BoundingBox
from rasterio.transform import array_bounds
from shapely.geometry import box as shapely_box
from scipy.ndimage import rotate as ndimage_rotate
import geopandas as gpd
from multiprocessing import Pool
import builtins


# =============================================================================
# SAFE PRINT
# =============================================================================

_original_print = builtins.print

def safe_print(*args, **kwargs):
    try:
        _original_print(*args, **kwargs, flush=True)
    except OSError:
        pass

builtins.print = safe_print


# =============================================================================
# CONFIG — your exact paths preserved
# =============================================================================
# Directory containing water mask mosaics (*_mosaic.tif)
MASK_DIR = Path("/home/geomorph/california_rivers/naip/outputs/gee/red_bluff_colusa_utm")

# Directory containing per-section NAIP tiles
# Structure: NAIP_DIR/{section_name}/naip_{year}.tif
NAIP_DIR = Path("/home/geomorph/california_rivers/naip/gee_naip/red_bluff_colusa_utm")

# Where to save output PNG images
OUTPUT_DIR = Path("/home/geomorph/california_rivers/naip/outputs/comparisons/red_bluff_colusa_rotate")

# River miles shapefile
RIVER_MILES_SHP = Path("/home/geomorph/california_rivers/naip/river_miles/river_mile_markers_sacriver_2012.shp")

CENTERLINE_SHP  = Path("/home/geomorph/california_rivers/naip/shapefiles/lines/sacramento_line.shp")


MARKER_COL       = "MARKER"
OVERLAY_ALPHA    = 0.5
DPI              = 150
MAX_PIXELS       = 20_000_000
N_WORKERS        = 4
TARGET_CRS       = "EPSG:26910"
FORCE_RERUN      = False
SEGMENT_LENGTH_M = 5000


# =============================================================================
# SETUP
# =============================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Masks from:    {MASK_DIR}")
print(f"NAIP from:     {NAIP_DIR}")
print(f"Output to:     {OUTPUT_DIR}")
print(f"River miles:   {RIVER_MILES_SHP}")
print(f"Centerline:    {CENTERLINE_SHP}")
print(f"Display CRS:   {TARGET_CRS}")
print(f"Workers:       {N_WORKERS}")
print(f"Force rerun:   {FORCE_RERUN}\n")

if not MASK_DIR.exists():
    print(f"ERROR: Mask directory does not exist: {MASK_DIR}")
    raise SystemExit(1)

if not NAIP_DIR.exists():
    print(f"ERROR: NAIP directory does not exist: {NAIP_DIR}")
    raise SystemExit(1)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_year_from_mask(path):
    match = re.search(r"_(\d{4})_mosaic", path.stem)
    return int(match.group(1)) if match else None


def parse_section_from_mask(path):
    return re.sub(r"_\d{4}_mosaic$", "", path.stem)


def parse_section_num(section_name):
    match = re.search(r"_(\d+)$", section_name)
    return int(match.group(1)) if match else 0


def get_centerline_angle(centerline_shp, section_num, segment_length_m,
                          working_crs=TARGET_CRS):
    """
    Get the angle of the river centerline slice for this section.
    Uses same slicing logic as gpkg_maker.py so sections align correctly.
    Returns angle in degrees measured from east (standard math convention).
    """
    try:
        from shapely.ops import substring
        from shapely.geometry import LineString, MultiLineString

        gdf   = gpd.read_file(centerline_shp).to_crs(working_crs)
        lines = []
        for geom in gdf.geometry:
            if geom.geom_type == "MultiLineString":
                lines.extend(list(geom.geoms))
            elif geom.geom_type == "LineString":
                lines.append(geom)

        all_pieces = []
        for line in lines:
            total = line.length
            start = 0
            while start < total:
                end   = min(start + segment_length_m, total)
                piece = substring(line, start, end)
                if piece.length > 0:
                    all_pieces.append(piece)
                start += segment_length_m

        idx = section_num - 1
        if not (0 <= idx < len(all_pieces)):
            return None

        piece  = all_pieces[idx]
        coords = list(piece.coords)
        p1, p2 = coords[0], coords[-1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = np.degrees(np.arctan2(dy, dx))
        return angle

    except Exception as e:
        print(f"    WARNING: could not get centerline angle: {e}")
        return None


def rotate_image_and_mask(rgb, overlay, angle_deg):
    """
    Rotate NAIP RGB and overlay arrays so river runs horizontally
    with upstream on the left. Uses scipy.ndimage.rotate with
    reshape=True so no content is clipped during rotation.
    """
    rot_angle = -angle_deg  # scipy rotates counterclockwise so negate

    rgb_rotated = ndimage_rotate(
        rgb, rot_angle, axes=(0, 1),
        reshape=True, mode='constant', cval=0.0
    )
    overlay_rotated = ndimage_rotate(
        overlay, rot_angle, axes=(0, 1),
        reshape=True, mode='constant', cval=0.0
    )
    return rgb_rotated, overlay_rotated


def load_naip_rgb(naip_path, target_crs=TARGET_CRS):
    """
    Load NAIP tile as RGB normalized 0-1.
    All tiles already in EPSG:26910 so no reprojection needed —
    loaded at full native resolution with no quality loss.
    """
    with rasterio.open(naip_path) as src:
        src_crs  = src.crs
        src_epsg = src_crs.to_epsg()
        tgt_epsg = int(target_crs.split(":")[1])
        needs_reproject = src_epsg != tgt_epsg

        if needs_reproject:
            print(f"    Reprojecting from EPSG:{src_epsg} to {target_crs}...")
            transform, width, height = calculate_default_transform(
                src_crs, target_crs, src.width, src.height, *src.bounds
            )
            if height * width > MAX_PIXELS:
                scale  = (MAX_PIXELS / (height * width)) ** 0.5
                height = max(1, int(height * scale))
                width  = max(1, int(width  * scale))
                transform, _, _ = calculate_default_transform(
                    src_crs, target_crs, width, height, *src.bounds
                )

            r = np.zeros((height, width), dtype=np.uint8)
            g = np.zeros((height, width), dtype=np.uint8)
            b = np.zeros((height, width), dtype=np.uint8)

            for band_idx, arr in enumerate([r, g, b], start=1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=arr,
                    src_transform=src.transform,
                    src_crs=src_crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=RasterioResampling.bilinear
                )

            left, bottom, right, top = array_bounds(height, width, transform)
            bounds    = BoundingBox(left, bottom, right, top)
            out_crs   = CRS.from_string(target_crs)
            out_shape = (height, width)

        else:
            h, w = src.height, src.width
            if h * w > MAX_PIXELS:
                scale = (MAX_PIXELS / (h * w)) ** 0.5
                h = max(1, int(h * scale))
                w = max(1, int(w  * scale))

            r = src.read(1, out_shape=(h, w),
                         resampling=RasterioResampling.bilinear)
            g = src.read(2, out_shape=(h, w),
                         resampling=RasterioResampling.bilinear)
            b = src.read(3, out_shape=(h, w),
                         resampling=RasterioResampling.bilinear)
            transform = src.transform
            bounds    = src.bounds
            out_crs   = src_crs
            out_shape = (h, w)

    rgb = np.stack([r, g, b], axis=-1).astype(float)
    rgb = np.clip(rgb / 255.0, 0, 1)
    return rgb, out_shape, transform, out_crs, bounds


def load_mask_aligned(mask_path, target_shape, naip_bounds, naip_crs):
    """Load water mask aligned to NAIP tile shape and extent."""
    dst_transform = transform_from_bounds(
        naip_bounds.left, naip_bounds.bottom,
        naip_bounds.right, naip_bounds.top,
        target_shape[1], target_shape[0]
    )
    with rasterio.open(mask_path) as src:
        aligned = np.zeros(target_shape, dtype=np.uint8)
        reproject(
            source=rasterio.band(src, 1),
            destination=aligned,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=naip_crs,
            resampling=WarpResampling.nearest,
        )
    return aligned > 0


def find_naip_for_year(naip_section_dir, year):
    candidate = naip_section_dir / f"naip_{year}.tif"
    return candidate if candidate.exists() else None


def make_overlay_rgba(first_mask, last_mask, alpha=OVERLAY_ALPHA):
    """Build RGBA overlay from two water masks."""
    h, w = first_mask.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    overlay[first_mask & ~last_mask] = [1.0, 0.5, 0.0, alpha]
    overlay[~first_mask & last_mask] = [1.0, 0.1, 0.7, alpha]
    overlay[first_mask & last_mask]  = [0.7, 0.7, 0.7, alpha]
    stats = {
        "lost":       int(np.sum(first_mask & ~last_mask)),
        "gained":     int(np.sum(~first_mask & last_mask)),
        "persistent": int(np.sum(first_mask & last_mask)),
    }
    return overlay, stats


def get_river_mile_points(river_miles_gdf, naip_crs, bounds, pad_frac=0.02):
    """
    Reproject and clip river mile markers to tile bounds.
    Returns list of (x, y, label) in naip_crs coordinates.
    Handles both Point and MultiPoint geometries.
    """
    if river_miles_gdf is None or river_miles_gdf.empty:
        return []

    rm_reprojected = river_miles_gdf.to_crs(naip_crs)
    pad_x    = (bounds.right  - bounds.left)   * pad_frac
    pad_y    = (bounds.top    - bounds.bottom)  * pad_frac
    tile_box = shapely_box(
        bounds.left   - pad_x, bounds.bottom - pad_y,
        bounds.right  + pad_x, bounds.top    + pad_y
    )
    rm_clipped = rm_reprojected[rm_reprojected.geometry.intersects(tile_box)]

    if rm_clipped.empty:
        return []

    def _as_point(geom):
        if geom.geom_type == "MultiPoint":
            pts = list(geom.geoms)
            return pts[0] if pts else None
        return geom

    result     = []
    marker_col = MARKER_COL if MARKER_COL in rm_clipped.columns else \
                 next((c for c in rm_clipped.columns
                       if c.lower() == MARKER_COL.lower()), None)

    for _, row in rm_clipped.iterrows():
        pt        = _as_point(row.geometry)
        label_val = row[marker_col] if marker_col else ""
        if pt is None:
            continue
        result.append((pt.x, pt.y, str(label_val)))

    return result


def save_overlay_image(naip_rgb, overlay, stats, bounds, naip_crs,
                       section_name, first_year, last_year,
                       bg_year, output_path,
                       display_bounds=None,
                       river_mile_points=None,
                       rotation_angle=None):
    """
    Compose and save overlay image.
    If rotation_angle provided, rotates so river flows left to right
    with upstream on left. Figure size adapts to true aspect ratio.
    """
    plot_bounds = display_bounds if display_bounds is not None else bounds

    # ── Optional rotation ─────────────────────────────────────────────────────
    if rotation_angle is not None:
        naip_rgb_plot, overlay_plot = rotate_image_and_mask(
            naip_rgb, overlay, rotation_angle
        )

        h_orig, w_orig = naip_rgb.shape[:2]
        h_rot,  w_rot  = naip_rgb_plot.shape[:2]
        geo_w = plot_bounds.right  - plot_bounds.left
        geo_h = plot_bounds.top    - plot_bounds.bottom

        cx_orig   = w_orig / 2
        cy_orig   = h_orig / 2
        cx_rot    = w_rot  / 2
        cy_rot    = h_rot  / 2
        angle_rad = np.radians(-rotation_angle)
        cos_a     = np.cos(angle_rad)
        sin_a     = np.sin(angle_rad)

        rotated_mile_points = []
        if river_mile_points:
            for x_geo, y_geo, label in river_mile_points:
                # Convert geographic coords to pixel coords
                px = (x_geo - plot_bounds.left) / geo_w * w_orig
                py = (plot_bounds.top - y_geo)   / geo_h * h_orig
                # Rotate around original image center
                dx = px - cx_orig
                dy = py - cy_orig
                px_r = cx_rot + cos_a * dx - sin_a * dy
                py_r = cy_rot + sin_a * dx + cos_a * dy
                rotated_mile_points.append((px_r, py_r, label))

        use_pixel_coords  = True
        mile_points_final = rotated_mile_points

    else:
        naip_rgb_plot    = naip_rgb
        overlay_plot     = overlay
        use_pixel_coords = False
        mile_points_final = river_mile_points
        h_rot, w_rot     = naip_rgb.shape[:2]

    # ── Figure size adapts to true aspect ratio — no stretching ──────────────
    if use_pixel_coords:
        aspect_ratio = h_rot / max(w_rot, 1)
    else:
        data_w       = plot_bounds.right - plot_bounds.left
        data_h       = plot_bounds.top   - plot_bounds.bottom
        aspect_ratio = data_h / max(data_w, 1)

    MAX_DIM = 16
    if aspect_ratio >= 1:
        fig_height = MAX_DIM
        fig_width  = MAX_DIM / aspect_ratio
    else:
        fig_width  = MAX_DIM
        fig_height = MAX_DIM * aspect_ratio
    fig_height += 2.0  # extra for title and legend

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor("black")
    fig.suptitle(
        f"{section_name.replace('_', ' ')}  —  {first_year} vs {last_year}"
        f"  (background: {bg_year})",
        fontsize=18, fontweight="bold", color="white"
    )
    ax.set_facecolor("black")

    # ── Draw imagery ──────────────────────────────────────────────────────────
    if use_pixel_coords:
        ax.imshow(naip_rgb_plot, origin="upper", aspect="equal",
                  interpolation="bilinear", zorder=1)
        ax.imshow(overlay_plot,  origin="upper", aspect="equal",
                  interpolation="nearest", zorder=2)
        ax.set_xlim(0, w_rot)
        ax.set_ylim(h_rot, 0)
        ax.set_xlabel("← Downstream       Upstream →",
                      color="white", fontsize=13, labelpad=8)
        ax.set_ylabel("", color="white")
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.imshow(naip_rgb_plot,
                  extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                  origin="upper", aspect="equal",
                  interpolation="bilinear", zorder=1)
        ax.imshow(overlay_plot,
                  extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                  origin="upper", aspect="equal",
                  interpolation="nearest", zorder=2)
        ax.set_xlim(plot_bounds.left,   plot_bounds.right)
        ax.set_ylim(plot_bounds.bottom, plot_bounds.top)
        ax.set_xlabel("Easting (m)",  color="white", fontsize=13, labelpad=8)
        ax.set_ylabel("Northing (m)", color="white", fontsize=13, labelpad=8)

    ax.tick_params(colors="white", labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("white")

    # ── River mile markers ────────────────────────────────────────────────────
    if mile_points_final:
        for x, y, label_val in mile_points_final:
            ax.plot(x, y, "o", color="white", markersize=10,
                    markeredgecolor="black", markeredgewidth=1.5, zorder=5)
            ax.annotate(
                label_val, (x, y),
                textcoords="offset points", xytext=(8, 4),
                fontsize=11, color="white", fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black",
                          alpha=0.5, edgecolor="none")
            )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor=[1.0, 0.5, 0.0], alpha=OVERLAY_ALPHA,
                       label=f"Water Present {first_year} ({stats['lost']:,} px)"),
        mpatches.Patch(facecolor=[1.0, 0.1, 0.7], alpha=OVERLAY_ALPHA,
                       label=f"New Water Present in {last_year} ({stats['gained']:,} px)"),
        mpatches.Patch(facecolor=[0.7, 0.7, 0.7], alpha=OVERLAY_ALPHA,
                       label=f"Persistent Water ({stats['persistent']:,} px)"),
        mpatches.Patch(facecolor=[0.0, 0.0, 0.0],
                       label="NAIP Background"),
        plt.Line2D([0], [0], color="white", linewidth=1,
                   marker="o", markersize=8, markerfacecolor="white",
                   label="River mile marker"),
    ]
    ax.legend(
        handles=legend_elements, loc="upper right",
        facecolor="black", edgecolor="white", labelcolor="white",
        framealpha=0.9, fontsize=11, markerscale=1.5
    )

    # ── North arrow ───────────────────────────────────────────────────────────
    # In UTM north is straight up. If rotated, north arrow rotates
    # by the negative of the image rotation angle.
    north_angle = 90.0
    if rotation_angle is not None:
        north_angle = 90.0 - rotation_angle

    arrow_x  = 0.05
    arrow_y0 = 0.06
    arr_len  = 0.07
    dx_arr   = arr_len * np.cos(np.radians(north_angle))
    dy_arr   = arr_len * np.sin(np.radians(north_angle))

    ax.annotate(
        "",
        xy=(arrow_x + dx_arr, arrow_y0 + dy_arr),
        xytext=(arrow_x, arrow_y0),
        xycoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>", color="white",
            lw=2.5, mutation_scale=22
        ),
        zorder=10
    )
    ax.text(
        arrow_x + dx_arr * 1.35,
        arrow_y0 + dy_arr * 1.35,
        "N",
        transform=ax.transAxes,
        color="white", fontsize=13, fontweight="bold",
        ha="center", va="center", zorder=10
    )

    plt.tight_layout()
    plt.savefig(output_path, facecolor="black", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {output_path.name}")


# =============================================================================
# PROCESS_SECTION
# =============================================================================

def process_section(args):
    """
    Process a single section. Produces four images per section:
      - Original orientation, first year background
      - Original orientation, last year background
      - Rotated (river left to right), first year background
      - Rotated (river left to right), last year background
    """
    (section_name, year_dict, naip_section_dir,
     out_first, out_last,
     out_first_rot, out_last_rot,
     first_year, last_year,
     river_miles_gdf, marker_col,
     centerline_shp, segment_length_m,
     force_rerun) = args

    messages = []

    try:
        naip_first_path = find_naip_for_year(naip_section_dir, first_year)
        naip_last_path  = find_naip_for_year(naip_section_dir, last_year)

        if naip_first_path is None:
            return f"  WARNING: no NAIP tile for {first_year} in {section_name}"
        if naip_last_path is None:
            return f"  WARNING: no NAIP tile for {last_year} in {section_name}"

        # Get rotation angle from centerline
        section_num    = parse_section_num(section_name)
        rotation_angle = get_centerline_angle(
            centerline_shp, section_num, segment_length_m
        )
        if rotation_angle is not None:
            print(f"  {section_name} centerline angle: {rotation_angle:.1f}°")
        else:
            print(f"  {section_name}: no centerline angle — skipping rotation")

        # Load first year — bounds are reference for consistent zoom
        naip_rgb_first, out_shape_first, _, naip_crs_first, bounds_first = \
            load_naip_rgb(naip_first_path)

        river_mile_points = get_river_mile_points(
            river_miles_gdf, naip_crs_first, bounds_first
        ) if river_miles_gdf is not None else []

        # Pre-compute masks for first year background
        first_mask_f = load_mask_aligned(
            year_dict[first_year], out_shape_first,
            bounds_first, naip_crs_first
        )
        last_mask_f = load_mask_aligned(
            year_dict[last_year], out_shape_first,
            bounds_first, naip_crs_first
        )
        overlay_f, stats_f = make_overlay_rgba(first_mask_f, last_mask_f)
        del first_mask_f, last_mask_f

        # ── First year background — original ──────────────────────────────
        if not out_first.exists() or force_rerun:
            save_overlay_image(
                naip_rgb_first, overlay_f, stats_f,
                bounds_first, naip_crs_first,
                section_name, first_year, last_year,
                bg_year=first_year, output_path=out_first,
                display_bounds=bounds_first,
                river_mile_points=river_mile_points,
                rotation_angle=None
            )
            messages.append(f"  Saved {out_first.name}")

        # ── First year background — rotated ───────────────────────────────
        if rotation_angle is not None and (not out_first_rot.exists() or force_rerun):
            save_overlay_image(
                naip_rgb_first, overlay_f, stats_f,
                bounds_first, naip_crs_first,
                section_name, first_year, last_year,
                bg_year=first_year, output_path=out_first_rot,
                display_bounds=bounds_first,
                river_mile_points=river_mile_points,
                rotation_angle=rotation_angle
            )
            messages.append(f"  Saved {out_first_rot.name}")

        del overlay_f

        # Load last year tile
        naip_rgb_last, out_shape_last, _, naip_crs_last, bounds_last = \
            load_naip_rgb(naip_last_path)

        first_mask_l = load_mask_aligned(
            year_dict[first_year], out_shape_last,
            bounds_last, naip_crs_last
        )
        last_mask_l = load_mask_aligned(
            year_dict[last_year], out_shape_last,
            bounds_last, naip_crs_last
        )
        overlay_l, stats_l = make_overlay_rgba(first_mask_l, last_mask_l)
        del first_mask_l, last_mask_l

        # ── Last year background — original ───────────────────────────────
        if not out_last.exists() or force_rerun:
            save_overlay_image(
                naip_rgb_last, overlay_l, stats_l,
                bounds_last, naip_crs_last,
                section_name, first_year, last_year,
                bg_year=last_year, output_path=out_last,
                display_bounds=bounds_first,
                river_mile_points=river_mile_points,
                rotation_angle=None
            )
            messages.append(f"  Saved {out_last.name}")

        # ── Last year background — rotated ────────────────────────────────
        if rotation_angle is not None and (not out_last_rot.exists() or force_rerun):
            save_overlay_image(
                naip_rgb_last, overlay_l, stats_l,
                bounds_last, naip_crs_last,
                section_name, first_year, last_year,
                bg_year=last_year, output_path=out_last_rot,
                display_bounds=bounds_first,
                river_mile_points=river_mile_points,
                rotation_angle=rotation_angle
            )
            messages.append(f"  Saved {out_last_rot.name}")

        del naip_rgb_first, naip_rgb_last, overlay_l

        if not messages:
            return f"  Skipped {section_name} — all outputs already exist"
        return "\n".join(messages)

    except Exception as e:
        import traceback
        return f"  ERROR on {section_name}: {e}\n{traceback.format_exc()}"


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # Load river miles
    river_miles_gdf = None
    if RIVER_MILES_SHP.exists():
        print(f"Loading river miles from {RIVER_MILES_SHP}...")
        river_miles_gdf = gpd.read_file(RIVER_MILES_SHP)
        if river_miles_gdf.geometry.geom_type.eq("MultiPoint").any():
            print("  Converting MultiPoint to Point geometries...")
            river_miles_gdf = river_miles_gdf.explode(index_parts=False)
            river_miles_gdf = river_miles_gdf.reset_index(drop=True)
        print(f"  Loaded {len(river_miles_gdf)} river mile markers")
        print(f"  CRS: {river_miles_gdf.crs}\n")
    else:
        print(f"WARNING: No river miles shapefile found at {RIVER_MILES_SHP}\n")

    # Find all mosaic files
    all_mosaics = sorted(MASK_DIR.glob("*_mosaic.tif"))
    print(f"Found {len(all_mosaics)} mosaic files\n")

    if not all_mosaics:
        print("ERROR: No mosaic files found.")
        raise SystemExit(1)

    sections = defaultdict(dict)
    for path in all_mosaics:
        year    = parse_year_from_mask(path)
        section = parse_section_from_mask(path)
        if year is not None and section:
            sections[section][year] = path

    print(f"Found {len(sections)} sections\n")

    section_args = []
    for section_name, year_dict in sorted(sections.items()):
        years = sorted(year_dict.keys())

        if len(years) < 2:
            print(f"  Skipping {section_name} — need at least 2 years")
            continue

        first_year = 2009
        last_year  = 2024

        out_first     = OUTPUT_DIR / f"{section_name}_bg{first_year}.png"
        out_last      = OUTPUT_DIR / f"{section_name}_bg{last_year}.png"
        out_first_rot = OUTPUT_DIR / f"{section_name}_bg{first_year}_rotated.png"
        out_last_rot  = OUTPUT_DIR / f"{section_name}_bg{last_year}_rotated.png"

        all_exist = all([
            out_first.exists(), out_last.exists(),
            out_first_rot.exists(), out_last_rot.exists()
        ])
        if all_exist and not FORCE_RERUN:
            print(f"  Skipping {section_name} — all outputs already exist")
            continue

        naip_section_dir = NAIP_DIR / section_name

        section_args.append((
            section_name, year_dict, naip_section_dir,
            out_first, out_last,
            out_first_rot, out_last_rot,
            first_year, last_year,
            river_miles_gdf, MARKER_COL,
            CENTERLINE_SHP, SEGMENT_LENGTH_M,
            FORCE_RERUN
        ))

    print(f"Processing {len(section_args)} sections with {N_WORKERS} workers\n")

    if section_args:
        with Pool(processes=N_WORKERS) as pool:
            results = pool.map(process_section, section_args)

        for result in results:
            if result:
                print(result)

    print("\nAll done!")
