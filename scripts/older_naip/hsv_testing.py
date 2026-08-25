"""
================================================================================
NAIP RGB -> HSV WATER / NON-WATER SAMPLING AND HUE-VS-VALUE SCATTER PLOT
================================================================================

WORKFLOW OVERVIEW
------------------
    1. Load a 4-band NAIP GeoTIFF (Red, Green, Blue, NIR)
       -> Only the RGB bands are used; NIR is read but intentionally ignored.
    2. Load your existing binary water mask GeoTIFF (1 = water, 0 = not water)
       -> Reproject/align it onto the NAIP pixel grid if needed.
    3. Convert the RGB image to HSV (Hue, Saturation, Value) color space.
    4. Randomly sample a balanced set of water and non-water pixels,
       using the mask to label each sampled pixel.
    5. Plot Hue (x-axis) vs. Value (y-axis) for the sampled pixels,
       color-coded blue (water) / orange (not water).

================================================================================
"""
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt


# ==============================================================================
# SETUP
# ==============================================================================

#Establish NAIP Path
NAIP_PATH = "/home/geomorph/california_rivers/naip/gee_naip/sacramento/sacramento_30/naip_2020.tif"

#Path to Specific Mask
MASK_PATH = "/home/geomorph/california_rivers/naip/outputs/gee/red_bluff_colusa_outputs/sacramento_30_2020_mosaic.tif"

#Number of Samples Taken (Pixels Per Individual Type)
N_SAMPLES = 5000

#Seed for Random Number Generator
RANDOM_SEED = 42

#Where Output Plot Should Go
OUTPUT_PLOT = "/home/geomorph/california_rivers/naip/older_naip/hue_vs_value_scatter.png"

OUTPUT_LOCATIONS_PLOT = "/home/geomorph/california_rivers/naip/older_naip/sample_locations_overlay.png"

# ==============================================================================
# LOAD NAIP RGB BANDS
# ==============================================================================

#Load the RGB Bands from 4 Band NAIP Tif
#REMINDER: this is a test to see if this will work on 3-band older naip images so have to filer out NIR
def load_naip_rgb(naip_path):

    with rasterio.open(naip_path) as src:
        # --- Read only bands 1, 2, 3 (R, G, B) -----------------------------
        rgb = src.read([1, 2, 3]).astype(np.float32)
    valid_mask = src.dataset_mask() > 0

        # --- Capture georeferencing info for later mask alignment ---------
        transform = src.transform #pulls the geotransform
        crs = src.crs #pulls the coordinate reference system
        shape = src.shape #pulls the pixel dimensions of the raster 

    # --- Normalize pixel values to the 0-1 range ---------------------------
    # NAIP imagery can be delivered as 8-bit (0-255) or occasionally 16-bit.
    # Dividing by the array's own max (rather than a hardcoded 255) makes
    # this work correctly regardless of bit depth.
    rgb /= rgb.max()

    # --- Reorder axes from (band, row, col) to (row, col, band) -----------
    # skimage's rgb2hsv() and standard image conventions expect the color
    # channel to be the LAST axis, not the first.
    rgb_img = np.transpose(rgb, (1, 2, 0))

    return rgb_img, transform, crs, shape, valid_mask


# ==============================================================================
# LOAD AND ALIGN THE WATER MASK
# ==============================================================================
#Loading in the Mask

def load_aligned_mask(mask_path, ref_transform, ref_crs, ref_shape):
    with rasterio.open(mask_path) as mask_src:

        #Checks to See if Matches Already
        needs_alignment = (
            mask_src.crs != ref_crs
            or mask_src.transform != ref_transform
            or mask_src.shape != ref_shape
        )
    
    #If Is Not Aligned, Do This 
        if needs_alignment:
            print("Mask grid does not match NAIP grid - reprojecting mask onto NAIP grid.")

            # Preallocate the output array at the NAIP image's shape.
            mask = np.zeros(ref_shape, dtype=np.uint8)

            # --- Reproject the mask onto the NAIP grid ---------------------
            # resampling=Resampling.nearest is IMPORTANT here: this mask is
            # binary (0/1). Any other resampling method (bilinear, cubic,
            # etc.) would interpolate BETWEEN 0 and 1 at mask edges,
            # producing meaningless fractional "mask" values like 0.4.
            # Nearest-neighbor preserves clean 0/1 values everywhere.
            reproject(
                source=rasterio.band(mask_src, 1),
                destination=mask,
                src_transform=mask_src.transform,
                src_crs=mask_src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.nearest,
            )
        else:
            # Grids already match -- just read the mask band directly.
            mask = mask_src.read(1)

    return mask


# ==============================================================================
# SAMPLE HSV PIXELS BY CLASS (WATER / NOT WATER)
# ==============================================================================
#Randomly Sample Water and Non-Water Pixels from the HSV Image
#Classifications of water and not water are assigned based on the input watermask 
def sample_hsv_by_class(hsv_img, mask, n_samples, seed=42):
    """
    Returns
    -------
    water_sample : np.ndarray, shape (n_water, 3)
        Sampled HSV pixel values for pixels labeled "water" by the mask.
    nonwater_sample : np.ndarray, shape (n_nonwater, 3)
        Sampled HSV pixel values for pixels labeled "not water" by the mask.
    """
    #Establish the Random Number Generator 
    rng = np.random.default_rng(seed)

    # --- Combine the class mask with the valid-data mask --------------------
    # A pixel only counts as "water" or "not water" if it ALSO has real
    # image data. This is what keeps black mosaic-seam borders out of the
    # non-water sample pool.
    is_water = (mask == 1) & valid_mask
    is_nonwater = (mask == 0) & valid_mask
 
    # --- Get (row, col) coordinates of every water / non-water pixel -------
    # np.argwhere returns an (N, 2) array of [row, col] pairs for every
    # True entry in the boolean array. This is what lets us trace each
    # sampled HSV value back to a location in the original image.
    water_coords_all = np.argwhere(is_water)
    nonwater_coords_all = np.argwhere(is_nonwater)
 
    # --- Boolean-index the HSV image using the combined mask -----------------
    # hsv_img[is_water] flattens all pixels where is_water==True into a
    # single (N, 3) array of [Hue, Saturation, Value] rows. Same for
    # is_nonwater. Because both this and np.argwhere() iterate over the
    # array in the same (row-major) order, index i in water_pixels
    # corresponds to index i in water_coords_all.
    water_pixels = hsv_img[is_water]
    nonwater_pixels = hsv_img[is_nonwater]
 
    print(f"Available pixels - water: {len(water_pixels)}, non-water: {len(nonwater_pixels)}")
 
    # --- Cap sample size per class ------------------------------------------
    # Prevents a huge class (e.g. millions of "not water" land pixels) from
    # overwhelming a small class (e.g. a single small pond), and keeps the
    # resulting scatter plot fast to render and easy to read.
    n_water = min(n_samples, len(water_pixels))
    n_nonwater = min(n_samples, len(nonwater_pixels))
 
    # --- Randomly choose indices WITHOUT replacement ------------------------
    # replace=False ensures no pixel is sampled twice. The SAME random
    # indices are used to pull from both the HSV values and the coordinates,
    # so each sampled HSV row stays matched to its correct pixel location.
    water_idx = rng.choice(len(water_pixels), n_water, replace=False)
    nonwater_idx = rng.choice(len(nonwater_pixels), n_nonwater, replace=False)
 
    water_sample = water_pixels[water_idx]
    nonwater_sample = nonwater_pixels[nonwater_idx]
 
    water_coords = water_coords_all[water_idx]
    nonwater_coords = nonwater_coords_all[nonwater_idx]
 
    print(f"Sampled - water: {len(water_sample)}, non-water: {len(nonwater_sample)}")
    return water_sample, nonwater_sample, water_coords, nonwater_coords 


# ==============================================================================
# PLOT HUE VS. VALUE, COLOR-CODED BY CLASS
# ==============================================================================
#Create the Scatter Plot of Hue and Value
#Each individual poitn is color coded by the class 
def plot_hue_vs_value(water_sample, nonwater_sample, output_path):

    fig, ax = plt.subplots(figsize=(8, 6))

    # --- Plot non-water FIRST, water SECOND ---------------------------------
    # Drawing order matters for readability: whichever class is plotted last
    # sits visually "on top" in overlapping regions. Water is usually the
    # smaller / more important class, so it's drawn on top here.
    ax.scatter(
        nonwater_sample[:, 0],  # Hue channel (column index 0)
        nonwater_sample[:, 2],  # Value channel (column index 2)
        c="orange", alpha=0.4, s=10, label="not water"
    )
    ax.scatter(
        water_sample[:, 0],
        water_sample[:, 2],
        c="blue", alpha=0.4, s=10, label="water"
    )

    # --- Axis labels, title, and fixed 0-1 limits ---------------------------
    # Fixing the axis limits to the full 0-1 HSV range (rather than letting
    # matplotlib auto-scale) makes it easy to compare plots across different
    # NAIP tiles or runs, since the scale never shifts.
    ax.set_xlabel("Hue")
    ax.set_ylabel("Value")
    ax.set_title("Hue vs. Value - sampled NAIP pixels by water mask class")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")
    plt.show()

# ==============================================================================
# SECTION 5b: PLOT SAMPLE LOCATIONS ON TOP OF THE ORIGINAL IMAGE
# ==============================================================================
 
def plot_sample_locations(rgb_img, water_coords, nonwater_coords, output_path):
    """
    Display the original NAIP RGB image with the sampled pixel locations
    overlaid as points, color-coded by class: blue = water, orange = not
    water. This lets you visually confirm the sampled points are actually
    falling where you expect (e.g. inside the river/pond, and on land
    elsewhere) rather than trusting the mask blindly.
 
    Parameters
    ----------
    rgb_img : np.ndarray, shape (H, W, 3)
        The original RGB image (0-1 normalized), as returned by
        load_naip_rgb(). Used as the background.
    water_coords : np.ndarray, shape (n_water, 2)
        (row, col) pixel coordinates of sampled water pixels, as returned
        by sample_hsv_by_class().
    nonwater_coords : np.ndarray, shape (n_nonwater, 2)
        (row, col) pixel coordinates of sampled non-water pixels.
    output_path : str
        File path to save the resulting PNG image to.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
 
    # --- Draw the RGB image as the background -------------------------------
    # imshow expects (row, col, channel) order, which matches rgb_img as
    # returned by load_naip_rgb() -- no transposing needed here.
    ax.imshow(rgb_img)
 
    # --- Overlay sampled points ----------------------------------------------
    # IMPORTANT: coordinates from np.argwhere() are (row, col), but imshow's
    # x-axis is COLUMN and y-axis is ROW. So we plot column on x and row on y
    # (i.e. coords[:, 1] on x, coords[:, 0] on y) -- getting this backwards
    # is the most common mistake when overlaying pixel coordinates on imshow.
    ax.scatter(
        nonwater_coords[:, 1], nonwater_coords[:, 0],  # x=col, y=row
        c="orange", alpha=0.5, s=8, label="not water"
    )
    ax.scatter(
        water_coords[:, 1], water_coords[:, 0],  # x=col, y=row
        c="blue", alpha=0.5, s=8, label="water"
    )
 
    ax.set_title("Sampled pixel locations overlaid on NAIP RGB image")
    ax.legend(loc="upper right")
    ax.set_xticks([])  # pixel-index ticks aren't meaningful to show here
    ax.set_yticks([])
 
    plt.tight_layout()
 
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Saved sample location plot to {output_path}")
    plt.show()
 
 
# ==============================================================================
# MAIN WORKFLOW
# ==============================================================================
#Run Each Thing From Above
def main():
    """
    Run the full workflow end-to-end, in order:
    load NAIP RGB -> load/align mask -> convert to HSV -> sample -> plot.
    """
    print("Loading NAIP RGB bands...")
    rgb_img, transform, crs, shape, valid_mask = load_naip_rgb(NAIP_PATH)
 
    print("Loading and aligning water mask...")
    mask = load_aligned_mask(MASK_PATH, transform, crs, shape)
 
    print("Converting RGB to HSV...")
    # rgb2hsv expects and returns float arrays with values in [0, 1],
    # which matches how we normalized rgb_img in Section 2.
    hsv_img = rgb2hsv(rgb_img)
 
    print("Sampling pixels by class...")
    water_sample, nonwater_sample, water_coords, nonwater_coords = sample_hsv_by_class(
        hsv_img, mask, valid_mask, N_SAMPLES, seed=RANDOM_SEED
    )
 
    print("Plotting Hue vs. Value...")
    plot_hue_vs_value(water_sample, nonwater_sample, OUTPUT_PLOT)
 
    print("Plotting sample locations on original image...")
    plot_sample_locations(rgb_img, water_coords, nonwater_coords, OUTPUT_LOCATIONS_PLOT) 
# ==============================================================================
# SCRIPT ENTRY POINT
# ==============================================================================
# The `if __name__ == "__main__":` guard ensures main() only runs when this
# file is executed directly (e.g. `python naip_hsv_water_sampling.py`), and
# NOT when this file is imported as a module into another script -- which
# lets you reuse the individual functions above elsewhere without triggering
# the full workflow automatically.
# ==============================================================================

if __name__ == "__main__":
    main()
