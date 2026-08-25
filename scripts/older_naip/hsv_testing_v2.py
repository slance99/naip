"""
================================================================================
NAIP RGB -> HSV WATER / NON-WATER SAMPLING AND HUE-VS-VALUE SCATTER PLOT
================================================================================

PURPOSE
-------
This script helps you visually explore whether Hue and Value (from an HSV
color-space transform of NAIP RGB imagery) can separate "water" from
"not water" pixels, using a binary water mask you've already created as your
ground-truth labels.

WHY THIS MATTERS
-----------------
Standard water indices like NDWI require a Near-Infrared (NIR) band. NAIP
imagery sometimes is distributed or used as RGB-only (3-band), so this
script instead builds a labeled "point cloud" of Hue/Saturation/Value pixel
values directly from your own imagery and mask, so you can pick thresholds
empirically instead of guessing them.

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

	A valid-data mask is also used throughout so that nodata/fill pixels
	(e.g. black borders where separate NAIP tiles were mosaicked together)
	are excluded from both classes, rather than being sampled as if they
	were real land.

HOW TO RUN
-----------
	1. Edit the file paths and parameters in the CONFIG section below.
	2. Run from the command line:
		   python naip_hsv_water_sampling.py
	3. Two PNGs will be saved and displayed: the Hue-vs-Value scatter plot,
	   and an overlay of sample locations on the original image.

NOTE ON INDENTATION
---------------------
This file uses TABS for indentation, consistently throughout. Do not mix
in space-indented lines when editing -- Python will raise a TabError if
tabs and spaces are mixed within the same block.
================================================================================
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt


# ==============================================================================
# SECTION 1: CONFIGURATION
# ==============================================================================
# Edit these values before running. These are the only lines you should need
# to change for a basic run
# ------------------------------------------------------------------------------
NAIP_PATH = "/home/geomorph/california_rivers/naip/gee_naip/sacramento/sacramento_38/naip_2009.tif"
MASK_PATH = "/home/geomorph/california_rivers/naip/outputs/gee/red_bluff_colusa_outputs/sacramento_38_2009_mosaic.tif"
N_SAMPLES = 5000
RANDOM_SEED = 42
OUTPUT_PLOT = "/home/geomorph/california_rivers/naip/older_naip/scatter_38_2009.png"
OUTPUT_LOCATIONS_PLOT = "/home/geomorph/california_rivers/naip/older_naip/overlay_38_2009.png"

OUTPUT_PANELS_PLOT = "/home/geomorph/california_rivers/naip/older_naip/panel_38_2009.png"
# ==============================================================================
# SECTION 2: LOAD NAIP RGB BANDS
# ==============================================================================

def load_naip_rgb(naip_path):
	"""
	Load the Red, Green, and Blue bands from a 4-band NAIP GeoTIFF.

	The 4th band (NIR) is intentionally NOT loaded here, since this workflow
	is specifically testing what can be done with RGB-only data.

	Parameters
	----------
	naip_path : str
		Path to the 4-band NAIP GeoTIFF file.

	Returns
	-------
	rgb_img : np.ndarray, shape (H, W, 3)
		RGB image normalized to the 0-1 range, arranged as (row, col, band)
		so it's ready for skimage's rgb2hsv().
	transform : affine.Affine
		The raster's geotransform (maps pixel coordinates to map coordinates).
		Needed later to align the water mask onto this same grid.
	crs : rasterio.crs.CRS
		The raster's coordinate reference system. Needed for mask alignment.
	shape : tuple (H, W)
		The raster's pixel dimensions (height, width). Needed for mask
		alignment and to preallocate the reprojected mask array.
	valid_mask : np.ndarray, shape (H, W), dtype bool
		True where the pixel has real image data, False where it's
		nodata/fill (e.g. black borders where separate NAIP tiles were
		mosaicked together with no coverage). Used later to exclude these
		fill pixels from the "not water" sample -- otherwise pure-black
		nodata pixels get sampled as if they were real land, which
		corrupts the HSV point cloud with meaningless values.
	"""
	with rasterio.open(naip_path) as src:
		# --- Read only bands 1, 2, 3 (R, G, B) -----------------------------
		# rasterio band indices are 1-based, and .read() returns an array
		# shaped (bands, rows, cols) -- i.e. "band-first" order.
		rgb = src.read([1, 2, 3]).astype(np.float32)

		# --- Build a valid-data mask BEFORE normalizing ---------------------
		# rasterio's dataset_mask() combines the file's declared nodata value
		# (if any) with each band's own internal mask, returning 255 for
		# valid pixels and 0 for nodata/fill pixels. This correctly flags
		# mosaic-seam black borders even when the file's nodata metadata
		# tag is missing or inconsistent, because it also treats
		# all-bands-zero pixels as invalid.
		valid_mask = src.dataset_mask() > 0

		# --- Capture georeferencing info for later mask alignment ---------
		transform = src.transform
		crs = src.crs
		shape = src.shape

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
# SECTION 3: LOAD AND ALIGN THE WATER MASK
# ==============================================================================

def load_aligned_mask(mask_path, ref_transform, ref_crs, ref_shape):
	"""
	Load a binary water mask and, if necessary, reproject/resample it so
	its pixel grid exactly matches the NAIP image's pixel grid.

	WHY THIS STEP IS NECESSARY
	---------------------------
	Water masks are often generated at a different resolution, extent, or
	even coordinate reference system (CRS) than the source imagery. If the
	mask and image grids don't line up pixel-for-pixel, indexing one array
	with the other (as done in Section 4) would silently pull water/non-water
	labels from the WRONG spatial location -- a subtle bug that produces
	misleading results without raising any errors.

	Parameters
	----------
	mask_path : str
		Path to the binary water mask GeoTIFF (1 = water, 0 = not water).
	ref_transform : affine.Affine
		Geotransform of the reference grid (the NAIP image) to align to.
	ref_crs : rasterio.crs.CRS
		CRS of the reference grid to align to.
	ref_shape : tuple (H, W)
		Pixel dimensions of the reference grid to align to.

	Returns
	-------
	mask : np.ndarray, shape (H, W), dtype uint8
		Binary water mask aligned to the NAIP pixel grid (1 = water,
		0 = not water).
	"""
	with rasterio.open(mask_path) as mask_src:
		# --- Check whether the mask already matches the NAIP grid ---------
		needs_alignment = (
			mask_src.crs != ref_crs
			or mask_src.transform != ref_transform
			or mask_src.shape != ref_shape
		)

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
# SECTION 4: SAMPLE HSV PIXELS BY CLASS (WATER / NOT WATER)
# ==============================================================================

def sample_hsv_by_class(hsv_img, mask, valid_mask, n_samples, seed=42):
	"""
	Randomly sample a balanced set of water and non-water pixels from the
	HSV image, using the aligned binary mask to assign class labels.

	Pixels flagged as nodata/fill by valid_mask (e.g. black mosaic-seam
	borders where separate NAIP tiles don't overlap) are excluded from
	BOTH classes -- most importantly from "not water", since those pixels
	would otherwise be sampled as if they were real land and would pollute
	the HSV point cloud with meaningless near-zero values.

	In addition to the sampled HSV values, this also returns the (row, col)
	pixel coordinates of each sampled point, so the sample locations can
	later be plotted back on top of the original image (see
	plot_sample_locations() in Section 5b).

	Parameters
	----------
	hsv_img : np.ndarray, shape (H, W, 3)
		HSV-converted image (Hue, Saturation, Value channels, each in 0-1).
	mask : np.ndarray, shape (H, W)
		Aligned binary water mask (1 = water, 0 = not water).
	valid_mask : np.ndarray, shape (H, W), dtype bool
		True where the pixel has real image data (not nodata/fill), as
		returned by load_naip_rgb().
	n_samples : int
		Maximum number of pixels to sample per class.
	seed : int
		Random seed for reproducibility.

	Returns
	-------
	water_sample : np.ndarray, shape (n_water, 3)
		Sampled HSV pixel values for pixels labeled "water" by the mask.
	nonwater_sample : np.ndarray, shape (n_nonwater, 3)
		Sampled HSV pixel values for pixels labeled "not water" by the mask.
	water_coords : np.ndarray, shape (n_water, 2)
		(row, col) pixel coordinates of each sampled water pixel, in the
		same order as water_sample.
	nonwater_coords : np.ndarray, shape (n_nonwater, 2)
		(row, col) pixel coordinates of each sampled non-water pixel, in
		the same order as nonwater_sample.
	"""
	# --- Use a local random generator (not global np.random) --------------
	# This avoids side effects on any other code that might rely on
	# NumPy's global random state, and makes results reproducible.
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
# SECTION 5: PLOT HUE VS. VALUE, COLOR-CODED BY CLASS
# ==============================================================================

def plot_hue_vs_value(water_sample, nonwater_sample, output_path):
	"""
	Create and save a scatter plot of Hue (x-axis) vs. Value (y-axis) for
	the sampled pixels, color-coded by class: blue = water, orange = not
	water.

	Parameters
	----------
	water_sample : np.ndarray, shape (n_water, 3)
		Sampled HSV values for water pixels. Columns are [Hue, Saturation,
		Value] -- only columns 0 (Hue) and 2 (Value) are plotted here.
	nonwater_sample : np.ndarray, shape (n_nonwater, 3)
		Sampled HSV values for non-water pixels, same column layout.
	output_path : str
		File path to save the resulting PNG plot to.
	"""
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

	# --- Create the output directory if it doesn't already exist -----------
	# matplotlib's savefig() will NOT create missing parent directories on
	# its own and will raise FileNotFoundError if they don't exist.
	Path(output_path).parent.mkdir(parents=True, exist_ok=True)

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
# SECTION 5c: PLOT ALL THREE HSV PAIRWISE COMPARISONS
# ==============================================================================

def plot_hsv_panels(water_sample, nonwater_sample, output_path):
	"""
	Create a 3-panel figure showing all three pairwise HSV comparisons side
	by side: Hue vs. Saturation, Hue vs. Value, and Saturation vs. Value.

	This exists because Hue vs. Value alone can hide separation that shows
	up once Saturation is considered -- e.g. shaded water and shaded tree
	canopy can land in a similar Hue/Value range, but water is often less
	saturated than vegetation even in shadow.

	Parameters
	----------
	water_sample : np.ndarray, shape (n_water, 3)
		Sampled HSV values for water pixels. Columns are [Hue, Saturation, Value].
	nonwater_sample : np.ndarray, shape (n_nonwater, 3)
		Sampled HSV values for non-water pixels, same column layout.
	output_path : str
		File path to save the resulting PNG plot to.
	"""
	fig, axes = plt.subplots(1, 3, figsize=(18, 6))

	# Each tuple is (x_column_index, y_column_index, x_label, y_label).
	# Columns are 0=Hue, 1=Saturation, 2=Value.
	panel_configs = [
		(0, 1, "Hue", "Saturation"),
		(0, 2, "Hue", "Value"),
		(1, 2, "Saturation", "Value"),
	]

	for ax, (x_idx, y_idx, x_label, y_label) in zip(axes, panel_configs):
		# Non-water plotted first so water sits visually on top where they overlap.
		ax.scatter(
			nonwater_sample[:, x_idx], nonwater_sample[:, y_idx],
			c="orange", alpha=0.3, s=8, label="not water"
		)
		ax.scatter(
			water_sample[:, x_idx], water_sample[:, y_idx],
			c="blue", alpha=0.3, s=8, label="water"
		)
		ax.set_xlabel(x_label)
		ax.set_ylabel(y_label)
		ax.set_xlim(0, 1)
		ax.set_ylim(0, 1)

	axes[0].legend(loc="upper right")
	fig.suptitle("HSV pairwise comparisons - sampled NAIP pixels by water mask class")

	plt.tight_layout()

	Path(output_path).parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, dpi=150)
	print(f"Saved 3-panel HSV plot to {output_path}")
	plt.show()
# ==============================================================================
# SECTION 6: MAIN WORKFLOW
# ==============================================================================

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

	print("Plotting 3-panel HSV comparison...")
	plot_hsv_panels(water_sample, nonwater_sample, OUTPUT_PANELS_PLOT)
# ==============================================================================
# SECTION 7: SCRIPT ENTRY POINT
# ==============================================================================
# The `if __name__ == "__main__":` guard ensures main() only runs when this
# file is executed directly (e.g. `python naip_hsv_water_sampling.py`), and
# NOT when this file is imported as a module into another script -- which
# lets you reuse the individual functions above elsewhere without triggering
# the full workflow automatically.
# ==============================================================================

if __name__ == "__main__":
	main()
