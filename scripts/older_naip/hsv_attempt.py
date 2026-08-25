"""
================================================================================
NAIP RGB -> HSV UNIFORM (UNCLASSED) SAMPLING ACROSS THE WHOLE IMAGE
================================================================================

PURPOSE
-------
Companion script to naip_hsv_water_sampling.py. Instead of sampling
water/non-water pixels using a binary mask, this script samples a fixed
number of points UNIFORMLY across the entire valid (non-nodata) area of a
NAIP image, with no class distinction at all.

------------------
	1. Load a 4-band NAIP GeoTIFF (Red, Green, Blue, NIR)
	   -> Only the RGB bands are used; NIR is read but intentionally ignored.
	2. Convert the RGB image to HSV (Hue, Saturation, Value) color space.
	3. Randomly sample N points from anywhere in the image, excluding only
	   nodata/fill pixels (e.g. black mosaic-seam borders) -- no water mask
	   is used at all.
	4. Plot all three pairwise HSV comparisons (Hue-Saturation, Hue-Value,
	   Saturation-Value) side by side.
	5. Plot the sampled point locations overlaid on the original image.
================================================================================
"""

from pathlib import Path

import numpy as np
import rasterio
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt


# ==============================================================================
# SECTION 1: CONFIGURATION
# ==============================================================================
# Edit these values before running. These are the only lines you should need
# to change for a basic run.
# ------------------------------------------------------------------------------

NAIP_PATH = "naip_4band.tif"

N_SAMPLES_UNIFORM = 5000

RANDOM_SEED = 42

OUTPUT_PANELS_UNIFORM_PLOT = "hsv_three_panel_uniform.png"

OUTPUT_LOCATIONS_UNIFORM_PLOT = "sample_locations_overlay_uniform.png"


# ==============================================================================
# SECTION 2: LOAD NAIP RGB BANDS
# ==============================================================================

def load_naip_rgb(naip_path):
	"""
	Load the Red, Green, and Blue bands from a 4-band NAIP GeoTIFF, along
	with a valid-data mask that flags nodata/fill pixels.

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
	valid_mask : np.ndarray, shape (H, W), dtype bool
		True where the pixel has real image data, False where it's
		nodata/fill (e.g. black borders where separate NAIP tiles were
		mosaicked together with no coverage). Used to exclude these fill
		pixels from sampling -- otherwise pure-black nodata pixels get
		sampled as if they were real scene content, which would corrupt
		the HSV point cloud with meaningless values.
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

	# --- Normalize pixel values to the 0-1 range ---------------------------
	# NAIP imagery can be delivered as 8-bit (0-255) or occasionally 16-bit.
	# Dividing by the array's own max (rather than a hardcoded 255) makes
	# this work correctly regardless of bit depth.
	rgb /= rgb.max()

	# --- Reorder axes from (band, row, col) to (row, col, band) -----------
	# skimage's rgb2hsv() and standard image conventions expect the color
	# channel to be the LAST axis, not the first.
	rgb_img = np.transpose(rgb, (1, 2, 0))

	return rgb_img, valid_mask


# ==============================================================================
# SECTION 3: SAMPLE HSV PIXELS UNIFORMLY ACROSS THE WHOLE IMAGE (NO MASK)
# ==============================================================================

def sample_hsv_uniform(hsv_img, valid_mask, n_samples, seed=42):
	"""
	Randomly sample n_samples pixels from anywhere in the image, with no
	water/non-water class distinction -- only excluding nodata/fill pixels
	via valid_mask.

	Parameters
	----------
	hsv_img : np.ndarray, shape (H, W, 3)
		HSV-converted image (Hue, Saturation, Value channels, each in 0-1).
	valid_mask : np.ndarray, shape (H, W), dtype bool
		True where the pixel has real image data (not nodata/fill), as
		returned by load_naip_rgb().
	n_samples : int
		Maximum number of pixels to sample.
	seed : int
		Random seed for reproducibility.

	Returns
	-------
	sample : np.ndarray, shape (n, 3)
		Sampled HSV pixel values.
	coords : np.ndarray, shape (n, 2)
		(row, col) pixel coordinates of each sampled point, in the same
		order as sample. Because both np.argwhere(valid_mask) and
		hsv_img[valid_mask] iterate over the array in the same (row-major)
		order, index i in one corresponds to index i in the other.
	"""
	rng = np.random.default_rng(seed)

	coords_all = np.argwhere(valid_mask)
	pixels_all = hsv_img[valid_mask]

	print(f"Available valid pixels: {len(pixels_all)}")

	n = min(n_samples, len(pixels_all))
	idx = rng.choice(len(pixels_all), n, replace=False)

	sample = pixels_all[idx]
	coords = coords_all[idx]

	print(f"Sampled: {len(sample)}")
	return sample, coords


# ==============================================================================
# SECTION 4: PLOT 3-PANEL HSV COMPARISON (UNCLASSED)
# ==============================================================================

def plot_hsv_panels_uniform(sample, output_path):
	"""
	Create a 3-panel figure showing all three pairwise HSV comparisons side
	by side: Hue vs. Saturation, Hue vs. Value, and Saturation vs. Value.
	Since there is no water mask here, all points are plotted in a single
	color -- this shows the overall shape/spread of the scene's HSV
	distribution, useful as a baseline to compare against a masked
	water/non-water sample.

	Parameters
	----------
	sample : np.ndarray, shape (n, 3)
		Sampled HSV values, as returned by sample_hsv_uniform().
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
		ax.scatter(
			sample[:, x_idx], sample[:, y_idx],
			c="teal", alpha=0.3, s=8
		)
		ax.set_xlabel(x_label)
		ax.set_ylabel(y_label)
		ax.set_xlim(0, 1)
		ax.set_ylim(0, 1)

	fig.suptitle("HSV pairwise comparisons - uniform sample, whole image (no mask)")

	plt.tight_layout()

	# --- Create the output directory if it doesn't already exist -----------
	# matplotlib's savefig() will NOT create missing parent directories on
	# its own and will raise FileNotFoundError if they don't exist.
	Path(output_path).parent.mkdir(parents=True, exist_ok=True)

	plt.savefig(output_path, dpi=150)
	print(f"Saved uniform 3-panel HSV plot to {output_path}")
	plt.show()


# ==============================================================================
# SECTION 5: PLOT SAMPLE LOCATIONS ON TOP OF THE ORIGINAL IMAGE (UNCLASSED)
# ==============================================================================

def plot_sample_locations_uniform(rgb_img, coords, output_path):
	"""
	Display the original NAIP RGB image with uniformly sampled pixel
	locations overlaid as single-color points (no class distinction).

	Parameters
	----------
	rgb_img : np.ndarray, shape (H, W, 3)
		The original RGB image (0-1 normalized), as returned by
		load_naip_rgb(). Used as the background.
	coords : np.ndarray, shape (n, 2)
		(row, col) pixel coordinates of sampled points, as returned by
		sample_hsv_uniform().
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
		coords[:, 1], coords[:, 0],  # x=col, y=row
		c="teal", alpha=0.5, s=8
	)

	ax.set_title("Uniform sample locations overlaid on NAIP RGB image (no mask)")
	ax.set_xticks([])  # pixel-index ticks aren't meaningful to show here
	ax.set_yticks([])

	plt.tight_layout()

	Path(output_path).parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, dpi=150)
	print(f"Saved uniform sample location plot to {output_path}")
	plt.show()


# ==============================================================================
# SECTION 6: MAIN WORKFLOW
# ==============================================================================

def main():
	"""
	Run the full workflow end-to-end, in order:
	load NAIP RGB -> convert to HSV -> sample uniformly -> plot panels ->
	plot sample locations.
	"""
	print("Loading NAIP RGB bands...")
	rgb_img, valid_mask = load_naip_rgb(NAIP_PATH)

	print("Converting RGB to HSV...")
	# rgb2hsv expects and returns float arrays with values in [0, 1],
	# which matches how we normalized rgb_img in Section 2.
	hsv_img = rgb2hsv(rgb_img)

	print("Sampling uniformly across whole image (no mask)...")
	sample, coords = sample_hsv_uniform(
		hsv_img, valid_mask, N_SAMPLES_UNIFORM, seed=RANDOM_SEED
	)

	print("Plotting uniform 3-panel HSV comparison...")
	plot_hsv_panels_uniform(sample, OUTPUT_PANELS_UNIFORM_PLOT)

	print("Plotting uniform sample locations on original image...")
	plot_sample_locations_uniform(rgb_img, coords, OUTPUT_LOCATIONS_UNIFORM_PLOT)


# ==============================================================================
# SECTION 7: SCRIPT ENTRY POINT
# ==============================================================================
# The `if __name__ == "__main__":` guard ensures main() only runs when this
# file is executed directly (e.g. `python naip_hsv_uniform_sampling.py`), and
# NOT when this file is imported as a module into another script -- which
# lets you reuse the individual functions above elsewhere without triggering
# the full workflow automatically.
# ==============================================================================

if __name__ == "__main__":
	main()
