# check_bands.py
#
# Quick diagnostic to check if NAIP bands 1, 2, 3 (R, G, B) are genuinely
# different from each other, or if they're accidentally duplicates.
#
# Usage: python check_bands.py /path/to/naip_2010.tif

import sys
import rasterio
import numpy as np

def check_bands(tif_path):
    print(f"Checking: {tif_path}\n")

    with rasterio.open(tif_path) as src:
        n_bands = src.count
        print(f"Number of bands: {n_bands}")

        # Read bands one at a time to keep memory use low
        r = src.read(1).astype(np.float32)
        g = src.read(2).astype(np.float32)
        b = src.read(3).astype(np.float32)
        if n_bands >= 4:
            n = src.read(4).astype(np.float32)
        else:
            n = None

    print(f"\nBand shapes: {r.shape}")
    print(f"\n--- Comparing bands ---")
    print(f"R vs G - mean abs diff: {np.mean(np.abs(r - g)):.3f}  | identical pixels: {np.mean(r == g) * 100:.1f}%")
    print(f"R vs B - mean abs diff: {np.mean(np.abs(r - b)):.3f}  | identical pixels: {np.mean(r == b) * 100:.1f}%")
    print(f"G vs B - mean abs diff: {np.mean(np.abs(g - b)):.3f}  | identical pixels: {np.mean(g == b) * 100:.1f}%")

    if n is not None:
        print(f"R vs N - mean abs diff: {np.mean(np.abs(r - n)):.3f}  | identical pixels: {np.mean(r == n) * 100:.1f}%")

    print(f"\n--- Per-band stats ---")
    for name, band in [("R", r), ("G", g), ("B", b)] + ([("N", n)] if n is not None else []):
        print(f"{name}: min={band.min():.1f}, max={band.max():.1f}, mean={band.mean():.1f}, std={band.std():.1f}")

    print(f"\n--- Verdict ---")
    if np.mean(r == g) > 0.95 and np.mean(r == b) > 0.95:
        print("Bands 1, 2, 3 appear to be DUPLICATES of each other — likely a real bug.")
    else:
        print("Bands 1, 2, 3 are genuinely different — earlier visual similarity was likely just a low-contrast display artifact from -scale, not a real problem.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_bands.py /path/to/naip_file.tif")
        sys.exit(1)

    check_bands(sys.argv[1])
