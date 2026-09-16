"""
RIVGRAPH
"""

import re
from pathlib import Path
from rivgraph.classes import river


# ---------------------------------------------------------------------------
# 1. CONFIGURE
# ---------------------------------------------------------------------------
#Where do masks live 
MASK_DIR = Path(
    "/home/geomorph/california_rivers/naip/outputs/gee/red_bluff_colusa_utm"
)

#Where my Results Should Go 
RESULTS_DIR = "home/geomorph/california_rivers/naip/outputs/gee/rivgraph_results"

#Direction of Input 
EXIT_SIDES = "ns"
 
#Filename Pattern 
FILENAME_RE = re.compile(r"^sacramento_(\d+)_(\d{4})_mosaic\.tif$")
 
# ---------------------------------------------------------------------------
# 2. RUN
# ---------------------------------------------------------------------------
 
def process_mask(mask_path: Path, reach: str, year: str) -> None:
    name = f"sacramento_{reach}_{year}"
 
    out_folder = RESULTS_DIR / name
    out_folder.mkdir(parents=True, exist_ok=True)
 
    print(f"[run ] {name}  (exit_sides={EXIT_SIDES})")
 
    try:
        riv = river(name, str(mask_path), str(out_folder), exit_sides=EXIT_SIDES)
        riv.skeletonize()
        riv.compute_network()
        riv.compute_link_width_and_length()
        riv.prune_network()
        riv.compute_centerline()
        riv.to_geovectors()
    except Exception as exc:  # noqa: BLE001 -- keep batch going on failure
        print(f"[fail] {name}: {exc}")
 
 
def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
 
    masks = sorted(MASK_DIR.glob("*.tif"))
    if not masks:
        raise SystemExit(f"No .tif files found in {MASK_DIR}")
 
    for mask_path in masks:
        m = FILENAME_RE.match(mask_path.name)
        if not m:
            print(f"[skip] {mask_path.name}: doesn't match expected pattern")
            continue
        reach, year = m.groups()
        process_mask(mask_path, reach, year)
 
 
if __name__ == "__main__":
    main()
 
