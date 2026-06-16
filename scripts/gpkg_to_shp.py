# convert_gpkgs_to_shp.py
from pathlib import Path
import geopandas as gpd
import fiona

GPKG_DIR = Path("/home/geomorph/california_rivers/naip/gpkgs/rbc_small_gpkgs/")
SHP_DIR  = Path("/home/geomorph/california_rivers/naip/shapefiles/")
SHP_DIR.mkdir(parents=True, exist_ok=True)

for gpkg in sorted(GPKG_DIR.glob("*.gpkg")):
    layers = fiona.listlayers(str(gpkg))
    gdf = gpd.read_file(gpkg, layer=layers[0])
    out = SHP_DIR / f"{gpkg.stem}.shp"
    gdf.to_file(out, driver="ESRI Shapefile")
    print(f"Converted {gpkg.name} -> {out.name}")

print("Done!")
