import os
import geopandas as gpd

# GDAL env var: reconstruir .shx ausente antes de qualquer import geo
os.environ["SHAPE_RESTORE_SHX"] = "YES"

caminho_shp = "/Users/luizancard/Developer/nexus_project/data_files/curvas_lourdes_Isolado.shp"

gdf = gpd.read_file(caminho_shp)

print(f"CRS atual do Shapefile: {gdf.crs}")
print(f"Bounds (limites): {gdf.total_bounds}")
print(f"Total de feições: {len(gdf)}")