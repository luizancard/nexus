import os
import geopandas as gpd

# GDAL env var: reconstruir .shx ausente antes de qualquer import geo
os.environ["SHAPE_RESTORE_SHX"] = "YES"

# Caminho para o shapefile de curvas de nível
caminho_shp = "/Users/luizancard/Developer/nexus_project/data_files/curvas_lourdes_Isolado.shp"

print(f"Auditando shapefile: {caminho_shp}\n")

# Verificar quais sidecar files estão presentes
base = os.path.splitext(caminho_shp)[0]
for ext in [".shx", ".dbf", ".prj"]:
    existe = os.path.exists(base + ext)
    print(f"  {ext}: {'✓ encontrado' if existe else '✗ AUSENTE'}")
print()

try:
    gdf = gpd.read_file(caminho_shp)
    print(f"CRS: {gdf.crs}")
    print(f"Total de feições: {len(gdf)}")
    print(f"Tipo de geometria: {gdf.geom_type.unique()}")
    print("\nColunas disponíveis no arquivo:")
    print(gdf.columns.tolist())
    print("\nPrimeiras 5 linhas da tabela:")
    print(gdf.head())

    # Verifica coluna de altitude (ajuste o nome se necessário)
    coluna_alt = "Z"
    if coluna_alt in gdf.columns:
        print(f"\nNulos na coluna '{coluna_alt}': {gdf[coluna_alt].isna().sum()}")
        print(f"Valores únicos na coluna '{coluna_alt}':")
        print(sorted(gdf[coluna_alt].dropna().unique())[:20])  # Primeiros 20
    else:
        print(f"\nColuna '{coluna_alt}' não encontrada. Colunas disponíveis: {gdf.columns.tolist()}")

except Exception as e:
    print(f"Erro ao ler o Shapefile: {e}")
