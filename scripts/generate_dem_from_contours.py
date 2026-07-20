"""Gera o DEM (Modelo Digital de Elevacao) a partir das curvas de nivel.

O `lourdes_dem_1m.tif` anterior continha apenas zeros em todos os 277
milhoes de pixels (confirmado via inspecao direta) -- o processo externo que
o gerou (commit "created the .TIF ... using cKDTree") provavelmente buscava
uma coluna chamada 'Z', mas o shapefile de curvas usa 'COTA_CURVA'. A propria
`validate_dem.py` ja acusava isso ("ALERTA: A altitude media esta fora do
esperado"). Este script substitui aquele processo: densifica as linhas de
contorno reais (825-932m, coerente com a altitude de Belo Horizonte) em uma
nuvem de pontos e interpola um raster georreferenciado de verdade, com NoData
propriamente marcado (nao um zero implicito) fora do alcance real dos dados.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import osmnx as ox
import rasterio
from rasterio.transform import from_origin
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

RESOLUCAO_M = 1.0
BUFFER_M = 400.0  # margem alem do bairro, evita artefatos de borda na interpolacao
DISTANCIA_MAX_INTERPOLACAO_M = 150.0  # pixels mais longe que isso de qualquer dado real viram NoData
ESPACAMENTO_AMOSTRAGEM_M = 2.0  # intervalo de amostragem ao longo de cada linha de contorno
NODATA = -9999.0


def densificar_curvas(
    gdf: gpd.GeoDataFrame, espacamento_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converte linhas de contorno esparsas numa nuvem de pontos densa.

    Cada linha carrega uma unica elevacao (COTA_CURVA); amostrar pontos ao
    longo do seu comprimento da uma nuvem (x, y, z) adequada para
    interpolacao de dados dispersos (scipy.interpolate.griddata).
    """
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for geom, cota in zip(gdf.geometry, gdf["COTA_CURVA"]):
        comprimento = geom.length
        if comprimento == 0:
            continue
        n_pontos = max(2, int(comprimento / espacamento_m))
        for i in range(n_pontos + 1):
            ponto = geom.interpolate(i / n_pontos, normalized=True)
            xs.append(ponto.x)
            ys.append(ponto.y)
            zs.append(cota)
    return np.array(xs), np.array(ys), np.array(zs)


def gerar_dem(caminho_shapefile: Path, caminho_saida: Path, lugar: str) -> None:
    """Interpola um DEM real a partir das curvas de nivel, escopado ao redor
    de `lugar` -- evita gerar um raster do tamanho da extensao total (e quase
    vazia, fora do bairro) do shapefile de curvas.
    """
    print(f"[1/5] Lendo curvas de nivel de {caminho_shapefile}...")
    contornos = gpd.read_file(caminho_shapefile)
    print(
        f"      {len(contornos)} linhas, COTA_CURVA: "
        f"{contornos['COTA_CURVA'].min()}-{contornos['COTA_CURVA'].max()}m"
    )

    print(f"[2/5] Delimitando area de interesse ao redor de '{lugar}' (+{BUFFER_M}m)...")
    area = ox.geocode_to_gdf(lugar).to_crs(contornos.crs)
    area_buffer = area.buffer(BUFFER_M)
    minx, miny, maxx, maxy = area_buffer.total_bounds
    contornos_area = gpd.clip(contornos, area_buffer)
    if contornos_area.empty:
        raise ValueError(f"Nenhuma curva de nivel encontrada perto de '{lugar}'.")
    print(f"      {len(contornos_area)} linhas de contorno na area de interesse.")

    print(f"[3/5] Densificando curvas em nuvem de pontos (espacamento {ESPACAMENTO_AMOSTRAGEM_M}m)...")
    xs, ys, zs = densificar_curvas(contornos_area, ESPACAMENTO_AMOSTRAGEM_M)
    print(f"      {len(xs)} pontos amostrados.")

    print(f"[4/5] Interpolando grade regular de {RESOLUCAO_M}m...")
    n_cols = int(np.ceil((maxx - minx) / RESOLUCAO_M))
    n_rows = int(np.ceil((maxy - miny) / RESOLUCAO_M))
    grid_x, grid_y = np.meshgrid(
        minx + (np.arange(n_cols) + 0.5) * RESOLUCAO_M,
        maxy - (np.arange(n_rows) + 0.5) * RESOLUCAO_M,  # topo->baixo, casa com a ordem de linhas do raster
    )
    pontos = np.column_stack([xs, ys])
    grade_z = griddata(pontos, zs, (grid_x, grid_y), method="linear")

    faltando = np.isnan(grade_z)
    if faltando.any():
        grade_z[faltando] = griddata(pontos, zs, (grid_x[faltando], grid_y[faltando]), method="nearest")

    print("      Mascarando pixels longe de qualquer dado real como NoData...")
    arvore = cKDTree(pontos)
    distancias, _ = arvore.query(np.column_stack([grid_x.ravel(), grid_y.ravel()]))
    distancias = distancias.reshape(grade_z.shape)
    grade_z[distancias > DISTANCIA_MAX_INTERPOLACAO_M] = NODATA

    print(f"[5/5] Escrevendo {caminho_saida}...")
    transform = from_origin(minx, maxy, RESOLUCAO_M, RESOLUCAO_M)
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        caminho_saida,
        "w",
        driver="GTiff",
        height=n_rows,
        width=n_cols,
        count=1,
        dtype="float32",
        crs=contornos.crs,
        transform=transform,
        nodata=NODATA,
    ) as dst:
        dst.write(grade_z.astype(np.float32), 1)

    validos = grade_z[grade_z != NODATA]
    print(
        f"      OK -- {n_rows}x{n_cols} pixels, {len(validos)}/{grade_z.size} validos "
        f"({100 * len(validos) / grade_z.size:.1f}%), "
        f"altitude {validos.min():.1f}-{validos.max():.1f}m (media {validos.mean():.1f}m)"
    )


if __name__ == "__main__":
    RAIZ = Path(__file__).resolve().parent.parent
    gerar_dem(
        caminho_shapefile=RAIZ / "data_files" / "curvas_lourdes_Isolado.shp",
        caminho_saida=RAIZ / "data_files" / "lourdes_dem_1m.tif",
        lugar="Lourdes, Belo Horizonte, Minas Gerais, Brazil",
    )
