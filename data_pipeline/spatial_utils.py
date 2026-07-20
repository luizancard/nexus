"""Shared coordinate reference system (CRS) conversions for the NEXUS spatial pipeline.

Every data source in this pipeline arrives in a different CRS: Mapillary images
are WGS84 (EPSG:4326), the DEM and contour data are SIRGAS 2000 / UTM zone 23S
(EPSG:31983, confirmed from `data_files/lourdes_dem_1m.tif`), and the OSM graph
exists in *both* depending on pipeline stage (raw download vs. after
`ox.project_graph`). Mixing them silently produces plausible-looking but wrong
distances -- this module is the single place those conversions happen so every
call site (mapillary_client, geometric_attribute_extractor, edge_attribute_fusion)
agrees on the same transform.
"""

from __future__ import annotations

from functools import lru_cache

from pyproj import Transformer

WGS84 = "EPSG:4326"
DEM_CRS = "EPSG:31983"  # SIRGAS 2000 / UTM zone 23S


@lru_cache(maxsize=None)
def _transformer(source_crs: str, target_crs: str) -> Transformer:
    """Build (and cache) a pyproj Transformer for one CRS pair.

    Constructing a Transformer parses a PROJ pipeline, which is too expensive
    to redo per point when converting thousands of coordinates; `always_xy=True`
    pins the axis order to (lon, lat)/(x, y) regardless of how the underlying
    CRS defines its own axis order, which otherwise silently varies by CRS.
    """
    return Transformer.from_crs(source_crs, target_crs, always_xy=True)


def to_dem_crs(lat: float, lon: float, target_crs: str = DEM_CRS) -> tuple[float, float]:
    """Convert a WGS84 (lat, lon) point into a projected CRS.

    Args:
        lat: Latitude in decimal degrees (WGS84).
        lon: Longitude in decimal degrees (WGS84).
        target_crs: Destination CRS. Defaults to the project DEM CRS (EPSG:31983).

    Returns:
        (x, y) in meters under `target_crs`.
    """
    x, y = _transformer(WGS84, target_crs).transform(lon, lat)
    return x, y


def to_wgs84(x: float, y: float, source_crs: str = DEM_CRS) -> tuple[float, float]:
    """Convert a projected (x, y) point back into WGS84 (lat, lon).

    Args:
        x: Easting in meters under `source_crs`.
        y: Northing in meters under `source_crs`.
        source_crs: CRS the input is expressed in. Defaults to EPSG:31983.

    Returns:
        (lat, lon) in decimal degrees.
    """
    lon, lat = _transformer(source_crs, WGS84).transform(x, y)
    return lat, lon


def batch_to_dem_crs(
    points: list[tuple[float, float]], target_crs: str = DEM_CRS
) -> list[tuple[float, float]]:
    """Vectorized (lat, lon) -> (x, y) conversion for many points at once.

    Args:
        points: (lat, lon) pairs in WGS84.
        target_crs: Destination CRS. Defaults to EPSG:31983.

    Returns:
        (x, y) pairs in meters under `target_crs`, same order as `points`.
    """
    if not points:
        return []
    lats, lons = zip(*points)
    xs, ys = _transformer(WGS84, target_crs).transform(list(lons), list(lats))
    return list(zip(xs, ys))
