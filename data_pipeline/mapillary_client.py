"""Mapillary Graph API v4 client for fetching street-level image metadata.

Pulls image metadata (not the raw images themselves -- those are only ever
downloaded on the fly during inference, see `run_segmentation_inference.py`)
for a place, and writes it to the CSV schema `semantic_segmentation_training.py`
already expects.

API constraints this module works around (verified against the live Mapillary
developer docs):
    - `bbox` must be smaller than 0.01 degrees square, so a neighborhood-sized
      area has to be tiled into a grid of sub-threshold cells.
    - Pagination only works when filtering by `creator_username`; a plain bbox
      query caps at `limit=2000` with no next page, so tiles must stay small
      enough that no single one approaches that cap.

Empirical finding, not documented by Mapillary: tile density affects total
results even for tiles well under the 2000-result cap -- a coarse 4-tile
fetch of Lourdes returned 6,579 unique images; the same area at 16 tiles
(the DEFAULT_TILE_DEGREES below) returned 22,943. Something in Mapillary's
bbox search appears to thin/sample results in a way that scales with query
area, not just result count. See docs/METHODOLOGY.md Section 3.1 for the
full comparison. This is why DEFAULT_TILE_DEGREES is set well below the
API's own 0.01 limit -- not just as a boundary safety margin, but because
the coarser end of the legal range silently under-collects data.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from pathlib import Path
from typing import Any

import osmnx as ox
import requests

from data_pipeline.spatial_utils import batch_to_dem_crs

GRAPH_API_URL = "https://graph.mapillary.com/images"
MAX_TILE_DEGREES = 0.01
# 0.004, not just-under-0.01: the validated operating point from real testing
# (see module docstring) -- a coarser tile size legally satisfies the API's
# bbox limit but silently returns far fewer images per unit area.
DEFAULT_TILE_DEGREES = 0.004
NEAR_CAP_WARNING_THRESHOLD = 1800  # warn well before the 2000-result API cap
REQUEST_LIMIT = 2000  # Mapillary's documented max `limit` value per request
REQUEST_FIELDS = "id,captured_at,compass_angle,camera_type,geometry,thumb_1024_url"


def get_place_bbox(place: str) -> tuple[float, float, float, float]:
    """Resolve a place name to a (min_lon, min_lat, max_lon, max_lat) bbox.

    Uses the same OSMnx geocoding entry point `osm_extractor.py` uses for the
    same place string, so the Mapillary image search area and the OSM graph
    area are guaranteed to agree.

    Args:
        place: Nominatim-resolvable place name, e.g.
            "Lourdes, Belo Horizonte, Minas Gerais, Brazil".

    Returns:
        (min_lon, min_lat, max_lon, max_lat) in WGS84 degrees.
    """
    gdf = ox.geocode_to_gdf(place)
    min_lon, min_lat, max_lon, max_lat = gdf.total_bounds
    return float(min_lon), float(min_lat), float(max_lon), float(max_lat)


def tile_bbox(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    cell_size_deg: float = DEFAULT_TILE_DEGREES,
) -> list[tuple[float, float, float, float]]:
    """Split a bbox into a grid of cells each under Mapillary's 0.01-degree cap.

    Args:
        min_lon: Western edge, WGS84 degrees.
        min_lat: Southern edge, WGS84 degrees.
        max_lon: Eastern edge, WGS84 degrees.
        max_lat: Northern edge, WGS84 degrees.
        cell_size_deg: Target cell size. Must stay under `MAX_TILE_DEGREES`.

    Returns:
        List of (min_lon, min_lat, max_lon, max_lat) tiles covering the input
        bbox, row-major order.
    """
    if cell_size_deg >= MAX_TILE_DEGREES:
        raise ValueError(
            f"cell_size_deg must be < {MAX_TILE_DEGREES} (Mapillary API bbox limit)."
        )
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError("Invalid bbox: min must be less than max on both axes.")

    n_cols = max(1, math.ceil((max_lon - min_lon) / cell_size_deg))
    n_rows = max(1, math.ceil((max_lat - min_lat) / cell_size_deg))
    col_step = (max_lon - min_lon) / n_cols
    row_step = (max_lat - min_lat) / n_rows

    tiles: list[tuple[float, float, float, float]] = []
    for row in range(n_rows):
        for col in range(n_cols):
            tile_min_lon = min_lon + col * col_step
            tile_max_lon = min_lon + (col + 1) * col_step
            tile_min_lat = min_lat + row * row_step
            tile_max_lat = min_lat + (row + 1) * row_step
            tiles.append((tile_min_lon, tile_min_lat, tile_max_lon, tile_max_lat))
    return tiles


def fetch_images_in_tile(
    bbox: tuple[float, float, float, float],
    access_token: str,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Fetch all image records within one Mapillary API bbox tile.

    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat), each axis span < 0.01 deg.
        access_token: Mapillary client access token (starts with "MLY|").
        session: Optional shared `requests.Session` for connection reuse.

    Returns:
        Raw image dicts as returned by the API's `data` array.

    Raises:
        requests.HTTPError: On a non-2xx response (e.g. invalid token -> 401).
    """
    http = session or requests
    min_lon, min_lat, max_lon, max_lat = bbox
    params = {
        "access_token": access_token,
        "fields": REQUEST_FIELDS,
        "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "limit": REQUEST_LIMIT,
    }
    response = http.get(GRAPH_API_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    images = payload.get("data", [])
    if len(images) >= NEAR_CAP_WARNING_THRESHOLD:
        print(
            f"      [WARNING] Tile {bbox} returned {len(images)} images, "
            f"near the API's {REQUEST_LIMIT} cap -- results may be truncated. "
            "Reduce cell_size_deg and re-run."
        )
    return images


def fetch_images_for_place(
    place: str,
    access_token: str,
    cell_size_deg: float = DEFAULT_TILE_DEGREES,
    request_delay_s: float = 0.2,
) -> list[dict[str, Any]]:
    """Fetch and dedupe all Mapillary images covering a named place.

    Args:
        place: Nominatim-resolvable place name.
        access_token: Mapillary client access token.
        cell_size_deg: Tile size passed to `tile_bbox`.
        request_delay_s: Delay between tile requests (politeness margin; well
            under Mapillary's documented 10,000/min search-API rate limit).

    Returns:
        Deduplicated raw image dicts across all tiles.
    """
    min_lon, min_lat, max_lon, max_lat = get_place_bbox(place)
    tiles = tile_bbox(min_lon, min_lat, max_lon, max_lat, cell_size_deg=cell_size_deg)
    print(f"[1/2] Querying {len(tiles)} tile(s) covering '{place}'...")

    seen_ids: set[str] = set()
    images: list[dict[str, Any]] = []
    with requests.Session() as session:
        for index, tile in enumerate(tiles, start=1):
            tile_images = fetch_images_in_tile(tile, access_token, session=session)
            new_count = 0
            for image in tile_images:
                image_id = str(image.get("id", ""))
                if image_id and image_id not in seen_ids:
                    seen_ids.add(image_id)
                    images.append(image)
                    new_count += 1
            print(f"      tile {index}/{len(tiles)}: {len(tile_images)} images ({new_count} new)")
            if index < len(tiles):
                time.sleep(request_delay_s)

    print(f"[2/2] Total unique images: {len(images)}")
    return images


def images_to_rows(images: list[dict[str, Any]], city: str) -> list[dict[str, Any]]:
    """Normalize raw Mapillary image dicts into flat manifest rows.

    Args:
        images: Raw image dicts from `fetch_images_for_place`.
        city: City label to stamp on every row.

    Returns:
        Rows with `image_id`, `image_path`, `latitude`, `longitude`, `city`,
        `compass_angle`, `captured_at` -- missing-field records dropped.
    """
    rows: list[dict[str, Any]] = []
    skipped = 0
    for image in images:
        geometry = image.get("geometry") or {}
        coordinates = geometry.get("coordinates")
        image_id = image.get("id")
        thumb_url = image.get("thumb_1024_url")
        if not (image_id and thumb_url and coordinates and len(coordinates) == 2):
            skipped += 1
            continue
        lon, lat = coordinates
        rows.append(
            {
                "image_id": str(image_id),
                "image_path": thumb_url,
                "latitude": lat,
                "longitude": lon,
                "city": city,
                "compass_angle": image.get("compass_angle", ""),
                "captured_at": image.get("captured_at", ""),
            }
        )
    if skipped:
        print(f"      [INFO] Skipped {skipped} record(s) missing id/geometry/thumbnail.")
    return rows


def select_representative_images(
    rows: list[dict[str, Any]],
    cell_size_m: float = 20.0,
    max_per_cell_direction: int = 1,
) -> list[dict[str, Any]]:
    """Downselect a dense raw image set to a spatially representative subset.

    Mapillary coverage in a well-mapped neighborhood is dominated by many
    near-duplicate frames from repeated drive/walk-throughs (confirmed
    empirically for Lourdes: tile counts kept climbing well past the point
    where a naive read of the API's `limit=2000` would suggest completeness).
    Running segmentation inference on every raw image is neither necessary
    (adjacent frames of the same stretch of sidewalk carry near-identical
    accessibility signal) nor practical within the project timeline.

    Bins images into a `cell_size_m` x `cell_size_m` grid (in the DEM's
    projected CRS, so cell size is true meters, not degrees) crossed with a
    4-way compass quadrant (N/E/S/W), and keeps the `max_per_cell_direction`
    most recent image(s) per (cell, quadrant) -- preserving multiple viewing
    angles of the same physical location while collapsing redundant frames.

    Args:
        rows: Manifest rows from `images_to_rows` (must include `latitude`,
            `longitude`, `compass_angle`, `captured_at`).
        cell_size_m: Spatial grid resolution in meters.
        max_per_cell_direction: Images kept per (grid cell, quadrant).

    Returns:
        Downselected subset of `rows`, same schema.
    """
    if not rows:
        return []

    points = batch_to_dem_crs([(float(r["latitude"]), float(r["longitude"])) for r in rows])
    buckets: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for row, (x, y) in zip(rows, points):
        cell_x = int(x // cell_size_m)
        cell_y = int(y // cell_size_m)
        try:
            compass = float(row["compass_angle"]) % 360.0
        except (TypeError, ValueError):
            compass = 0.0
        quadrant = int(compass // 90.0)  # 0=N-ish, 1=E-ish, 2=S-ish, 3=W-ish
        buckets.setdefault((cell_x, cell_y, quadrant), []).append(row)

    selected: list[dict[str, Any]] = []
    for bucket_rows in buckets.values():
        bucket_rows.sort(key=lambda r: str(r.get("captured_at", "")), reverse=True)
        selected.extend(bucket_rows[:max_per_cell_direction])

    print(
        f"      [INFO] Representative downselection: {len(rows)} -> {len(selected)} images "
        f"({len(buckets)} (cell, direction) buckets, {cell_size_m}m grid)."
    )
    return selected


def write_metadata_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    """Write manifest rows to CSV, matching the schema
    `parse_mapillary_metadata()` in `semantic_segmentation_training.py` expects
    (`image_id`, `image_path`, `latitude`, `longitude`, `city`), plus
    `compass_angle` and `captured_at`.

    Args:
        rows: Rows from `images_to_rows` (optionally downselected).
        output_csv: Destination CSV path.
    """
    if not rows:
        raise ValueError("No valid image records to write (all missing required fields).")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} image records to {output_csv}")


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch Mapillary image metadata for a place into a manifest CSV."
    )
    parser.add_argument("--place", type=str, required=True)
    parser.add_argument("--city", type=str, required=True, help="City label stamped on every row.")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--cell-size-deg", type=float, default=DEFAULT_TILE_DEGREES)
    parser.add_argument(
        "--no-downselect",
        action="store_true",
        help="Skip representative downselection and write every raw fetched image.",
    )
    parser.add_argument("--downselect-cell-size-m", type=float, default=20.0)
    parser.add_argument("--max-per-cell-direction", type=int, default=1)
    return parser


def main() -> None:
    """CLI entrypoint."""
    from dotenv import load_dotenv

    load_dotenv()
    access_token = os.environ.get("MAPILLARY_ACCESS_TOKEN")
    if not access_token:
        raise RuntimeError(
            "MAPILLARY_ACCESS_TOKEN not set. Add it to a local .env file "
            "(see .env.example) -- never hardcode it or commit it."
        )

    args = _build_cli_parser().parse_args()
    images = fetch_images_for_place(
        place=args.place,
        access_token=access_token,
        cell_size_deg=args.cell_size_deg,
    )
    rows = images_to_rows(images, city=args.city)
    if not args.no_downselect:
        rows = select_representative_images(
            rows,
            cell_size_m=args.downselect_cell_size_m,
            max_per_cell_direction=args.max_per_cell_direction,
        )
    write_metadata_csv(rows, output_csv=args.output_csv)


if __name__ == "__main__":
    main()
