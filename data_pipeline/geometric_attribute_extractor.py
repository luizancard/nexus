"""Heuristic geometric attribute estimation from segmentation detections.

No monocular depth model, by design (see docs/METHODOLOGY.md) -- these are
categorical, best-effort estimates meant to match the cost formula's own
coarse thresholds (e.g. the <50cm width rule), not precise measurements.
Every assumption here is stated explicitly and gets checked empirically in
the Phase 1.6 validation pass, not just assumed to be good enough.

Two attributes, two different techniques:

- Sidewalk width: no reliable per-image camera calibration is available
  (Mapillary's `camera_parameters` field varies by capture device and isn't
  fetched by `mapillary_client.py`), so this uses a *stated, fixed*
  pinhole-camera assumption (camera height + vertical field of view) rather
  than a false precision. This is exactly the kind of approximation the
  Phase 1.6 validation exists to quantify, not hide.

- Ramp/curb-cut declivity: reuses the real DEM (see
  `scripts/generate_dem_from_contours.py`) rather than anything
  image-derived -- a slope raster gives genuine, surveyed-contour-based
  ground truth at any point, which no image heuristic could match.

Curb curvature (sharp vs. beveled) is deliberately NOT a separate geometric
computation here: Mapillary Vistas already distinguishes "Curb" (raised,
sharp by definition) from "Curb Cut" (the beveled accessibility ramp) as
distinct classes. Re-deriving that from contour geometry would be a less
reliable proxy for information the classifier already provides directly --
`edge_attribute_fusion.py` reads curb type straight off which class was
detected, not from this module.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from data_pipeline.spatial_utils import to_dem_crs

# Stated assumptions for the width pinhole projection -- NOT measured per
# image. Typical Mapillary contributor capture height (phone/chest mount);
# vertical FOV is a common mid-range value for phone/action-cam wide angle.
# Phase 1.6 validation quantifies how much error this introduces.
ASSUMED_CAMERA_HEIGHT_M = 1.5
ASSUMED_VERTICAL_FOV_DEG = 65.0

WIDTH_BUCKET_THRESHOLDS_M = (0.5, 0.9)  # matches the cost formula's own <50cm rule
SLOPE_NODATA = -9999.0
# Sanity bound, same defensive pattern osm_extractor.py already uses for
# elevation (< 0 or > 3000m -> NaN). Empirically, 99.8% of interpolated
# slope pixels fall under this; the rest are Delaunay-boundary interpolation
# artifacts at the edge of the contour data's convex hull, not real terrain
# -- no real Lourdes street approaches a 150% (~56 deg) grade.
MAX_PLAUSIBLE_SLOPE_PCT = 150.0

# NBR 9050 (Brazilian accessibility standard) ramp slope guidance: longer
# ramps must stay under ~8.33%; steeper is only tolerated for very short rises.
NBR9050_MAX_RAMP_SLOPE_PCT = 8.33


def _focal_length_px(image_height_px: int, vertical_fov_deg: float = ASSUMED_VERTICAL_FOV_DEG) -> float:
    """Pinhole focal length in pixels, from an assumed vertical field of view."""
    return (image_height_px / 2.0) / math.tan(math.radians(vertical_fov_deg) / 2.0)


def _ground_distance_m(
    pixel_row: float, image_height_px: int, camera_height_m: float = ASSUMED_CAMERA_HEIGHT_M
) -> float | None:
    """Distance from the camera to a ground-plane point at a given pixel row.

    Assumes a level camera (0 deg pitch, horizon at vertical center) -- a
    real Mapillary capture can be tilted, which is exactly the kind of error
    Phase 1.6's ground-truth comparison is meant to surface.

    Args:
        pixel_row: Row in the image (0 = top), e.g. the bottom of a mask.
        image_height_px: Full image height in pixels.
        camera_height_m: Assumed camera height above the ground plane.

    Returns:
        Distance in meters, or None if the row is above the horizon (no
        ground-plane intersection is possible there).
    """
    f_px = _focal_length_px(image_height_px)
    angle_below_horizon = math.atan((pixel_row - image_height_px / 2.0) / f_px)
    if angle_below_horizon <= 0:
        return None
    return camera_height_m / math.tan(angle_below_horizon)


def estimate_width_bucket(
    polygon_xy: list[tuple[float, float]], image_width_px: int, image_height_px: int
) -> str | None:
    """Estimate a sidewalk (or similar) mask's real-world width bucket.

    Takes the horizontal pixel extent of the mask near its bottom-most edge
    (closest to the camera -> largest, most scale-stable) and converts it to
    meters via the pinhole ground-plane projection in `_ground_distance_m`.

    Args:
        polygon_xy: Mask polygon in original image pixel coordinates.
        image_width_px: Full image width in pixels.
        image_height_px: Full image height in pixels.

    Returns:
        One of "under_50cm" / "50_to_90cm" / "over_90cm", or None if no
        reliable estimate could be made (e.g. mask entirely above the
        assumed horizon).
    """
    if len(polygon_xy) < 3:
        return None

    ys = [p[1] for p in polygon_xy]
    bottom_y = max(ys)
    # points within a thin band of the bottom edge -- avoids the overall
    # bbox width, which perspective can make misleading for a tapering mask
    band = [p for p in polygon_xy if bottom_y - p[1] <= max(2.0, 0.02 * image_height_px)]
    if len(band) < 2:
        band = polygon_xy
    xs = [p[0] for p in band]

    distance_m = _ground_distance_m(bottom_y, image_height_px)
    if distance_m is None:
        return None

    f_px = _focal_length_px(image_height_px)
    meters_per_px = distance_m / f_px
    width_m = (max(xs) - min(xs)) * meters_per_px

    low, high = WIDTH_BUCKET_THRESHOLDS_M
    if width_m < low:
        return "under_50cm"
    if width_m < high:
        return "50_to_90cm"
    return "over_90cm"


def generate_slope_raster(dem_path: Path, output_path: Path) -> None:
    """Derive a slope-percent raster from the DEM via a finite-difference gradient.

    Persisted once and reused, rather than recomputed per query -- same
    pattern the project already uses for the DEM itself. A pixel is marked
    NoData if it or any neighbor used in its gradient is NoData in the
    source DEM (an unreliable gradient is worse than a missing one).

    Args:
        dem_path: Source elevation raster (see `generate_dem_from_contours.py`).
        output_path: Destination for the slope-percent raster.
    """
    with rasterio.open(dem_path) as src:
        elevation = src.read(1)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs
        profile = src.profile

    valid = elevation != nodata if nodata is not None else np.ones_like(elevation, dtype=bool)
    elevation_masked = np.where(valid, elevation, np.nan)

    res_x = transform.a
    res_y = -transform.e  # e is negative for north-up rasters
    dz_dy, dz_dx = np.gradient(elevation_masked, res_y, res_x)
    slope_pct = np.sqrt(dz_dx**2 + dz_dy**2) * 100.0

    unreliable = ~np.isfinite(slope_pct) | ~valid | (slope_pct > MAX_PLAUSIBLE_SLOPE_PCT)
    slope_pct = np.where(unreliable, SLOPE_NODATA, slope_pct)

    profile.update(dtype="float32", nodata=SLOPE_NODATA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(slope_pct.astype(np.float32), 1)

    valid_slope = slope_pct[slope_pct != SLOPE_NODATA]
    print(
        f"Slope raster written to {output_path} -- "
        f"{len(valid_slope)}/{slope_pct.size} valid pixels, "
        f"mean {valid_slope.mean():.2f}%, max {valid_slope.max():.2f}%"
    )


def sample_declivity(lat: float, lon: float, slope_raster_path: Path) -> float | None:
    """Sample fine-grained (1m) declivity at a specific point, e.g. a
    detected ramp/curb-cut's location -- independent of and finer-grained
    than the coarse node-to-node edge `grade` already computed by
    `osm_extractor.py` (see module docstring: a short, steep curb ramp
    inside an otherwise-flat street segment is invisible to that edge-level
    average).

    Args:
        lat: WGS84 latitude of the feature (e.g. from the source image's GPS).
        lon: WGS84 longitude of the feature.
        slope_raster_path: Path to a raster written by `generate_slope_raster`.

    Returns:
        Slope in percent, or None if the point falls outside the raster's
        valid-data coverage.
    """
    with rasterio.open(slope_raster_path) as src:
        x, y = to_dem_crs(lat, lon, target_crs=str(src.crs))
        sample = next(src.sample([(x, y)]))
        value = float(sample[0])
    if value == SLOPE_NODATA:
        return None
    return value


def classify_ramp_compliance(slope_pct: float) -> str:
    """Classify a sampled ramp declivity against NBR 9050 guidance.

    Args:
        slope_pct: Declivity from `sample_declivity`.

    Returns:
        "compliant" if within NBR 9050's general ramp threshold, else
        "excessive" -- a coarse pass/fail, not the standard's full
        length-dependent table.
    """
    return "compliant" if slope_pct <= NBR9050_MAX_RAMP_SLOPE_PCT else "excessive"


def extract_geometric_attributes(
    image_result: dict[str, Any],
    image_lat: float,
    image_lon: float,
    slope_raster_path: Path | None = None,
) -> dict[str, Any]:
    """Derive geometric attributes for one image's segmentation result.

    Args:
        image_result: One entry from `run_segmentation_inference.py`'s
            output (`image_id`, `image_width`, `image_height`, `detections`).
        image_lat: WGS84 latitude the image was captured at (used as the
            ramp/curb-cut location proxy -- these features are typically
            within a few meters of the capture point).
        image_lon: WGS84 longitude the image was captured at.
        slope_raster_path: Path to the slope raster (see
            `generate_slope_raster`); ramp declivity is skipped if omitted.

    Returns:
        `{"width_bucket": ..., "ramp_declivity_pct": ..., "ramp_compliance": ...}`,
        any key omitted if it could not be estimated.
    """
    attributes: dict[str, Any] = {}
    width_px = image_result["image_width"]
    height_px = image_result["image_height"]

    sidewalk_widths = [
        estimate_width_bucket(det["polygon_xy"], width_px, height_px)
        for det in image_result["detections"]
        if det["class_name"] in ("Sidewalk", "Pedestrian Area")
    ]
    sidewalk_widths = [w for w in sidewalk_widths if w is not None]
    if sidewalk_widths:
        # pessimistic aggregation within a single image too: narrowest observed wins
        order = ["under_50cm", "50_to_90cm", "over_90cm"]
        attributes["width_bucket"] = min(sidewalk_widths, key=order.index)

    has_ramp_feature = any(det["class_name"] == "Curb Cut" for det in image_result["detections"])
    if has_ramp_feature and slope_raster_path is not None:
        declivity = sample_declivity(image_lat, image_lon, slope_raster_path)
        if declivity is not None:
            attributes["ramp_declivity_pct"] = round(declivity, 2)
            attributes["ramp_compliance"] = classify_ramp_compliance(declivity)

    return attributes
