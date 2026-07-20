"""Fuses OSM tags, Mapillary-derived accessibility signals, and DEM-based
declivity into one unified per-edge attribute table.

Merge priority per attribute, in order: OSM tag (official, if present) >
image-derived signal (Mapillary segmentation + geometric heuristics) >
`imputation_engine`'s pessimistic default. This is the final data contract
the future routing phase (`core/impedance_model.py`, out of scope here)
consumes -- kept attribute-level and additive rather than tied to one
specific cost formula, since the formula itself is only a baseline.

Spatial join: every Mapillary image is snapped to its nearest graph edge
via `osmnx.distance.nearest_edges`, using `spatial_utils` to reproject the
image's WGS84 point into the graph's own projected CRS first -- mixing
degree-based and meter-based coordinates in the same nearest-neighbor
search is the same class of silent bug flagged in `geometric_attribute_extractor`
and `osm_extractor`, just at the join step instead of the sampling step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import osmnx as ox

from data_pipeline.geometric_attribute_extractor import extract_geometric_attributes
from data_pipeline.imputation_engine import impute_missing_attributes
from data_pipeline.semantic_segmentation_training import MapillaryImageEntry
from data_pipeline.spatial_utils import to_dem_crs

# Segmentation class -> canonical attribute this image contributes evidence for.
CLASS_TO_PRESENCE_ATTR = {
    "Curb Cut": "ramp_present",
    "Crosswalk - Plain": "marked_crossing_present",
    "Lane Marking - Crosswalk": "marked_crossing_present",
    "Barrier": "fixed_obstacle_present",
    "Bench": "fixed_obstacle_present",
    "Bike Rack": "fixed_obstacle_present",
    "Fire Hydrant": "fixed_obstacle_present",
    "Mailbox": "fixed_obstacle_present",
    "Manhole": "fixed_obstacle_present",
    "Phone Booth": "fixed_obstacle_present",
    "Pothole": "fixed_obstacle_present",
    "Trash Can": "fixed_obstacle_present",
}

# OSM surface/smoothness raw values -> canonical tier. Anything not listed
# stays unmapped (falls through to imputation) rather than guessing.
OSM_SURFACE_TO_TIER = {
    "asphalt": "good", "concrete": "good", "paving_stones": "good", "paved": "good",
    "sett": "fair", "compacted": "fair", "fine_gravel": "fair",
    "cobblestone": "poor", "unpaved": "poor", "dirt": "poor", "ground": "poor",
    "gravel": "poor", "grass": "poor", "sand": "poor", "earth": "poor",
}
OSM_SMOOTHNESS_TO_TIER = {
    "excellent": "good", "good": "good",
    "intermediate": "fair",
    "bad": "poor", "very_bad": "poor", "horrible": "poor",
    "very_horrible": "poor", "impassable": "poor",
}


def osm_tags_to_canonical(edge_data: dict[str, Any]) -> dict[str, Any]:
    """Convert an edge's raw OSM tags into the canonical attribute schema.

    Args:
        edge_data: Edge attribute dict from the graph (see
            `osm_extractor.TAGS_ACESSIBILIDADE_WAY` for which tags exist).

    Returns:
        Canonical attribute dict, keys omitted (not None) where OSM has no
        relevant tag at all -- `impute_missing_attributes` treats missing
        keys and None identically, so omitting is equivalent and clearer.
    """
    canonical: dict[str, Any] = {}

    surface = edge_data.get("surface")
    if surface in OSM_SURFACE_TO_TIER:
        canonical["surface_material_tier"] = OSM_SURFACE_TO_TIER[surface]

    smoothness = edge_data.get("smoothness")
    if smoothness in OSM_SMOOTHNESS_TO_TIER:
        canonical["smoothness_tier"] = OSM_SMOOTHNESS_TO_TIER[smoothness]

    width = edge_data.get("width")
    if width is not None:
        try:
            width_m = float(width)
            canonical["width_bucket"] = (
                "under_50cm" if width_m < 0.5 else "50_to_90cm" if width_m < 0.9 else "over_90cm"
            )
        except (TypeError, ValueError):
            pass  # non-numeric width tag (e.g. "narrow") -- not a value we can bucket

    if edge_data.get("tactile_paving") in ("yes",):
        canonical["tactile_paving_present"] = "present"
    elif edge_data.get("tactile_paving") in ("no",):
        canonical["tactile_paving_present"] = "absent"

    if edge_data.get("handrail") in ("yes",):
        canonical["handrail_present"] = "present"
    elif edge_data.get("handrail") in ("no",):
        canonical["handrail_present"] = "absent"

    kerb = edge_data.get("kerb")
    if kerb in ("lowered", "flush"):
        canonical["ramp_present"] = "present"
    elif kerb == "raised":
        canonical["ramp_present"] = "absent"

    if edge_data.get("crossing") is not None:
        canonical["marked_crossing_present"] = "present"

    if edge_data.get("barrier") is not None:
        canonical["fixed_obstacle_present"] = "present"

    if edge_data.get("highway") == "steps":
        canonical["steps_present"] = "present"
    elif edge_data.get("highway") in ("footway", "path", "pedestrian", "residential", "living_street"):
        # a way tagged as one of these ordinary pedestrian types is, by
        # definition, not itself a flight of steps -- distinct from "we
        # never checked", which is what an untagged/other highway value means
        canonical["steps_present"] = "absent"

    return canonical


def snap_images_to_edges(
    graph: nx.MultiDiGraph, entries: list[MapillaryImageEntry]
) -> dict[tuple[Any, Any, int], list[MapillaryImageEntry]]:
    """Assign each Mapillary image to its nearest graph edge.

    Args:
        graph: Projected graph (see `osm_extractor.injetar_topografia_e_calcular_esforco`).
        entries: Manifest entries with `latitude`/`longitude` in WGS84.

    Returns:
        edge_id (u, v, key) -> list of entries snapped to that edge.
    """
    graph_crs = graph.graph.get("crs")
    xs, ys = [], []
    for entry in entries:
        x, y = to_dem_crs(entry.latitude, entry.longitude, target_crs=str(graph_crs))
        xs.append(x)
        ys.append(y)

    nearest = ox.distance.nearest_edges(graph, X=xs, Y=ys)
    assignment: dict[tuple[Any, Any, int], list[MapillaryImageEntry]] = {}
    for entry, edge_id in zip(entries, nearest):
        assignment.setdefault(edge_id, []).append(entry)
    return assignment


def aggregate_image_evidence(
    image_results_by_id: dict[str, dict[str, Any]],
    images_on_edge: list[MapillaryImageEntry],
    slope_raster_path: Path | None,
) -> dict[str, Any]:
    """Pessimistically aggregate every image snapped to one edge into a
    single evidence dict: presence attributes are OR'd (any image showing a
    barrier means the edge has it), width takes the narrowest bucket seen.

    Args:
        image_results_by_id: image_id -> that image's inference result.
        images_on_edge: Manifest entries snapped to this edge.
        slope_raster_path: Passed through to `extract_geometric_attributes`.

    Returns:
        Canonical attribute dict for this edge, keys present only where at
        least one image contributed evidence.
    """
    width_order = ["under_50cm", "50_to_90cm", "over_90cm"]
    evidence: dict[str, Any] = {}
    narrowest_width: str | None = None
    ramp_declivities: list[float] = []

    for entry in images_on_edge:
        result = image_results_by_id.get(entry.image_id)
        if result is None:
            continue

        detected_classes = {d["class_name"] for d in result["detections"]}
        for class_name in detected_classes:
            attr = CLASS_TO_PRESENCE_ATTR.get(class_name)
            if attr:
                evidence[attr] = "present"
        if "Curb Cut" in detected_classes:
            evidence.setdefault("ramp_present", "present")

        geo_attrs = extract_geometric_attributes(
            result, entry.latitude, entry.longitude, slope_raster_path
        )
        width = geo_attrs.get("width_bucket")
        if width and (narrowest_width is None or width_order.index(width) < width_order.index(narrowest_width)):
            narrowest_width = width
        if "ramp_declivity_pct" in geo_attrs:
            ramp_declivities.append(geo_attrs["ramp_declivity_pct"])

    if narrowest_width:
        evidence["width_bucket"] = narrowest_width
    if ramp_declivities:
        evidence["ramp_declivity_pct"] = max(ramp_declivities)  # worst-case = steepest

    return evidence


def fuse_edge_attributes(
    graph: nx.MultiDiGraph,
    predictions_json: Path,
    manifest_entries: list[MapillaryImageEntry],
    slope_raster_path: Path | None = None,
) -> nx.MultiDiGraph:
    """Fuse OSM tags + image evidence + imputation into final edge attributes.

    Mutates and returns `graph` with every `IMPUTATION_POLICY` attribute set
    on every edge, plus `<attr>_imputed` / `<attr>_source` companions
    recording where each value actually came from.

    Args:
        graph: Projected graph with OSM tags + elevation/grade already set.
        predictions_json: Output of `run_segmentation_inference.py`.
        manifest_entries: The manifest those predictions were generated from.
        slope_raster_path: Path from `geometric_attribute_extractor.generate_slope_raster`.

    Returns:
        The same graph object, edges enriched in place.
    """
    image_results = json.loads(predictions_json.read_text(encoding="utf-8"))
    image_results_by_id = {r["image_id"]: r for r in image_results}

    entries_with_predictions = [e for e in manifest_entries if e.image_id in image_results_by_id]
    edge_assignment = snap_images_to_edges(graph, entries_with_predictions)

    for u, v, key, edge_data in graph.edges(keys=True, data=True):
        osm_attrs = osm_tags_to_canonical(edge_data)
        image_attrs = aggregate_image_evidence(
            image_results_by_id, edge_assignment.get((u, v, key), []), slope_raster_path
        )

        merged: dict[str, Any] = {}
        source: dict[str, str] = {}
        for attr in set(osm_attrs) | set(image_attrs):
            if attr in osm_attrs:
                merged[attr] = osm_attrs[attr]
                source[attr] = "osm"
            else:
                merged[attr] = image_attrs[attr]
                source[attr] = "imagery"

        final = impute_missing_attributes(merged)
        for key_name, value in final.items():
            edge_data[key_name] = value
        for attr, src in source.items():
            edge_data[f"{attr}_source"] = src
        edge_data["n_images_observed"] = len(edge_assignment.get((u, v, key), []))

    return graph


def export_edges_geodataframe(graph: nx.MultiDiGraph) -> gpd.GeoDataFrame:
    """Flatten the enriched graph's edges into a GeoDataFrame (CSV/GeoJSON-ready).

    Args:
        graph: Graph already processed by `fuse_edge_attributes`.

    Returns:
        One row per edge, all attributes as columns -- feeds the "heat map
        of Lourdes" deliverable and general debugging/inspection.
    """
    _, edges = ox.graph_to_gdfs(graph)
    return edges.reset_index()
