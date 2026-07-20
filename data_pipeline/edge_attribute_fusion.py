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

import argparse
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import osmnx as ox

from data_pipeline.geometric_attribute_extractor import WIDTH_BUCKET_ORDER, extract_geometric_attributes
from data_pipeline.imputation_engine import (
    ABSENT,
    IMPUTATION_POLICY,
    PRESENT,
    impute_missing_attributes,
    summarize_imputation_rate,
)
from data_pipeline.osm_extractor import carregar_grafo, highway_values, salvar_grafo
from data_pipeline.semantic_segmentation_training import MapillaryImageEntry, parse_mapillary_metadata
from data_pipeline.spatial_utils import to_dem_crs

# ox.save_graphml/load_graphml round-trips every attribute as a string by
# design (OSM tag values are inherently heterogeneous) and requires the
# caller to specify dtypes to restore them -- otherwise e.g. the boolean
# `False` comes back as the Python string "False", which is truthy
# (`bool("False") is True`), silently inverting every `_imputed` flag for
# any downstream code (the future impedance model included) that reads the
# saved graph and trusts its own type system. `ox.io._convert_bool_string`
# is OSMnx's own converter for exactly this trap.
FUSED_EDGE_DTYPES: dict[str, Any] = {"n_images_observed": int, "ramp_declivity_pct": float}
for _attr in IMPUTATION_POLICY:
    FUSED_EDGE_DTYPES[f"{_attr}_imputed"] = ox.io._convert_bool_string


def load_fused_graph(path: Path) -> nx.MultiDiGraph:
    """Load a graph written by `fuse_edge_attributes` with correct attribute types.

    Args:
        path: GraphML file written via `osm_extractor.salvar_grafo` after fusion.

    Returns:
        The graph with `n_images_observed` (int), `ramp_declivity_pct`
        (float), and every `<attr>_imputed` flag (real bool) restored --
        plain `carregar_grafo` leaves these as unconverted strings.
    """
    return ox.io.load_graphml(filepath=path, edge_dtypes=FUSED_EDGE_DTYPES)

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
TIER_ORDER = ["poor", "fair", "good"]  # worst-first, for pessimistic selection


def _tag_values(raw: Any) -> list[Any]:
    """Normalize any OSM tag value into a list, exactly like
    `osm_extractor.highway_values` does for `highway`: OSMnx's simplify=True
    merges consecutive way segments and stores disagreeing tags as a *list*.
    The Lourdes graph has 12 real list-valued `highway` edges today; the same
    merge can produce a list for `surface`/`smoothness`/`kerb`/etc. after any
    future OSM edit + weekly refresh. Before this helper existed,
    `surface in OSM_SURFACE_TO_TIER` raised TypeError (lists are unhashable)
    on such an edge -- killing the whole fusion run -- and the scalar `==`
    comparisons for tactile_paving/kerb/handrail silently dropped the evidence.
    """
    if raw is None:
        return []
    return raw if isinstance(raw, list) else [raw]


def _first_tag(*raws: Any) -> Any | None:
    """First non-empty tag value across edge/node candidates, preserving the
    edge > node-u > node-v priority the scalar `or`-chain used to encode."""
    for raw in raws:
        if _tag_values(raw):
            return raw
    return None


def osm_tags_to_canonical(
    edge_data: dict[str, Any],
    node_u_data: dict[str, Any] | None = None,
    node_v_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert an edge's raw OSM tags into the canonical attribute schema.

    Args:
        edge_data: Edge attribute dict from the graph (see
            `osm_extractor.TAGS_ACESSIBILIDADE_WAY` for which tags exist).
        node_u_data: Attribute dict for the edge's start node, if available.
        node_v_data: Attribute dict for the edge's end node, if available.

    Returns:
        Canonical attribute dict, keys omitted (not None) where OSM has no
        relevant tag at all -- `impute_missing_attributes` treats missing
        keys and None identically, so omitting is equivalent and clearer.
    """
    # `crossing`, `kerb`, and `tactile_paving` are frequently tagged on the
    # NODE where a footway meets a road (see osm_extractor.TAGS_ACESSIBILIDADE_NODE),
    # not on the way itself -- checking only edge_data here silently missed
    # every node-level tag. Concretely: 41/278 Lourdes nodes carry a
    # `crossing` tag, but marked_crossing_present's OSM contribution was
    # measuring exactly 0% before this fix, because it only ever looked at
    # edges. A crossing at either endpoint is evidence for the edge that
    # leads to/from it.
    node_u_data = node_u_data or {}
    node_v_data = node_v_data or {}

    canonical: dict[str, Any] = {}

    # Every tag below goes through _tag_values so a list-valued tag (OSMnx's
    # merged-segment representation, see the helper's docstring) is handled;
    # when merged segments disagree, the pessimistic value wins -- the same
    # worst-case-wins rule `aggregate_image_evidence` applies to imagery.
    surface_tiers = [
        OSM_SURFACE_TO_TIER[s] for s in _tag_values(edge_data.get("surface")) if s in OSM_SURFACE_TO_TIER
    ]
    if surface_tiers:
        canonical["surface_material_tier"] = min(surface_tiers, key=TIER_ORDER.index)

    smoothness_tiers = [
        OSM_SMOOTHNESS_TO_TIER[s] for s in _tag_values(edge_data.get("smoothness")) if s in OSM_SMOOTHNESS_TO_TIER
    ]
    if smoothness_tiers:
        canonical["smoothness_tier"] = min(smoothness_tiers, key=TIER_ORDER.index)

    widths_m = []
    for width in _tag_values(edge_data.get("width")):
        try:
            widths_m.append(float(width))
        except (TypeError, ValueError):
            pass  # non-numeric width tag (e.g. "narrow") -- not a value we can bucket
    if widths_m:
        width_m = min(widths_m)  # pessimistic: narrowest merged segment wins
        canonical["width_bucket"] = (
            "under_50cm" if width_m < 0.5 else "50_to_90cm" if width_m < 0.9 else "over_90cm"
        )

    tactile = _tag_values(
        _first_tag(edge_data.get("tactile_paving"), node_u_data.get("tactile_paving"), node_v_data.get("tactile_paving"))
    )
    if "no" in tactile:  # pessimistic: any merged segment without it -> not fully present
        canonical["tactile_paving_present"] = ABSENT
    elif "yes" in tactile:
        canonical["tactile_paving_present"] = PRESENT

    handrail = _tag_values(edge_data.get("handrail"))
    if "no" in handrail:
        canonical["handrail_present"] = ABSENT
    elif "yes" in handrail:
        canonical["handrail_present"] = PRESENT

    kerb = _tag_values(_first_tag(edge_data.get("kerb"), node_u_data.get("kerb"), node_v_data.get("kerb")))
    if "raised" in kerb:  # pessimistic: a raised kerb anywhere along the edge blocks it
        canonical["ramp_present"] = ABSENT
    elif any(k in ("lowered", "flush") for k in kerb):
        canonical["ramp_present"] = PRESENT

    if (
        edge_data.get("crossing") is not None
        or node_u_data.get("crossing") is not None
        or node_v_data.get("crossing") is not None
    ):
        canonical["marked_crossing_present"] = PRESENT

    if (
        edge_data.get("barrier") is not None
        or node_u_data.get("barrier") is not None
        or node_v_data.get("barrier") is not None
    ):
        canonical["fixed_obstacle_present"] = PRESENT

    # highway_values() handles OSMnx's list-valued tags from merged way
    # segments -- a plain `edge_data.get("highway") == "steps"` silently
    # missed every real staircase in this graph (see highway_values' docstring).
    hw_values = highway_values(edge_data.get("highway"))
    non_steps_types = {"footway", "path", "pedestrian", "residential", "living_street"}
    if "steps" in hw_values:
        canonical["steps_present"] = PRESENT
    elif hw_values and all(h in non_steps_types for h in hw_values):
        # every merged segment is a recognized non-steps type -- distinct
        # from "we never checked", which is what an untagged/unrecognized
        # highway value means
        canonical["steps_present"] = ABSENT

    return canonical


# An image farther than this from its nearest edge is discarded, not snapped.
# The Mapillary fetch covers the *rectangular* geocoded bbox of Lourdes, but
# the graph covers the neighborhood polygon -- measured on the real dataset,
# the snap-distance distribution is bimodal: 2,179/3,689 images within 25m
# (on-graph streets: GPS error plus half a street width from the edge
# centerline), a thin trough at 25-50m (233 images), then a second population
# of 1,277 images at 50-501m, 83% of which lie outside the graph's convex
# hull entirely -- photos of *other neighborhoods'* streets inside the bbox
# corners. Without this cutoff those far images attached their detections to
# whatever boundary edge happened to be nearest (81 edges received evidence
# from >50m away), fabricating attribute evidence from streets the edge
# doesn't represent. 25m sits in the measured trough between the two
# populations.
MAX_SNAP_DISTANCE_M = 25.0


def snap_images_to_edges(
    graph: nx.MultiDiGraph,
    entries: list[MapillaryImageEntry],
    max_snap_distance_m: float = MAX_SNAP_DISTANCE_M,
) -> dict[tuple[Any, Any, int], list[MapillaryImageEntry]]:
    """Assign each Mapillary image to its nearest graph edge.

    An image only counts if its nearest edge is within `max_snap_distance_m`
    (see that constant's rationale). Because street-level evidence describes
    the physical street segment, not a direction of travel, every image list
    is mirrored onto the reverse twin edge (v, u, key) when it exists --
    OSMnx's walk network stores each two-way segment as two directed edges
    with identical geometry, and `ox.distance.nearest_edges` arbitrarily
    returns only one of the two equidistant twins. Without mirroring, the
    same physical segment reported e.g. "obstacle present" in one walking
    direction and "unknown" in the other (measured on the real fused graph:
    245 of 416 twin pairs disagreed on n_images_observed).

    Args:
        graph: Projected graph (see `osm_extractor.injetar_topografia_e_calcular_esforco`).
        entries: Manifest entries with `latitude`/`longitude` in WGS84.
        max_snap_distance_m: Maximum image-to-edge distance for a valid snap.

    Returns:
        edge_id (u, v, key) -> list of entries snapped to that edge (or its twin).
    """
    graph_crs = graph.graph.get("crs")
    xs, ys = [], []
    for entry in entries:
        x, y = to_dem_crs(entry.latitude, entry.longitude, target_crs=str(graph_crs))
        xs.append(x)
        ys.append(y)

    nearest, distances = ox.distance.nearest_edges(graph, X=xs, Y=ys, return_dist=True)
    assignment: dict[tuple[Any, Any, int], list[MapillaryImageEntry]] = {}
    n_discarded = 0
    for entry, edge_id, distance in zip(entries, nearest, distances):
        if distance > max_snap_distance_m:
            n_discarded += 1
            continue
        assignment.setdefault(edge_id, []).append(entry)
    if n_discarded:
        print(
            f"      [INFO] Discarded {n_discarded}/{len(entries)} image(s) farther than "
            f"{max_snap_distance_m}m from any graph edge (outside-neighborhood imagery)."
        )

    mirrored: dict[tuple[Any, Any, int], list[MapillaryImageEntry]] = {}
    for edge_id, edge_entries in assignment.items():
        u, v, key = edge_id
        mirrored.setdefault(edge_id, []).extend(edge_entries)
        if graph.has_edge(v, u, key):
            mirrored.setdefault((v, u, key), []).extend(edge_entries)
    return mirrored


def aggregate_image_evidence(
    image_results_by_id: dict[str, dict[str, Any]],
    images_on_edge: list[MapillaryImageEntry],
    slope_raster_path: Path | None,
) -> dict[str, Any]:
    """Pessimistically aggregate every image snapped to one edge into a
    single evidence dict: presence attributes are OR'd (any image showing a
    barrier means the edge has it), width takes the narrowest bucket seen.

    NOTE on width: the pinhole width estimate is written under
    `width_bucket_uncalibrated_estimate`, NOT `width_bucket` -- it does not
    feed the canonical schema. A full-sample check after fusion (832 edges)
    found 90.9% of all real (non-imputed) width estimates landing in the
    single most extreme bucket, `under_50cm`, including on well-known wide
    boulevards (Avenida do Contorno, Rua da Bahia) that are not plausibly
    sub-50cm almost everywhere. Combined with dashcam watermarks observed
    during visual QA (a common wide-angle-lens source, ~140-160deg FOV vs.
    the assumed 65deg here), this points to a *systematic* bias, not just
    noise: an assumed FOV that's too narrow overestimates focal length,
    which underestimates real-world size per pixel, which shrinks every
    width estimate in the same direction. The 57.8%-cross-method-agreement
    validation (see docs/METHODOLOGY.md) measured *consistency* between two
    methods, not *calibration* against reality -- two methods can agree
    while sharing the same directional bias, which is what happened here.
    Guessing a new FOV without real per-image calibration data would just
    swap one unvalidated assumption for another, so `width_bucket` instead
    falls through to OSM (0% tag coverage) -> imputation's neutral default
    until real camera calibration or field validation is available -- an
    honestly-unknown value is safer to route on than a confidently wrong one.

    Args:
        image_results_by_id: image_id -> that image's inference result.
        images_on_edge: Manifest entries snapped to this edge.
        slope_raster_path: Passed through to `extract_geometric_attributes`.

    Returns:
        Canonical attribute dict for this edge, keys present only where at
        least one image contributed evidence.
    """
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
                evidence[attr] = PRESENT
        if "Curb Cut" in detected_classes:
            evidence.setdefault("ramp_present", PRESENT)

        geo_attrs = extract_geometric_attributes(
            result, entry.latitude, entry.longitude, slope_raster_path
        )
        width = geo_attrs.get("width_bucket_uncalibrated_estimate")
        if width and (
            narrowest_width is None
            or WIDTH_BUCKET_ORDER.index(width) < WIDTH_BUCKET_ORDER.index(narrowest_width)
        ):
            narrowest_width = width
        if "ramp_declivity_pct" in geo_attrs:
            ramp_declivities.append(geo_attrs["ramp_declivity_pct"])

    if narrowest_width:
        evidence["width_bucket_uncalibrated_estimate"] = narrowest_width
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
        osm_attrs = osm_tags_to_canonical(edge_data, graph.nodes[u], graph.nodes[v])
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


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fuse OSM tags + Mapillary imagery + imputation into the final edge attribute graph."
    )
    parser.add_argument("--input-graph", type=Path, required=True, help="Output of osm_extractor (or refresh.py).")
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--predictions-json", type=Path, required=True)
    parser.add_argument("--slope-raster", type=Path, default=None)
    parser.add_argument("--city-filter", type=str, default=None)
    parser.add_argument("--output-graph", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser


def main() -> None:
    """CLI entrypoint -- the missing driver that regenerates
    `data_files/lourdes_graph_fused.graphml` and `lourdes_edges_fused.csv`
    from scratch. Prior to this, those files existed only as the product of
    ad hoc interactive calls during development, with no committed script
    able to reproduce them -- exactly the kind of reproducibility gap this
    project's own emphasis on verifying everything against real execution
    argues against leaving in place.
    """
    args = _build_cli_parser().parse_args()

    graph = carregar_grafo(args.input_graph)
    entries = parse_mapillary_metadata(
        metadata_csv=args.metadata_csv, train_ratio=0.85, city_filter=args.city_filter
    )
    fused = fuse_edge_attributes(
        graph,
        predictions_json=args.predictions_json,
        manifest_entries=entries,
        slope_raster_path=args.slope_raster,
    )

    salvar_grafo(fused, args.output_graph)
    if args.output_csv:
        export_edges_geodataframe(fused).to_csv(args.output_csv, index=False)
        print(f"Exported edges to {args.output_csv}")

    rows = [d for _, _, d in fused.edges(data=True)]
    print(f"\nImputation rate ({len(rows)} edges):")
    for attr, pct in summarize_imputation_rate(rows).items():
        print(f"  {attr}: {pct}%")


if __name__ == "__main__":
    main()
