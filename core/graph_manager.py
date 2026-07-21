"""Builds the routing-condition graphs and the origin-destination sample the
Phase 2 experiment compares.

The controlled comparison hinges on one invariant: every condition is the SAME
graph -- same nodes, edges, topology, length, grade, surface, steps -- and
differs ONLY in the two imagery-derived attributes (`ramp_present`,
`fixed_obstacle_present`). So a condition is fully described by two independent
choices, a *source* for each of those two attributes:

- ``baseline``  -- no imagery: ramp -> 'absent' (imputed), obstacle -> 'unknown'
  (imputed). Reproduces the OSM-alone column of METHODOLOGY SS6.
- ``validated`` -- hand-validated ground truth (SS9.4): use `<attr>_validated_value`
  where it exists (real evidence, imputed=False); every uncertain/untouched edge
  falls back to the pessimistic imputed default. Raw model output is NOT trusted.
- ``raw``       -- raw post-Barrier imagery exactly as the fused graph carries it.

The five named conditions are then just combinations of those two sources, which
makes the ablation fall out for free:

    baseline       = (ramp=baseline,  obstacle=baseline)
    full_validated = (ramp=validated, obstacle=validated)   <- headline
    full_raw       = (ramp=raw,       obstacle=raw)          <- detection-error probe
    ramp_only      = (ramp=validated, obstacle=baseline)     <- ablation
    obstacle_only  = (ramp=baseline,  obstacle=validated)    <- ablation

Every built graph carries a per-edge float ``cost`` from `impedance_model.edge_cost`,
ready for the Dijkstra wrapper. Guard assertions here are load-bearing: they are
the same "verify against the real data, never trust by construction" discipline
the SS7/SS8 audits established.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import osmnx as ox

from core.impedance_model import ExperimentConfig, WHEELCHAIR_PROFILE, edge_cost
from data_pipeline.edge_attribute_fusion import load_fused_graph
from data_pipeline.imputation_engine import ABSENT, PRESENT, UNKNOWN
from data_pipeline.spatial_utils import to_dem_crs

DATA_DIR = Path("data_files")
FUSED_GRAPH_PATH = DATA_DIR / "lourdes_graph_fused.graphml"
VALIDATED_GRAPH_PATH = DATA_DIR / "lourdes_graph_validated.graphml"
BASELINE_GRAPH_PATH = DATA_DIR / "lourdes_graph_baseline.graphml"

# The two imagery-derived attributes -- the ONLY thing that varies by condition.
RAMP = "ramp_present"
OBSTACLE = "fixed_obstacle_present"

# Named conditions -> (ramp_source, obstacle_source). See module docstring.
CONDITIONS: dict[str, tuple[str, str]] = {
    "baseline": ("baseline", "baseline"),
    "full_validated": ("validated", "validated"),
    "full_raw": ("raw", "raw"),
    "ramp_only": ("validated", "baseline"),
    "obstacle_only": ("baseline", "validated"),
}

# Expected hand-validated coverage (SS9.4) -- asserted as a join guard.
EXPECTED_RAMP_VALIDATED_EDGES = 260
EXPECTED_OBSTACLE_VALIDATED_EDGES = 294


@dataclass(frozen=True)
class Landmark:
    """A named place used as an illustrative OD endpoint, in WGS84."""

    name: str
    lat: float
    lon: float


# Real landmarks in / bordering the Lourdes pilot area (central Belo Horizonte).
# Used ONLY for the illustrative route-overlay figures -- never for the headline
# statistics, which come from the reproducible stratified random sample below.
# Each is snapped to its nearest graph node with the snap distance reported, so
# any coordinate that lands off-graph is transparent, not hidden.
LANDMARKS: tuple[Landmark, ...] = (
    Landmark("Praca Raul Soares", -19.92470, -43.94320),
    Landmark("Hospital Mater Dei", -19.93700, -43.94450),
    Landmark("Praca da Assembleia", -19.93300, -43.94100),
    Landmark("Igreja de Lourdes", -19.93000, -43.94200),
    Landmark("Av. do Contorno x Olegario Maciel", -19.92600, -43.94500),
)


# --------------------------------------------------------------------------- #
# Condition graph construction
# --------------------------------------------------------------------------- #
def _validated_values(attr: str) -> dict[tuple[Any, Any, int], str]:
    """Map each directed edge to its hand-validated value for `attr`, if any.

    Keys on the presence of `<attr>_validated_value` (a clean 'present'/'absent'
    string) rather than the `<attr>_validated` flag, which round-trips as the
    truthy string 'True' and would silently mark every edge validated (SS7 #10,
    SS11.6). Only edges that were actually adjudicated appear in the result.
    """
    g = load_fused_graph(VALIDATED_GRAPH_PATH)
    key = f"{attr}_validated_value"
    return {
        (u, v, k): d[key]
        for u, v, k, d in g.edges(keys=True, data=True)
        if d.get(key) in (ABSENT, PRESENT)
    }


def _apply_source(
    graph: nx.MultiDiGraph,
    attr: str,
    source: str,
    validated: dict[tuple[Any, Any, int], str],
    absent_default: str,
) -> None:
    """Overwrite `attr` (and its `_imputed` flag) on every edge per `source`.

    - ``raw``: leave the fused graph's imagery value untouched.
    - ``baseline``: force the pessimistic imputed default (imputed=True).
    - ``validated``: use the adjudicated value where present (imputed=False);
      otherwise fall back to the pessimistic default (imputed=True).

    The `_imputed` flag is set as a real bool because `impedance_model.ramp_factor`
    keys on `is False` -- getting this wrong is exactly the string-'False'
    truthiness trap, so it is set explicitly here, never inherited as a string.
    """
    imputed_key = f"{attr}_imputed"
    for u, v, k, d in graph.edges(keys=True, data=True):
        if source == "raw":
            continue
        if source == "baseline":
            d[attr] = absent_default
            d[imputed_key] = True
            continue
        # source == "validated"
        val = validated.get((u, v, k))
        if val is not None:
            d[attr] = val
            d[imputed_key] = False
        else:
            d[attr] = absent_default
            d[imputed_key] = True


def build_condition_graph(
    condition: str, config: ExperimentConfig = WHEELCHAIR_PROFILE
) -> nx.MultiDiGraph:
    """Return a fresh graph for `condition` with a float ``cost`` on every edge.

    Starts from the fused graph (which already carries every OSM/DEM attribute
    plus raw imagery) and overrides only `ramp_present` / `fixed_obstacle_present`
    according to the condition's two sources, then stamps `edge_cost`.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected one of {list(CONDITIONS)}")
    ramp_source, obstacle_source = CONDITIONS[condition]

    graph = load_fused_graph(FUSED_GRAPH_PATH)
    ramp_validated = _validated_values(RAMP) if "validated" in (ramp_source,) else {}
    obstacle_validated = _validated_values(OBSTACLE) if "validated" in (obstacle_source,) else {}

    _apply_source(graph, RAMP, ramp_source, ramp_validated, absent_default=ABSENT)
    _apply_source(graph, OBSTACLE, obstacle_source, obstacle_validated, absent_default=UNKNOWN)

    for _u, _v, _k, d in graph.edges(keys=True, data=True):
        d["cost"] = edge_cost(d, config)
    return graph


def build_all_conditions(
    config: ExperimentConfig = WHEELCHAIR_PROFILE,
) -> dict[str, nx.MultiDiGraph]:
    """Build every named condition graph. Runs the coverage guards once."""
    graphs = {name: build_condition_graph(name, config) for name in CONDITIONS}
    _assert_guards(graphs)
    return graphs


def load_ground_truth_graph() -> nx.MultiDiGraph:
    """The validated graph, for scoring routes against hand-checked ground truth.

    Kept separate from the cost graphs on purpose: metric #2 must read validated
    hazards WITHOUT ever touching the impedance weights, so improvement is
    measured independently of the model that produced the routes.
    """
    return load_fused_graph(VALIDATED_GRAPH_PATH)


# --------------------------------------------------------------------------- #
# Guard assertions -- verified, not trusted
# --------------------------------------------------------------------------- #
def _assert_guards(graphs: dict[str, nx.MultiDiGraph]) -> None:
    """Fail loudly if any condition graph violates a known-true invariant."""
    base = graphs["baseline"]
    # 1. Baseline is pure OSM+DEM: no ramp evidence, no obstacle evidence.
    assert all(d[RAMP] == ABSENT for *_, d in base.edges(keys=True, data=True)), (
        "baseline ramp_present must be all 'absent'"
    )
    assert all(d[OBSTACLE] == UNKNOWN for *_, d in base.edges(keys=True, data=True)), (
        "baseline fixed_obstacle_present must be all 'unknown'"
    )
    # 2. Every condition shares one connected topology and non-negative costs.
    for name, g in graphs.items():
        assert nx.number_weakly_connected_components(g) == 1, f"{name}: graph not connected"
        assert all(d["cost"] >= 0 for *_, d in g.edges(keys=True, data=True)), (
            f"{name}: negative edge cost"
        )
    # 3. Validated coverage matches SS9.4 exactly (join guard).
    fv = graphs["full_validated"]
    n_ramp = sum(1 for *_, d in fv.edges(keys=True, data=True) if d[f"{RAMP}_imputed"] is False)
    n_obs = sum(1 for *_, d in fv.edges(keys=True, data=True) if d[f"{OBSTACLE}_imputed"] is False)
    assert n_ramp == EXPECTED_RAMP_VALIDATED_EDGES, (
        f"expected {EXPECTED_RAMP_VALIDATED_EDGES} validated ramp edges, got {n_ramp}"
    )
    assert n_obs == EXPECTED_OBSTACLE_VALIDATED_EDGES, (
        f"expected {EXPECTED_OBSTACLE_VALIDATED_EDGES} validated obstacle edges, got {n_obs}"
    )


# --------------------------------------------------------------------------- #
# OD-pair sampling
# --------------------------------------------------------------------------- #
def straight_line_distance(graph: nx.MultiDiGraph, u: Any, v: Any) -> float:
    """Euclidean distance in metres between two nodes (graph is projected UTM)."""
    a, b = graph.nodes[u], graph.nodes[v]
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def sample_od_pairs(
    graph: nx.MultiDiGraph,
    n_pairs: int = 1000,
    n_strata: int = 4,
    seed: int = 42,
) -> list[tuple[Any, Any]]:
    """Draw `n_pairs` distinct node pairs stratified by straight-line distance.

    The pairwise-distance distribution over all nodes is split into `n_strata`
    equal-count bands; pairs are drawn by seeded rejection sampling until each
    band holds ~`n_pairs / n_strata`. Stratifying stops trivially-short adjacent
    pairs from dominating the cost distribution -- the failure mode of naive
    all-pairs enumeration -- while staying fully reproducible under `seed`.
    """
    rng = random.Random(seed)
    nodes = list(graph.nodes())

    # Band thresholds from a large seeded sample of the pairwise-distance CDF
    # (exact all-pairs is ~38k here and cheap, but sampling keeps this O(pairs)).
    probe = _sample_distances(graph, nodes, rng, n=5000)
    probe.sort()
    # Interior band boundaries at the equal-count quantiles of the distance CDF.
    thresholds = [probe[min(int(q * len(probe)), len(probe) - 1)] for q in _quantile_edges(n_strata)[1:-1]]

    # Per-band quotas summing to EXACTLY n_pairs: distribute any remainder across
    # the first bands so an indivisible n_pairs is not silently under-sampled.
    base = n_pairs // n_strata
    remainder = n_pairs % n_strata
    band_targets = [base + (1 if i < remainder else 0) for i in range(n_strata)]
    target = n_pairs

    bands: list[list[tuple[Any, Any]]] = [[] for _ in range(n_strata)]
    seen: set[tuple[Any, Any]] = set()

    # Rejection-sample until every band hits its quota (or a generous attempt cap).
    attempts = 0
    max_attempts = target * 500
    while sum(len(b) for b in bands) < target and attempts < max_attempts:
        attempts += 1
        u, v = rng.sample(nodes, 2)
        key = (u, v) if str(u) <= str(v) else (v, u)
        if key in seen:
            continue
        d = straight_line_distance(graph, u, v)
        band = _band_index(d, thresholds)
        if len(bands[band]) >= band_targets[band]:
            continue
        seen.add(key)
        bands[band].append((u, v))

    pairs = [p for band in bands for p in band]
    rng.shuffle(pairs)
    return pairs


def _sample_distances(
    graph: nx.MultiDiGraph, nodes: list[Any], rng: random.Random, n: int
) -> list[float]:
    out = []
    for _ in range(n):
        u, v = rng.sample(nodes, 2)
        out.append(straight_line_distance(graph, u, v))
    return out


def _quantile_edges(n_strata: int) -> list[float]:
    return [i / n_strata for i in range(n_strata + 1)]


def _band_index(distance: float, thresholds: list[float]) -> int:
    for i, t in enumerate(thresholds):
        if distance < t:
            return i
    return len(thresholds)


# --------------------------------------------------------------------------- #
# Landmark resolution (illustrative figures only)
# --------------------------------------------------------------------------- #
def resolve_landmarks(
    graph: nx.MultiDiGraph, landmarks: Iterable[Landmark] = LANDMARKS
) -> list[tuple[Landmark, Any, float]]:
    """Snap each landmark to its nearest graph node, reporting the snap distance.

    Returns (landmark, node_id, snap_distance_m). The landmark's WGS84 point is
    reprojected into the graph's UTM CRS via `spatial_utils.to_dem_crs` before
    the nearest-node search, so degree/metre coordinates are never mixed -- the
    same reprojection discipline the fusion spatial join uses.
    """
    resolved = []
    for lm in landmarks:
        x, y = to_dem_crs(lm.lat, lm.lon)
        node = ox.distance.nearest_nodes(graph, X=x, Y=y)
        snap = math.hypot(graph.nodes[node]["x"] - x, graph.nodes[node]["y"] - y)
        resolved.append((lm, node, snap))
    return resolved


__all__ = [
    "CONDITIONS",
    "Landmark",
    "LANDMARKS",
    "build_condition_graph",
    "build_all_conditions",
    "load_ground_truth_graph",
    "sample_od_pairs",
    "straight_line_distance",
    "resolve_landmarks",
    "FUSED_GRAPH_PATH",
    "VALIDATED_GRAPH_PATH",
    "BASELINE_GRAPH_PATH",
]
