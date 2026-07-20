"""Idempotent weekly refresh: re-pull the OSM graph, re-fuse elevation and
accessibility tags, and diff against the previous snapshot so community
edits to OSM (the project's "VGI feedback loop" concept) are actually
absorbed over time -- not captured once and left stale.

Scoped to OSM + re-fusion only. Mapillary imagery is NOT re-segmented on
this cadence -- street-level scenes don't change week to week the way OSM
tags do from community edits; re-run `run_segmentation_inference.py`
manually/occasionally instead (e.g. when new Mapillary coverage appears).

Actual weekly scheduling (cron, or the `/schedule` skill) is a deliberate
follow-up action once this script has been run and checked manually -- not
wired up here. Recurring automated jobs deserve their own explicit go-ahead.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import networkx as nx

from data_pipeline.osm_extractor import (
    TAGS_ACESSIBILIDADE_WAY,
    carregar_grafo,
    diagnosticar_cobertura_tags,
    extrair_malha_pedestres,
    injetar_topografia_e_calcular_esforco,
    salvar_grafo,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data_files"
SNAPSHOTS_DIR = DATA_DIR / "graph_snapshots"
LATEST_GRAPH_PATH = DATA_DIR / "lourdes_graph_latest.graphml"
CHANGELOG_PATH = SNAPSHOTS_DIR / "changelog.md"


def _snapshot_path(date: datetime.date) -> Path:
    return SNAPSHOTS_DIR / f"lourdes_graph_{date.isoformat()}.graphml"


def diff_snapshots(old: nx.MultiDiGraph | None, new: nx.MultiDiGraph) -> list[str]:
    """Compare accessibility tags edge-by-edge between two graph snapshots.

    OSM node/way IDs are stable across re-downloads of an unedited feature,
    so matching edges by their (u, v, key) identity is reliable for
    detecting real tag changes -- it's exactly what changed under an OSM
    edit, not just what changed due to a different simplify() pass.

    Args:
        old: Previous snapshot, or None on the very first run.
        new: Freshly extracted snapshot.

    Returns:
        Human-readable change lines, e.g. "edge (123, 456, 0): surface
        None -> asphalt".
    """
    if old is None:
        return [f"Initial snapshot: {new.number_of_edges()} edges, nothing to compare against."]

    changes: list[str] = []
    new_edges = set(new.edges(keys=True))
    old_edges = set(old.edges(keys=True))

    for edge_id in sorted(new_edges - old_edges):
        changes.append(f"edge {edge_id}: new (not present in previous snapshot)")
    for edge_id in sorted(old_edges - new_edges):
        changes.append(f"edge {edge_id}: removed (was present in previous snapshot)")

    for edge_id in sorted(new_edges & old_edges):
        u, v, k = edge_id
        old_data = old.edges[u, v, k]
        new_data = new.edges[u, v, k]
        for tag in TAGS_ACESSIBILIDADE_WAY:
            old_val = old_data.get(tag)
            new_val = new_data.get(tag)
            if old_val != new_val:
                changes.append(f"edge {edge_id}: {tag} {old_val!r} -> {new_val!r}")

    return changes


def refresh_lourdes_graph(
    lugar: str = "Lourdes, Belo Horizonte, Minas Gerais, Brazil",
    caminho_dem: Path | None = None,
    today: datetime.date | None = None,
) -> Path:
    """Idempotent refresh entrypoint: re-pull OSM, re-fuse DEM, snapshot, diff.

    Safe to run repeatedly (e.g. weekly) -- each run produces a new dated
    snapshot and updates the "latest" pointer other modules read from,
    without needing any state beyond what's already on disk.

    Args:
        lugar: OSMnx-resolvable place name.
        caminho_dem: DEM path; defaults to data_files/lourdes_dem_1m.tif.
        today: Override for the snapshot date (testing only); defaults to
            the real current date.

    Returns:
        Path to the refreshed "latest" graph.
    """
    caminho_dem = caminho_dem or (DATA_DIR / "lourdes_dem_1m.tif")
    today = today or datetime.date.today()
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    grafo_bruto = extrair_malha_pedestres(lugar)
    grafo_novo = injetar_topografia_e_calcular_esforco(grafo_bruto, str(caminho_dem))
    diagnosticar_cobertura_tags(grafo_novo)

    grafo_antigo = carregar_grafo(LATEST_GRAPH_PATH) if LATEST_GRAPH_PATH.exists() else None
    changes = diff_snapshots(grafo_antigo, grafo_novo)

    salvar_grafo(grafo_novo, _snapshot_path(today))
    salvar_grafo(grafo_novo, LATEST_GRAPH_PATH)

    with CHANGELOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {today.isoformat()}\n\n")
        if changes:
            for line in changes:
                handle.write(f"- {line}\n")
        else:
            handle.write("- No accessibility tag changes since previous snapshot.\n")

    print(f"[refresh] {len(changes)} change(s) logged to {CHANGELOG_PATH}")
    return LATEST_GRAPH_PATH


if __name__ == "__main__":
    refresh_lourdes_graph()
