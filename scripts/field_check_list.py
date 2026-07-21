"""Generate a walkable field-check list from the validation labels.

Turns a stratified sample of hand-validated segments into a physical
visit list: each item has the segment's real-world coordinate (WGS84), a
Google Maps link, the exact Mapillary photo the model/adjudicator saw, the
recorded verdict, and what to confirm on the ground. Items are ordered by a
greedy nearest-neighbour walk so the visit is one continuous route through
Lourdes.

Purpose (see docs/METHODOLOGY.md §9.5 / §10.2): a small field sample
upgrades the precision claim from "cross-model visual adjudication of
thumbnails" to "field-anchored on a sample". You do NOT need all 105
uncertain segments -- ~15 stratified points, checked against reality, let
you state real agreement between the pipeline and the ground.

Sampling strategy (stratified, not random -- each stratum answers a
different question):
  - ramp present   -> does the model's "present" match reality? (anchors 90.8%)
  - ramp uncertain -> resolve an unconfirmable segment
  - obstacle present   -> does "present" match reality? (anchors 64%)
  - obstacle uncertain -> resolve an unconfirmable segment

Output: prints the ordered list and writes CSV + GeoJSON to data_files/
(git-ignored, regenerable). Load the GeoJSON into Google My Maps or any
phone GIS app; or tap the Google Maps links one by one.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from data_pipeline.edge_attribute_fusion import load_fused_graph
from data_pipeline.spatial_utils import to_wgs84
from scripts.edge_validation import _seg_id, build_worklist

DATA = Path("data_files")

# stratum -> how many to sample. Tune with --counts if you want more/fewer.
DEFAULT_STRATA = {
    ("ramp_present", "present"): 5,
    ("ramp_present", "uncertain"): 3,
    ("fixed_obstacle_present", "present"): 4,
    ("fixed_obstacle_present", "uncertain"): 3,
}

CHECK_PROMPT = {
    "ramp_present": "Is there a lowered curb ramp / rebaixamento de guia on this segment (corner or driveway)?",
    "fixed_obstacle_present": "Is there a fixed obstacle blocking the sidewalk here (post, bin, hydrant, manhole, dumpster, bench)?",
}


def _seg_midpoints_wgs84(graph) -> dict[str, tuple[float, float]]:
    """segment_id -> (lat, lon) of the segment midpoint, in WGS84."""
    crs = str(graph.graph.get("crs"))
    mids: dict[str, tuple[float, float]] = {}
    for u, v, k in graph.edges(keys=True):
        sid = _seg_id(u, v, k)
        if sid in mids:
            continue
        xu, yu = graph.nodes[u]["x"], graph.nodes[u]["y"]
        xv, yv = graph.nodes[v]["x"], graph.nodes[v]["y"]
        lat, lon = to_wgs84((xu + xv) / 2.0, (yu + yv) / 2.0, source_crs=crs)
        mids[sid] = (lat, lon)
    return mids


def _greedy_walk_order(items: list[dict]) -> list[dict]:
    """Order points by nearest-neighbour from the northern-most, so the
    field visit is one continuous walk rather than crisscrossing. Distances
    use lat/lon directly -- fine at neighbourhood scale for ordering only."""
    if not items:
        return []
    remaining = items[:]
    start = max(remaining, key=lambda it: it["lat"])  # northmost
    ordered = [start]
    remaining.remove(start)
    while remaining:
        last = ordered[-1]
        nxt = min(remaining, key=lambda it: (it["lat"] - last["lat"]) ** 2 + (it["lon"] - last["lon"]) ** 2)
        ordered.append(nxt)
        remaining.remove(nxt)
    return ordered


def build_field_list(seed: int, strata: dict[tuple[str, str], int]) -> list[dict]:
    graph = load_fused_graph(DATA / "lourdes_graph_fused.graphml")
    rows = {r["key"]: r for r in build_worklist(graph, DATA / "lourdes_predictions.json", DATA / "mapillary_metadata.csv")}
    labels = json.loads((DATA / "edge_validation_labels.json").read_text())
    mids = _seg_midpoints_wgs84(graph)

    meta = {}
    with (DATA / "mapillary_metadata.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            meta[r["image_id"]] = r

    # group labelled keys by stratum, deterministic order
    import random
    rng = random.Random(seed)
    by_stratum: dict[tuple[str, str], list[str]] = {}
    for key, lab in labels.items():
        attr = key.split("|")[1]
        by_stratum.setdefault((attr, lab["verdict"]), []).append(key)

    picked: list[dict] = []
    for stratum, n in strata.items():
        pool = sorted(by_stratum.get(stratum, []))
        rng.shuffle(pool)
        for key in pool[:n]:
            r = rows.get(key)
            sid = key.split("|")[0]
            attr = key.split("|")[1]
            lat, lon = mids.get(sid, (None, None))
            img = r["image_id"] if r else None
            picked.append({
                "segment_id": sid,
                "attribute": attr,
                "model_verdict": labels[key]["verdict"],
                "lat": round(lat, 6) if lat else None,
                "lon": round(lon, 6) if lon else None,
                "gmaps": f"https://www.google.com/maps/search/?api=1&query={lat},{lon}" if lat else "",
                "streetview": f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}" if lat else "",
                "reference_photo": meta.get(img, {}).get("image_path", "") if img else "",
                "check": CHECK_PROMPT[attr],
                "note_from_review": labels[key].get("note", ""),
            })
    return _greedy_walk_order(picked)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out-csv", type=Path, default=DATA / "field_check_lourdes.csv")
    p.add_argument("--out-geojson", type=Path, default=DATA / "field_check_lourdes.geojson")
    args = p.parse_args()

    items = build_field_list(args.seed, DEFAULT_STRATA)

    with args.out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(items[0].keys()))
        w.writeheader()
        w.writerows(items)
    geojson = {"type": "FeatureCollection", "features": [
        {"type": "Feature",
         "geometry": {"type": "Point", "coordinates": [it["lon"], it["lat"]]},
         "properties": {k: v for k, v in it.items() if k not in ("lat", "lon")}}
        for it in items if it["lat"]]}
    args.out_geojson.write_text(json.dumps(geojson, indent=1), encoding="utf-8")

    print(f"{len(items)} field-check points (walk order), written to {args.out_csv} and {args.out_geojson}\n")
    for i, it in enumerate(items, 1):
        print(f"{i:>2}. [{it['attribute'].replace('_present','')}: model says {it['model_verdict'].upper()}]  {it['lat']},{it['lon']}")
        print(f"    check: {it['check']}")
        print(f"    map:   {it['gmaps']}")
        print(f"    photo the model saw: {it['reference_photo'][:90]}")
        print()


if __name__ == "__main__":
    main()
