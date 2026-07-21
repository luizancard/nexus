"""Manual per-edge validation of the imagery-only presence attributes.

Decision recorded in docs/METHODOLOGY.md §9.3, option 3: rather than trust
`ramp_present`/`fixed_obstacle_present` at the ~20-45% detection precision
measured in §9, hand-check the imagery-touched edges and build a *validated
subset* the routing experiment can consume.

Unit of work: one undirected physical segment (the two directed OSMnx twins
collapsed) per attribute. An edge is imagery-`present` for an attribute iff
at least one snapped image carries a triggering detection; to confirm the
segment it suffices to confirm one such detection, so this renders the
single largest-area detection per (segment, attribute) -- §9 found large
detections at corners are the reliable-true ones, tiny ones the false
positives, so the largest is the fairest single test of whether the
segment's `present` is real.

State lives in a JSON label store (`--labels`, default
data_files/edge_validation_labels.json) keyed by a stable
`segment_id|attribute` so the work is resumable across sessions and nothing
is lost: re-running skips already-labelled items. `apply` then writes the
validated subset back onto the fused graph as `<attr>_validated` (bool) and
`<attr>_validated_value` (present/absent), leaving the original imagery
value untouched for comparison.

Subcommands:
  worklist  -- print the deterministic (segment, attr) work order + counts
  render    -- render review montages for the next N unlabelled items
  label     -- record a verdict for one segment_id|attribute
  status    -- how many of each attribute are validated so far
  apply     -- write validated flags onto a copy of the fused graph
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from data_pipeline.edge_attribute_fusion import (
    CLASS_TO_PRESENCE_ATTR,
    load_fused_graph,
    snap_images_to_edges,
)
from data_pipeline.semantic_segmentation_training import parse_mapillary_metadata, polygon_area

DATA = Path("data_files")
DEFAULT_LABELS = DATA / "edge_validation_labels.json"
RAMP_CLASSES = {c for c, a in CLASS_TO_PRESENCE_ATTR.items() if a == "ramp_present"}
OBSTACLE_CLASSES = {c for c, a in CLASS_TO_PRESENCE_ATTR.items() if a == "fixed_obstacle_present"}
ATTR_CLASSES = {"ramp_present": RAMP_CLASSES, "fixed_obstacle_present": OBSTACLE_CLASSES}


def _area(poly: list[list[float]]) -> float:
    return polygon_area([tuple(p) for p in poly])


def _seg_id(u: Any, v: Any, k: int) -> str:
    a, b = sorted([str(u), str(v)])
    return f"{a}_{b}_{k}"


def build_worklist(graph, predictions_json: Path, metadata_csv: Path) -> list[dict[str, Any]]:
    """Deterministic list of {segment_id, attribute, image_id, poly, conf, area,
    n_detections, n_images} -- one row per (segment, attribute), the largest
    triggering detection chosen as the representative."""
    preds = json.loads(predictions_json.read_text(encoding="utf-8"))
    by_id = {p["image_id"]: p for p in preds}
    entries = [e for e in parse_mapillary_metadata(metadata_csv, train_ratio=0.85) if e.image_id in by_id]
    assign = snap_images_to_edges(graph, entries)

    # segment_id -> attribute -> {best detection, counts}
    acc: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(
        lambda: {"best": None, "n_detections": 0, "images": set()}))
    for (u, v, k), imgs in assign.items():
        sid = _seg_id(u, v, k)
        for entry in imgs:
            for det in by_id[entry.image_id]["detections"]:
                for attr, classes in ATTR_CLASSES.items():
                    if det["class_name"] in classes:
                        slot = acc[sid][attr]
                        slot["n_detections"] += 1
                        slot["images"].add(entry.image_id)
                        a = _area(det["polygon_xy"])
                        if slot["best"] is None or a > slot["best"]["area"]:
                            slot["best"] = {"image_id": entry.image_id, "poly": det["polygon_xy"],
                                            "conf": det["confidence"], "area": a,
                                            "class_name": det["class_name"]}

    rows: list[dict[str, Any]] = []
    for sid in sorted(acc):
        for attr in ("ramp_present", "fixed_obstacle_present"):
            if attr in acc[sid]:
                slot = acc[sid][attr]
                b = slot["best"]
                rows.append({"key": f"{sid}|{attr}", "segment_id": sid, "attribute": attr,
                             "image_id": b["image_id"], "poly": b["poly"], "conf": b["conf"],
                             "area": b["area"], "class_name": b["class_name"],
                             "n_detections": slot["n_detections"], "n_images": len(slot["images"])})
    return rows


def _load_labels(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save_labels(path: Path, labels: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(labels, indent=1, sort_keys=True), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", type=Path, default=DATA / "lourdes_graph_fused.graphml")
    p.add_argument("--predictions-json", type=Path, default=DATA / "lourdes_predictions.json")
    p.add_argument("--metadata-csv", type=Path, default=DATA / "mapillary_metadata.csv")
    p.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("worklist")
    sub.add_parser("status")
    r = sub.add_parser("render")
    r.add_argument("--attribute", choices=list(ATTR_CLASSES), default=None)
    r.add_argument("--limit", type=int, default=25)
    r.add_argument("--per-sheet", type=int, default=5)
    r.add_argument("--out-dir", type=Path, default=Path("scripts/_edgeval_sheets"))
    lb = sub.add_parser("label")
    lb.add_argument("--key", required=True, help="segment_id|attribute")
    lb.add_argument("--verdict", required=True, choices=["present", "absent", "uncertain"])
    lb.add_argument("--note", default="")
    ap = sub.add_parser("apply")
    ap.add_argument("--out-graph", type=Path, default=DATA / "lourdes_graph_validated.graphml")
    args = p.parse_args()

    graph = load_fused_graph(args.graph)

    if args.cmd == "label":
        labels = _load_labels(args.labels)
        labels[args.key] = {"verdict": args.verdict, "note": args.note}
        _save_labels(args.labels, labels)
        print(f"recorded {args.key} = {args.verdict}")
        return

    rows = build_worklist(graph, args.predictions_json, args.metadata_csv)
    labels = _load_labels(args.labels)

    if args.cmd == "worklist":
        by_attr = defaultdict(int)
        for row in rows:
            by_attr[row["attribute"]] += 1
        print(f"{len(rows)} (segment, attribute) items across {len({r['segment_id'] for r in rows})} segments")
        for attr, n in by_attr.items():
            print(f"  {attr}: {n}")
        return

    if args.cmd == "status":
        done = defaultdict(lambda: defaultdict(int))
        for row in rows:
            v = labels.get(row["key"], {}).get("verdict")
            done[row["attribute"]][v or "UNLABELLED"] += 1
        for attr in ATTR_CLASSES:
            d = done[attr]
            tot = sum(d.values())
            lab = tot - d.get("UNLABELLED", 0)
            print(f"{attr}: {lab}/{tot} labelled  {dict(d)}")
        return

    if args.cmd == "render":
        from scripts.validate_detection_precision import _load_meta, render_sheet
        meta = _load_meta(args.metadata_csv)
        cache = Path(__file__).resolve().parent / "_precision_cache"
        todo = [r for r in rows if r["key"] not in labels
                and (args.attribute is None or r["attribute"] == args.attribute)][:args.limit]
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "batch_keys.json").write_text(json.dumps([r["key"] for r in todo], indent=1))
        for j in range(0, len(todo), args.per_sheet):
            chunk = todo[j:j + args.per_sheet]
            items = [(r["image_id"], r["poly"], r["conf"],
                      f"{r['key']}  n_img={r['n_images']} n_det={r['n_detections']}") for r in chunk]
            render_sheet(items, args.out_dir / f"eval_{j // args.per_sheet}.png", meta, cache)
        print(f"rendered {len(todo)} items; keys in {args.out_dir/'batch_keys.json'}")
        return

    if args.cmd == "apply":
        val = {"ramp_present": 0, "fixed_obstacle_present": 0}
        by_seg = defaultdict(dict)
        for row in rows:
            lab = labels.get(row["key"])
            if lab and lab["verdict"] in ("present", "absent"):
                by_seg[row["segment_id"]][row["attribute"]] = lab["verdict"]
        for u, v, k, d in graph.edges(keys=True, data=True):
            sid = _seg_id(u, v, k)
            for attr in ATTR_CLASSES:
                if attr in by_seg.get(sid, {}):
                    d[f"{attr}_validated"] = True
                    d[f"{attr}_validated_value"] = by_seg[sid][attr]
                    val[attr] += 1
        from data_pipeline.osm_extractor import salvar_grafo
        salvar_grafo(graph, args.out_graph)
        print(f"wrote {args.out_graph}; validated edge-attrs applied: {val}")
        return


if __name__ == "__main__":
    main()
