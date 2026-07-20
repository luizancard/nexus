"""Phase 1.7 validation: how precise are the imagery-derived PRESENCE
attributes, really?

`ramp_present` and `fixed_obstacle_present` are the only two attributes in
the fused schema that come entirely from imagery (0% OSM coverage, see
docs/METHODOLOGY.md Section 6), and they are high-impact terms in the cost
formula. Every other attribute has either an OSM cross-check or, for
`width_bucket`, a cross-method agreement check (Section 3.4). These two had
neither: they rested entirely on the pretrained Mask2Former's "Curb Cut"
and obstacle-class outputs being correct, unvalidated by anything beyond
the 3-image visual QA in Section 3.2.

This script makes that validation reproducible. It does NOT compute a
precision number by itself -- there is no ground-truth label file, and the
adjudication is a human (or vision-capable model) looking at the rendered
overlays. What it does is produce a *fixed, seeded* sample of real
detections rendered as full-frame + zoomed-inset overlays on the real
Mapillary images, so the manual precision assessment recorded in Section 9
can be regenerated and re-checked rather than taken on faith.

Two samples per class:
- an unbiased random sample of detections (the precision estimate), and
- the smallest-area detections (characterizes the dominant failure mode:
  in fusion, ANY Curb Cut detection -- even a 24px sliver -- flips
  `ramp_present` to present, so tiny false positives do maximum damage).

Requires network access (downloads real Mapillary thumbnails on the fly,
cached under scripts/_precision_cache/) and matplotlib. Deliberately kept
out of the fusion import path so a headless fusion run never depends on it.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

from data_pipeline.semantic_segmentation_training import polygon_area

OBSTACLE_CLASSES = {
    "Barrier", "Bench", "Bike Rack", "Fire Hydrant", "Mailbox",
    "Manhole", "Phone Booth", "Pothole", "Trash Can",
}


def _load_meta(metadata_csv: Path) -> dict[str, dict[str, str]]:
    meta: dict[str, dict[str, str]] = {}
    with metadata_csv.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            meta[row["image_id"]] = row
    return meta


def _get_image(image_id: str, meta: dict[str, dict[str, str]], cache: Path) -> Image.Image:
    fp = cache / f"{image_id}.jpg"
    if fp.exists():
        return Image.open(fp).convert("RGB")
    resp = requests.get(meta[image_id]["image_path"], timeout=60)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    cache.mkdir(parents=True, exist_ok=True)
    img.save(fp, "JPEG", quality=90)
    return img


def _area(polygon: list[list[float]]) -> float:
    return polygon_area([tuple(p) for p in polygon])


def render_sheet(
    items: list[tuple[str, list[list[float]], float, str]],
    out_path: Path,
    meta: dict[str, dict[str, str]],
    cache: Path,
) -> None:
    """Render one review sheet: one row per detection (full frame + zoom)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPoly, Rectangle

    n = len(items)
    fig, axes = plt.subplots(n, 2, figsize=(13, 3.1 * n), gridspec_kw={"width_ratios": [1.4, 1]})
    if n == 1:
        axes = axes.reshape(1, 2)
    for i, (image_id, poly, conf, tag) in enumerate(items):
        img = np.array(_get_image(image_id, meta, cache))
        h, w = img.shape[:2]
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        cx0, cx1, cy0, cy1 = min(xs), max(xs), min(ys), max(ys)
        a = _area(poly)
        axes[i, 0].imshow(img)
        axes[i, 0].axis("off")
        axes[i, 0].add_patch(MplPoly(poly, closed=True, fill=False, edgecolor="red", linewidth=2))
        axes[i, 0].add_patch(Rectangle((cx0, cy0), cx1 - cx0, cy1 - cy0, fill=False,
                                       edgecolor="yellow", linewidth=1, linestyle="--"))
        axes[i, 0].set_title(
            f"[{tag}] conf={conf:.3f} area={a:.0f}px ({100*a/(w*h):.2f}% frame) id={image_id}",
            fontsize=9, loc="left",
        )
        pad = max(40, 0.5 * max(cx1 - cx0, cy1 - cy0))
        zx0, zx1 = int(max(0, cx0 - pad)), int(min(w, cx1 + pad))
        zy0, zy1 = int(max(0, cy0 - pad)), int(min(h, cy1 + pad))
        axes[i, 1].imshow(img[zy0:zy1, zx0:zx1])
        axes[i, 1].axis("off")
        axes[i, 1].set_title("zoom", fontsize=9)
        axes[i, 1].add_patch(MplPoly([(x - zx0, y - zy0) for x, y in poly], closed=True,
                                     fill=False, edgecolor="red", linewidth=2))
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=85, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path} ({n} rows)")


def build_samples(
    predictions_json: Path, seed: int
) -> dict[str, list[tuple[str, list[list[float]], float, str]]]:
    """Build the fixed, seeded review samples (no rendering, no network)."""
    preds = json.loads(predictions_json.read_text(encoding="utf-8"))
    rng = random.Random(seed)

    curb = [
        (p["image_id"], d["polygon_xy"], d["confidence"])
        for p in preds for d in p["detections"] if d["class_name"] == "Curb Cut"
    ]
    rng.shuffle(curb)
    cc_random = [(i, poly, c, "RAND") for (i, poly, c) in curb[:30]]
    cc_tiny = [(i, poly, c, "TINY") for (i, poly, c) in sorted(curb, key=lambda t: _area(t[1]))[:10]]

    obstacles = [
        (p["image_id"], d["polygon_xy"], d["confidence"], d["class_name"])
        for p in preds for d in p["detections"] if d["class_name"] in OBSTACLE_CLASSES
    ]
    rng.shuffle(obstacles)
    ob_sample = obstacles[:22]

    return {"cc_random": cc_random, "cc_tiny": cc_tiny, "obstacles": ob_sample}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-json", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("scripts/_precision_sheets"))
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    meta = _load_meta(args.metadata_csv)
    cache = Path(__file__).resolve().parent / "_precision_cache"
    samples = build_samples(args.predictions_json, args.seed)

    for j in range(0, len(samples["cc_random"]), 5):
        render_sheet(samples["cc_random"][j:j + 5], args.out_dir / f"cc_random_{j//5}.png", meta, cache)
    render_sheet(samples["cc_tiny"], args.out_dir / "cc_tiny.png", meta, cache)
    for j in range(0, len(samples["obstacles"]), 5):
        render_sheet(samples["obstacles"][j:j + 5], args.out_dir / f"obstacle_{j//5}.png", meta, cache)
    print(f"Done. Review sheets in {args.out_dir} -- adjudicate manually per docs/METHODOLOGY.md Section 9.")


if __name__ == "__main__":
    main()
