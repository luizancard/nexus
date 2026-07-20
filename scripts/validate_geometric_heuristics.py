"""Phase 1.6 validation: how good are the geometric heuristics, really?

No physical field access is available to collect independently-measured
ground truth (tape-measure width, etc.) for this validation pass, so this
is honestly a *cross-method agreement* check, not a comparison against a
gold-standard reference: for images where both a Curb and a Sidewalk are
detected, sidewalk width is estimated two independent ways --

1. `geometric_attribute_extractor.estimate_width_bucket` -- the pipeline's
   primary method, a fixed assumed camera height + vertical FOV pinhole
   projection (see that module's docstring for the exact assumptions).
2. `estimate_width_via_curb_reference` (this script only) -- uses the
   detected curb's own pixel height as a *local, per-image* scale
   reference, assuming a standard ~15cm curb height (common Brazilian
   municipal standard, consistent with NBR 9050).

Agreement between two independent methods is evidence of a real signal;
disagreement quantifies exactly how much the primary method's fixed-camera
assumption costs in practice. This does NOT replace real ground truth --
if a field visit or higher-quality reference data becomes available, this
script's rate should be re-checked against it, not treated as final.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data_pipeline.geometric_attribute_extractor import (
    WIDTH_BUCKET_ORDER,
    WIDTH_BUCKET_THRESHOLDS_M,
    bottom_band_points,
    estimate_width_bucket,
)

ASSUMED_CURB_HEIGHT_M = 0.15
MIN_CURB_HEIGHT_PX = 2  # below this, the curb mask is too thin/noisy to use as a scale reference


def estimate_width_via_curb_reference(
    sidewalk_polygon: list[tuple[float, float]],
    curb_polygon: list[tuple[float, float]],
    image_height_px: int,
    assumed_curb_height_m: float = ASSUMED_CURB_HEIGHT_M,
) -> str | None:
    """Independent width estimate using the detected curb as a local scale reference.

    Args:
        sidewalk_polygon: Sidewalk mask polygon, image pixel coordinates.
        curb_polygon: Curb mask polygon from the same image.
        image_height_px: Full image height in pixels.
        assumed_curb_height_m: Stated reference height for a standard curb.

    Returns:
        A width bucket string (see WIDTH_BUCKET_THRESHOLDS_M), or None if
        the curb's vertical extent is degenerate (e.g. a sliver mask).
    """
    curb_ys = [p[1] for p in curb_polygon]
    curb_height_px = max(curb_ys) - min(curb_ys)
    if curb_height_px < MIN_CURB_HEIGHT_PX:
        return None

    meters_per_px = assumed_curb_height_m / curb_height_px
    band = bottom_band_points(sidewalk_polygon, image_height_px)
    xs = [p[0] for p in band]
    width_m = (max(xs) - min(xs)) * meters_per_px

    low, high = WIDTH_BUCKET_THRESHOLDS_M
    if width_m < low:
        return "under_50cm"
    if width_m < high:
        return "50_to_90cm"
    return "over_90cm"


def compare_methods(predictions_json: Path) -> dict[str, Any]:
    """Run both width-estimation methods on every eligible image and compare.

    Args:
        predictions_json: Output of `run_segmentation_inference.py`.

    Returns:
        Summary dict with per-bucket agreement counts, a confusion matrix,
        and the overall agreement rate.
    """
    results = json.loads(predictions_json.read_text(encoding="utf-8"))

    confusion = {b1: {b2: 0 for b2 in WIDTH_BUCKET_ORDER} for b1 in WIDTH_BUCKET_ORDER}
    n_compared = 0
    n_eligible = 0

    for item in results:
        sidewalks = [d["polygon_xy"] for d in item["detections"] if d["class_name"] == "Sidewalk"]
        curbs = [d["polygon_xy"] for d in item["detections"] if d["class_name"] == "Curb"]
        if not sidewalks or not curbs:
            continue
        n_eligible += 1

        primary = estimate_width_bucket(sidewalks[0], item["image_width"], item["image_height"])
        reference = estimate_width_via_curb_reference(sidewalks[0], curbs[0], item["image_height"])
        if primary is None or reference is None:
            continue
        n_compared += 1
        confusion[primary][reference] += 1

    agree = sum(confusion[b][b] for b in WIDTH_BUCKET_ORDER)
    summary = {
        "n_images_with_both_sidewalk_and_curb": n_eligible,
        "n_images_both_methods_produced_estimate": n_compared,
        "agreement_rate_pct": round(100 * agree / n_compared, 1) if n_compared else None,
        "confusion_matrix (rows=primary_pinhole, cols=curb_reference)": confusion,
    }
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-json", type=Path, required=True)
    args = parser.parse_args()

    summary = compare_methods(args.predictions_json)
    print(json.dumps(summary, indent=2))
