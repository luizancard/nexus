"""Runs pretrained Mask2Former (Mapillary Vistas) inference over a Mapillary
image manifest and emits predictions in the exact JSON shape
`parse_prediction_records()` in `semantic_segmentation_training.py` expects.

This is the "hybrid" fast path: no training happens here, just inference with
`facebook/mask2former-swin-large-mapillary-vistas-semantic` (verified live on
HuggingFace, ~216M params, Vistas v1.2's 65-class taxonomy). Only classes that
`MAPILLARY_TO_ACCESSIBILITY` actually maps to something are extracted as
polygons -- the other ~41 Vistas classes (sky, vehicles, buildings, ...) are
irrelevant to accessibility scoring and would only cost compute for no
downstream use.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from data_pipeline.semantic_segmentation_training import (
    MAPILLARY_TO_ACCESSIBILITY,
    MapillaryImageEntry,
    _normalize_label,
    parse_mapillary_metadata,
)

MODEL_NAME = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
MIN_CONTOUR_AREA_PX = 50.0
POLYGON_SIMPLIFY_EPSILON_PX = 2.0


def pick_device() -> str:
    """Pick the fastest available torch device: MPS (Apple GPU) > CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(
    device: str,
) -> tuple[Mask2FormerForUniversalSegmentation, AutoImageProcessor, dict[int, str]]:
    """Load the pretrained Vistas semantic segmentation model.

    Returns:
        (model, image_processor, relevant_id2label) where `relevant_id2label`
        is restricted to the class ids that `MAPILLARY_TO_ACCESSIBILITY`
        actually maps to something -- the ids worth extracting polygons for.
    """
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    relevant_id2label = {
        class_id: label
        for class_id, label in model.config.id2label.items()
        if _normalize_label(label) in MAPILLARY_TO_ACCESSIBILITY
    }
    return model, processor, relevant_id2label


@torch.inference_mode()
def segment_image(
    image: Image.Image,
    model: Mask2FormerForUniversalSegmentation,
    processor: AutoImageProcessor,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one image through the model and return (class_map, confidence_map).

    Replicates the same query-fusion HF's own `post_process_semantic_segmentation`
    uses internally (softmax class probs per query x sigmoid mask probs per
    query, summed over queries), but keeps the per-pixel max *value* alongside
    the argmax *class* -- the convenience post-processing method only returns
    the class map and discards the confidence, which we need per detection.

    Returns:
        class_map: (H, W) int array of predicted Vistas class ids, at the
            original image resolution.
        confidence_map: (H, W) float array in [0, 1], the winning class's
            fused query score at each pixel.
    """
    original_size = (image.height, image.width)
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)

    class_queries_logits = outputs.class_queries_logits[0]  # (num_queries, num_classes + 1)
    masks_queries_logits = outputs.masks_queries_logits[0]  # (num_queries, h, w)

    class_probs = class_queries_logits.softmax(dim=-1)[..., :-1]  # drop "no object"
    mask_probs = masks_queries_logits.sigmoid()

    pixel_class_scores = torch.einsum("qc,qhw->chw", class_probs, mask_probs)
    pixel_class_scores = torch.nn.functional.interpolate(
        pixel_class_scores.unsqueeze(0),
        size=original_size,
        mode="bilinear",
        align_corners=False,
    )[0]

    confidence_map, class_map = pixel_class_scores.max(dim=0)
    # `pixel_class_scores` sums contributions across queries (the same fusion
    # HF's own post_process_semantic_segmentation uses internally to pick the
    # argmax class), so when multiple queries agree on one pixel the winning
    # score can exceed 1.0. Clamp so this stays a genuine [0, 1] confidence --
    # `summarize_barriers_per_image` multiplies it directly against
    # BARRIER_WEIGHTS and assumes that range.
    confidence_map = confidence_map.clamp(max=1.0)
    return class_map.cpu().numpy(), confidence_map.cpu().numpy()


def extract_polygons(
    class_map: np.ndarray,
    confidence_map: np.ndarray,
    relevant_id2label: dict[int, str],
) -> list[dict[str, Any]]:
    """Convert per-pixel class/confidence maps into per-class polygon detections.

    Only classes in `relevant_id2label` are processed. Each disjoint connected
    region of a class becomes its own detection (a photo can show two separate
    sidewalk segments, for instance).

    Args:
        class_map: (H, W) predicted class ids.
        confidence_map: (H, W) per-pixel confidence of the winning class.
        relevant_id2label: class id -> raw Vistas label, accessibility-relevant only.

    Returns:
        List of `{class_name, confidence, polygon_xy}` dicts.
    """
    detections: list[dict[str, Any]] = []
    for class_id, label in relevant_id2label.items():
        binary_mask = (class_map == class_id).astype(np.uint8)
        if not binary_mask.any():
            continue
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_CONTOUR_AREA_PX:
                continue
            simplified = cv2.approxPolyDP(contour, POLYGON_SIMPLIFY_EPSILON_PX, closed=True)
            polygon_xy = simplified.reshape(-1, 2).tolist()
            if len(polygon_xy) < 3:
                continue

            region_mask = np.zeros_like(binary_mask)
            cv2.drawContours(region_mask, [contour], -1, 1, thickness=cv2.FILLED)
            region_confidence = float(confidence_map[region_mask.astype(bool)].mean())

            detections.append(
                {
                    "class_name": label,
                    "confidence": round(region_confidence, 4),
                    "polygon_xy": polygon_xy,
                }
            )
    return detections


def download_image(
    url: str, session: requests.Session, timeout: tuple[int, int] = (10, 30)
) -> Image.Image:
    """Download an image on the fly -- never persisted to disk (keeps this
    repo's "no full local dataset" design, matching the manifest/shard step).

    Args:
        timeout: (connect_timeout, read_timeout) in seconds -- split
            explicitly rather than a single value so a stalled read on one
            image can't silently stall the whole run past what the caller
            expects.
    """
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def run_inference(
    entries: list[MapillaryImageEntry],
    output_json: Path,
    device: str | None = None,
    progress_every: int = 25,
) -> dict[str, int]:
    """Run segmentation over a list of manifest entries and write predictions JSON.

    Args:
        entries: Parsed manifest entries (see `parse_mapillary_metadata`).
        output_json: Destination path, written in the `parse_prediction_records`
            schema: a list of `{image_id, detections: [...]}`.
        device: Torch device; auto-picked via `pick_device()` if None.
        progress_every: Print a progress line every N images.

    Returns:
        Summary counts: {"images_processed", "images_failed", "images_no_detections"}.
    """
    device = device or pick_device()
    print(f"[1/3] Loading {MODEL_NAME} on device={device}...")
    model, processor, relevant_id2label = load_model(device)
    print(f"      {len(relevant_id2label)} of {len(model.config.id2label)} Vistas classes are accessibility-relevant.")

    print(f"[2/3] Running inference over {len(entries)} image(s)...")
    results: list[dict[str, Any]] = []
    stats = {"images_processed": 0, "images_failed": 0, "images_no_detections": 0}
    start = time.time()

    with requests.Session() as session:
        for index, entry in enumerate(entries, start=1):
            try:
                image = download_image(entry.image_path, session=session)
                class_map, confidence_map = segment_image(image, model, processor, device)
                detections = extract_polygons(class_map, confidence_map, relevant_id2label)
                if detections:
                    results.append(
                        {
                            "image_id": entry.image_id,
                            # image_width/image_height are extra fields ignored by
                            # parse_prediction_records() but read directly by
                            # geometric_attribute_extractor.py's pinhole-projection
                            # width estimate, which needs to know pixel-to-frame scale.
                            "image_width": image.width,
                            "image_height": image.height,
                            "detections": detections,
                        }
                    )
                else:
                    stats["images_no_detections"] += 1
                stats["images_processed"] += 1
            except Exception as exc:  # network/model failures shouldn't kill the whole run
                stats["images_failed"] += 1
                print(f"      [WARNING] image_id={entry.image_id} failed: {exc}")

            if index % progress_every == 0 or index == len(entries):
                elapsed = time.time() - start
                rate = index / elapsed if elapsed > 0 else 0.0
                eta_min = (len(entries) - index) / rate / 60 if rate > 0 else float("nan")
                print(
                    f"      {index}/{len(entries)} "
                    f"({rate:.2f} img/s, ETA {eta_min:.1f} min)"
                )

    print(f"[3/3] Writing {len(results)} image result(s) to {output_json}")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return stats


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run pretrained Mapillary Vistas segmentation over a manifest CSV."
    )
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--city-filter", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N images (testing).")
    parser.add_argument("--device", type=str, default=None, choices=["mps", "cpu", "cuda"])
    return parser


def main() -> None:
    """CLI entrypoint."""
    args = _build_cli_parser().parse_args()
    entries = parse_mapillary_metadata(
        metadata_csv=args.metadata_csv,
        train_ratio=0.85,  # split field unused for inference, value is a no-op here
        city_filter=args.city_filter,
    )
    if args.limit:
        entries = entries[: args.limit]

    stats = run_inference(entries, output_json=args.output_json, device=args.device)
    print(f"Done: {stats}")


if __name__ == "__main__":
    main()
