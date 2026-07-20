"""Semantic segmentation pipeline for accessibility barrier mining.

This module prepares Mapillary manifests without full local download, defines
class remapping for accessibility barriers, and summarizes YOLO segmentation
predictions for Lourdes accessibility scoring.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ACCESSIBILITY_CLASSES: list[str] = [
    "street",
    "sidewalk",
    "curb",
    "crossing",
    "steps",
    "ramp",
    "handrail",
    "tactile_paving",
    "lighting",
    "surface_obstacle",
]

# Mapillary Vistas v1.2 semantic labels (verified against the real 65-class
# id2label of facebook/mask2former-swin-large-mapillary-vistas-semantic) to
# Nexus accessibility classes. Keys are pre-normalized -- see
# `_normalize_label` -- so multi-word/punctuated Vistas names like
# "Crosswalk - Plain" or "Traffic Sign (Back)" match correctly.
MAPILLARY_TO_ACCESSIBILITY: dict[str, str] = {
    # street / road surface
    "road": "street",
    "service-lane": "street",
    "bike-lane": "street",
    "lane-marking-general": "street",
    # sidewalk / pedestrian-exclusive ground
    "sidewalk": "sidewalk",
    "pedestrian-area": "sidewalk",
    # curb + curb ramps
    "curb": "curb",
    "curb-cut": "ramp",
    # crossings
    "crosswalk-plain": "crossing",
    "lane-marking-crosswalk": "crossing",
    # lighting -- approximate: Street Light is a direct match; Utility Pole is
    # a weaker proxy (common in Brazilian streetscapes to carry light
    # fixtures, but not guaranteed). Generic "Pole", traffic signals, and
    # traffic signs are deliberately excluded to avoid diluting this signal.
    "street-light": "lighting",
    "utility-pole": "lighting",
    # fixed street-furniture obstructions -- real, literature-documented
    # accessibility complaints: these narrow already-tight sidewalks
    "bench": "surface_obstacle",
    "bike-rack": "surface_obstacle",
    "fire-hydrant": "surface_obstacle",
    "mailbox": "surface_obstacle",
    "manhole": "surface_obstacle",
    "phone-booth": "surface_obstacle",
    "pothole": "surface_obstacle",
    "trash-can": "surface_obstacle",
    "barrier": "surface_obstacle",
}
# `steps`, `handrail`, and `tactile_paving` are deliberately absent: Mapillary
# Vistas v1.2 (the taxonomy this pretrained checkpoint was trained on) has no
# equivalent class for any of them -- confirmed against the full 65-class
# list. Note "Guard Rail" is intentionally NOT mapped to `handrail` despite
# superficial name similarity: Vistas' Guard Rail is a roadside vehicle
# barrier, not pedestrian stair/ramp support, and conflating them would
# fabricate a false accessibility-positive signal. These 3 gap classes stay
# genuinely undetected until the stretch-goal fine-tune (see README).

# Higher value = stronger negative impact for mobility-impaired users.
BARRIER_WEIGHTS: dict[str, float] = {
    "street": 0.20,
    "sidewalk": -0.15,
    "curb": 0.55,
    "crossing": -0.20,
    "steps": 0.95,
    "ramp": -0.40,
    "handrail": -0.25,
    "tactile_paving": -0.30,
    "lighting": -0.10,
    "surface_obstacle": 0.65,
}


@dataclass(frozen=True)
class BoundingBox:
    """Lat/lon bounding box."""

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def contains(self, latitude: float, longitude: float) -> bool:
        """Return True if a point is inside the box."""
        return (
            self.min_lat <= latitude <= self.max_lat
            and self.min_lon <= longitude <= self.max_lon
        )


@dataclass(frozen=True)
class MapillaryImageEntry:
    """Metadata for one Mapillary image record."""

    image_id: str
    image_path: str
    latitude: float
    longitude: float
    city: str
    split: str
    compass_angle: float | None = None


@dataclass(frozen=True)
class DetectionRecord:
    """Single segmentation detection extracted from model output."""

    image_id: str
    class_name: str
    confidence: float
    polygon_xy: list[tuple[float, float]]


def _required_field(row: dict[str, str], candidates: list[str]) -> str:
    for name in candidates:
        value = row.get(name, "").strip()
        if value:
            return value
    raise ValueError(
        f"Missing required metadata field. Expected one of: {', '.join(candidates)}"
    )


def _hash_to_split(image_id: str, train_ratio: float) -> str:
    digest = hashlib.sha256(image_id.encode("utf-8")).hexdigest()
    normalized = int(digest[:8], 16) / float(0xFFFFFFFF)
    return "train" if normalized < train_ratio else "val"


def parse_mapillary_metadata(
    metadata_csv: Path,
    train_ratio: float,
    city_filter: str | None = None,
) -> list[MapillaryImageEntry]:
    """Parse Mapillary metadata CSV into typed entries.

    Args:
        metadata_csv: CSV file with image metadata.
        train_ratio: Deterministic train split ratio in [0, 1].
        city_filter: Optional city name filter.

    Returns:
        Parsed metadata entries.
    """
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_csv}")
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1 (exclusive).")

    entries: list[MapillaryImageEntry] = []
    with metadata_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader, start=2):
            try:
                image_id = _required_field(row, ["image_id", "id"])
                image_path = _required_field(row, ["image_path", "path", "url"])
                latitude = float(_required_field(row, ["latitude", "lat"]))
                longitude = float(_required_field(row, ["longitude", "lon", "lng"]))
                city = row.get("city", "").strip()
                compass_raw = row.get("compass_angle", "").strip()
                compass_angle = float(compass_raw) if compass_raw else None
            except Exception as exc:
                raise ValueError(f"Invalid metadata row {row_index}: {exc}") from exc

            if city_filter and city.lower() != city_filter.lower():
                continue

            split = _hash_to_split(image_id=image_id, train_ratio=train_ratio)
            entries.append(
                MapillaryImageEntry(
                    image_id=image_id,
                    image_path=image_path,
                    latitude=latitude,
                    longitude=longitude,
                    city=city,
                    split=split,
                    compass_angle=compass_angle,
                )
            )

    if not entries:
        filter_note = f" (city filter: {city_filter})" if city_filter else ""
        raise ValueError(f"No entries loaded from metadata CSV{filter_note}.")
    return entries


def filter_by_bbox(
    entries: Iterable[MapillaryImageEntry],
    bbox: BoundingBox,
) -> list[MapillaryImageEntry]:
    """Filter records by geospatial bounding box."""
    filtered = [
        entry
        for entry in entries
        if bbox.contains(latitude=entry.latitude, longitude=entry.longitude)
    ]
    if not filtered:
        raise ValueError("Bounding box filter returned zero images.")
    return filtered


def write_sharded_manifests(
    entries: list[MapillaryImageEntry],
    output_dir: Path,
    shard_size: int,
) -> dict[str, int]:
    """Write train/val sharded manifest files.

    Each manifest is a JSON list of image records. These shards can be consumed
    by cloud jobs that fetch and process only one chunk at a time.
    """
    if shard_size <= 0:
        raise ValueError("shard_size must be > 0.")
    output_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, int] = {"train": 0, "val": 0}
    for split in ("train", "val"):
        split_entries = [entry for entry in entries if entry.split == split]
        stats[split] = len(split_entries)
        for shard_idx in range(0, len(split_entries), shard_size):
            shard = split_entries[shard_idx : shard_idx + shard_size]
            shard_name = f"{split}_shard_{(shard_idx // shard_size):04d}.json"
            shard_path = output_dir / shard_name
            payload = [
                {
                    "image_id": entry.image_id,
                    "image_path": entry.image_path,
                    "latitude": entry.latitude,
                    "longitude": entry.longitude,
                    "city": entry.city,
                    "split": entry.split,
                }
                for entry in shard
            ]
            shard_path.write_text(
                json.dumps(payload, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
    return stats


def write_yolo_dataset_yaml(
    output_yaml: Path,
    train_images_dir: str,
    val_images_dir: str,
) -> None:
    """Write Ultralytics-style dataset YAML for segmentation training."""
    names_block = "\n".join(
        f"  {index}: {name}" for index, name in enumerate(ACCESSIBILITY_CLASSES)
    )
    content = (
        f"train: {train_images_dir}\n"
        f"val: {val_images_dir}\n"
        f"nc: {len(ACCESSIBILITY_CLASSES)}\n"
        f"names:\n{names_block}\n"
    )
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    output_yaml.write_text(content, encoding="utf-8")


def _normalize_label(raw_label: str) -> str:
    """Normalize a raw Vistas class name into a lookup key.

    Collapses any run of whitespace/punctuation into a single hyphen (e.g.
    "Crosswalk - Plain" -> "crosswalk-plain", "Traffic Sign (Back)" ->
    "traffic-sign-back"). A plain `.replace(" ", "-")` would leave stray
    repeated hyphens on every multi-word Vistas name that also contains its
    own hyphen, silently breaking the lookup for exactly the classes this
    remap depends on.
    """
    return re.sub(r"[^a-z0-9]+", "-", raw_label.strip().lower()).strip("-")


def remap_mapillary_label(raw_label: str) -> str | None:
    """Convert a raw Mapillary/Vistas label into the Nexus class taxonomy."""
    return MAPILLARY_TO_ACCESSIBILITY.get(_normalize_label(raw_label))


def polygon_area(points: list[tuple[float, float]]) -> float:
    """Compute polygon area with Shoelace formula.

    Formula: A = |Σ(x_i*y_{i+1} - y_i*x_{i+1})| / 2
    """
    if len(points) < 3:
        return 0.0
    area = 0.0
    for index, (x_i, y_i) in enumerate(points):
        x_j, y_j = points[(index + 1) % len(points)]
        area += (x_i * y_j) - (y_i * x_j)
    return abs(area) * 0.5


def parse_prediction_records(predictions_json: Path) -> list[DetectionRecord]:
    """Parse predictions emitted by an inference pipeline.

    Expected JSON format:
    [
        {
        "image_id": "...",
        "detections": [
            {"class_name": "...", "confidence": 0.9, "polygon_xy": [[x, y], ...]}
        ]
        }
    ]
    """
    if not predictions_json.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_json}")
    payload = json.loads(predictions_json.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Predictions JSON must be a list of image prediction items.")

    records: list[DetectionRecord] = []
    for image_item in payload:
        image_id = str(image_item.get("image_id", "")).strip()
        if not image_id:
            raise ValueError("Every prediction item must include image_id.")
        detections = image_item.get("detections", [])
        if not isinstance(detections, list):
            raise ValueError("detections must be a list.")
        for det in detections:
            class_name = str(det.get("class_name", "")).strip()
            confidence = float(det.get("confidence", 0.0))
            polygon_raw = det.get("polygon_xy", [])
            if not class_name:
                continue
            if confidence <= 0.0:
                continue
            polygon_xy: list[tuple[float, float]] = []
            for item in polygon_raw:
                if not isinstance(item, list) or len(item) != 2:
                    continue
                polygon_xy.append((float(item[0]), float(item[1])))
            if not polygon_xy:
                continue
            records.append(
                DetectionRecord(
                    image_id=image_id,
                    class_name=class_name,
                    confidence=confidence,
                    polygon_xy=polygon_xy,
                )
            )
    if not records:
        raise ValueError("No valid detection records found in predictions JSON.")
    return records


def summarize_barriers_per_image(
    detections: list[DetectionRecord],
    image_area: float,
    confidence_threshold: float,
) -> list[dict[str, Any]]:
    """Aggregate detections into per-image accessibility barrier score.

    Score formula:
        score = Σ (w_c * conf * area_ratio)
    where area_ratio = polygon_area / image_area.
    """
    if image_area <= 0:
        raise ValueError("image_area must be > 0.")

    grouped: dict[str, list[DetectionRecord]] = {}
    for det in detections:
        grouped.setdefault(det.image_id, []).append(det)

    rows: list[dict[str, Any]] = []
    for image_id, image_detections in grouped.items():
        class_counts = {class_name: 0 for class_name in ACCESSIBILITY_CLASSES}
        score = 0.0
        for det in image_detections:
            if det.confidence < confidence_threshold:
                continue
            mapped = remap_mapillary_label(det.class_name)
            if not mapped:
                continue
            class_counts[mapped] += 1
            area_ratio = min(polygon_area(det.polygon_xy) / image_area, 1.0)
            score += BARRIER_WEIGHTS[mapped] * det.confidence * area_ratio

        rows.append(
            {
                "image_id": image_id,
                "accessibility_score": round(score, 6),
                **class_counts,
            }
        )
    return rows


def write_scores_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    """Write accessibility scores to CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write to CSV.")
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_bbox(bbox_raw: str) -> BoundingBox:
    """Parse bbox string 'min_lon,min_lat,max_lon,max_lat'."""
    parts = [part.strip() for part in bbox_raw.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must have four comma-separated values.")
    min_lon, min_lat, max_lon, max_lat = map(float, parts)
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError("Invalid bbox extents.")
    return BoundingBox(
        min_lon=min_lon, min_lat=min_lat, max_lon=max_lon, max_lat=max_lat
    )


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mapillary + YOLO segmentation pipeline for accessibility barriers."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-manifests")
    prepare.add_argument("--metadata-csv", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--train-ratio", type=float, default=0.85)
    prepare.add_argument("--shard-size", type=int, default=5000)
    prepare.add_argument("--city-filter", type=str, default=None)
    prepare.add_argument("--bbox", type=str, default=None)

    dataset = subparsers.add_parser("build-yolo-config")
    dataset.add_argument("--output-yaml", type=Path, required=True)
    dataset.add_argument("--train-images-dir", type=str, required=True)
    dataset.add_argument("--val-images-dir", type=str, required=True)

    score = subparsers.add_parser("score-lourdes")
    score.add_argument("--predictions-json", type=Path, required=True)
    score.add_argument("--output-csv", type=Path, required=True)
    score.add_argument("--image-area", type=float, default=1280.0 * 720.0)
    score.add_argument("--confidence-threshold", type=float, default=0.30)
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = _build_cli_parser()
    args = parser.parse_args()

    if args.command == "prepare-manifests":
        entries = parse_mapillary_metadata(
            metadata_csv=args.metadata_csv,
            train_ratio=args.train_ratio,
            city_filter=args.city_filter,
        )
        if args.bbox:
            bbox = parse_bbox(args.bbox)
            entries = filter_by_bbox(entries=entries, bbox=bbox)
        stats = write_sharded_manifests(
            entries=entries,
            output_dir=args.output_dir,
            shard_size=args.shard_size,
        )
        print(
            "Manifest generation complete | "
            f"train={stats['train']} val={stats['val']} shards_dir={args.output_dir}"
        )
        return

    if args.command == "build-yolo-config":
        write_yolo_dataset_yaml(
            output_yaml=args.output_yaml,
            train_images_dir=args.train_images_dir,
            val_images_dir=args.val_images_dir,
        )
        print(f"YOLO dataset YAML written to {args.output_yaml}")
        return

    if args.command == "score-lourdes":
        detections = parse_prediction_records(args.predictions_json)
        rows = summarize_barriers_per_image(
            detections=detections,
            image_area=args.image_area,
            confidence_threshold=args.confidence_threshold,
        )
        write_scores_csv(rows=rows, output_csv=args.output_csv)
        print(f"Lourdes accessibility scores written to {args.output_csv}")
        return

    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
