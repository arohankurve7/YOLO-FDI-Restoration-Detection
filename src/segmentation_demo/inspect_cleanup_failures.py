from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

from src.segmentation_demo.evaluate_cleanup_thresholds import (
    MODEL_PATH,
    IMAGE_DIR,
    extract_predictions,
    mask_iou,
    get_gt_count,
)

THRESHOLD = 0.50


def cleanup_with_log(predictions):
    ordered = sorted(
        predictions,
        key=lambda x: x["confidence"],
        reverse=True,
    )

    retained = []
    suppressed = []

    for candidate in ordered:
        best_match = None
        best_iou = 0.0

        for existing in retained:
            overlap = mask_iou(
                candidate["mask"],
                existing["mask"],
            )

            if overlap > best_iou:
                best_iou = overlap
                best_match = existing

        if best_match is not None and best_iou >= THRESHOLD:
            suppressed.append({
                "candidate": candidate,
                "kept": best_match,
                "iou": best_iou,
            })
        else:
            retained.append(candidate)

    return retained, suppressed


model = YOLO(str(MODEL_PATH))

rows = []

for image_path in sorted(IMAGE_DIR.iterdir()):

    if image_path.suffix.lower() not in {
        ".png", ".jpg", ".jpeg"
    }:
        continue

    gt = get_gt_count(image_path)

    if gt is None:
        continue

    raw = extract_predictions(
        model,
        image_path,
    )

    clean, suppressed = cleanup_with_log(raw)

    error = len(clean) - gt

    if error != 0:
        rows.append(
            (
                abs(error),
                error,
                image_path,
                gt,
                raw,
                clean,
                suppressed,
            )
        )


for _, error, image_path, gt, raw, clean, suppressed in sorted(
    rows,
    reverse=True,
):
    print()
    print("=" * 90)
    print(
        image_path.name,
        f"GT={gt}",
        f"RAW={len(raw)}",
        f"CLEAN={len(clean)}",
        f"ERROR={error:+d}",
    )

    print("-" * 90)
    print("SUPPRESSED PREDICTIONS")

    for item in sorted(
        suppressed,
        key=lambda x: x["iou"],
        reverse=True,
    ):
        candidate = item["candidate"]
        kept = item["kept"]

        print(
            f'{candidate["predicted_fdi"]:>2s} '
            f'conf={candidate["confidence"]:.3f}'
            f'  -> kept '
            f'{kept["predicted_fdi"]:>2s} '
            f'conf={kept["confidence"]:.3f}'
            f'  IoU={item["iou"]:.3f}'
        )
