from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "fdi_seg_v1" / "best.pt"

CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 640
DEVICE = "mps"

# Two highly overlapping predictions are treated as predictions
# of the same physical tooth, irrespective of the predicted class.
DUPLICATE_IOU_THRESHOLD = 0.70


def mask_iou(mask_a, mask_b):
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def extract_predictions(model, image_path):
    result = model.predict(
        source=str(image_path),
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        device=DEVICE,
        verbose=False,
    )[0]

    if result.masks is None or result.boxes is None:
        return []

    height, width = result.orig_img.shape[:2]

    masks = result.masks.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()

    predictions = []

    for mask, class_id, confidence in zip(
        masks,
        classes,
        confidences,
    ):
        mask = cv2.resize(
            mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

        binary_mask = mask > 0.5
        ys, xs = np.where(binary_mask)

        if len(xs) == 0:
            continue

        predictions.append({
            "mask": binary_mask,
            "confidence": float(confidence),
            "class_id": int(class_id),
            "predicted_fdi": str(model.names[class_id]),
            "centroid_x": float(xs.mean()),
            "centroid_y": float(ys.mean()),
        })

    return predictions


def remove_spatial_duplicates(predictions):
    # Confidence-first class-agnostic suppression.
    ordered = sorted(
        predictions,
        key=lambda item: item["confidence"],
        reverse=True,
    )

    retained = []

    for candidate in ordered:
        duplicate = False

        for existing in retained:
            overlap = mask_iou(
                candidate["mask"],
                existing["mask"],
            )

            if overlap >= DUPLICATE_IOU_THRESHOLD:
                duplicate = True
                break

        if not duplicate:
            retained.append(candidate)

    return retained


def gt_count_for_image(image_path):
    label_path = (
        ROOT
        / "datasets"
        / "processed"
        / "dual_yolo_seg"
        / "test"
        / "labels"
        / f"{image_path.stem}.txt"
    )

    if not label_path.exists():
        return None

    return len([
        line
        for line in label_path.read_text().splitlines()
        if line.strip()
    ])


def main():
    model = YOLO(str(MODEL_PATH))
    test_root = ROOT / "TEST"

    image_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
    }

    images = sorted(
        path
        for path in test_root.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in image_extensions
        )
    )

    print()
    print("CLASS-AGNOSTIC INSTANCE CLEANUP")
    print("=" * 105)

    print(
        f"{'Category':30s} "
        f"{'Image':45s} "
        f"{'GT':>4s} "
        f"{'Raw':>5s} "
        f"{'Clean':>6s} "
        f"{'Error':>6s}"
    )

    print("-" * 105)

    for image_path in images:
        predictions = extract_predictions(
            model,
            image_path,
        )

        cleaned = remove_spatial_duplicates(
            predictions
        )

        gt_count = gt_count_for_image(
            image_path
        )

        if gt_count is None:
            gt_text = "-"
            error_text = "-"
        else:
            gt_text = str(gt_count)
            error_text = str(
                len(cleaned) - gt_count
            )

        print(
            f"{image_path.parent.name:30s} "
            f"{image_path.name:45s} "
            f"{gt_text:>4s} "
            f"{len(predictions):>5d} "
            f"{len(cleaned):>6d} "
            f"{error_text:>6s}"
        )


if __name__ == "__main__":
    main()
