from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "fdi_seg_v1" / "best.pt"

IMAGE_DIR = (
    ROOT
    / "datasets"
    / "processed"
    / "dual_yolo_seg"
    / "test"
    / "images"
)

LABEL_DIR = (
    ROOT
    / "datasets"
    / "processed"
    / "dual_yolo_seg"
    / "test"
    / "labels"
)

CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 640
DEVICE = "mps"

IOU_THRESHOLDS = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]


def mask_iou(mask_a, mask_b):
    intersection = np.logical_and(
        mask_a,
        mask_b,
    ).sum()

    union = np.logical_or(
        mask_a,
        mask_b,
    ).sum()

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

        if not np.any(binary_mask):
            continue

        predictions.append({
            "mask": binary_mask,
            "confidence": float(confidence),
            "class_id": int(class_id),
            "predicted_fdi": str(model.names[class_id]),
        })

    return predictions


def remove_spatial_duplicates(
    predictions,
    iou_threshold,
):
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

            if overlap >= iou_threshold:
                duplicate = True
                break

        if not duplicate:
            retained.append(candidate)

    return retained


def get_gt_count(image_path):
    label_path = (
        LABEL_DIR
        / f"{image_path.stem}.txt"
    )

    if not label_path.exists():
        return None

    return sum(
        1
        for line in label_path.read_text().splitlines()
        if line.strip()
    )


def main():

    model = YOLO(str(MODEL_PATH))

    images = sorted(
        p
        for p in IMAGE_DIR.iterdir()
        if p.suffix.lower()
        in {".png", ".jpg", ".jpeg"}
    )

    print()
    print(
        "INSTANCE CLEANUP THRESHOLD EVALUATION"
    )
    print("=" * 78)

    print(
        f"Images evaluated : {len(images)}"
    )

    print(
        f"Detection conf   : "
        f"{CONFIDENCE_THRESHOLD}"
    )

    print()
    print(
        f"{'IoU':>6s} "
        f"{'Exact':>8s} "
        f"{'Accuracy':>10s} "
        f"{'MAE':>10s} "
        f"{'TotalErr':>10s} "
        f"{'MaxErr':>8s}"
    )

    print("-" * 78)

    # Run YOLO only once per image.
    cached = []

    for image_path in images:

        predictions = extract_predictions(
            model,
            image_path,
        )

        gt_count = get_gt_count(
            image_path
        )

        if gt_count is None:
            continue

        cached.append(
            (
                image_path,
                predictions,
                gt_count,
            )
        )

    for threshold in IOU_THRESHOLDS:

        errors = []
        exact = 0

        for (
            image_path,
            predictions,
            gt_count,
        ) in cached:

            cleaned = remove_spatial_duplicates(
                predictions,
                threshold,
            )

            error = (
                len(cleaned)
                - gt_count
            )

            errors.append(error)

            if error == 0:
                exact += 1

        absolute_errors = [
            abs(error)
            for error in errors
        ]

        mae = float(
            np.mean(absolute_errors)
        )

        total_error = int(
            np.sum(absolute_errors)
        )

        max_error = int(
            np.max(absolute_errors)
        )

        accuracy = (
            exact / len(errors) * 100
        )

        print(
            f"{threshold:>6.2f} "
            f"{exact:>5d}/{len(errors):<2d} "
            f"{accuracy:>9.1f}% "
            f"{mae:>10.3f} "
            f"{total_error:>10d} "
            f"{max_error:>8d}"
        )


if __name__ == "__main__":
    main()
