from pathlib import Path
import argparse
import json

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL = (
    ROOT
    / "runs/segment/runs/restorations_seg_v1/weights/best.pt"
)


# Final V1 per-class thresholds derived from held-out test evaluation.
CLASS_THRESHOLDS = {
    "implant": 0.30,
    "bridge": 0.45,
    "Crown": 0.40,
    "Dental_Fillings": 0.40,
}


def run_inference(image_path, model_path, conf=0.25):

    image_path = Path(image_path)
    model_path = Path(model_path)

    if not image_path.is_absolute():
        image_path = ROOT / image_path

    if not model_path.is_absolute():
        model_path = ROOT / model_path

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    model = YOLO(str(model_path))

    # Low global threshold is intentionally retained here.
    # Final filtering is performed using class-specific thresholds below.
    results = model.predict(
        source=str(image_path),
        conf=conf,
        verbose=False,
    )

    result = results[0]

    image = result.orig_img.copy()
    overlay = image.copy()

    h, w = image.shape[:2]

    detections = []

    if (
        result.boxes is not None
        and result.masks is not None
    ):

        classes = (
            result.boxes.cls
            .cpu()
            .numpy()
            .astype(int)
        )

        confidences = (
            result.boxes.conf
            .cpu()
            .numpy()
        )

        masks = (
            result.masks.data
            .cpu()
            .numpy()
        )

        rng = np.random.default_rng(123)

        colors = rng.integers(
            70,
            255,
            size=(len(model.names), 3),
            dtype=np.uint8,
        )

        for mask, cls, score in zip(
            masks,
            classes,
            confidences
        ):

            if cls not in model.names:
                continue

            label = str(model.names[cls])

            # V1 scope:
            # implant, bridge, Crown, Dental_Fillings only.
            # RCT / Endodontic Post and other classes are ignored.
            threshold = CLASS_THRESHOLDS.get(label)

            if threshold is None:
                continue

            if float(score) < threshold:
                continue

            mask = cv2.resize(
                mask,
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )

            binary = mask > 0.5

            if np.count_nonzero(binary) == 0:
                continue

            ys, xs = np.where(binary)

            x1 = int(xs.min())
            y1 = int(ys.min())
            x2 = int(xs.max())
            y2 = int(ys.max())

            cx = int(xs.mean())
            cy = int(ys.mean())

            contours, _ = cv2.findContours(
                binary.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            polygon = []

            if contours:

                largest = max(
                    contours,
                    key=cv2.contourArea
                )

                epsilon = (
                    0.004
                    * cv2.arcLength(
                        largest,
                        True
                    )
                )

                approx = cv2.approxPolyDP(
                    largest,
                    epsilon,
                    True
                )

                polygon = [
                    {
                        "x": int(pt[0][0]),
                        "y": int(pt[0][1]),
                    }
                    for pt in approx
                ]

            color = tuple(
                int(v)
                for v in colors[cls]
            )

            overlay[binary] = color

            cv2.drawContours(
                image,
                contours,
                -1,
                color,
                2,
            )

            cv2.putText(
                image,
                label,
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            detections.append({
                "class_id": int(cls),
                "class_name": label,
                "confidence": float(score),
                "threshold_used": float(threshold),

                "mask_area_px":
                    int(np.count_nonzero(binary)),

                "centroid": {
                    "x": cx,
                    "y": cy,
                },

                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },

                "mask_polygon": polygon,
            })

    final = cv2.addWeighted(
        overlay,
        0.35,
        image,
        0.65,
        0,
    )

    case_id = image_path.stem

    case_dir = (
        ROOT
        / "results/restoration_inference"
        / case_id
    )

    case_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    overlay_path = (
        case_dir
        / "restoration_overlay.png"
    )

    cv2.imwrite(
        str(overlay_path),
        final
    )

    summary = {
        "image": str(
            image_path.relative_to(ROOT)
            if image_path.is_relative_to(ROOT)
            else image_path
        ),

        "model": str(model_path),

        "base_confidence_threshold": conf,

        "class_thresholds": CLASS_THRESHOLDS,

        "supported_classes": [
            "implant",
            "bridge",
            "Crown",
            "Dental_Fillings",
        ],

        "number_detections":
            len(detections),

        "detections":
            detections,
    }

    json_path = (
        case_dir
        / "prediction.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2
        )

    print()
    print("================================")
    print("RESTORATION INFERENCE COMPLETE")
    print("================================")
    print("Detections:", len(detections))
    print()

    for det in detections:
        print(
            f'{det["class_name"]:28s} '
            f'{det["confidence"]:.3f} '
            f'(threshold={det["threshold_used"]:.2f})'
        )

    print()
    print("Overlay:")
    print(overlay_path.resolve())

    print()
    print("JSON:")
    print(json_path.resolve())


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True
    )

    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL)
    )

    # YOLO pre-filter only.
    # Final filtering uses CLASS_THRESHOLDS.
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25
    )

    args = parser.parse_args()

    run_inference(
        args.image,
        args.model,
        args.conf
    )
