import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models/fdi_seg_v1/best.pt"
OUTPUT_ROOT = ROOT / "results/fdi_inference"

FDI = [
    "11", "12", "13", "14", "15", "16", "17", "18",
    "21", "22", "23", "24", "25", "26", "27", "28",
    "31", "32", "33", "34", "35", "36", "37", "38",
    "41", "42", "43", "44", "45", "46", "47", "48",
]


def run_inference(image_path, conf=0.25, imgsz=640):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    case_dir = OUTPUT_ROOT / image_path.stem
    case_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))

    results = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        device="mps",
        verbose=False,
    )

    result = results[0]
    image = result.orig_img.copy()
    overlay = image.copy()

    present = []
    detections = []

    if result.boxes is not None and result.masks is not None:
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        masks = result.masks.data.cpu().numpy()

        h, w = image.shape[:2]

        rng = np.random.default_rng(42)
        colors = rng.integers(
            60,
            255,
            size=(32, 3),
            dtype=np.uint8,
        )

        for mask, cls, score in zip(masks, classes, confidences):
            if cls < 0 or cls >= 32:
                continue

            fdi = str(model.names[cls])

            mask = cv2.resize(
                mask,
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )

            binary = mask > 0.5

            if np.count_nonzero(binary) == 0:
                continue

            color = tuple(int(v) for v in colors[cls])

            overlay[binary] = color

            contours, _ = cv2.findContours(
                binary.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            cv2.drawContours(
                image,
                contours,
                -1,
                color,
                2,
            )

            ys, xs = np.where(binary)

            cx = int(xs.mean())
            cy = int(ys.mean())

            # Bounding box from the final binary mask
            x1 = int(xs.min())
            y1 = int(ys.min())
            x2 = int(xs.max())
            y2 = int(ys.max())

            # Keep only the largest contour as the primary tooth polygon.
            # Simplify it so prediction.json stays compact enough for UI use.
            mask_polygon = []

            if contours:
                largest = max(contours, key=cv2.contourArea)

                epsilon = 0.004 * cv2.arcLength(
                    largest,
                    True
                )

                approx = cv2.approxPolyDP(
                    largest,
                    epsilon,
                    True
                )

                mask_polygon = [
                    {
                        "x": int(pt[0][0]),
                        "y": int(pt[0][1]),
                    }
                    for pt in approx
                ]

            cv2.putText(
                image,
                fdi,
                (cx - 10, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            present.append(fdi)

            detections.append({
                "fdi": fdi,
                "confidence": float(score),
                "mask_area_px": int(np.count_nonzero(binary)),
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
                "mask_polygon": mask_polygon,
            })

    final = cv2.addWeighted(
        overlay,
        0.35,
        image,
        0.65,
        0,
    )

    # Keep only the highest-confidence detection for each FDI.
    best_by_fdi = {}

    for det in detections:
        fdi = det["fdi"]

        if (
            fdi not in best_by_fdi
            or det["confidence"] > best_by_fdi[fdi]["confidence"]
        ):
            best_by_fdi[fdi] = det

    detections = list(best_by_fdi.values())

    present = sorted(
        best_by_fdi.keys(),
        key=lambda x: FDI.index(x) if x in FDI else 999,
    )

    missing = [
        fdi
        for fdi in FDI
        if fdi not in present
    ]

    summary = {
        "image": str(image_path),
        "model": str(MODEL_PATH),
        "confidence_threshold": conf,
        "present_teeth": present,
        "missing_teeth_candidates": missing,
        "number_present": len(present),
        "number_missing_candidates": len(missing),
        "detections": detections,
    }

    overlay_path = case_dir / "segmentation_fdi_overlay.png"
    json_path = case_dir / "prediction.json"

    cv2.imwrite(str(overlay_path), final)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n================================")
    print("FDI INFERENCE COMPLETE")
    print("================================")
    print("Present:", present)
    print("Present count:", len(present))
    print("Missing candidates:", missing)
    print("\nOverlay:")
    print(overlay_path)
    print("\nJSON:")
    print(json_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help="Path to panoramic radiograph",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )

    args = parser.parse_args()

    run_inference(
        args.image,
        conf=args.conf,
        imgsz=args.imgsz,
    )
