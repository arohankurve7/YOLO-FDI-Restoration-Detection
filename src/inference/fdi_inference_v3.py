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

# IMPORTANT:
# V2 writes to a completely separate result directory.
OUTPUT_ROOT = ROOT / "results/fdi_inference_v2"

FDI = [
    "11", "12", "13", "14", "15", "16", "17", "18",
    "21", "22", "23", "24", "25", "26", "27", "28",
    "31", "32", "33", "34", "35", "36", "37", "38",
    "41", "42", "43", "44", "45", "46", "47", "48",
]

CROSS_FDI_IOU_THRESHOLD = 0.75


def mask_iou(a, b):
    intersection = np.logical_and(
        a,
        b
    ).sum()

    union = np.logical_or(
        a,
        b
    ).sum()

    if union == 0:
        return 0.0

    return float(
        intersection / union
    )


def run_inference(
    image_path,
    conf=0.25,
    imgsz=640
):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    case_dir = (
        OUTPUT_ROOT
        / image_path.stem
    )

    case_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model = YOLO(
        str(MODEL_PATH)
    )

    results = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        device="mps",
        verbose=False,
    )

    result = results[0]

    original = result.orig_img.copy()

    h, w = original.shape[:2]

    raw_detections = []

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

        for mask, cls, score in zip(
            masks,
            classes,
            confidences
        ):
            if cls < 0 or cls >= 32:
                continue

            fdi = str(
                model.names[cls]
            )

            mask = cv2.resize(
                mask,
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )

            binary = (
                mask > 0.5
            )

            if (
                np.count_nonzero(binary)
                == 0
            ):
                continue

            ys, xs = np.where(
                binary
            )

            cx = int(
                xs.mean()
            )

            cy = int(
                ys.mean()
            )

            x1 = int(
                xs.min()
            )

            y1 = int(
                ys.min()
            )

            x2 = int(
                xs.max()
            )

            y2 = int(
                ys.max()
            )

            contours, _ = cv2.findContours(
                binary.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            mask_polygon = []

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

                mask_polygon = [
                    {
                        "x": int(pt[0][0]),
                        "y": int(pt[0][1]),
                    }
                    for pt in approx
                ]

            raw_detections.append({
                "fdi": fdi,
                "confidence": float(score),

                "mask_area_px":
                    int(
                        np.count_nonzero(
                            binary
                        )
                    ),

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

                "mask_polygon":
                    mask_polygon,

                # Internal only.
                "_binary_mask":
                    binary,
            })

    # =====================================================
    # STAGE 1
    # Same-FDI duplicate suppression
    # =====================================================

    best_by_fdi = {}

    for det in raw_detections:
        fdi = det["fdi"]

        if (
            fdi not in best_by_fdi
            or det["confidence"]
            > best_by_fdi[fdi]["confidence"]
        ):
            best_by_fdi[fdi] = det

    detections = list(
        best_by_fdi.values()
    )

    # =====================================================
    # STAGE 2
    # Detect cross-FDI spatial conflicts.
    #
    # IMPORTANT:
    # We do NOT blindly delete one based only on confidence.
    # 17 vs 18 etc. requires anatomical reasoning.
    # =====================================================

    conflicts = []

    conflicting_fdis = set()

    for i in range(
        len(detections)
    ):
        for j in range(
            i + 1,
            len(detections)
        ):
            a = detections[i]
            b = detections[j]

            if (
                a["fdi"]
                == b["fdi"]
            ):
                continue

            overlap = mask_iou(
                a["_binary_mask"],
                b["_binary_mask"]
            )

            if (
                overlap
                < CROSS_FDI_IOU_THRESHOLD
            ):
                continue

            conflicting_fdis.add(
                a["fdi"]
            )

            conflicting_fdis.add(
                b["fdi"]
            )

            conflicts.append({
                "fdi_a":
                    a["fdi"],

                "confidence_a":
                    round(
                        a["confidence"],
                        4
                    ),

                "fdi_b":
                    b["fdi"],

                "confidence_b":
                    round(
                        b["confidence"],
                        4
                    ),

                "mask_iou":
                    round(
                        overlap,
                        4
                    ),

                "centroid_a":
                    a["centroid"],

                "centroid_b":
                    b["centroid"],

                "status":
                    "requires_anatomical_resolution",
            })


    # =====================================================
    # STAGE 2B
    # Conservative anatomical conflict resolution
    # =====================================================

    QUADRANTS = {
        "1": ["18","17","16","15","14","13","12","11"],
        "2": ["21","22","23","24","25","26","27","28"],
        "4": ["48","47","46","45","44","43","42","41"],
        "3": ["31","32","33","34","35","36","37","38"],
    }

    by_fdi = {
        det["fdi"]: det
        for det in detections
    }

    conflict_fdis = {
        fdi
        for conflict in conflicts
        for fdi in (
            conflict["fdi_a"],
            conflict["fdi_b"],
        )
    }

    def anatomical_expected_x(target_fdi):
        quadrant = target_fdi[0]
        sequence = QUADRANTS.get(quadrant)

        if sequence is None:
            return None

        anchors = []

        for idx, fdi_num in enumerate(sequence):
            if fdi_num not in by_fdi:
                continue

            if fdi_num in conflict_fdis:
                continue

            anchors.append(
                (
                    idx,
                    float(
                        by_fdi[fdi_num]
                        ["centroid"]["x"]
                    )
                )
            )

        if len(anchors) < 2:
            return None

        indices = np.array(
            [x[0] for x in anchors],
            dtype=float
        )

        xs = np.array(
            [x[1] for x in anchors],
            dtype=float
        )

        slope, intercept = np.polyfit(
            indices,
            xs,
            1
        )

        target_idx = sequence.index(
            target_fdi
        )

        expected_x = (
            slope * target_idx
            + intercept
        )

        return {
            "expected_x": float(expected_x),
            "anchor_count": len(anchors),
        }

    resolved_drop = set()
    unresolved_conflicts = []
    resolved_conflicts = []

    for conflict in conflicts:

        fdi_a = conflict["fdi_a"]
        fdi_b = conflict["fdi_b"]

        # Cross-quadrant conflicts near the midline are
        # deliberately NOT auto-resolved.
        if fdi_a[0] != fdi_b[0]:
            conflict["resolution"] = "unresolved_cross_quadrant"
            unresolved_conflicts.append(conflict)
            continue

        expected_a = anatomical_expected_x(
            fdi_a
        )

        expected_b = anatomical_expected_x(
            fdi_b
        )

        if (
            expected_a is None
            or expected_b is None
        ):
            conflict["resolution"] = "unresolved_insufficient_anchors"
            unresolved_conflicts.append(conflict)
            continue

        actual_a = float(
            by_fdi[fdi_a]["centroid"]["x"]
        )

        actual_b = float(
            by_fdi[fdi_b]["centroid"]["x"]
        )

        error_a = abs(
            actual_a
            - expected_a["expected_x"]
        )

        error_b = abs(
            actual_b
            - expected_b["expected_x"]
        )

        error_gap = abs(
            error_a - error_b
        )

        # Conservative rule:
        # only resolve when one option is clearly
        # more anatomically consistent.
        if error_gap < 8.0:
            conflict["resolution"] = "unresolved_close_scores"
            conflict["anatomical_error_a_px"] = round(error_a, 2)
            conflict["anatomical_error_b_px"] = round(error_b, 2)
            unresolved_conflicts.append(conflict)
            continue

        if error_a < error_b:
            keep_fdi = fdi_a
            drop_fdi = fdi_b
        else:
            keep_fdi = fdi_b
            drop_fdi = fdi_a

        resolved_drop.add(
            drop_fdi
        )

        conflict["resolution"] = "resolved_anatomically"
        conflict["keep_fdi"] = keep_fdi
        conflict["drop_fdi"] = drop_fdi
        conflict["anatomical_error_a_px"] = round(error_a, 2)
        conflict["anatomical_error_b_px"] = round(error_b, 2)

        resolved_conflicts.append(
            conflict
        )

    detections = [
        det
        for det in detections
        if det["fdi"] not in resolved_drop
    ]

    best_by_fdi = {
        det["fdi"]: det
        for det in detections
    }

    unresolved_fdis = {
        fdi
        for conflict in unresolved_conflicts
        for fdi in (
            conflict["fdi_a"],
            conflict["fdi_b"],
        )
    }

    # =====================================================
    # STAGE 3
    # Detection-state semantics
    #
    # Not detected != clinically missing.
    # =====================================================

    # Only anatomically resolved, non-conflicting labels
    # are considered confidently detected.
    detected_fdis = sorted(
        [
            fdi
            for fdi in best_by_fdi.keys()
            if fdi not in unresolved_fdis
        ],
        key=lambda x:
            FDI.index(x)
            if x in FDI
            else 999
    )

    suppressed_fdis = sorted(
        resolved_drop,
        key=lambda x:
            FDI.index(x)
            if x in FDI
            else 999
    )

    unresolved_fdi_teeth = sorted(
        unresolved_fdis,
        key=lambda x:
            FDI.index(x)
            if x in FDI
            else 999
    )

    # IMPORTANT:
    # A suppressed/conflicting label is NOT a clinically missing tooth.
    not_detected_fdis = [
        fdi
        for fdi in FDI
        if (
            fdi not in detected_fdis
            and fdi not in suppressed_fdis
            and fdi not in unresolved_fdis
        )
    ]

    # =====================================================
    # Draw overlay AFTER cleanup
    # =====================================================

    image = original.copy()
    overlay = original.copy()

    rng = np.random.default_rng(
        42
    )

    colors = rng.integers(
        60,
        255,
        size=(32, 3),
        dtype=np.uint8,
    )

    clean_json_detections = []

    for det in detections:
        fdi = det["fdi"]

        try:
            cls_idx = int(
                next(
                    k
                    for k, v
                    in model.names.items()
                    if str(v) == fdi
                )
            )
        except StopIteration:
            cls_idx = 0

        color = tuple(
            int(v)
            for v in colors[
                cls_idx % len(colors)
            ]
        )

        binary = det[
            "_binary_mask"
        ]

        overlay[
            binary
        ] = color

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

        cx = det[
            "centroid"
        ]["x"]

        cy = det[
            "centroid"
        ]["y"]

        label = fdi

        if (
            fdi
            in conflicting_fdis
        ):
            label = (
                f"{fdi}?"
            )

        cv2.putText(
            image,
            label,
            (
                cx - 10,
                cy
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        clean = {
            key: value
            for key, value
            in det.items()
            if key
            != "_binary_mask"
        }

        clean[
            "fdi_status"
        ] = (
            "conflict"
            if fdi in unresolved_fdis
            else "detected"
        )

        clean_json_detections.append(
            clean
        )

    final = cv2.addWeighted(
        overlay,
        0.35,
        image,
        0.65,
        0,
    )

    summary = {
        "image":
            str(image_path),

        "model":
            str(MODEL_PATH),

        "confidence_threshold":
            conf,

        "cross_fdi_iou_threshold":
            CROSS_FDI_IOU_THRESHOLD,

        "detected_teeth":
            detected_fdis,

        "number_detected":
            len(detected_fdis),

        # Deliberately renamed:
        # these are NOT clinically confirmed missing teeth.
        "not_detected_teeth":
            not_detected_fdis,

        "number_not_detected":
            len(not_detected_fdis),

        "conflicting_fdi_predictions":
            conflicts,

        "number_conflicts":
            len(conflicts),

        "resolved_conflicts":
            resolved_conflicts,

        "unresolved_conflicts":
            unresolved_conflicts,

        "suppressed_fdi_labels":
            suppressed_fdis,

        "unresolved_fdi_teeth":
            unresolved_fdi_teeth,

        "detections":
            clean_json_detections,
    }

    overlay_path = (
        case_dir
        / "segmentation_fdi_overlay_v2.png"
    )

    json_path = (
        case_dir
        / "prediction_v2.json"
    )

    cv2.imwrite(
        str(overlay_path),
        final
    )

    json_path.write_text(
        json.dumps(
            summary,
            indent=2
        )
    )

    print()
    print(
        "================================"
    )
    print(
        "FDI INFERENCE V2 COMPLETE"
    )
    print(
        "================================"
    )

    print(
        "Detected:",
        detected_fdis
    )

    print(
        "Detected count:",
        len(detected_fdis)
    )

    print(
        "Not detected:",
        not_detected_fdis
    )

    print()

    print(
        "Cross-FDI conflicts:",
        len(conflicts)
    )

    for conflict in conflicts:
        print(
            f'  {conflict["fdi_a"]} '
            f'vs {conflict["fdi_b"]} '
            f'IoU={conflict["mask_iou"]:.3f}'
        )

    print()
    print("Overlay:")
    print(
        overlay_path.resolve()
    )

    print()
    print("JSON:")
    print(
        json_path.resolve()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640
    )

    args = parser.parse_args()

    run_inference(
        args.image,
        args.conf,
        args.imgsz
    )
