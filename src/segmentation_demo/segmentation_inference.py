from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# =============================================================================
# Dental X-ray Analysis
# Physical Tooth Instance Segmentation - Inference Module
# =============================================================================
#
# Pipeline:
#   1. Run YOLO FDI segmentation model
#   2. Extract all raw segmentation hypotheses
#   3. Resolve highly overlapping predictions class-agnostically
#   4. Retain one mask per detected physical tooth
#   5. Produce tooth-instance overlay and structured metadata
#
# IMPORTANT:
# The YOLO model can produce multiple FDI-class hypotheses for the same
# physical tooth. Raw prediction count therefore MUST NOT be interpreted
# as physical tooth count.
#
# =============================================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "fdi_seg_v1" / "best.pt"

CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 640
DEVICE = "mps"

# Conservative class-agnostic duplicate suppression.
#
# Evaluation on the 50-image held-out Dual test split showed:
#   IoU 0.50 -> 35/50 exact counts, MAE 0.360
#   IoU 0.70 -> 34/50 exact counts, MAE 0.380
#
# 0.70 is intentionally retained as the conservative operating threshold
# to reduce the risk of merging genuinely adjacent tooth instances.
DUPLICATE_IOU_THRESHOLD = 0.70

MASK_COLORS = [
    (255, 80, 80),
    (80, 255, 80),
    (80, 160, 255),
    (255, 180, 80),
    (220, 80, 255),
    (80, 255, 220),
    (255, 120, 180),
    (180, 255, 80),
]


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Segmentation model not found: {MODEL_PATH}"
    )


model = YOLO(str(MODEL_PATH))


def _get_arch_from_fdi(fdi):
    fdi_str = str(fdi)

    if len(fdi_str) < 2:
        return "unknown"

    quadrant = fdi_str[0]

    if quadrant in ("1", "2"):
        return "upper"

    if quadrant in ("3", "4"):
        return "lower"

    return "unknown"


def _mask_iou(mask_a, mask_b):
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


def _extract_raw_predictions(result):
    """
    Convert raw YOLO output into full-resolution physical-mask candidates.
    """

    if result.masks is None or result.boxes is None:
        return []

    original = result.orig_img
    height, width = original.shape[:2]

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

        predicted_fdi = str(
            model.names[class_id]
        )

        predictions.append(
            {
                "mask": binary_mask,
                "class_id": int(class_id),
                "predicted_fdi": predicted_fdi,
                "confidence": float(confidence),
                "centroid_x": float(xs.mean()),
                "centroid_y": float(ys.mean()),
                "pixel_area": int(binary_mask.sum()),
            }
        )

    return predictions


def _resolve_physical_instances(predictions):
    """
    Resolve multiple class hypotheses representing the same physical tooth.

    Suppression is deliberately class-agnostic because a single physical
    tooth can be predicted simultaneously as multiple FDI classes.

    Highest-confidence hypothesis is retained when masks overlap above the
    configured duplicate IoU threshold.
    """

    ordered = sorted(
        predictions,
        key=lambda item: item["confidence"],
        reverse=True,
    )

    retained = []
    suppressed = []

    for candidate in ordered:
        duplicate_match = None
        duplicate_iou = 0.0

        for existing in retained:
            overlap = _mask_iou(
                candidate["mask"],
                existing["mask"],
            )

            if overlap > duplicate_iou:
                duplicate_iou = overlap
                duplicate_match = existing

        if (
            duplicate_match is not None
            and duplicate_iou >= DUPLICATE_IOU_THRESHOLD
        ):
            suppressed.append(
                {
                    "predicted_fdi":
                        candidate["predicted_fdi"],

                    "confidence":
                        candidate["confidence"],

                    "duplicate_of_fdi":
                        duplicate_match["predicted_fdi"],

                    "duplicate_of_confidence":
                        duplicate_match["confidence"],

                    "mask_iou":
                        duplicate_iou,
                }
            )

            continue

        retained.append(candidate)

    # Stable anatomical visualization order.
    retained = sorted(
        retained,
        key=lambda item: (
            item["centroid_y"],
            item["centroid_x"],
        ),
    )

    return retained, suppressed


def run_segmentation(image_path, opacity=0.35):
    """
    Perform physical tooth instance segmentation on a panoramic radiograph.

    Returns
    -------
    original_image : numpy.ndarray
    segmented_image : numpy.ndarray
    summary : dict

    Notes
    -----
    teeth_detected represents CLEAN PHYSICAL INSTANCES after class-agnostic
    duplicate suppression, not the number of raw YOLO predictions.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

    original_image = cv2.imread(
        str(image_path)
    )

    if original_image is None:
        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    results = model.predict(
        source=str(image_path),
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        device=DEVICE,
        verbose=False,
    )

    if not results:
        return (
            original_image,
            original_image.copy(),
            {
                "raw_predictions": 0,
                "suppressed_duplicates": 0,
                "teeth_detected": 0,
                "upper_arch": 0,
                "lower_arch": 0,
                "unknown_arch": 0,
                "instances": [],
                "suppression_log": [],
            },
        )

    result = results[0]

    raw_predictions = _extract_raw_predictions(
        result
    )

    physical_instances, suppressed = (
        _resolve_physical_instances(
            raw_predictions
        )
    )

    overlay = original_image.copy()

    upper_arch = 0
    lower_arch = 0
    unknown_arch = 0

    instances = []

    for instance_index, prediction in enumerate(
        physical_instances,
        start=1,
    ):
        predicted_fdi = prediction[
            "predicted_fdi"
        ]

        arch = _get_arch_from_fdi(
            predicted_fdi
        )

        if arch == "upper":
            upper_arch += 1

        elif arch == "lower":
            lower_arch += 1

        else:
            unknown_arch += 1

        binary_mask = prediction["mask"]

        color = MASK_COLORS[
            (instance_index - 1)
            % len(MASK_COLORS)
        ]

        overlay[binary_mask] = color

        instances.append(
            {
                "instance_id": instance_index,

                # This remains model metadata.
                # Final anatomical FDI resolution is a later stage.
                "predicted_fdi":
                    predicted_fdi,

                "confidence":
                    prediction["confidence"],

                "arch":
                    arch,

                "centroid": {
                    "x":
                        int(
                            round(
                                prediction[
                                    "centroid_x"
                                ]
                            )
                        ),

                    "y":
                        int(
                            round(
                                prediction[
                                    "centroid_y"
                                ]
                            )
                        ),
                },

                "pixel_area":
                    prediction["pixel_area"],
            }
        )

    opacity = float(
        np.clip(
            opacity,
            0.0,
            1.0,
        )
    )

    segmented_image = cv2.addWeighted(
        overlay,
        opacity,
        original_image,
        1.0 - opacity,
        0,
    )

    summary = {
        "raw_predictions":
            len(raw_predictions),

        "suppressed_duplicates":
            len(suppressed),

        "teeth_detected":
            len(physical_instances),

        "upper_arch":
            upper_arch,

        "lower_arch":
            lower_arch,

        "unknown_arch":
            unknown_arch,

        "duplicate_iou_threshold":
            DUPLICATE_IOU_THRESHOLD,

        "instances":
            instances,

        "suppression_log":
            suppressed,
    }

    return (
        original_image,
        segmented_image,
        summary,
    )


def save_segmentation_result(
    segmented_image,
    output_path,
):
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(output_path),
        segmented_image,
    )

    if not success:
        raise IOError(
            f"Unable to save segmentation result: "
            f"{output_path}"
        )

    return output_path
