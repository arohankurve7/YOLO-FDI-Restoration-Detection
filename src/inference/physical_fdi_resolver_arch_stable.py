from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT
    / "models"
    / "fdi_seg_v1"
    / "best.pt"
)

CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 640
DEVICE = "mps"

# Already established on the 50-image Dual evaluation.
PHYSICAL_DUPLICATE_IOU_THRESHOLD = 0.70


FDI_ORDER = [
    "18", "17", "16", "15", "14", "13", "12", "11",
    "21", "22", "23", "24", "25", "26", "27", "28",
    "48", "47", "46", "45", "44", "43", "42", "41",
    "31", "32", "33", "34", "35", "36", "37", "38",
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


def extract_raw_predictions(model, image_path):
    result = model.predict(
        source=str(image_path),
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        device=DEVICE,
        verbose=False,
    )[0]

    if (
        result.boxes is None
        or result.masks is None
    ):
        return [], None

    image = result.orig_img
    height, width = image.shape[:2]

    masks = (
        result.masks.data
        .cpu()
        .numpy()
    )

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

    detections = []

    for mask, cls, confidence in zip(
        masks,
        classes,
        confidences,
    ):
        mask = cv2.resize(
            mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

        binary = mask > 0.5

        ys, xs = np.where(binary)

        if len(xs) == 0:
            continue

        predicted_fdi = str(
            model.names[int(cls)]
        )

        detections.append({
            "predicted_fdi":
                predicted_fdi,

            "confidence":
                float(confidence),

            "centroid_x":
                float(xs.mean()),

            "centroid_y":
                float(ys.mean()),

            "mask":
                binary,

            "pixel_area":
                int(binary.sum()),
        })

    return detections, image


def physical_instance_cleanup(raw_detections):
    """
    Collapse multiple class hypotheses describing the
    SAME physical tooth.

    IMPORTANT:
    This is class-agnostic. FDI labels do not decide
    whether two predictions represent the same tooth.
    """

    ordered = sorted(
        raw_detections,
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

            if (
                overlap
                >= PHYSICAL_DUPLICATE_IOU_THRESHOLD
            ):
                duplicate = True
                break

        if not duplicate:
            retained.append(candidate)

    return retained


def split_arches(instances, image_height):
    """
    Separate maxillary and mandibular physical instances
    using spatial row structure rather than the image
    horizontal midpoint.

    For each tooth, compare its vertical position with
    nearby teeth in x. Panoramic upper/lower arches form
    two locally separated rows even though their global
    y-ranges overlap.
    """

    if not instances:
        return [], []

    ordered = sorted(
        instances,
        key=lambda item: item["centroid_x"],
    )

    xs = np.array(
        [item["centroid_x"] for item in ordered],
        dtype=float,
    )

    ys = np.array(
        [item["centroid_y"] for item in ordered],
        dtype=float,
    )

    # Normalize x/y so clustering is not dominated by
    # the much larger horizontal image dimension.
    x_scale = max(float(np.std(xs)), 1.0)
    y_scale = max(float(np.std(ys)), 1.0)

    features = np.column_stack([
        xs / x_scale,
        ys / y_scale,
    ]).astype(np.float32)

    # OpenCV k-means: two spatial dental arches.
    criteria = (
        cv2.TERM_CRITERIA_EPS
        + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.01,
    )

    _, labels, centers = cv2.kmeans(
        features,
        2,
        None,
        criteria,
        20,
        cv2.KMEANS_PP_CENTERS,
    )

    labels = labels.ravel()

    cluster_0 = [
        item
        for item, label in zip(ordered, labels)
        if label == 0
    ]

    cluster_1 = [
        item
        for item, label in zip(ordered, labels)
        if label == 1
    ]

    mean_y_0 = np.mean([
        item["centroid_y"]
        for item in cluster_0
    ])

    mean_y_1 = np.mean([
        item["centroid_y"]
        for item in cluster_1
    ])

    if mean_y_0 < mean_y_1:
        upper = cluster_0
        lower = cluster_1
    else:
        upper = cluster_1
        lower = cluster_0

    # -------------------------------------------------
    # Conservative arch reconciliation
    #
    # Spatial clustering remains primary. The model FDI
    # class is used only as weak evidence when the split
    # is imbalanced. We move only enough candidates to
    # restore balance, and only when their predicted FDI
    # clearly belongs to the opposite arch.
    # -------------------------------------------------

    def predicted_arch(item):
        fdi = str(item["predicted_fdi"])

        if fdi.startswith(("1", "2")):
            return "upper"

        if fdi.startswith(("3", "4")):
            return "lower"

        return "unknown"

    total = len(upper) + len(lower)

    # For complete 32-instance dentitions, a 16/16 split
    # is anatomically expected. This correction does NOT
    # run on missing-tooth cases.
    if total == 32:

        while len(upper) > 16 and len(lower) < 16:

            candidates = [
                item
                for item in upper
                if predicted_arch(item) == "lower"
            ]

            if not candidates:
                break

            # Prefer the candidate closest vertically to
            # the mandibular cluster.
            lower_mean_y = float(np.mean([
                item["centroid_y"]
                for item in lower
            ]))

            candidate = min(
                candidates,
                key=lambda item:
                abs(
                    item["centroid_y"]
                    - lower_mean_y
                ),
            )

            upper.remove(candidate)
            lower.append(candidate)

        while len(lower) > 16 and len(upper) < 16:

            candidates = [
                item
                for item in lower
                if predicted_arch(item) == "upper"
            ]

            if not candidates:
                break

            upper_mean_y = float(np.mean([
                item["centroid_y"]
                for item in upper
            ]))

            candidate = min(
                candidates,
                key=lambda item:
                abs(
                    item["centroid_y"]
                    - upper_mean_y
                ),
            )

            lower.remove(candidate)
            upper.append(candidate)

    # -------------------------------------------------
    # Missing-dentition conservative reconciliation
    #
    # With very sparse arches, spatial clustering alone
    # can fail because surviving teeth may be far apart.
    #
    # Use the model only for ARCH evidence here:
    # quadrants 1/2 = upper, 3/4 = lower.
    # Exact FDI numbering is still NOT trusted.
    # -------------------------------------------------

    if total < 32:

        ARCH_RECONCILE_CONFIDENCE = 0.70

        move_to_lower = [
            item
            for item in upper
            if (
                predicted_arch(item) == "lower"
                and item["confidence"]
                >= ARCH_RECONCILE_CONFIDENCE
            )
        ]

        move_to_upper = [
            item
            for item in lower
            if (
                predicted_arch(item) == "upper"
                and item["confidence"]
                >= ARCH_RECONCILE_CONFIDENCE
            )
        ]

        for item in move_to_lower:
            upper.remove(item)
            lower.append(item)

        for item in move_to_upper:
            lower.remove(item)
            upper.append(item)

    upper.sort(
        key=lambda item: item["centroid_x"]
    )

    lower.sort(
        key=lambda item: item["centroid_x"]
    )

    return upper, lower

def print_arch(title, instances):
    print()
    print(title)
    print("-" * 80)

    for index, item in enumerate(
        instances,
        start=1,
    ):
        print(
            f"{index:02d}  "
            f"x={item['centroid_x']:8.1f}  "
            f"y={item['centroid_y']:8.1f}  "
            f"model={item['predicted_fdi']:>2s}  "
            f"conf={item['confidence']:.3f}"
        )


def run(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            image_path
        )

    model = YOLO(
        str(MODEL_PATH)
    )

    raw, image = extract_raw_predictions(
        model,
        image_path,
    )

    if image is None:
        print("No detections.")
        return

    physical = physical_instance_cleanup(
        raw
    )

    height, width = image.shape[:2]

    upper, lower = split_arches(
        physical,
        height,
    )

    print()
    print("=" * 80)
    print("PHYSICAL-FIRST FDI RESOLVER - DIAGNOSTIC")
    print("=" * 80)

    print("Image       :", image_path.name)
    print("Image size  :", width, "x", height)
    print("Raw         :", len(raw))
    print("Physical    :", len(physical))
    print("Upper       :", len(upper))
    print("Lower       :", len(lower))

    print_arch(
        "UPPER PHYSICAL INSTANCES - LEFT TO RIGHT",
        upper,
    )

    print_arch(
        "LOWER PHYSICAL INSTANCES - LEFT TO RIGHT",
        lower,
    )

    print()
    print("=" * 80)
    print(
        "NOTE: No final FDI slots are assigned yet."
    )
    print(
        "This diagnostic verifies that physical teeth "
        "survive BEFORE anatomical numbering."
    )
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
    )

    args = parser.parse_args()

    run(args.image)
