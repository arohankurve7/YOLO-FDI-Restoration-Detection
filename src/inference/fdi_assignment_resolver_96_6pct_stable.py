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

    # -------------------------------------------------
    # Sparse-arch spatial outlier reconciliation
    #
    # A prediction is moved only when BOTH:
    #   1. its model quadrant indicates the opposite arch
    #   2. its vertical position is a strong outlier from
    #      its current spatial cluster and closer to the
    #      opposite arch cluster
    #
    # This is deliberately conservative.
    # -------------------------------------------------

    if total < 32 and upper and lower:

        upper_median_y = float(np.median([
            item["centroid_y"]
            for item in upper
        ]))

        lower_median_y = float(np.median([
            item["centroid_y"]
            for item in lower
        ]))

        separation = abs(
            lower_median_y
            - upper_median_y
        )

        # Require meaningful vertical arch separation.
        if separation >= 40.0:

            move_to_lower = []

            for item in upper:

                if predicted_arch(item) != "lower":
                    continue

                y = float(item["centroid_y"])

                dist_upper = abs(
                    y - upper_median_y
                )

                dist_lower = abs(
                    y - lower_median_y
                )

                if (
                    dist_lower + 20.0
                    < dist_upper
                ):
                    move_to_lower.append(item)

            move_to_upper = []

            for item in lower:

                if predicted_arch(item) != "upper":
                    continue

                y = float(item["centroid_y"])

                dist_lower = abs(
                    y - lower_median_y
                )

                dist_upper = abs(
                    y - upper_median_y
                )

                if (
                    dist_upper + 20.0
                    < dist_lower
                ):
                    move_to_upper.append(item)

            for item in move_to_lower:
                upper.remove(item)
                lower.append(item)

            for item in move_to_upper:
                lower.remove(item)
                upper.append(item)

    # -------------------------------------------------
    # Boundary-ambiguous arch reconciliation
    #
    # Some posterior teeth lie very close to the vertical
    # boundary between the two arches. In that narrow
    # ambiguity zone, use the model quadrant only as a
    # tie-breaker when geometry is also suspicious.
    # -------------------------------------------------

    if total < 32 and upper and lower:

        upper_median_y = float(np.median([
            item["centroid_y"]
            for item in upper
        ]))

        lower_median_y = float(np.median([
            item["centroid_y"]
            for item in lower
        ]))

        separation = abs(
            lower_median_y - upper_median_y
        )

        boundary_y = (
            upper_median_y
            + lower_median_y
        ) / 2.0

        AMBIGUITY_BAND_PX = 30.0
        MIN_MODEL_CONF = 0.30
        MIN_CURRENT_ARCH_DEVIATION = (
            0.35 * separation
        )

        move_to_lower = []

        for item in upper:

            if predicted_arch(item) != "lower":
                continue

            if item["confidence"] < MIN_MODEL_CONF:
                continue

            y = float(item["centroid_y"])

            if (
                abs(y - boundary_y)
                <= AMBIGUITY_BAND_PX
                and abs(y - upper_median_y)
                >= MIN_CURRENT_ARCH_DEVIATION
            ):
                move_to_lower.append(item)

        move_to_upper = []

        for item in lower:

            if predicted_arch(item) != "upper":
                continue

            if item["confidence"] < MIN_MODEL_CONF:
                continue

            y = float(item["centroid_y"])

            if (
                abs(y - boundary_y)
                <= AMBIGUITY_BAND_PX
                and abs(y - lower_median_y)
                >= MIN_CURRENT_ARCH_DEVIATION
            ):
                move_to_upper.append(item)

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



UPPER_FULL_ORDER = [
    "18", "17", "16", "15", "14", "13", "12", "11",
    "21", "22", "23", "24", "25", "26", "27", "28",
]

LOWER_FULL_ORDER = [
    "48", "47", "46", "45", "44", "43", "42", "41",
    "31", "32", "33", "34", "35", "36", "37", "38",
]


def assign_full_dentition_fdi(upper, lower):
    """
    Deterministic anatomical assignment for a complete
    32-tooth physical dentition.

    Requires exactly 16 physical instances in each arch.
    """

    if len(upper) != 16 or len(lower) != 16:
        return None

    assignments = []

    for instance, resolved_fdi in zip(
        upper,
        UPPER_FULL_ORDER,
    ):
        assignments.append({
            "resolved_fdi": resolved_fdi,
            "predicted_fdi": instance["predicted_fdi"],
            "confidence": instance["confidence"],
            "centroid_x": instance["centroid_x"],
            "centroid_y": instance["centroid_y"],
        })

    for instance, resolved_fdi in zip(
        lower,
        LOWER_FULL_ORDER,
    ):
        assignments.append({
            "resolved_fdi": resolved_fdi,
            "predicted_fdi": instance["predicted_fdi"],
            "confidence": instance["confidence"],
            "centroid_x": instance["centroid_x"],
            "centroid_y": instance["centroid_y"],
        })

    return assignments


def print_missing_aware_assignment(assignments):
    print()
    print("FULL-DENTITION FDI ASSIGNMENT")
    print("-" * 72)

    print(
        f"{'Resolved':>10s} "
        f"{'Model':>8s} "
        f"{'Conf':>8s} "
        f"{'X':>9s} "
        f"{'Y':>9s}"
    )

    print("-" * 72)

    for item in assignments:
        marker = (
            ""
            if item["resolved_fdi"] == item["predicted_fdi"]
            else "*"
        )

        print(
            f"{item['resolved_fdi']:>10s} "
            f"{item['predicted_fdi']:>8s} "
            f"{item['confidence']:8.3f} "
            f"{item['centroid_x']:9.1f} "
            f"{item['centroid_y']:9.1f} "
            f"{marker}"
        )



def _arch_order(arch):
    if arch == "upper":
        return [
            "18", "17", "16", "15", "14", "13", "12", "11",
            "21", "22", "23", "24", "25", "26", "27", "28",
        ]

    return [
        "48", "47", "46", "45", "44", "43", "42", "41",
        "31", "32", "33", "34", "35", "36", "37", "38",
    ]


def assign_sparse_arch_fdi(instances, arch):
    """
    Dependency-free monotonic FDI assignment.

    Each physical instance receives exactly one unique FDI slot.
    Final assignments must preserve anatomical left-to-right order.

    Model FDI predictions are treated as priors, not absolute truth.
    Dynamic programming finds the globally lowest-cost ordered
    assignment of N physical teeth into the 16 anatomical slots.
    """

    if not instances:
        return []

    order = _arch_order(arch)

    instances = sorted(
        instances,
        key=lambda item: item["centroid_x"],
    )

    n = len(instances)
    m = len(order)

    if n > m:
        return None

    # -------------------------------------------------
    # Cost of assigning one physical instance to one
    # anatomical FDI slot.
    # -------------------------------------------------

    def assignment_cost(instance, slot_index):
        target_fdi = order[slot_index]
        predicted_fdi = str(
            instance["predicted_fdi"]
        )

        confidence = float(
            instance["confidence"]
        )

        # Position prior:
        # compare normalized physical rank against
        # normalized anatomical slot position.
        if n == 1:
            physical_position = 0.5
        else:
            physical_position = (
                instance["_rank"] / (n - 1)
            )

        anatomical_position = (
            slot_index / (m - 1)
        )

        position_cost = (
            abs(
                physical_position
                - anatomical_position
            )
            * 2.0
        )

        # Model FDI prior.
        #
        # Correct model prediction receives zero penalty.
        # Wrong prediction receives confidence-weighted penalty.
        # High-confidence model labels therefore matter more,
        # while weak labels can be corrected anatomically.
        if predicted_fdi == target_fdi:
            model_cost = 0.0
        else:
            model_cost = (
                0.55
                + 1.25 * confidence
            )

            # If prediction belongs to this same arch,
            # penalize according to anatomical distance
            # from the proposed slot.
            if predicted_fdi in order:
                predicted_index = order.index(
                    predicted_fdi
                )

                distance = abs(
                    predicted_index
                    - slot_index
                )

                model_cost += (
                    0.10 * distance
                )
            else:
                # Cross-arch prediction should already
                # have been handled by arch reconciliation.
                model_cost += 0.75

        return (
            position_cost
            + model_cost
        )

    working = []

    for rank, instance in enumerate(instances):
        item = dict(instance)
        item["_rank"] = rank
        working.append(item)

    # -------------------------------------------------
    # Dynamic programming:
    #
    # dp[i][j] = minimum cost for assigning first i
    # physical instances using first j anatomical slots.
    #
    # A slot may be skipped => missing tooth.
    # A physical instance may NOT be skipped.
    # -------------------------------------------------

    inf = float("inf")

    dp = [
        [inf] * (m + 1)
        for _ in range(n + 1)
    ]

    parent = [
        [None] * (m + 1)
        for _ in range(n + 1)
    ]

    for j in range(m + 1):
        dp[0][j] = 0.0

        if j > 0:
            parent[0][j] = (
                0,
                j - 1,
                "skip",
            )

    for i in range(1, n + 1):
        for j in range(1, m + 1):

            # Option 1:
            # anatomical slot j-1 is missing.
            if dp[i][j - 1] < dp[i][j]:
                dp[i][j] = dp[i][j - 1]

                parent[i][j] = (
                    i,
                    j - 1,
                    "skip",
                )

            # Option 2:
            # assign physical instance i-1
            # to anatomical slot j-1.
            if dp[i - 1][j - 1] != inf:

                cost = (
                    dp[i - 1][j - 1]
                    + assignment_cost(
                        working[i - 1],
                        j - 1,
                    )
                )

                if cost < dp[i][j]:
                    dp[i][j] = cost

                    parent[i][j] = (
                        i - 1,
                        j - 1,
                        "assign",
                    )

    # -------------------------------------------------
    # Backtrack optimal assignment.
    # -------------------------------------------------

    i = n
    j = m

    assigned = []

    while i > 0 or j > 0:

        step = parent[i][j]

        if step is None:
            raise RuntimeError(
                "Unable to reconstruct sparse FDI assignment."
            )

        prev_i, prev_j, action = step

        if action == "assign":
            instance = working[i - 1]

            assigned.append({
                "resolved_fdi":
                    order[j - 1],

                "predicted_fdi":
                    str(
                        instance[
                            "predicted_fdi"
                        ]
                    ),

                "confidence":
                    float(
                        instance[
                            "confidence"
                        ]
                    ),

                "centroid_x":
                    float(
                        instance[
                            "centroid_x"
                        ]
                    ),

                "centroid_y":
                    float(
                        instance[
                            "centroid_y"
                        ]
                    ),

                "assignment_cost":
                    float(
                        assignment_cost(
                            instance,
                            j - 1,
                        )
                    ),
            })

        i = prev_i
        j = prev_j

    assigned.reverse()

    return assigned


def assign_missing_aware_fdi(upper, lower):
    """
    Unified assignment entry point.

    Complete dentition:
        deterministic 16/16 anatomical assignment.

    Missing dentition:
        monotonic model-assisted dynamic programming.
    """

    if len(upper) == 16 and len(lower) == 16:
        return assign_full_dentition_fdi(
            upper,
            lower,
        )

    upper_assignments = assign_sparse_arch_fdi(
        upper,
        "upper",
    )

    lower_assignments = assign_sparse_arch_fdi(
        lower,
        "lower",
    )

    if (
        upper_assignments is None
        or lower_assignments is None
    ):
        return None

    return (
        upper_assignments
        + lower_assignments
    )


def print_missing_aware_assignment(assignments):
    print()
    print("MISSING-AWARE FDI ASSIGNMENT")
    print("-" * 82)

    print(
        f"{'Resolved':>10s} "
        f"{'Model':>8s} "
        f"{'Conf':>8s} "
        f"{'X':>9s} "
        f"{'Y':>9s} "
        f"{'Cost':>9s}"
    )

    print("-" * 82)

    for item in assignments:

        changed = (
            ""
            if (
                item["resolved_fdi"]
                == item["predicted_fdi"]
            )
            else "*"
        )

        cost = item.get(
            "assignment_cost",
            0.0,
        )

        print(
            f"{item['resolved_fdi']:>10s} "
            f"{item['predicted_fdi']:>8s} "
            f"{item['confidence']:8.3f} "
            f"{item['centroid_x']:9.1f} "
            f"{item['centroid_y']:9.1f} "
            f"{cost:9.3f} "
            f"{changed}"
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

    # -------------------------------------------------
    # Full-dentition anatomical FDI assignment
    # -------------------------------------------------

    assignments = assign_missing_aware_fdi(
        upper,
        lower,
    )

    if assignments is not None:
        print_missing_aware_assignment(assignments)

        resolved = [
            item["resolved_fdi"]
            for item in assignments
        ]

        print()
        print("Resolved FDI count :", len(resolved))
        print(
            "Unique FDI count   :",
            len(set(resolved)),
        )

        print(
            "Resolved FDI       :",
            sorted(
                resolved,
                key=lambda x: (
                    int(x[0]),
                    int(x[1]),
                ),
            ),
        )
    else:
        print()
        print(
            "Full-dentition assignment skipped "
            "(arch is not 16/16)."
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
