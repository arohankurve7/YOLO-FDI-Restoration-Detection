from pathlib import Path
import json
import math

import cv2
import numpy as np
from ultralytics import YOLO

from src.inference.fdi_assignment_resolver_96_6pct_stable import (
    MODEL_PATH,
    extract_raw_predictions,
    physical_instance_cleanup,
    split_arches,
    assign_missing_aware_fdi,
)


ROOT = Path(__file__).resolve().parents[2]

OUTPUT_ROOT = (
    ROOT
    / "results"
    / "final_fdi_inference"
)


# =============================================================================
# FDI permanent dentition
# =============================================================================

FDI_ORDER = [
    "18", "17", "16", "15", "14", "13", "12", "11",
    "21", "22", "23", "24", "25", "26", "27", "28",
    "48", "47", "46", "45", "44", "43", "42", "41",
    "31", "32", "33", "34", "35", "36", "37", "38",
]


# =============================================================================
# Locked tooth-type color scheme
#
# BGR because OpenCV uses BGR, not RGB.
# Same colors must later be used by the dashboard tooth chart.
# =============================================================================

TOOTH_TYPE_COLORS = {
    "Third Molar":      (40, 90, 210),    # dark orange
    "Second Molar":     (20, 145, 245),   # orange
    "First Molar":      (40, 205, 245),   # yellow
    "Second Premolar":  (80, 180, 70),    # green
    "First Premolar":   (210, 180, 70),   # cyan
    "Canine":           (220, 100, 60),   # blue
    "Lateral Incisor":  (190, 80, 170),   # purple
    "Central Incisor":  (80, 70, 220),    # red
}


def tooth_type_from_fdi(fdi):
    position = int(str(fdi)[1])

    mapping = {
        1: "Central Incisor",
        2: "Lateral Incisor",
        3: "Canine",
        4: "First Premolar",
        5: "Second Premolar",
        6: "First Molar",
        7: "Second Molar",
        8: "Third Molar",
    }

    return mapping[position]


def quadrant_metadata(fdi):
    quadrant = str(fdi)[0]

    mapping = {
        "1": {
            "quadrant": "Q1",
            "arch": "Maxillary",
            "side": "Right",
        },
        "2": {
            "quadrant": "Q2",
            "arch": "Maxillary",
            "side": "Left",
        },
        "3": {
            "quadrant": "Q3",
            "arch": "Mandibular",
            "side": "Left",
        },
        "4": {
            "quadrant": "Q4",
            "arch": "Mandibular",
            "side": "Right",
        },
    }

    return mapping[quadrant]


def anatomical_name(fdi):
    meta = quadrant_metadata(fdi)
    tooth_type = tooth_type_from_fdi(fdi)

    if tooth_type == "Third Molar":
        tooth_type = "Third Molar (Wisdom Tooth)"

    return (
        f'{meta["arch"]} '
        f'{meta["side"]} '
        f'{tooth_type}'
    )


def mask_to_polygon(binary_mask):
    mask = (
        binary_mask.astype(np.uint8)
        * 255
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return []

    largest = max(
        contours,
        key=cv2.contourArea,
    )

    epsilon = (
        0.003
        * cv2.arcLength(
            largest,
            True,
        )
    )

    approx = cv2.approxPolyDP(
        largest,
        epsilon,
        True,
    )

    return [
        {
            "x": int(point[0][0]),
            "y": int(point[0][1]),
        }
        for point in approx
    ]


def nearest_physical_instance(
    assignment,
    physical_instances,
):
    """
    Assignment results retain physical centroids but do not carry masks.

    Reconnect the resolved FDI assignment to its original physical instance
    so the dashboard can receive segmentation geometry.
    """

    ax = float(
        assignment["centroid_x"]
    )

    ay = float(
        assignment["centroid_y"]
    )

    return min(
        physical_instances,
        key=lambda item:
        math.hypot(
            float(item["centroid_x"]) - ax,
            float(item["centroid_y"]) - ay,
        ),
    )


def build_detected_tooth(
    assignment,
    physical,
):
    fdi = str(
        assignment["resolved_fdi"]
    )

    meta = quadrant_metadata(fdi)
    tooth_type = tooth_type_from_fdi(fdi)

    binary_mask = physical["mask"]

    ys, xs = np.where(binary_mask)

    if len(xs):
        bbox = {
            "x1": int(xs.min()),
            "y1": int(ys.min()),
            "x2": int(xs.max()),
            "y2": int(ys.max()),
        }
    else:
        bbox = None

    return {
        "fdi": fdi,

        "anatomical_name":
            anatomical_name(fdi),

        "tooth_type":
            tooth_type,

        "quadrant":
            meta["quadrant"],

        "arch":
            meta["arch"],

        "side":
            meta["side"],

        "status":
            "present",

        "selectable":
            True,

        "centroid": {
            "x": int(
                round(
                    physical["centroid_x"]
                )
            ),
            "y": int(
                round(
                    physical["centroid_y"]
                )
            ),
        },

        "bbox":
            bbox,

        "mask_polygon":
            mask_to_polygon(
                binary_mask
            ),

        "pixel_area":
            int(
                physical["pixel_area"]
            ),
    }


def build_not_detected_tooth(fdi):
    meta = quadrant_metadata(fdi)

    return {
        "fdi":
            fdi,

        "anatomical_name":
            anatomical_name(fdi),

        "tooth_type":
            tooth_type_from_fdi(fdi),

        "quadrant":
            meta["quadrant"],

        "arch":
            meta["arch"],

        "side":
            meta["side"],

        "status":
            "not_detected",

        # Dashboard rule:
        # greyed out and non-clickable.
        "selectable":
            False,

        "centroid":
            None,

        "bbox":
            None,

        "mask_polygon":
            [],
    }


def draw_overlay(
    image,
    detected_teeth,
    detected_map,
    opacity=0.35,
):
    overlay = image.copy()
    output = image.copy()

    for tooth in detected_teeth:

        fdi = tooth["fdi"]

        physical = detected_map[fdi]

        binary_mask = physical["mask"]

        tooth_type = tooth[
            "tooth_type"
        ]

        color = TOOTH_TYPE_COLORS[
            tooth_type
        ]

        overlay[binary_mask] = color

    output = cv2.addWeighted(
        overlay,
        opacity,
        output,
        1.0 - opacity,
        0,
    )

    # Draw clean contours + resolved FDI labels.
    for tooth in detected_teeth:

        fdi = tooth["fdi"]
        physical = detected_map[fdi]

        binary_mask = (
            physical["mask"]
            .astype(np.uint8)
            * 255
        )

        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        color = TOOTH_TYPE_COLORS[
            tooth["tooth_type"]
        ]

        cv2.drawContours(
            output,
            contours,
            -1,
            color,
            2,
        )

        cx = tooth["centroid"]["x"]
        cy = tooth["centroid"]["y"]

        cv2.putText(
            output,
            fdi,
            (
                cx - 12,
                cy,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return output


def run_final_fdi_inference(
    image_path,
):
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

    # -------------------------------------------------
    # Zero-detection / edentulous-safe image fallback
    #
    # Some detector paths return image=None when there
    # are no tooth detections. That is a valid clinical
    # outcome for an edentulous OPG, not an inference
    # failure.
    # -------------------------------------------------

    if image is None:
        image = cv2.imread(
            str(image_path)
        )

    if image is None:
        raise RuntimeError(
            "Unable to read input OPG image."
        )

    physical = physical_instance_cleanup(
        raw
    )

    # -------------------------------------------------
    # Valid edentulous result
    #
    # No physical teeth detected:
    #   detected = 0
    #   missing  = 32
    #   chart    = all grey / disabled
    #   overlay  = original OPG
    # -------------------------------------------------

    if not physical:

        registry = [
            build_not_detected_tooth(fdi)
            for fdi in FDI_ORDER
        ]

        case_id = image_path.stem

        case_dir = (
            OUTPUT_ROOT
            / case_id
        )

        case_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        overlay_path = (
            case_dir
            / "fdi_segmentation_overlay.png"
        )

        json_path = (
            case_dir
            / "tooth_registry.json"
        )

        # No masks exist, so the "overlay" is simply
        # the original panoramic radiograph.
        cv2.imwrite(
            str(overlay_path),
            image,
        )

        summary = {
            "case_id": case_id,

            "image": str(image_path),

            "radiograph_type":
                "Panoramic OPG",

            "summary": {
                "teeth_detected": 0,
                "upper_arch": 0,
                "lower_arch": 0,
                "not_detected": 32,
            },

            "detected_fdi": [],

            "not_detected_fdi":
                list(FDI_ORDER),

            "tooth_registry":
                registry,

            "ui": {
                "numbering_system":
                    "FDI Two-Digit Tooth Numbering System (ISO 3950)",

                "interaction_instruction":
                    "No detected teeth are available for selection.",

                "not_detected_behavior": {
                    "color": "grey",
                    "selectable": False,
                },

                "confidence_visible":
                    False,
            },

            "screening_disclaimer":
                (
                    "Panoramic images are intended for screening. "
                    "Findings should be interpreted in clinical context."
                ),
        }

        with json_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                summary,
                file,
                indent=2,
            )

        return {
            "summary":
                summary,

            "overlay_path":
                overlay_path,

            "json_path":
                json_path,
        }

    height, width = image.shape[:2]

    upper, lower = split_arches(
        physical,
        height,
    )

    assignments = assign_missing_aware_fdi(
        upper,
        lower,
    )

    if assignments is None:
        raise RuntimeError(
            "Unable to resolve FDI assignments."
        )

    detected_teeth = []
    detected_map = {}

    used_physical_ids = set()

    for assignment in assignments:

        # Find the matching physical instance.
        candidates = [
            instance
            for instance in physical
            if id(instance)
            not in used_physical_ids
        ]

        physical_instance = (
            nearest_physical_instance(
                assignment,
                candidates,
            )
        )

        used_physical_ids.add(
            id(physical_instance)
        )

        tooth = build_detected_tooth(
            assignment,
            physical_instance,
        )

        detected_teeth.append(
            tooth
        )

        detected_map[
            tooth["fdi"]
        ] = physical_instance

    detected_teeth.sort(
        key=lambda item:
        FDI_ORDER.index(
            item["fdi"]
        )
    )

    detected_fdis = {
        tooth["fdi"]
        for tooth in detected_teeth
    }

    registry = []

    for fdi in FDI_ORDER:

        if fdi in detected_fdis:
            tooth = next(
                item
                for item in detected_teeth
                if item["fdi"] == fdi
            )

            registry.append(
                tooth
            )

        else:
            registry.append(
                build_not_detected_tooth(
                    fdi
                )
            )

    not_detected = [
        item
        for item in registry
        if item["status"]
        == "not_detected"
    ]

    upper_present = sum(
        1
        for item in detected_teeth
        if item["arch"]
        == "Maxillary"
    )

    lower_present = sum(
        1
        for item in detected_teeth
        if item["arch"]
        == "Mandibular"
    )

    overlay = draw_overlay(
        image,
        detected_teeth,
        detected_map,
    )

    case_id = image_path.stem

    case_dir = (
        OUTPUT_ROOT
        / case_id
    )

    case_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    overlay_path = (
        case_dir
        / "fdi_segmentation_overlay.png"
    )

    json_path = (
        case_dir
        / "tooth_registry.json"
    )

    cv2.imwrite(
        str(overlay_path),
        overlay,
    )

    summary = {
        "case_id":
            case_id,

        "image":
            str(image_path),

        "radiograph_type":
            "Panoramic OPG",

        "summary": {
            "teeth_detected":
                len(detected_teeth),

            "upper_arch":
                upper_present,

            "lower_arch":
                lower_present,

            "not_detected":
                len(not_detected),
        },

        "detected_fdi": [
            item["fdi"]
            for item in detected_teeth
        ],

        "not_detected_fdi": [
            item["fdi"]
            for item in not_detected
        ],

        # Fixed 32-position registry used by the dashboard.
        "tooth_registry":
            registry,

        "ui": {
            "numbering_system":
                "FDI Two-Digit Tooth Numbering System (ISO 3950)",

            "interaction_instruction":
                "Select a detected tooth to view its details and zoomed view.",

            "not_detected_behavior": {
                "color":
                    "grey",

                "selectable":
                    False,
            },

            # No model confidence fields are exposed to the UI.
            "confidence_visible":
                False,
        },

        "screening_disclaimer":
            (
                "Panoramic images are intended for screening. "
                "Findings should be interpreted in clinical context."
            ),
    }

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    return {
        "summary":
            summary,

        "overlay_path":
            overlay_path,

        "json_path":
            json_path,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
    )

    args = parser.parse_args()

    result = run_final_fdi_inference(
        args.image
    )

    summary = result[
        "summary"
    ]["summary"]

    print()
    print("=" * 72)
    print("FINAL FDI INFERENCE COMPLETE")
    print("=" * 72)
    print(
        "Teeth detected :",
        summary["teeth_detected"],
    )
    print(
        "Upper arch     :",
        summary["upper_arch"],
    )
    print(
        "Lower arch     :",
        summary["lower_arch"],
    )
    print(
        "Not detected   :",
        summary["not_detected"],
    )
    print()
    print(
        "Overlay:",
        result["overlay_path"].resolve(),
    )
    print(
        "Registry:",
        result["json_path"].resolve(),
    )
