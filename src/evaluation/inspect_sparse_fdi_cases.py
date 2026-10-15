from pathlib import Path

from ultralytics import YOLO

from src.inference.fdi_assignment_resolver import (
    MODEL_PATH,
    extract_raw_predictions,
    physical_instance_cleanup,
    split_arches,
)


ROOT = Path(__file__).resolve().parents[2]

IMAGE_ROOT = (
    ROOT
    / "datasets"
    / "processed"
    / "dual_yolo_seg"
    / "test"
    / "images"
)

LABEL_ROOT = (
    ROOT
    / "datasets"
    / "processed"
    / "dual_yolo_seg"
    / "test"
    / "labels"
)

FDI_NAMES = [
    "11","12","13","14","15","16","17","18",
    "21","22","23","24","25","26","27","28",
    "31","32","33","34","35","36","37","38",
    "41","42","43","44","45","46","47","48",
]


def load_gt(stem):
    label_path = LABEL_ROOT / f"{stem}.txt"

    result = []

    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue

        cls = int(line.split()[0])
        result.append(FDI_NAMES[cls])

    return result


def ordered_gt(gt, arch):
    if arch == "upper":
        order = [
            "18","17","16","15","14","13","12","11",
            "21","22","23","24","25","26","27","28",
        ]
    else:
        order = [
            "48","47","46","45","44","43","42","41",
            "31","32","33","34","35","36","37","38",
        ]

    gt_set = set(gt)

    return [
        fdi
        for fdi in order
        if fdi in gt_set
    ]


def show_arch(title, instances, gt_order):
    print()
    print(title)
    print("-" * 78)

    print("GT ORDER:")
    print(" ".join(gt_order))

    print()
    print("PHYSICAL INSTANCES:")

    for i, item in enumerate(instances, 1):
        print(
            f"{i:02d} "
            f"x={item['centroid_x']:8.1f} "
            f"y={item['centroid_y']:8.1f} "
            f"model={item['predicted_fdi']:>2s} "
            f"conf={item['confidence']:.3f}"
        )


def main():
    model = YOLO(str(MODEL_PATH))

    # Representative sparse cases:
    # extreme missing + moderate missing + difficult duplicates.
    cases = [
        "1016",
        "1052",
        "1071",
        "2013",
        "4415",
    ]

    for stem in cases:
        image_path = IMAGE_ROOT / f"{stem}.png"

        raw, image = extract_raw_predictions(
            model,
            image_path,
        )

        physical = physical_instance_cleanup(raw)

        upper, lower = split_arches(
            physical,
            image.shape[0],
        )

        gt = load_gt(stem)

        print()
        print("=" * 90)
        print(stem)
        print("=" * 90)

        print(
            f"GT={len(gt)} "
            f"Physical={len(physical)} "
            f"Upper={len(upper)} "
            f"Lower={len(lower)}"
        )

        show_arch(
            "UPPER",
            upper,
            ordered_gt(gt, "upper"),
        )

        show_arch(
            "LOWER",
            lower,
            ordered_gt(gt, "lower"),
        )


if __name__ == "__main__":
    main()
