from pathlib import Path
from collections import Counter

from ultralytics import YOLO

from src.inference.physical_fdi_resolver import (
    MODEL_PATH,
    extract_raw_predictions,
    physical_instance_cleanup,
    split_arches,
)


ROOT = Path(__file__).resolve().parents[2]

TEST_IMAGES = (
    ROOT
    / "datasets"
    / "processed"
    / "dual_yolo_seg"
    / "test"
    / "images"
)

TEST_LABELS = (
    ROOT
    / "datasets"
    / "processed"
    / "dual_yolo_seg"
    / "test"
    / "labels"
)

FDI_NAMES = [
    "11", "12", "13", "14", "15", "16", "17", "18",
    "21", "22", "23", "24", "25", "26", "27", "28",
    "31", "32", "33", "34", "35", "36", "37", "38",
    "41", "42", "43", "44", "45", "46", "47", "48",
]


def load_gt_fdis(stem):
    path = TEST_LABELS / f"{stem}.txt"

    if not path.exists():
        return []

    fdis = []

    for line in path.read_text().splitlines():
        if not line.strip():
            continue

        cls = int(line.split()[0])

        if 0 <= cls < len(FDI_NAMES):
            fdis.append(FDI_NAMES[cls])

    return fdis


def main():
    model = YOLO(str(MODEL_PATH))

    images = sorted(
        list(TEST_IMAGES.glob("*.png"))
        + list(TEST_IMAGES.glob("*.jpg"))
        + list(TEST_IMAGES.glob("*.jpeg"))
    )

    print()
    print("PHYSICAL-FIRST MODEL-FDI BASELINE")
    print("=" * 92)

    print(
        f"{'Image':12s}"
        f"{'GT':>5s}"
        f"{'Physical':>10s}"
        f"{'PredUnique':>12s}"
        f"{'GT∩Pred':>10s}"
        f"{'Recall':>10s}"
        f"{'Duplicates':>12s}"
    )

    print("-" * 92)

    total_gt = 0
    total_overlap = 0
    exact_sets = 0

    for image_path in images:
        raw, image = extract_raw_predictions(
            model,
            image_path,
        )

        physical = physical_instance_cleanup(raw)

        if image is not None:
            split_arches(
                physical,
                image.shape[0],
            )

        predicted = [
            item["predicted_fdi"]
            for item in physical
        ]

        gt = load_gt_fdis(
            image_path.stem
        )

        pred_counter = Counter(predicted)
        gt_counter = Counter(gt)

        # Multiset overlap, so duplicated GT annotations do
        # not silently distort the comparison.
        overlap = sum(
            min(
                gt_counter[fdi],
                pred_counter[fdi],
            )
            for fdi in set(gt_counter) | set(pred_counter)
        )

        duplicates = sum(
            count - 1
            for count in pred_counter.values()
            if count > 1
        )

        recall = (
            overlap / len(gt)
            if gt
            else 0.0
        )

        if gt_counter == pred_counter:
            exact_sets += 1

        total_gt += len(gt)
        total_overlap += overlap

        print(
            f"{image_path.name:12s}"
            f"{len(gt):5d}"
            f"{len(physical):10d}"
            f"{len(pred_counter):12d}"
            f"{overlap:10d}"
            f"{recall * 100:9.1f}%"
            f"{duplicates:12d}"
        )

    print("-" * 92)

    overall_recall = (
        total_overlap / total_gt
        if total_gt
        else 0.0
    )

    print()
    print("Images evaluated :", len(images))
    print("Exact FDI sets   :", exact_sets)
    print(
        "Overall overlap  : "
        f"{total_overlap}/{total_gt} "
        f"({overall_recall * 100:.1f}%)"
    )


if __name__ == "__main__":
    main()
