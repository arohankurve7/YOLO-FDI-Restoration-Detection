from pathlib import Path
from collections import Counter

from ultralytics import YOLO

from src.inference.fdi_assignment_resolver import (
    MODEL_PATH,
    extract_raw_predictions,
    physical_instance_cleanup,
    split_arches,
    assign_missing_aware_fdi,
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

    result = []

    for line in path.read_text().splitlines():
        if not line.strip():
            continue

        cls = int(line.split()[0])

        if 0 <= cls < len(FDI_NAMES):
            result.append(FDI_NAMES[cls])

    return result


def overlap_count(gt, prediction):
    gt_counter = Counter(gt)
    pred_counter = Counter(prediction)

    return sum(
        min(
            gt_counter[fdi],
            pred_counter[fdi],
        )
        for fdi in set(gt_counter) | set(pred_counter)
    )


def main():
    model = YOLO(str(MODEL_PATH))

    images = sorted(
        list(TEST_IMAGES.glob("*.png"))
        + list(TEST_IMAGES.glob("*.jpg"))
        + list(TEST_IMAGES.glob("*.jpeg"))
    )

    total_gt = 0
    total_baseline = 0
    total_resolved = 0

    baseline_exact = 0
    resolved_exact = 0

    improved = []
    regressed = []
    unchanged = []

    print()
    print("MODEL FDI vs MISSING-AWARE RESOLVER")
    print("=" * 105)

    print(
        f"{'Image':12s}"
        f"{'GT':>5s}"
        f"{'Phys':>6s}"
        f"{'Base':>7s}"
        f"{'Resolved':>10s}"
        f"{'Delta':>8s}"
        f"{'Base%':>9s}"
        f"{'Res%':>9s}"
    )

    print("-" * 105)

    for image_path in images:

        raw, image = extract_raw_predictions(
            model,
            image_path,
        )

        physical = physical_instance_cleanup(raw)

        upper, lower = split_arches(
            physical,
            image.shape[0],
        )

        baseline_prediction = [
            item["predicted_fdi"]
            for item in physical
        ]

        assignments = assign_missing_aware_fdi(
            upper,
            lower,
        )

        if assignments is None:
            resolved_prediction = []
        else:
            resolved_prediction = [
                item["resolved_fdi"]
                for item in assignments
            ]

        gt = load_gt_fdis(
            image_path.stem
        )

        base_overlap = overlap_count(
            gt,
            baseline_prediction,
        )

        resolved_overlap = overlap_count(
            gt,
            resolved_prediction,
        )

        base_counter = Counter(
            baseline_prediction
        )

        resolved_counter = Counter(
            resolved_prediction
        )

        gt_counter = Counter(gt)

        if base_counter == gt_counter:
            baseline_exact += 1

        if resolved_counter == gt_counter:
            resolved_exact += 1

        delta = (
            resolved_overlap
            - base_overlap
        )

        if delta > 0:
            improved.append(
                (image_path.name, delta)
            )
        elif delta < 0:
            regressed.append(
                (image_path.name, delta)
            )
        else:
            unchanged.append(
                image_path.name
            )

        total_gt += len(gt)
        total_baseline += base_overlap
        total_resolved += resolved_overlap

        base_pct = (
            100.0 * base_overlap / len(gt)
            if gt else 0.0
        )

        resolved_pct = (
            100.0 * resolved_overlap / len(gt)
            if gt else 0.0
        )

        print(
            f"{image_path.name:12s}"
            f"{len(gt):5d}"
            f"{len(physical):6d}"
            f"{base_overlap:7d}"
            f"{resolved_overlap:10d}"
            f"{delta:+8d}"
            f"{base_pct:8.1f}%"
            f"{resolved_pct:8.1f}%"
        )

    baseline_pct = (
        100.0 * total_baseline / total_gt
    )

    resolved_pct = (
        100.0 * total_resolved / total_gt
    )

    print("-" * 105)
    print()

    print(
        "Baseline overlap : "
        f"{total_baseline}/{total_gt} "
        f"({baseline_pct:.1f}%)"
    )

    print(
        "Resolved overlap : "
        f"{total_resolved}/{total_gt} "
        f"({resolved_pct:.1f}%)"
    )

    print(
        "Net improvement  : "
        f"{total_resolved - total_baseline:+d} teeth "
        f"({resolved_pct - baseline_pct:+.1f} pp)"
    )

    print()
    print(
        "Baseline exact   :",
        baseline_exact,
        "/",
        len(images),
    )

    print(
        "Resolved exact   :",
        resolved_exact,
        "/",
        len(images),
    )

    print()
    print("Improved images  :", len(improved))
    print("Regressed images :", len(regressed))
    print("Unchanged images :", len(unchanged))

    print()
    print("IMPROVED")
    print("-" * 50)

    for name, delta in improved:
        print(
            f"{name:15s} {delta:+d}"
        )

    print()
    print("REGRESSED")
    print("-" * 50)

    for name, delta in regressed:
        print(
            f"{name:15s} {delta:+d}"
        )


if __name__ == "__main__":
    main()
