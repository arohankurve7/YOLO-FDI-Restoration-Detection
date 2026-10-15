from pathlib import Path
import csv

from src.segmentation_demo.segmentation_inference import (
    run_segmentation,
    save_segmentation_result,
)


ROOT = Path(__file__).resolve().parents[2]

TEST_ROOT = ROOT / "TEST"

OUTPUT_ROOT = (
    ROOT
    / "results"
    / "segmentation_demo"
    / "full_demo_suite"
)

CSV_PATH = OUTPUT_ROOT / "segmentation_summary.csv"

DUAL_LABEL_DIR = (
    ROOT
    / "datasets"
    / "processed"
    / "dual_yolo_seg"
    / "test"
    / "labels"
)

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
}


def get_source(image_path):
    # In the current curated TEST suite:
    # Dual test images are PNG and restoration-dataset images are JPG/JPEG.
    if image_path.suffix.lower() == ".png":
        return "DUAL"

    if image_path.suffix.lower() in {
        ".jpg",
        ".jpeg",
    }:
        return "RESTORATION"

    return "UNKNOWN"


def get_gt_count(image_path, source):
    if source != "DUAL":
        return None

    label_path = (
        DUAL_LABEL_DIR
        / f"{image_path.stem}.txt"
    )

    if not label_path.exists():
        return None

    return sum(
        1
        for line
        in label_path.read_text().splitlines()
        if line.strip()
    )


def main():
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    images = sorted(
        path
        for path in TEST_ROOT.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )

    print()
    print("=" * 120)
    print("FULL SEGMENTATION DEMO SUITE")
    print("=" * 120)
    print("Entries found:", len(images))
    print()

    rows = []

    for index, image_path in enumerate(
        images,
        start=1,
    ):
        category = image_path.parent.name
        source = get_source(image_path)

        gt_count = get_gt_count(
            image_path,
            source,
        )

        try:
            (
                original,
                segmented,
                summary,
            ) = run_segmentation(
                image_path,
                opacity=0.35,
            )

            output_dir = (
                OUTPUT_ROOT
                / category
            )

            output_path = (
                output_dir
                / image_path.name
            )

            save_segmentation_result(
                segmented,
                output_path,
            )

            raw = summary[
                "raw_predictions"
            ]

            suppressed = summary[
                "suppressed_duplicates"
            ]

            clean = summary[
                "teeth_detected"
            ]

            upper = summary[
                "upper_arch"
            ]

            lower = summary[
                "lower_arch"
            ]

            unknown = summary[
                "unknown_arch"
            ]

            if gt_count is None:
                error = None
                abs_error = None
                exact = None
            else:
                error = clean - gt_count
                abs_error = abs(error)
                exact = int(error == 0)

            status = "OK"

        except Exception as exc:
            raw = None
            suppressed = None
            clean = None
            upper = None
            lower = None
            unknown = None
            error = None
            abs_error = None
            exact = None
            status = f"ERROR: {exc}"

        rows.append({
            "category": category,
            "source": source,
            "image": image_path.name,
            "gt_count": gt_count,
            "raw_predictions": raw,
            "suppressed_duplicates": suppressed,
            "clean_teeth": clean,
            "upper_arch": upper,
            "lower_arch": lower,
            "unknown_arch": unknown,
            "error": error,
            "absolute_error": abs_error,
            "exact_count": exact,
            "status": status,
        })

        gt_text = (
            str(gt_count)
            if gt_count is not None
            else "-"
        )

        clean_text = (
            str(clean)
            if clean is not None
            else "ERR"
        )

        error_text = (
            f"{error:+d}"
            if error is not None
            else "-"
        )

        print(
            f"[{index:03d}/{len(images):03d}] "
            f"{category:28s} "
            f"{source:11s} "
            f"{image_path.name[:38]:38s} "
            f"GT={gt_text:>2s} "
            f"RAW={str(raw):>3s} "
            f"CLEAN={clean_text:>3s} "
            f"ERR={error_text:>3s}"
        )

    fieldnames = [
        "category",
        "source",
        "image",
        "gt_count",
        "raw_predictions",
        "suppressed_duplicates",
        "clean_teeth",
        "upper_arch",
        "lower_arch",
        "unknown_arch",
        "error",
        "absolute_error",
        "exact_count",
        "status",
    ]

    with CSV_PATH.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    # =====================================================
    # GLOBAL SUMMARY
    # =====================================================

    successful = [
        row
        for row in rows
        if row["status"] == "OK"
    ]

    dual_gt = [
        row
        for row in successful
        if row["gt_count"] is not None
    ]

    exact_rows = [
        row
        for row in dual_gt
        if row["error"] == 0
    ]

    over_rows = [
        row
        for row in dual_gt
        if row["error"] > 0
    ]

    under_rows = [
        row
        for row in dual_gt
        if row["error"] < 0
    ]

    total_abs_error = sum(
        row["absolute_error"]
        for row in dual_gt
    )

    mae = (
        total_abs_error
        / len(dual_gt)
        if dual_gt
        else 0.0
    )

    print()
    print("=" * 120)
    print("OVERALL SUMMARY")
    print("=" * 120)

    print(
        "Total category entries :",
        len(rows),
    )

    print(
        "Successful inference   :",
        len(successful),
    )

    print(
        "Failed inference       :",
        len(rows) - len(successful),
    )

    print(
        "Dual GT entries        :",
        len(dual_gt),
    )

    if dual_gt:
        print(
            "Exact tooth count      :",
            f"{len(exact_rows)}/{len(dual_gt)} "
            f"({len(exact_rows) / len(dual_gt) * 100:.1f}%)"
        )

        print(
            "Over-count cases       :",
            len(over_rows),
        )

        print(
            "Under-count cases      :",
            len(under_rows),
        )

        print(
            "Mean absolute error    :",
            f"{mae:.3f}",
        )

        print(
            "Maximum absolute error :",
            max(
                row["absolute_error"]
                for row in dual_gt
            ),
        )

    # =====================================================
    # CATEGORY SUMMARY
    # =====================================================

    print()
    print("=" * 120)
    print("CATEGORY SUMMARY")
    print("=" * 120)

    categories = sorted(
        set(
            row["category"]
            for row in rows
        )
    )

    for category in categories:
        category_rows = [
            row
            for row in successful
            if row["category"] == category
        ]

        dual_rows = [
            row
            for row in category_rows
            if row["gt_count"] is not None
        ]

        restoration_rows = [
            row
            for row in category_rows
            if row["source"] == "RESTORATION"
        ]

        print()
        print(category)
        print("-" * 80)

        print(
            "Entries             :",
            len(category_rows),
        )

        print(
            "Dual GT entries     :",
            len(dual_rows),
        )

        print(
            "Restoration entries :",
            len(restoration_rows),
        )

        if dual_rows:
            category_exact = sum(
                row["error"] == 0
                for row in dual_rows
            )

            category_mae = sum(
                row["absolute_error"]
                for row in dual_rows
            ) / len(dual_rows)

            print(
                "Exact GT count      :",
                f"{category_exact}/{len(dual_rows)} "
                f"({category_exact / len(dual_rows) * 100:.1f}%)"
            )

            print(
                "Category MAE        :",
                f"{category_mae:.3f}",
            )

        if restoration_rows:
            counts = [
                row["clean_teeth"]
                for row in restoration_rows
                if row["clean_teeth"] is not None
            ]

            if counts:
                print(
                    "Restoration clean  :",
                    f"min={min(counts)}, "
                    f"max={max(counts)}, "
                    f"mean={sum(counts) / len(counts):.1f}"
                )

    print()
    print("=" * 120)
    print("OUTPUT")
    print("=" * 120)
    print("CSV      :", CSV_PATH.resolve())
    print("Overlays :", OUTPUT_ROOT.resolve())


if __name__ == "__main__":
    main()
