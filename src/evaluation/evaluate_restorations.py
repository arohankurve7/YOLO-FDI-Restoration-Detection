from pathlib import Path
import json
import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

DATASET = (
    ROOT
    / "datasets/raw/restorations/"
    / "Annotated OPG Dataset for Dental Fillings, Prostheses (Crowns & Bridges), Endodontic Treatments, Endodontic Posts, and Dental Implants"
    / "Dataset/test"
)

PRED_ROOT = (
    ROOT
    / "results/restoration_inference"
)

CLASSES = {
    0: "Crown",
    1: "Dental_Fillings",
    4: "bridge",
    5: "implant",
}

IOU_THRESHOLD = 0.50


def polygon_to_mask(points, width, height):
    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    if len(points) < 3:
        return mask

    pts = np.array(
        points,
        dtype=np.int32
    )

    cv2.fillPoly(
        mask,
        [pts],
        1
    )

    return mask


def load_gt(label_path, width, height):
    gt = []

    for line in label_path.read_text().splitlines():

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        cls = int(parts[0])

        if cls not in CLASSES:
            continue

        values = list(
            map(float, parts[1:])
        )

        if len(values) < 6:
            continue

        coords = []

        for i in range(
            0,
            len(values),
            2
        ):
            x = int(
                round(values[i] * width)
            )

            y = int(
                round(values[i + 1] * height)
            )

            x = max(
                0,
                min(width - 1, x)
            )

            y = max(
                0,
                min(height - 1, y)
            )

            coords.append(
                (x, y)
            )

        gt.append({
            "class_id": cls,
            "class_name": CLASSES[cls],
            "mask": polygon_to_mask(
                coords,
                width,
                height
            )
        })

    return gt


def load_predictions(
    pred_path,
    width,
    height
):
    data = json.loads(
        pred_path.read_text()
    )

    preds = []

    for det in data.get(
        "detections",
        []
    ):

        class_name = det[
            "class_name"
        ]

        if class_name not in CLASSES.values():
            continue

        polygon = det.get(
            "mask_polygon",
            []
        )

        coords = [
            (
                int(p["x"]),
                int(p["y"])
            )
            for p in polygon
        ]

        preds.append({
            "class_name":
                class_name,

            "confidence":
                det["confidence"],

            "mask":
                polygon_to_mask(
                    coords,
                    width,
                    height
                )
        })

    return preds


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


def match_class(
    gt_items,
    pred_items,
    iou_threshold
):
    matches = []

    used_gt = set()
    used_pred = set()

    candidates = []

    for gi, gt in enumerate(
        gt_items
    ):
        for pi, pred in enumerate(
            pred_items
        ):
            iou = mask_iou(
                gt["mask"],
                pred["mask"]
            )

            candidates.append(
                (
                    iou,
                    gi,
                    pi
                )
            )

    candidates.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    for iou, gi, pi in candidates:

        if iou < iou_threshold:
            break

        if gi in used_gt:
            continue

        if pi in used_pred:
            continue

        used_gt.add(
            gi
        )

        used_pred.add(
            pi
        )

        matches.append(
            (
                gi,
                pi,
                iou
            )
        )

    tp = len(matches)

    fp = (
        len(pred_items)
        - tp
    )

    fn = (
        len(gt_items)
        - tp
    )

    return tp, fp, fn, matches


def safe_div(a, b):
    return (
        a / b
        if b
        else 0.0
    )


def main():

    totals = {
        name: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "ious": []
        }
        for name
        in CLASSES.values()
    }

    case_rows = []

    image_dir = (
        DATASET
        / "images"
    )

    label_dir = (
        DATASET
        / "labels"
    )

    images = sorted(
        image_dir.glob("*")
    )

    evaluated_cases = 0
    skipped_cases = 0

    for image_path in images:

        label_path = (
            label_dir
            / (
                image_path.stem
                + ".txt"
            )
        )

        if not label_path.exists():
            continue

        pred_path = (
            PRED_ROOT
            / image_path.stem
            / "prediction.json"
        )

        if not pred_path.exists():
            skipped_cases += 1
            continue

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        height, width = (
            image.shape[:2]
        )

        gt = load_gt(
            label_path,
            width,
            height
        )

        preds = load_predictions(
            pred_path,
            width,
            height
        )

        evaluated_cases += 1

        for class_name in CLASSES.values():

            gt_class = [
                x
                for x in gt
                if x["class_name"]
                == class_name
            ]

            pred_class = [
                x
                for x in preds
                if x["class_name"]
                == class_name
            ]

            tp, fp, fn, matches = (
                match_class(
                    gt_class,
                    pred_class,
                    IOU_THRESHOLD
                )
            )

            totals[class_name][
                "tp"
            ] += tp

            totals[class_name][
                "fp"
            ] += fp

            totals[class_name][
                "fn"
            ] += fn

            totals[class_name][
                "ious"
            ].extend(
                [
                    m[2]
                    for m in matches
                ]
            )

            if (
                gt_class
                or pred_class
            ):
                case_rows.append({
                    "case":
                        image_path.stem,

                    "class":
                        class_name,

                    "gt":
                        len(gt_class),

                    "pred":
                        len(pred_class),

                    "tp":
                        tp,

                    "fp":
                        fp,

                    "fn":
                        fn,
                })

    print()
    print(
        "============================================"
    )
    print(
        "RESTORATION MASK EVALUATION"
    )
    print(
        "============================================"
    )
    print(
        f"IoU threshold : {IOU_THRESHOLD:.2f}"
    )
    print(
        f"Cases evaluated: {evaluated_cases}"
    )
    print(
        f"Cases skipped  : {skipped_cases}"
    )
    print()

    header = (
        f'{"CLASS":<24}'
        f'{"TP":>6}'
        f'{"FP":>6}'
        f'{"FN":>6}'
        f'{"PREC":>10}'
        f'{"RECALL":>10}'
        f'{"F1":>10}'
        f'{"MEAN_IOU":>12}'
    )

    print(header)
    print(
        "-" * len(header)
    )

    for class_name in [
        "implant",
        "bridge",
        "Crown",
        "Dental_Fillings",
    ]:

        d = totals[
            class_name
        ]

        tp = d["tp"]
        fp = d["fp"]
        fn = d["fn"]

        precision = safe_div(
            tp,
            tp + fp
        )

        recall = safe_div(
            tp,
            tp + fn
        )

        f1 = safe_div(
            2 * precision * recall,
            precision + recall
        )

        mean_iou = (
            float(
                np.mean(
                    d["ious"]
                )
            )
            if d["ious"]
            else 0.0
        )

        print(
            f"{class_name:<24}"
            f"{tp:>6}"
            f"{fp:>6}"
            f"{fn:>6}"
            f"{precision:>10.3f}"
            f"{recall:>10.3f}"
            f"{f1:>10.3f}"
            f"{mean_iou:>12.3f}"
        )

    print()

    out = (
        ROOT
        / "results"
        / "restoration_evaluation"
    )

    out.mkdir(
        parents=True,
        exist_ok=True
    )

    report_path = (
        out
        / "mask_iou_eval.json"
    )

    report = {
        "iou_threshold":
            IOU_THRESHOLD,

        "evaluated_cases":
            evaluated_cases,

        "skipped_cases":
            skipped_cases,

        "totals":
            totals,

        "cases":
            case_rows,
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2
        )
    )

    print(
        "Saved report:"
    )
    print(
        report_path.resolve()
    )


if __name__ == "__main__":
    main()
