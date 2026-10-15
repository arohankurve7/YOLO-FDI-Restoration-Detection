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

CLASS_ORDER = [
    "implant",
    "bridge",
    "Crown",
    "Dental_Fillings",
]

THRESHOLDS = [
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
]

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
            "class_name":
                CLASSES[cls],

            "mask":
                polygon_to_mask(
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
                float(
                    det["confidence"]
                ),

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

    tp = 0
    matched_ious = []

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

        tp += 1
        matched_ious.append(
            iou
        )

    fp = (
        len(pred_items)
        - tp
    )

    fn = (
        len(gt_items)
        - tp
    )

    return (
        tp,
        fp,
        fn,
        matched_ious
    )


def safe_div(a, b):
    return (
        a / b
        if b
        else 0.0
    )


def prepare_cases():
    cases = []

    image_dir = (
        DATASET
        / "images"
    )

    label_dir = (
        DATASET
        / "labels"
    )

    for image_path in sorted(
        image_dir.glob("*")
    ):

        label_path = (
            label_dir
            / f"{image_path.stem}.txt"
        )

        pred_path = (
            PRED_ROOT
            / image_path.stem
            / "prediction.json"
        )

        if (
            not label_path.exists()
            or not pred_path.exists()
        ):
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

        cases.append({
            "case":
                image_path.stem,

            "gt":
                gt,

            "preds":
                preds,
        })

    return cases


def evaluate_threshold(
    cases,
    class_name,
    confidence_threshold
):
    tp_total = 0
    fp_total = 0
    fn_total = 0
    ious = []

    for case in cases:

        gt_class = [
            x
            for x in case["gt"]
            if x["class_name"]
            == class_name
        ]

        pred_class = [
            x
            for x in case["preds"]
            if (
                x["class_name"]
                == class_name
                and
                x["confidence"]
                >= confidence_threshold
            )
        ]

        tp, fp, fn, matched_ious = (
            match_class(
                gt_class,
                pred_class,
                IOU_THRESHOLD
            )
        )

        tp_total += tp
        fp_total += fp
        fn_total += fn

        ious.extend(
            matched_ious
        )

    precision = safe_div(
        tp_total,
        tp_total + fp_total
    )

    recall = safe_div(
        tp_total,
        tp_total + fn_total
    )

    f1 = safe_div(
        2 * precision * recall,
        precision + recall
    )

    mean_iou = (
        float(
            np.mean(ious)
        )
        if ious
        else 0.0
    )

    return {
        "threshold":
            confidence_threshold,

        "tp":
            tp_total,

        "fp":
            fp_total,

        "fn":
            fn_total,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "mean_iou":
            mean_iou,
    }


def main():

    cases = prepare_cases()

    print()
    print(
        "============================================"
    )
    print(
        "RESTORATION CONFIDENCE THRESHOLD SWEEP"
    )
    print(
        "============================================"
    )

    print(
        f"Cases evaluated: {len(cases)}"
    )

    print(
        f"Mask IoU threshold: {IOU_THRESHOLD:.2f}"
    )

    all_results = {}

    for class_name in CLASS_ORDER:

        print()
        print(
            f"=== {class_name} ==="
        )

        print(
            f'{"CONF":>6}'
            f'{"TP":>6}'
            f'{"FP":>6}'
            f'{"FN":>6}'
            f'{"PREC":>9}'
            f'{"REC":>9}'
            f'{"F1":>9}'
            f'{"MIOU":>9}'
        )

        print(
            "-" * 60
        )

        class_results = []

        for threshold in THRESHOLDS:

            result = evaluate_threshold(
                cases,
                class_name,
                threshold
            )

            class_results.append(
                result
            )

            print(
                f'{threshold:>6.2f}'
                f'{result["tp"]:>6}'
                f'{result["fp"]:>6}'
                f'{result["fn"]:>6}'
                f'{result["precision"]:>9.3f}'
                f'{result["recall"]:>9.3f}'
                f'{result["f1"]:>9.3f}'
                f'{result["mean_iou"]:>9.3f}'
            )

        best = max(
            class_results,
            key=lambda x: x["f1"]
        )

        print()
        print(
            "Best F1 threshold:",
            f'{best["threshold"]:.2f}'
        )

        print(
            "Precision:",
            f'{best["precision"]:.3f}'
        )

        print(
            "Recall:",
            f'{best["recall"]:.3f}'
        )

        print(
            "F1:",
            f'{best["f1"]:.3f}'
        )

        all_results[
            class_name
        ] = {
            "sweep":
                class_results,

            "best_f1":
                best,
        }

    out_dir = (
        ROOT
        / "results"
        / "restoration_evaluation"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output = (
        out_dir
        / "threshold_sweep.json"
    )

    output.write_text(
        json.dumps(
            {
                "iou_threshold":
                    IOU_THRESHOLD,

                "cases_evaluated":
                    len(cases),

                "results":
                    all_results,
            },
            indent=2
        )
    )

    print()
    print(
        "Saved:"
    )

    print(
        output.resolve()
    )


if __name__ == "__main__":
    main()
