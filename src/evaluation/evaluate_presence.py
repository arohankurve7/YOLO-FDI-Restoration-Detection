from pathlib import Path
import json
from collections import defaultdict

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

MODEL = ROOT / "models/fdi_seg_v1/best.pt"
TEST_IMAGES = ROOT / "datasets/processed/dual_yolo_seg/test/images"
GT_LABELS = ROOT / "datasets/raw/dual/labels"

FDI = [
    "11","12","13","14","15","16","17","18",
    "21","22","23","24","25","26","27","28",
    "31","32","33","34","35","36","37","38",
    "41","42","43","44","45","46","47","48"
]

CONF = 0.25

model = YOLO(str(MODEL))

TP_missing = 0
FP_missing = 0
FN_missing = 0
TN_present = 0

exact_cases = 0
total_cases = 0

per_fdi = defaultdict(lambda: {
    "correct": 0,
    "wrong": 0
})

case_errors = []

results = model.predict(
    source=str(TEST_IMAGES),
    imgsz=640,
    conf=CONF,
    device="mps",
    stream=True,
    verbose=False
)

for result in results:

    stem = Path(result.path).stem
    gt_path = GT_LABELS / f"{stem}.json"

    if not gt_path.exists():
        print("Missing GT:", stem)
        continue

    gt = json.loads(gt_path.read_text())

    gt_present = {
        str(s.get("label", "")).strip()
        for s in gt.get("shapes", [])
        if str(s.get("label", "")).strip() in FDI
    }

    if result.boxes is not None:
        classes = result.boxes.cls.cpu().numpy().astype(int)

        pred_present = {
            str(model.names[c])
            for c in classes
            if 0 <= c < len(FDI)
        }
    else:
        pred_present = set()

    gt_missing = set(FDI) - gt_present
    pred_missing = set(FDI) - pred_present

    total_cases += 1

    if gt_present == pred_present:
        exact_cases += 1

    false_present = sorted(
        pred_present - gt_present,
        key=FDI.index
    )

    missed_present = sorted(
        gt_present - pred_present,
        key=FDI.index
    )

    if false_present or missed_present:
        case_errors.append({
            "image": stem,
            "false_present": false_present,
            "missed_present": missed_present
        })

    for tooth in FDI:

        gt_is_missing = tooth in gt_missing
        pred_is_missing = tooth in pred_missing

        if gt_is_missing and pred_is_missing:
            TP_missing += 1

        elif not gt_is_missing and pred_is_missing:
            FP_missing += 1

        elif gt_is_missing and not pred_is_missing:
            FN_missing += 1

        else:
            TN_present += 1

        if gt_is_missing == pred_is_missing:
            per_fdi[tooth]["correct"] += 1
        else:
            per_fdi[tooth]["wrong"] += 1


total_slots = TP_missing + FP_missing + FN_missing + TN_present

accuracy = (
    (TP_missing + TN_present) / total_slots
    if total_slots else 0
)

precision = (
    TP_missing / (TP_missing + FP_missing)
    if TP_missing + FP_missing else 0
)

recall = (
    TP_missing / (TP_missing + FN_missing)
    if TP_missing + FN_missing else 0
)

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0
)


print("\n======================================")
print("MISSING / PRESENCE EVALUATION")
print("======================================")

print("Test images          :", total_cases)
print("FDI slots evaluated  :", total_slots)

print()
print("Exact case matches   :", exact_cases)

if total_cases:
    print(
        "Exact case accuracy  :",
        f"{100 * exact_cases / total_cases:.2f}%"
    )

print()
print("Tooth-slot accuracy  :", f"{accuracy * 100:.2f}%")

print()
print("Missing Precision    :", f"{precision * 100:.2f}%")
print("Missing Recall       :", f"{recall * 100:.2f}%")
print("Missing F1           :", f"{f1 * 100:.2f}%")

print()
print("TP missing :", TP_missing)
print("FP missing :", FP_missing)
print("FN missing :", FN_missing)
print("TN present :", TN_present)

print("\n--- PER-FDI ACCURACY ---")

for tooth in FDI:
    c = per_fdi[tooth]["correct"]
    w = per_fdi[tooth]["wrong"]

    acc = c / (c + w) if c + w else 0

    print(
        f"{tooth}: "
        f"{acc * 100:6.2f}% "
        f"(errors={w})"
    )

print("\n--- CASES WITH ERRORS ---")

for item in case_errors:
    print(
        f"{item['image']}: "
        f"false_present={item['false_present']} "
        f"missed_present={item['missed_present']}"
    )
