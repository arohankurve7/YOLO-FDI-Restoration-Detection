from pathlib import Path
import argparse
import json
import math


def bbox_overlap(a, b):
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)

    area_a = max(1, (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]))
    area_b = max(1, (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]))

    return inter / min(area_a, area_b)


def point_box_distance(point, box):
    x = point["x"]
    y = point["y"]

    dx = max(box["x1"] - x, 0, x - box["x2"])
    dy = max(box["y1"] - y, 0, y - box["y2"])

    return math.hypot(dx, dy)


def centroid_distance(a, b):
    return math.hypot(
        a["x"] - b["x"],
        a["y"] - b["y"]
    )


def association_score(rest, tooth):
    rest_c = rest["centroid"]
    tooth_c = tooth["centroid"]

    overlap = bbox_overlap(
        rest["bbox"],
        tooth["bbox"]
    )

    box_dist = point_box_distance(
        rest_c,
        tooth["bbox"]
    )

    center_dist = centroid_distance(
        rest_c,
        tooth_c
    )

    # Spatial score:
    # overlap is strongest evidence,
    # then distance of restoration centroid to tooth region,
    # then centroid proximity.
    score = 0.0

    score += overlap * 100.0
    score += max(0.0, 35.0 - box_dist) * 1.5
    score += max(0.0, 70.0 - center_dist) * 0.35

    return score, {
        "bbox_overlap": round(overlap, 4),
        "bbox_distance_px": round(box_dist, 2),
        "centroid_distance_px": round(center_dist, 2),
    }


def associate(fdi_json, restoration_json, output_json):
    fdi = json.loads(Path(fdi_json).read_text())
    rest = json.loads(Path(restoration_json).read_text())

    if fdi["image"] != rest["image"]:
        raise ValueError("FDI and restoration JSON refer to different images")

    teeth = fdi["detections"]
    restorations = rest["detections"]

    findings = {
        tooth["fdi"]: []
        for tooth in teeth
    }

    unmatched = []

    for r in restorations:
        candidates = []

        for tooth in teeth:
            score, details = association_score(
                r,
                tooth
            )

            candidates.append({
                "fdi": tooth["fdi"],
                "score": score,
                "details": details,
            })

        candidates.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        best = candidates[0]

        # Conservative first-pass threshold.
        # Helps reject image markers/background false positives.
        matched = (
            best["score"] >= 22
            and best["details"]["centroid_distance_px"] <= 90
        )

        result = {
            "class_name": r["class_name"],
            "confidence": r["confidence"],
            "centroid": r["centroid"],
            "bbox": r["bbox"],
            "mask_polygon": r["mask_polygon"],
            "association": {
                "score": round(best["score"], 2),
                **best["details"],
            }
        }

        if matched:
            result["fdi"] = best["fdi"]
            findings[best["fdi"]].append(result)
        else:
            result["reason"] = "No reliable FDI spatial match"
            unmatched.append(result)

    # Remove empty teeth to keep JSON compact
    findings = {
        fdi: items
        for fdi, items in findings.items()
        if items
    }

    summary = {
        "image": fdi["image"],
        "tooth_findings": findings,
        "unmatched_restorations": unmatched,
    }

    output_json = Path(output_json)
    output_json.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_json.write_text(
        json.dumps(summary, indent=2)
    )

    print("================================")
    print("RESTORATION → FDI ASSOCIATION")
    print("================================")

    for fdi_num, items in findings.items():
        for item in items:
            print(
                f'FDI {fdi_num}: '
                f'{item["class_name"]} '
                f'conf={item["confidence"]:.3f} '
                f'association={item["association"]["score"]:.1f}'
            )

    print()
    print("Unmatched:", len(unmatched))
    for item in unmatched:
        print(
            f'  {item["class_name"]} '
            f'conf={item["confidence"]:.3f}'
        )

    print()
    print("Saved:")
    print(output_json.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fdi",
        required=True
    )

    parser.add_argument(
        "--restorations",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    args = parser.parse_args()

    associate(
        args.fdi,
        args.restorations,
        args.output
    )
