from pathlib import Path
import argparse
import json
import math


FDI_ORDER = [
    "18","17","16","15","14","13","12","11",
    "21","22","23","24","25","26","27","28",
    "48","47","46","45","44","43","42","41",
    "31","32","33","34","35","36","37","38",
]


def center(box):
    return {
        "x": (box["x1"] + box["x2"]) / 2.0,
        "y": (box["y1"] + box["y2"]) / 2.0,
    }


def distance(a, b):
    return math.hypot(
        a["x"] - b["x"],
        a["y"] - b["y"]
    )


def bbox_iou(a, b):
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)

    area_a = max(
        1,
        (a["x2"] - a["x1"]) *
        (a["y2"] - a["y1"])
    )

    area_b = max(
        1,
        (b["x2"] - b["x1"]) *
        (b["y2"] - b["y1"])
    )

    return inter / float(area_a + area_b - inter)


def overlap_over_smaller(a, b):
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)

    area_a = max(
        1,
        (a["x2"] - a["x1"]) *
        (a["y2"] - a["y1"])
    )

    area_b = max(
        1,
        (b["x2"] - b["x1"]) *
        (b["y2"] - b["y1"])
    )

    return inter / min(area_a, area_b)


def suppress_duplicates(restorations):
    """
    Suppress likely duplicate detections of the same class.
    Conservative: only suppress when boxes strongly overlap
    or centroids are very close.
    """

    ordered = sorted(
        restorations,
        key=lambda x: x["confidence"],
        reverse=True
    )

    kept = []

    for det in ordered:
        duplicate = False

        for existing in kept:
            if det["class_name"] != existing["class_name"]:
                continue

            iou = bbox_iou(
                det["bbox"],
                existing["bbox"]
            )

            dist = distance(
                det["centroid"],
                existing["centroid"]
            )

            if iou >= 0.35 or dist <= 18:
                duplicate = True
                break

        if not duplicate:
            kept.append(det)

    return kept


def quadrant_slots():
    return {
        "1": ["18","17","16","15","14","13","12","11"],
        "2": ["21","22","23","24","25","26","27","28"],
        "4": ["48","47","46","45","44","43","42","41"],
        "3": ["31","32","33","34","35","36","37","38"],
    }


def infer_slot_positions(teeth):
    """
    Build expected x/y locations for all 32 FDI slots.

    Present teeth provide anchors.
    Missing slots are interpolated/extrapolated within each quadrant.
    """

    by_fdi = {
        t["fdi"]: t
        for t in teeth
    }

    slots = {}

    for quadrant, sequence in quadrant_slots().items():

        anchors = []

        for idx, fdi in enumerate(sequence):
            if fdi in by_fdi:
                c = by_fdi[fdi]["centroid"]
                anchors.append(
                    (idx, float(c["x"]), float(c["y"]))
                )

        if not anchors:
            continue

        # Typical y for that quadrant
        median_y = sorted(
            [a[2] for a in anchors]
        )[len(anchors) // 2]

        # Estimate tooth spacing from anchor pairs
        spacings = []

        for a, b in zip(
            anchors[:-1],
            anchors[1:]
        ):
            idx_a, x_a, _ = a
            idx_b, x_b, _ = b

            step_count = idx_b - idx_a

            if step_count > 0:
                spacings.append(
                    (x_b - x_a) / step_count
                )

        if spacings:
            spacing = sum(spacings) / len(spacings)
        else:
            spacing = 32.0

        for idx, fdi in enumerate(sequence):

            if fdi in by_fdi:
                c = by_fdi[fdi]["centroid"]
                slots[fdi] = {
                    "x": float(c["x"]),
                    "y": float(c["y"]),
                    "source": "detected"
                }
                continue

            left = None
            right = None

            for a in anchors:
                if a[0] < idx:
                    left = a
                if a[0] > idx:
                    right = a
                    break

            if left and right:
                i1, x1, y1 = left
                i2, x2, y2 = right

                ratio = (
                    (idx - i1) /
                    float(i2 - i1)
                )

                x = x1 + ratio * (x2 - x1)
                y = y1 + ratio * (y2 - y1)

            elif left:
                i1, x1, y1 = left
                x = x1 + spacing * (idx - i1)
                y = y1

            elif right:
                i2, x2, y2 = right
                x = x2 - spacing * (i2 - idx)
                y = y2

            else:
                continue

            slots[fdi] = {
                "x": float(x),
                "y": float(y if y is not None else median_y),
                "source": "interpolated"
            }

    return slots


def tooth_overlap_score(rest, tooth):
    overlap = overlap_over_smaller(
        rest["bbox"],
        tooth["bbox"]
    )

    cdist = distance(
        rest["centroid"],
        tooth["centroid"]
    )

    score = (
        overlap * 110.0
        + max(0.0, 70.0 - cdist) * 0.45
    )

    return score, overlap, cdist


def slot_score(rest, slot):
    d = distance(
        rest["centroid"],
        slot
    )

    # slot proximity is especially important for implants,
    # because an implant may replace a missing natural tooth
    score = max(0.0, 95.0 - d)

    return score, d


def classify_candidate(
    rest,
    teeth_by_fdi,
    slots
):
    candidates = []

    for fdi, slot in slots.items():

        s_slot, slot_dist = slot_score(
            rest,
            slot
        )

        overlap = 0.0
        tooth_dist = None
        s_tooth = 0.0

        if fdi in teeth_by_fdi:
            s_tooth, overlap, tooth_dist = (
                tooth_overlap_score(
                    rest,
                    teeth_by_fdi[fdi]
                )
            )

        # Hybrid weighting differs slightly for implants.
        if rest["class_name"] == "implant":
            final_score = (
                s_slot * 1.2
                + s_tooth * 0.35
            )
        else:
            final_score = (
                s_slot * 0.65
                + s_tooth * 0.85
            )

        candidates.append({
            "fdi": fdi,
            "score": final_score,
            "slot_distance_px": slot_dist,
            "tooth_overlap": overlap,
            "tooth_centroid_distance_px":
                tooth_dist,
            "slot_source":
                slots[fdi]["source"],
        })

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return candidates


def associate(
    fdi_json,
    restoration_json,
    output_json
):
    fdi = json.loads(
        Path(fdi_json).read_text()
    )

    restorations = json.loads(
        Path(restoration_json).read_text()
    )

    if fdi["image"] != restorations["image"]:
        raise ValueError(
            "FDI and restoration predictions use different images"
        )

    teeth = fdi["detections"]

    teeth_by_fdi = {
        t["fdi"]: t
        for t in teeth
    }

    slots = infer_slot_positions(
        teeth
    )

    raw_rest = restorations["detections"]

    deduped = suppress_duplicates(
        raw_rest
    )

    findings = {}
    unmatched = []

    for r in deduped:

        candidates = classify_candidate(
            r,
            teeth_by_fdi,
            slots
        )

        if not candidates:
            item = dict(r)
            item["reason"] = "No FDI slot geometry"
            unmatched.append(item)
            continue

        best = candidates[0]

        # Conservative rejection rules
        if r["class_name"] == "implant":
            max_slot_dist = 55.0
        elif r["class_name"] == "Dental_Fillings":
            max_slot_dist = 55.0
        else:
            max_slot_dist = 48.0

        reliable = (
            best["score"] >= 30.0
            and best["slot_distance_px"]
                <= max_slot_dist
        )

        item = {
            "class_name":
                r["class_name"],

            "confidence":
                r["confidence"],

            "centroid":
                r["centroid"],

            "bbox":
                r["bbox"],

            "mask_polygon":
                r["mask_polygon"],

            "association": {
                "score":
                    round(best["score"], 2),

                "slot_distance_px":
                    round(
                        best["slot_distance_px"],
                        2
                    ),

                "tooth_overlap":
                    round(
                        best["tooth_overlap"],
                        4
                    ),

                "slot_source":
                    best["slot_source"],
            },

            "candidate_fdi":
                best["fdi"],
        }

        if reliable:
            item["fdi"] = best["fdi"]

            findings.setdefault(
                best["fdi"],
                []
            ).append(item)

        else:
            item["reason"] = (
                "No reliable FDI association"
            )

            unmatched.append(item)

    # Demo-safe filtering:
    # 1) Suppress weak detections by class confidence.
    # 2) If an implant is assigned to an FDI slot, suppress overlapping filling
    #    detections on the same slot to avoid conflicting findings.
    min_conf = {
        "implant": 0.30,
        "bridge": 0.45,
        "Crown": 0.40,
        "Dental_Fillings": 0.40,
    }

    filtered_findings = {}

    for fdi_num, items in findings.items():
        kept = [
            item
            for item in items
            if item["confidence"] >= min_conf.get(
                item["class_name"],
                0.60
            )
        ]

        has_implant = any(
            item["class_name"] == "implant"
            for item in kept
        )

        if has_implant:
            kept = [
                item
                for item in kept
                if item["class_name"] != "Dental_Fillings"
            ]

        if kept:
            filtered_findings[fdi_num] = kept

    findings = filtered_findings

    summary = {
        "image":
            fdi["image"],

        "restoration_count_raw":
            len(raw_rest),

        "restoration_count_after_dedup":
            len(deduped),

        "estimated_fdi_slots":
            slots,

        "tooth_findings":
            findings,

        "unmatched_restorations":
            unmatched,
    }

    output_json = Path(output_json)

    output_json.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_json.write_text(
        json.dumps(
            summary,
            indent=2
        )
    )

    print(
        "========================================"
    )
    print(
        "RESTORATION → FDI ASSOCIATION V2"
    )
    print(
        "========================================"
    )

    print(
        "Raw restorations :",
        len(raw_rest)
    )

    print(
        "After dedup      :",
        len(deduped)
    )

    print()

    for fdi_num in sorted(
        findings,
        key=lambda x:
            FDI_ORDER.index(x)
            if x in FDI_ORDER
            else 999
    ):

        for item in findings[fdi_num]:
            print(
                f'FDI {fdi_num}: '
                f'{item["class_name"]} '
                f'conf={item["confidence"]:.3f} '
                f'score={item["association"]["score"]:.1f} '
                f'slot_dist={item["association"]["slot_distance_px"]:.1f}'
            )

    print()

    print(
        "Unmatched:",
        len(unmatched)
    )

    for item in unmatched:
        print(
            f'  {item["class_name"]} '
            f'conf={item["confidence"]:.3f} '
            f'candidate={item.get("candidate_fdi")} '
            f'score={item.get("association", {}).get("score")}'
        )

    print()

    print("Saved:")
    print(
        output_json.resolve()
    )


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
