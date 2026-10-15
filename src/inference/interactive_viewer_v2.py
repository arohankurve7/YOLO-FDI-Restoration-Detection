from pathlib import Path
import argparse
import base64
import json
import mimetypes


ROOT = Path(__file__).resolve().parents[2]

UPPER = [
    "18", "17", "16", "15", "14", "13", "12", "11",
    "21", "22", "23", "24", "25", "26", "27", "28"
]

LOWER = [
    "48", "47", "46", "45", "44", "43", "42", "41",
    "31", "32", "33", "34", "35", "36", "37", "38"
]


TOOTH_NAMES = {
    "11": "Maxillary Right Central Incisor",
    "12": "Maxillary Right Lateral Incisor",
    "13": "Maxillary Right Canine",
    "14": "Maxillary Right First Premolar",
    "15": "Maxillary Right Second Premolar",
    "16": "Maxillary Right First Molar",
    "17": "Maxillary Right Second Molar",
    "18": "Maxillary Right Third Molar",

    "21": "Maxillary Left Central Incisor",
    "22": "Maxillary Left Lateral Incisor",
    "23": "Maxillary Left Canine",
    "24": "Maxillary Left First Premolar",
    "25": "Maxillary Left Second Premolar",
    "26": "Maxillary Left First Molar",
    "27": "Maxillary Left Second Molar",
    "28": "Maxillary Left Third Molar",

    "31": "Mandibular Left Central Incisor",
    "32": "Mandibular Left Lateral Incisor",
    "33": "Mandibular Left Canine",
    "34": "Mandibular Left First Premolar",
    "35": "Mandibular Left Second Premolar",
    "36": "Mandibular Left First Molar",
    "37": "Mandibular Left Second Molar",
    "38": "Mandibular Left Third Molar",

    "41": "Mandibular Right Central Incisor",
    "42": "Mandibular Right Lateral Incisor",
    "43": "Mandibular Right Canine",
    "44": "Mandibular Right First Premolar",
    "45": "Mandibular Right Second Premolar",
    "46": "Mandibular Right First Molar",
    "47": "Mandibular Right Second Molar",
    "48": "Mandibular Right Third Molar",
}


def template_for(fdi: str):
    pos = int(fdi[1])

    if pos in (1, 2):
        return "11.svg"

    if pos == 3:
        return "13.svg"

    if pos in (4, 5):
        return "14.svg"

    return "16.svg"


def transform_for(fdi: str):
    quadrant = int(fdi[0])

    if quadrant == 1:
        return ""

    if quadrant == 2:
        return "scaleX(-1)"

    if quadrant == 3:
        return "rotate(180deg)"

    if quadrant == 4:
        return "rotate(180deg) scaleX(-1)"

    return ""


def file_to_data_uri(path: Path):
    mime, _ = mimetypes.guess_type(path.name)
    mime = mime or "application/octet-stream"

    payload = base64.b64encode(
        path.read_bytes()
    ).decode("ascii")

    return f"data:{mime};base64,{payload}"


def tooth_svg_data_uri(fdi: str):
    svg_path = (
        ROOT
        / "src"
        / "assets"
        / "teeth_clean"
        / template_for(fdi)
    )

    if not svg_path.exists():
        raise FileNotFoundError(
            f"Tooth SVG not found: {svg_path}"
        )

    return file_to_data_uri(svg_path)


def pretty_finding_name(class_name: str):
    mapping = {
        "implant": "Implant",
        "Dental_Fillings": "Dental Filling",
        "Crown": "Crown",
        "bridge": "Bridge",
        "Root_Canal_treated_tooth": "Root Canal",
        "Endodontic_post": "Endodontic Post",
    }

    return mapping.get(
        class_name,
        class_name.replace("_", " ")
    )


def tooth_cell(
    fdi: str,
    present: set[str],
    tooth_findings: dict,
):
    findings = tooth_findings.get(
        fdi,
        []
    )

    has_finding = len(findings) > 0

    has_implant = any(
        item.get("class_name") == "implant"
        for item in findings
    )

    has_bridge = any(
        item.get("class_name") == "bridge"
        for item in findings
    )

    has_crown = any(
        item.get("class_name") == "Crown"
        for item in findings
    )

    has_filling = any(
        item.get("class_name") == "Dental_Fillings"
        for item in findings
    )

    natural_present = fdi in present

    # IMPORTANT:
    # Grey only when neither a natural tooth nor
    # any restorative finding exists at this FDI slot.
    truly_missing = (
        not natural_present
        and not has_finding
    )

    classes = ["tooth-cell"]

    if truly_missing:
        classes.append("missing")

    if has_implant:
        classes.append("implant-state")

    if has_bridge:
        classes.append("bridge-state")

    if has_crown:
        classes.append("crown-state")

    if has_filling:
        classes.append("filling-state")

    if has_finding:
        classes.append("has-finding")

    state_class = " ".join(classes)

    img_src = tooth_svg_data_uri(
        fdi
    )

    transform = transform_for(
        fdi
    )

    is_upper = fdi[0] in (
        "1",
        "2",
    )

    number_html = (
        f'<div class="tooth-number">{fdi}</div>'
    )

    finding_icons = ""

    if has_implant:
        finding_icons += """
        <div
            class="implant-glyph"
            aria-label="Implant"
        >
            <div class="implant-head"></div>
            <div class="implant-body"></div>
        </div>
        """

    if has_bridge:
        finding_icons += """
        <div
            class="bridge-glyph"
            aria-label="Bridge"
        ></div>
        """

    if has_crown:
        finding_icons += """
        <div
            class="crown-glyph"
            aria-label="Crown"
        ></div>
        """

    if has_filling:
        finding_icons += """
        <div
            class="filling-glyph"
            aria-label="Dental Filling"
        ></div>
        """

    image_html = f"""
    <div class="tooth-image-wrap">

        <img
            src="{img_src}"
            class="tooth-image"
            style="transform:{transform};"
            alt="FDI {fdi}"
        />

        {finding_icons}

    </div>
    """

    if is_upper:
        content = (
            number_html
            + image_html
        )
    else:
        content = (
            image_html
            + number_html
        )

    return f"""
    <button
        class="{state_class}"
        data-fdi="{fdi}"
        onclick="selectTooth('{fdi}')"
        title="FDI {fdi}"
    >
        {content}
    </button>
    """


def build_dashboard(
    prediction_path: Path,
    output_path: Path,
    findings_path: Path | None = None,
):
    prediction = json.loads(
        prediction_path.read_text()
    )

    tooth_findings = {}

    if (
        findings_path is not None
        and findings_path.exists()
    ):
        combined = json.loads(
            findings_path.read_text()
        )

        tooth_findings = combined.get(
            "tooth_findings",
            {}
        )

    image_path = Path(
        prediction["image"]
    )

    if not image_path.is_absolute():
        image_path = (
            ROOT
            / image_path
        )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Original image not found: {image_path}"
        )

    image_uri = file_to_data_uri(
        image_path
    )

    present = set(
        prediction.get(
            "present_teeth",
            []
        )
    )

    detections = {
        det["fdi"]: det
        for det
        in prediction.get(
            "detections",
            []
        )
    }

    upper_html = "".join(
        tooth_cell(
            fdi,
            present,
            tooth_findings,
        )
        for fdi in UPPER
    )

    lower_html = "".join(
        tooth_cell(
            fdi,
            present,
            tooth_findings,
        )
        for fdi in LOWER
    )

    prediction_js = json.dumps(
        detections,
        separators=(",", ":")
    )

    findings_js = json.dumps(
        tooth_findings,
        separators=(",", ":")
    )

    tooth_names_js = json.dumps(
        TOOTH_NAMES,
        separators=(",", ":")
    )

    html = f"""<!DOCTYPE html>
<html>

<head>

<meta charset="utf-8">

<title>
PanoraDent Interactive Viewer
</title>

<style>

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;

    background: #f3f5f7;

    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        Arial,
        sans-serif;

    color: #252a31;
}}

.page {{
    width: 1240px;

    margin: 24px auto;
}}

.card {{
    background: #ffffff;

    border:
        1px solid #dce1e6;

    border-radius: 10px;

    box-shadow:
        0 2px 10px
        rgba(0,0,0,0.04);
}}

.viewer-card {{
    padding: 18px;
}}

.viewer-layout {{
    display: grid;

    grid-template-columns:
        1fr 310px;

    gap: 18px;
}}

.viewer {{
    position: relative;

    background: #111;

    border-radius: 8px;

    overflow: hidden;

    height: 430px;

    display: flex;

    align-items: center;

    justify-content: center;
}}

#opg {{
    display: block;

    width: auto;

    height: 100%;

    max-width: 100%;

    object-fit: contain;
}}

#overlay {{
    position: absolute;

    inset: 0;

    width: 100%;

    height: 100%;

    pointer-events: none;
}}

.side {{
    display: flex;

    flex-direction: column;

    gap: 12px;
}}

.panel {{
    border:
        1px solid #e1e5e9;

    border-radius: 8px;

    padding: 14px;
}}

.panel-title {{
    font-weight: 650;

    margin-bottom: 8px;
}}

.value {{
    font-size: 13px;

    color: #59636f;

    line-height: 1.55;
}}

#zoomCanvas {{
    width: 100%;

    border-radius: 6px;

    background: #f4f6f8;

    border: 1px solid #e3e7eb;
}}

.chart-card {{
    margin-top: 16px;

    padding:
        16px 20px 18px;
}}

.chart-header {{
    display: flex;

    justify-content:
        space-between;

    align-items: center;

    margin-bottom: 8px;
}}

.legend {{
    display: flex;

    gap: 14px;

    font-size: 12px;

    color: #68727f;
}}

.legend-item {{
    display: flex;

    gap: 6px;

    align-items: center;
}}

.dot {{
    width: 9px;

    height: 9px;

    border-radius: 50%;

    border:
        1px solid #8f98a3;

    background: #fff;
}}

.dot.missing-dot {{
    background: #c8cdd3;
}}

.dot.implant-dot {{
    background: #159ee4;

    border-color: #159ee4;
}}

.dot.bridge-dot {{
    background: #f28c28;

    border-color: #f28c28;
}}

.dot.crown-dot {{
    background: #f5c542;

    border-color: #d9a900;
}}

.dot.filling-dot {{
    background: #53b96b;

    border-color: #3b9651;
}}

.chart {{
    position: relative;
}}

.arch {{
    display: grid;

    grid-template-columns:
        repeat(16, 1fr);
}}

.arch.lower {{
    margin-top: -5px;
}}

.midline {{
    position: absolute;

    left: 50%;

    top: 20px;

    bottom: 20px;

    width: 1px;

    background: #e5e8ec;
}}

.tooth-cell {{
    height: 105px;

    padding: 0;

    border: 0;

    background: transparent;

    cursor: pointer;

    border-radius: 6px;
}}

.tooth-cell:hover {{
    background: #f3f7fb;
}}

.tooth-cell.selected {{
    background: #e5f0fa;
}}

.tooth-image-wrap {{
    position: relative;

    height: 88px;

    display: flex;

    justify-content: center;

    align-items: center;
}}

.tooth-image {{
    max-height: 88px;

    max-width: 58px;

    transform-origin:
        center center;
}}

.tooth-number {{
    height: 22px;

    display: flex;

    align-items: center;

    justify-content: center;

    font-size: 12px;

    font-weight: 600;

    color: #414a55;
}}


/* TRUE missing slot only */

.missing .tooth-image {{
    opacity: 0.28;

    filter:
        grayscale(1)
        contrast(0.65);
}}

.missing .tooth-number {{
    color: #a0a6ad;
}}


/*
   A restorative finding means the FDI slot
   is clinically occupied / relevant.

   Do NOT grey it simply because the natural
   tooth segmentation model failed.
*/

.has-finding .tooth-image {{
    opacity: 1;

    filter: none;
}}

.has-finding .tooth-number {{
    color: #414a55;
}}


/* Dental Filling */

.filling-state {{
    background: #f2fbf4;
}}

.filling-glyph {{
    position: absolute;

    left: 50%;
    top: 47%;

    width: 18px;
    height: 12px;

    transform: translate(-50%, -50%);

    border-radius: 45%;

    background: rgba(83, 185, 107, 0.82);

    border: 1.5px solid #3b9651;

    pointer-events: none;
}}


/* Crown */

.crown-state {{
    background: #fffaf0;
}}

.crown-glyph {{
    position: absolute;

    left: 50%;
    top: 25%;

    width: 30px;
    height: 18px;

    transform: translate(-50%, -50%);

    border-radius: 12px 12px 5px 5px;

    background: rgba(245, 197, 66, 0.82);

    border: 1.5px solid #d9a900;

    pointer-events: none;
}}


/* Bridge */

.bridge-state {{
    background: #fff7ef;
}}

.bridge-glyph {{
    position: absolute;

    left: 50%;
    top: 45%;

    width: 38px;
    height: 9px;

    transform: translate(-50%, -50%);

    border-radius: 5px;

    background: rgba(242, 140, 40, 0.85);

    border: 1.5px solid #d56f0b;

    pointer-events: none;
}}


/* Implant */

.implant-state {{
    background: #eef6ff;
}}

.implant-state .tooth-number {{
    color: #1677c8;

    font-weight: 700;
}}

.implant-glyph {{
    position: absolute;

    left: 50%;

    top: 50%;

    width: 18px;

    height: 56px;

    transform:
        translate(-50%, -46%);

    pointer-events: none;
}}

.implant-head {{
    width: 18px;

    height: 11px;

    border:
        2px solid #159ee4;

    border-radius:
        3px 3px 2px 2px;

    background: #ffffff;
}}

.implant-body {{
    width: 12px;

    height: 41px;

    margin: 0 auto;

    border:
        2px solid #159ee4;

    border-top: 0;

    border-radius:
        0 0 5px 5px;

    background:
        repeating-linear-gradient(
            180deg,
            #ffffff 0px,
            #ffffff 4px,
            #cfeeff 4px,
            #cfeeff 6px
        );
}}

</style>

</head>


<body>

<div class="page">


<div class="card viewer-card">

<div class="viewer-layout">


<div class="viewer">

<img
    id="opg"
    src="{image_uri}"
    alt="Panoramic radiograph"
/>

<canvas
    id="overlay"
></canvas>

</div>


<div class="side">


<div class="panel">

<div class="panel-title">
Selected Tooth
</div>

<div
    id="selectedInfo"
    class="value"
>
Click a tooth in the chart.
</div>

</div>


<div class="panel">

<div class="panel-title">
Tooth Zoom
</div>

<canvas
    id="zoomCanvas"
></canvas>

</div>


</div>


</div>

</div>


<div class="card chart-card">


<div class="chart-header">


<div style="font-weight:650">
Tooth Chart
</div>


<div class="legend">

<div class="legend-item">

<span class="dot"></span>

Present

</div>


<div class="legend-item">

<span class="dot missing-dot"></span>

Missing candidate

</div>


<div class="legend-item">

<span class="dot implant-dot"></span>

Implant

</div>


<div class="legend-item">

<span class="dot bridge-dot"></span>

Bridge

</div>


<div class="legend-item">

<span class="dot crown-dot"></span>

Crown

</div>


<div class="legend-item">

<span class="dot filling-dot"></span>

Filling

</div>

</div>


</div>


<div class="chart">

<div class="midline"></div>


<div class="arch upper">

{upper_html}

</div>


<div class="arch lower">

{lower_html}

</div>


</div>


</div>


</div>


<script>

const DETECTIONS =
    {prediction_js};

const FINDINGS =
    {findings_js};

const TOOTH_NAMES =
    {tooth_names_js};


const image =
    document.getElementById(
        "opg"
    );

const overlay =
    document.getElementById(
        "overlay"
    );

const ctx =
    overlay.getContext(
        "2d"
    );


const zoomCanvas =
    document.getElementById(
        "zoomCanvas"
    );

const zoomCtx =
    zoomCanvas.getContext(
        "2d"
    );


function resizeOverlay() {{

    overlay.width =
        image.naturalWidth;

    overlay.height =
        image.naturalHeight;

    const viewer =
        image.parentElement;

    const renderedWidth =
        image.clientWidth;

    const renderedHeight =
        image.clientHeight;

    overlay.style.width =
        renderedWidth + "px";

    overlay.style.height =
        renderedHeight + "px";

    overlay.style.left =
        ((viewer.clientWidth - renderedWidth) / 2)
        + "px";

    overlay.style.top =
        ((viewer.clientHeight - renderedHeight) / 2)
        + "px";

    overlay.style.right =
        "auto";

    overlay.style.bottom =
        "auto";
}}


image.addEventListener(
    "load",
    () => {{
        resizeOverlay();
    }}
);


if (image.complete) {{
    resizeOverlay();
}}


function clearSelection() {{

    document
        .querySelectorAll(
            ".tooth-cell"
        )
        .forEach(
            el =>
                el.classList.remove(
                    "selected"
                )
        );

    ctx.clearRect(
        0,
        0,
        overlay.width,
        overlay.height
    );
}}


function drawPolygon(
    polygon,
    fillStyle,
    strokeStyle,
    lineWidth
) {{

    if (
        !polygon
        || polygon.length < 3
    ) {{
        return;
    }}

    ctx.beginPath();

    ctx.moveTo(
        polygon[0].x,
        polygon[0].y
    );

    for (
        let i = 1;
        i < polygon.length;
        i++
    ) {{

        ctx.lineTo(
            polygon[i].x,
            polygon[i].y
        );
    }}

    ctx.closePath();

    ctx.fillStyle =
        fillStyle;

    ctx.strokeStyle =
        strokeStyle;

    ctx.lineWidth =
        lineWidth;

    ctx.fill();

    ctx.stroke();
}}


function prettyFindingName(
    className
) {{

    const names = {{
        "implant":
            "Implant",

        "Dental_Fillings":
            "Dental Filling",

        "Crown":
            "Crown",

        "bridge":
            "Bridge",

        "Root_Canal_treated_tooth":
            "Root Canal",

        "Endodontic_post":
            "Endodontic Post"
    }};

    return (
        names[className]
        ||
        className.replaceAll(
            "_",
            " "
        )
    );
}}


function selectTooth(fdi) {{

    clearSelection();


    const cell =
        document.querySelector(
            `[data-fdi="${{fdi}}"]`
        );


    if (cell) {{

        cell.classList.add(
            "selected"
        );
    }}


    const det =
        DETECTIONS[fdi];


    const findings =
        FINDINGS[fdi]
        || [];


    /*
       Draw raw natural-tooth segmentation
       when available.
    */

    if (det) {{

        drawPolygon(
            det.mask_polygon,
            "rgba(255, 215, 0, 0.30)",
            "rgba(255, 215, 0, 0.95)",
            3
        );
    }}


    /*
       Draw restoration masks.
    */

    findings.forEach(
        item => {{

            drawPolygon(
                item.mask_polygon,
                "rgba(0, 190, 255, 0.45)",
                "rgba(0, 190, 255, 1.0)",
                3
            );
        }}
    );


    /*
       DISPLAY RULE

       If there is any associated restorative
       finding, UI should not report the region
       as a detected natural tooth.

       Backend raw model output remains unchanged.
    */

    const toothName =
        TOOTH_NAMES[fdi]
        || "Permanent Tooth";


    const hasImplant =
        findings.some(
            item =>
                item.class_name === "implant"
        );

    const naturalToothDetected =
        (
            !!det
            &&
            !hasImplant
        );


    let info =
        `<b>FDI ${{fdi}} — ${{toothName}}</b>`
        +
        `<br>`
        +
        `Natural tooth: `
        +
        (
            naturalToothDetected
            ? "Detected"
            : "Not detected"
        );


    info +=
        `<br><br><b>Findings</b>`;


    if (
        findings.length > 0
    ) {{

        findings.forEach(
            item => {{

                info +=
                    `<br>${{
                        prettyFindingName(
                            item.class_name
                        )
                    }}`;
            }}
        );

    }} else {{

        info +=
            `<br>None`;
    }}


    document.getElementById(
        "selectedInfo"
    ).innerHTML =
        info;


    /*
       Zoom source.

       Natural tooth bbox first,
       restoration bbox fallback.
    */

    let box = null;


    if (det) {{

        box = det.bbox;

    }} else if (
        findings.length > 0
    ) {{

        box =
            findings[0].bbox;
    }}


    zoomCanvas.width =
        280;

    zoomCanvas.height =
        230;


    zoomCtx.clearRect(
        0,
        0,
        zoomCanvas.width,
        zoomCanvas.height
    );


    if (!box) {{

        zoomCtx.fillStyle = "#f4f6f8";

        zoomCtx.fillRect(
            0,
            0,
            zoomCanvas.width,
            zoomCanvas.height
        );

        zoomCtx.fillStyle = "#7a8490";

        zoomCtx.font =
            "14px -apple-system, BlinkMacSystemFont, Segoe UI, Arial";

        zoomCtx.textAlign = "center";
        zoomCtx.textBaseline = "middle";

        zoomCtx.fillText(
            "No region available for zoom",
            zoomCanvas.width / 2,
            zoomCanvas.height / 2
        );

        return;
    }}


    const padX =
        Math.max(
            25,
            (
                box.x2
                - box.x1
            )
            * 0.8
        );


    const padY =
        Math.max(
            25,
            (
                box.y2
                - box.y1
            )
            * 0.55
        );


    const sx =
        Math.max(
            0,
            box.x1 - padX
        );


    const sy =
        Math.max(
            0,
            box.y1 - padY
        );


    const sw =
        Math.min(
            image.naturalWidth
            - sx,

            (
                box.x2
                - box.x1
            )
            +
            2 * padX
        );


    const sh =
        Math.min(
            image.naturalHeight
            - sy,

            (
                box.y2
                - box.y1
            )
            +
            2 * padY
        );


    zoomCtx.drawImage(
        image,

        sx,
        sy,
        sw,
        sh,

        0,
        0,

        zoomCanvas.width,
        zoomCanvas.height
    );
}}

</script>


</body>

</html>
"""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path.write_text(
        html,
        encoding="utf-8"
    )

    print(
        "Saved interactive viewer:"
    )

    print(
        output_path.resolve()
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prediction",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    parser.add_argument(
        "--findings",
        required=False,
        default=None
    )

    args = parser.parse_args()

    build_dashboard(
        Path(
            args.prediction
        ),

        Path(
            args.output
        ),

        (
            Path(
                args.findings
            )
            if args.findings
            else None
        )
    )
