from pathlib import Path
import argparse
import base64
import html
import json


ROOT = Path(__file__).resolve().parents[2]

FDI_UPPER = [
    "18","17","16","15","14","13","12","11",
    "21","22","23","24","25","26","27","28",
]

FDI_LOWER = [
    "48","47","46","45","44","43","42","41",
    "31","32","33","34","35","36","37","38",
]

TOOTH_COLORS = {
    "Central Incisor": "#DC4652",
    "Lateral Incisor": "#9B51B6",
    "Canine": "#3977C3",
    "First Premolar": "#58BFD2",
    "Second Premolar": "#55A95B",
    "First Molar": "#E4BE32",
    "Second Molar": "#E58A2B",
    "Third Molar": "#D65A2D",
}


def image_to_data_uri(path):
    path = Path(path)

    suffix = path.suffix.lower()

    mime = (
        "image/png"
        if suffix == ".png"
        else "image/jpeg"
    )

    encoded = base64.b64encode(
        path.read_bytes()
    ).decode("ascii")

    return (
        f"data:{mime};base64,{encoded}"
    )


def tooth_svg_markup(fdi, tooth):
    template_map = {
        1: "11.svg",
        2: "11.svg",
        3: "13.svg",
        4: "14.svg",
        5: "14.svg",
        6: "16.svg",
        7: "16.svg",
        8: "16.svg",
    }

    pos = int(fdi[1])

    svg_path = (
        ROOT
        / "src"
        / "assets"
        / "teeth_clean"
        / template_map[pos]
    )

    svg = svg_path.read_text(
        encoding="utf-8"
    )

    if svg.startswith("<?xml"):
        svg = svg.split("?>", 1)[1]

    detected = (
        tooth["status"] == "present"
    )

    if detected:
        color = TOOTH_COLORS[
            tooth["tooth_type"]
        ]
        fill = color
        stroke = color
    else:
        fill = "#D9DDE2"
        stroke = "#AEB5BD"

    svg = svg.replace(
        "fill: #ffffff;",
        f"fill: {fill};"
    )

    svg = svg.replace(
        "stroke: #929aa3;",
        f"stroke: {stroke};"
    )

    svg = svg.replace(
        "fill: #fefefe;",
        f"fill: {fill};"
    )

    svg = svg.replace(
        "<svg ",
        '<svg class="tooth-svg" ',
        1,
    )

    return svg


def build_tooth_button(fdi, tooth):
    detected = (
        tooth["status"] == "present"
    )

    disabled = (
        ""
        if detected
        else "disabled"
    )

    css_class = (
        "tooth-button"
        if detected
        else "tooth-button missing"
    )

    return f"""
    <button
        class="{css_class}"
        data-fdi="{fdi}"
        {disabled}
        title="{html.escape(tooth['anatomical_name'])}"
    >
        <div class="tooth-number">{fdi}</div>
        <div class="tooth-svg-wrap">
            {tooth_svg_markup(fdi, tooth)}
        </div>
    </button>
    """


def build_dashboard(
    registry_json,
    overlay_image,
    output_html,
):
    registry_json = Path(
        registry_json
    )

    overlay_image = Path(
        overlay_image
    )

    output_html = Path(
        output_html
    )

    data = json.loads(
        registry_json.read_text()
    )

    registry_list = data[
        "tooth_registry"
    ]

    registry = {
        tooth["fdi"]: tooth
        for tooth in registry_list
    }

    summary = data["summary"]

    upper_chart = "".join(
        build_tooth_button(
            fdi,
            registry[fdi]
        )
        for fdi in FDI_UPPER
    )

    lower_chart = "".join(
        build_tooth_button(
            fdi,
            registry[fdi]
        )
        for fdi in FDI_LOWER
    )

    rows = []

    for fdi in FDI_UPPER + FDI_LOWER:
        tooth = registry[fdi]

        status_label = (
            "Present"
            if tooth["status"] == "present"
            else "Missing"
        )

        status_class = (
            "present-status"
            if tooth["status"] == "present"
            else "missing-status"
        )

        rows.append(
            f"""
            <tr data-row-fdi="{fdi}">
                <td>{fdi}</td>
                <td>
                    {html.escape(
                        tooth["anatomical_name"]
                    )}
                </td>
                <td class="{status_class}">
                    {status_label}
                </td>
            </tr>
            """
        )

    overlay_uri = image_to_data_uri(
        overlay_image
    )

    first_present = next(
        tooth
        for tooth in registry_list
        if tooth["status"] == "present"
    )

    json_blob = json.dumps(
        registry
    ).replace(
        "</",
        "<\\/"
    )

    dashboard = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">

<title>Dental X-ray Analysis Dashboard</title>

<style>

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    background: #f4f6f8;
    color: #243142;

    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        Arial,
        sans-serif;
}}

.page {{
    width: 1480px;
    margin: 18px auto 40px;
}}

.topbar {{
    display: flex;
    justify-content: space-between;
    align-items: center;

    background: white;
    border: 1px solid #dde3e9;
    border-radius: 12px;

    padding: 14px 18px;
    margin-bottom: 12px;
}}

.brand {{
    font-size: 19px;
    font-weight: 750;
}}

.meta {{
    color: #667383;
    font-size: 13px;
}}

.grid {{
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: 12px;
}}

.card {{
    background: white;
    border: 1px solid #dde3e9;
    border-radius: 12px;
    padding: 16px;
}}

.card-title {{
    font-size: 15px;
    font-weight: 700;
    margin-bottom: 12px;
}}

.opg {{
    width: 100%;
    max-height: 540px;
    object-fit: contain;
    background: #111;
    border-radius: 7px;
}}

.summary {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 9px;
}}

.metric {{
    border: 1px solid #e2e7ec;
    border-radius: 9px;
    padding: 12px;
}}

.metric-label {{
    color: #687688;
    font-size: 11px;
}}

.metric-value {{
    margin-top: 5px;
    font-size: 25px;
    font-weight: 750;
}}

.legend-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 9px 18px;
}}

.legend-item {{
    display: flex;
    align-items: center;
    gap: 8px;

    font-size: 12px;
}}

.swatch {{
    width: 11px;
    height: 11px;
    border-radius: 3px;
}}

.findings-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
}}

.findings-table th,
.findings-table td {{
    border-bottom: 1px solid #edf0f3;
    padding: 8px 6px;
    text-align: left;
}}

.present-status {{
    color: #238636;
}}

.missing-status {{
    color: #8b949e;
}}

.chart-card {{
    margin-top: 12px;
}}

.chart-header {{
    margin-bottom: 5px;
}}

.instruction {{
    font-size: 12px;
    color: #738092;
    margin-bottom: 10px;
}}

.chart {{
    position: relative;
}}

.arch {{
    display: grid;
    grid-template-columns: repeat(16, 1fr);
}}

.midline {{
    position: absolute;
    left: 50%;
    top: 8px;
    bottom: 8px;
    width: 1px;
    background: #dbe1e7;
}}

.tooth-button {{
    border: 0;
    background: transparent;
    cursor: pointer;
    border-radius: 7px;
    padding: 3px;
}}

.tooth-button:hover {{
    background: #f0f5f9;
}}

.tooth-button.selected {{
    outline: 2px solid #3977C3;
    background: #eef5fb;
}}

.tooth-button.missing {{
    cursor: not-allowed;
}}

.tooth-number {{
    font-size: 11px;
    font-weight: 650;
    text-align: center;
}}

.tooth-svg-wrap {{
    height: 83px;
    display: flex;
    align-items: center;
    justify-content: center;
}}

.tooth-svg {{
    width: 45px;
    height: 79px;
}}

.tooth-button.missing {{
    opacity: 0.42;
}}

.chart-label {{
    text-align: center;
    font-size: 12px;
    font-weight: 650;
    color: #617083;
}}

.bottom-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr 0.9fr;
    gap: 12px;
    margin-top: 12px;
}}

.detail-value {{
    font-weight: 650;
}}

.detail-row {{
    display: grid;
    grid-template-columns: 150px 1fr;
    margin-bottom: 9px;
    font-size: 13px;
}}

.zoom-box {{
    position: relative;
    overflow: hidden;
    min-height: 270px;
    background: #111;
    border-radius: 7px;
}}

.zoom-box img {{
    position: absolute;
    max-width: none;
}}

.disclaimer {{
    border: 1px solid #9fc7ef;
    background: #eef7ff;
    padding: 11px;
    border-radius: 8px;
    font-size: 12px;
    color: #24547d;
}}

.summary-list {{
    font-size: 13px;
    line-height: 1.7;
}}

.missing-note {{
    margin-top: 10px;
    color: #737f8c;
    font-size: 11px;
}}

</style>
</head>

<body>

<div class="page">

    <div class="topbar">

        <div>
            <div class="brand">
                Dental X-ray Analysis
            </div>
            <div class="meta">
                AI-Powered OPG Analysis
            </div>
        </div>

        <div class="meta">
            Modality: Panoramic OPG
        </div>

    </div>


    <div class="grid">

        <div class="card">

            <div class="card-title">
                1. Panoramic OPG with AI Annotations
            </div>

            <img
                id="main-opg"
                class="opg"
                src="{overlay_uri}"
                alt="Panoramic OPG segmentation overlay"
            />

        </div>


        <div>

            <div class="card">

                <div class="card-title">
                    2. Summary
                </div>

                <div class="summary">

                    <div class="metric">
                        <div class="metric-label">
                            Detected Teeth
                        </div>
                        <div class="metric-value">
                            {summary["teeth_detected"]}
                            <span style="font-size:13px;color:#718096">
                                / 32
                            </span>
                        </div>
                    </div>

                    <div class="metric">
                        <div class="metric-label">
                            Upper Arch
                        </div>
                        <div class="metric-value">
                            {summary["upper_arch"]}
                            <span style="font-size:13px;color:#718096">
                                / 16
                            </span>
                        </div>
                    </div>

                    <div class="metric">
                        <div class="metric-label">
                            Lower Arch
                        </div>
                        <div class="metric-value">
                            {summary["lower_arch"]}
                            <span style="font-size:13px;color:#718096">
                                / 16
                            </span>
                        </div>
                    </div>

                    <div class="metric">
                        <div class="metric-label">
                            Missing Teeth
                        </div>
                        <div class="metric-value">
                            {summary["not_detected"]}
                            <span style="font-size:13px;color:#718096">
                                / 32
                            </span>
                        </div>
                    </div>

                </div>

            </div>


            <div class="card" style="margin-top:12px">

                <div class="card-title">
                    3. FDI Tooth-Type Legend (ISO 3950)
                </div>

                <div class="legend-grid">

                    <div class="legend-item">
                        <span class="swatch"
                              style="background:#D65A2D"></span>
                        18/28/38/48 — Third Molar
                    </div>

                    <div class="legend-item">
                        <span class="swatch"
                              style="background:#E58A2B"></span>
                        17/27/37/47 — Second Molar
                    </div>

                    <div class="legend-item">
                        <span class="swatch"
                              style="background:#E4BE32"></span>
                        16/26/36/46 — First Molar
                    </div>

                    <div class="legend-item">
                        <span class="swatch"
                              style="background:#55A95B"></span>
                        15/25/35/45 — Second Premolar
                    </div>

                    <div class="legend-item">
                        <span class="swatch"
                              style="background:#58BFD2"></span>
                        14/24/34/44 — First Premolar
                    </div>

                    <div class="legend-item">
                        <span class="swatch"
                              style="background:#3977C3"></span>
                        13/23/33/43 — Canine
                    </div>

                    <div class="legend-item">
                        <span class="swatch"
                              style="background:#9B51B6"></span>
                        12/22/32/42 — Lateral Incisor
                    </div>

                    <div class="legend-item">
                        <span class="swatch"
                              style="background:#DC4652"></span>
                        11/21/31/41 — Central Incisor
                    </div>

                </div>

            </div>


            <div class="card" style="margin-top:12px">

                <div class="card-title">
                    4. Findings (Tooth-wise)
                </div>

                <div style="max-height:360px;overflow:auto">

                    <table class="findings-table">

                        <thead>
                            <tr>
                                <th>FDI</th>
                                <th>Anatomical Tooth Name</th>
                                <th>Status</th>
                            </tr>
                        </thead>

                        <tbody>
                            {''.join(rows)}
                        </tbody>

                    </table>

                </div>

            </div>

        </div>

    </div>


    <div class="card chart-card">

        <div class="card-title chart-header">
            5. FDI Two-Digit Tooth Numbering System (ISO 3950)
        </div>

        <div class="instruction">
            Select a detected tooth to view its details and zoomed view.
        </div>

        <div class="chart-label">
            Maxillary
        </div>

        <div class="chart">

            <div class="midline"></div>

            <div class="arch">
                {upper_chart}
            </div>

            <div class="arch">
                {lower_chart}
            </div>

        </div>

        <div class="chart-label">
            Mandibular
        </div>

        <div class="missing-note">
            Grey = Missing Teeth · Disabled
        </div>

    </div>


    <div class="bottom-grid">

        <div class="card">

            <div class="card-title">
                6. Tooth Details
            </div>

            <div class="detail-row">
                <div>FDI Number</div>
                <div id="detail-fdi" class="detail-value"></div>
            </div>

            <div class="detail-row">
                <div>Anatomical Tooth Name</div>
                <div id="detail-name" class="detail-value"></div>
            </div>

            <div class="detail-row">
                <div>Tooth Type</div>
                <div id="detail-type" class="detail-value"></div>
            </div>

            <div class="detail-row">
                <div>Status</div>
                <div id="detail-status" class="detail-value"></div>
            </div>

            <div class="detail-row">
                <div>Arch</div>
                <div id="detail-arch" class="detail-value"></div>
            </div>

            <div class="detail-row">
                <div>Quadrant</div>
                <div id="detail-quadrant" class="detail-value"></div>
            </div>

        </div>


        <div class="card">

            <div class="card-title">
                7. Zoomed View
            </div>

            <div
                id="zoom-box"
                class="zoom-box"
            >
                <img
                    id="zoom-image"
                    src="{overlay_uri}"
                    alt="Selected tooth zoom"
                />
            </div>

        </div>


        <div class="card">

            <div class="card-title">
                8. AI Findings Summary
            </div>

            <div class="summary-list">

                <strong>Dentition</strong>

                <ul>
                    <li>
                        {summary["teeth_detected"]}
                        tooth positions detected.
                    </li>

                    <li>
                        {summary["not_detected"]}
                        tooth positions missing.
                    </li>
                </ul>

                <strong>
                    Restorations
                </strong>

                <ul>
                    <li>
                        Restoration analysis will be
                        integrated in the next phase.
                    </li>
                </ul>

            </div>

            <div class="disclaimer">
                {html.escape(
                    data["screening_disclaimer"]
                )}
            </div>

        </div>

    </div>

</div>


<script>

const registry = {json_blob};

const imageNatural = {{
    width: 0,
    height: 0
}};

const mainImage =
    document.getElementById(
        "main-opg"
    );

const zoomImage =
    document.getElementById(
        "zoom-image"
    );

const zoomBox =
    document.getElementById(
        "zoom-box"
    );


function setDetails(tooth) {{

    document.getElementById(
        "detail-fdi"
    ).textContent =
        tooth.fdi;

    document.getElementById(
        "detail-name"
    ).textContent =
        tooth.anatomical_name;

    document.getElementById(
        "detail-type"
    ).textContent =
        tooth.tooth_type;

    document.getElementById(
        "detail-status"
    ).textContent =
        tooth.status === "present"
        ? "Present"
        : "Missing";

    document.getElementById(
        "detail-arch"
    ).textContent =
        tooth.arch;

    document.getElementById(
        "detail-quadrant"
    ).textContent =
        tooth.quadrant;
}}


function zoomToTooth(tooth) {{

    if (
        !tooth.bbox
        || !imageNatural.width
    ) {{
        return;
    }}

    const bbox =
        tooth.bbox;

    const cx =
        (bbox.x1 + bbox.x2) / 2;

    const cy =
        (bbox.y1 + bbox.y2) / 2;

    const boxWidth =
        bbox.x2 - bbox.x1;

    const boxHeight =
        bbox.y2 - bbox.y1;

    const cropWidth =
        Math.max(
            boxWidth * 4.0,
            imageNatural.width * 0.12
        );

    const cropHeight =
        Math.max(
            boxHeight * 3.2,
            imageNatural.height * 0.25
        );

    const scaleX =
        zoomBox.clientWidth
        / cropWidth;

    const scaleY =
        zoomBox.clientHeight
        / cropHeight;

    const scale =
        Math.min(
            scaleX,
            scaleY
        );

    zoomImage.style.width =
        (
            imageNatural.width
            * scale
        )
        + "px";

    zoomImage.style.height =
        (
            imageNatural.height
            * scale
        )
        + "px";

    zoomImage.style.left =
        (
            zoomBox.clientWidth / 2
            - cx * scale
        )
        + "px";

    zoomImage.style.top =
        (
            zoomBox.clientHeight / 2
            - cy * scale
        )
        + "px";
}}


function selectTooth(fdi) {{

    const tooth =
        registry[fdi];

    if (
        !tooth
        || !tooth.selectable
    ) {{
        return;
    }}

    document
        .querySelectorAll(
            ".tooth-button"
        )
        .forEach(button =>
            button.classList.remove(
                "selected"
            )
        );

    const button =
        document.querySelector(
            `[data-fdi="${{fdi}}"]`
        );

    if (button) {{
        button.classList.add(
            "selected"
        );
    }}

    setDetails(
        tooth
    );

    zoomToTooth(
        tooth
    );

    const row =
        document.querySelector(
            `[data-row-fdi="${{fdi}}"]`
        );

    if (row) {{
        row.scrollIntoView({{
            block: "nearest"
        }});
    }}
}}


mainImage.addEventListener(
    "load",
    () => {{

        imageNatural.width =
            mainImage.naturalWidth;

        imageNatural.height =
            mainImage.naturalHeight;

        selectTooth(
            "{first_present['fdi']}"
        );

    }}
);


document
    .querySelectorAll(
        ".tooth-button:not(.missing)"
    )
    .forEach(button => {{

        button.addEventListener(
            "click",
            () =>
                selectTooth(
                    button.dataset.fdi
                )
        );

    }});

</script>

</body>
</html>
"""

    output_html.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_html.write_text(
        dashboard,
        encoding="utf-8",
    )

    print(
        "Dashboard saved:"
    )

    print(
        output_html.resolve()
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--registry",
        required=True,
    )

    parser.add_argument(
        "--overlay",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    build_dashboard(
        args.registry,
        args.overlay,
        args.output,
    )
