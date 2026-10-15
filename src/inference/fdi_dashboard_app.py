from pathlib import Path
import json
import uuid

from flask import Flask, render_template_string, request, send_file

from src.inference.final_fdi_inference import run_final_fdi_inference

# IMPORTANT:
# Reuse the APPROVED anatomical chart. Do not redraw teeth here.
from src.inference.tooth_chart_dashboard import (
    UPPER,
    LOWER,
    cell_html,
)


ROOT = Path(__file__).resolve().parents[2]

UPLOAD_DIR = ROOT / "results" / "dashboard_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)


@app.route("/overlay/<case_id>")
def serve_overlay(case_id):
    path = (
        ROOT
        / "results"
        / "final_fdi_inference"
        / case_id
        / "fdi_segmentation_overlay.png"
    )

    return send_file(path, mimetype="image/png")


@app.route("/", methods=["GET", "POST"])
def dashboard():
    result = None
    uploaded_name = ""

    if request.method == "POST":
        upload = request.files.get("image")

        if upload and upload.filename:
            suffix = Path(upload.filename).suffix.lower()

            if suffix not in {".png", ".jpg", ".jpeg"}:
                return "Please select a PNG/JPG/JPEG panoramic OPG.", 400

            uploaded_name = upload.filename

            image_path = (
                UPLOAD_DIR
                / f"{uuid.uuid4().hex}{suffix}"
            )

            upload.save(image_path)

            result = run_final_fdi_inference(image_path)

    if result is None:
        return render_template_string(
            TEMPLATE,
            result=None,
        )

    data = result["summary"]

    registry_list = data["tooth_registry"]

    registry = {
        str(tooth["fdi"]): tooth
        for tooth in registry_list
    }

    # EXACT approved anatomical chart cells.
    upper_chart = "".join(
        cell_html(fdi, registry)
        for fdi in UPPER
    )

    lower_chart = "".join(
        cell_html(fdi, registry)
        for fdi in LOWER
    )

    findings_rows = "".join(
        f"""
        <tr data-fdi-row="{tooth['fdi']}">
            <td>{tooth['fdi']}</td>
            <td>{tooth['anatomical_name']}</td>
            <td class="{
                'status-present'
                if tooth['status'] == 'present'
                else 'status-missing'
            }">
                {
                    'Present'
                    if tooth['status'] == 'present'
                    else 'Missing'
                }
            </td>
        </tr>
        """
        for tooth in registry_list
    )

    first_present = next(
        tooth["fdi"]
        for tooth in registry_list
        if tooth["selectable"]
    )

    registry_json = json.dumps(
        registry
    ).replace("</", "<\\/")

    return render_template_string(
        TEMPLATE,
        result=data,
        filename=uploaded_name,
        upper_chart=upper_chart,
        lower_chart=lower_chart,
        findings_rows=findings_rows,
        registry_json=registry_json,
        first_present=first_present,
    )


TEMPLATE = r"""
<!doctype html>

<html>
<head>

<meta charset="utf-8">

<title>Dental X-ray Analysis</title>

<style>

* {
    box-sizing: border-box;
}

html,
body {
    margin: 0;
    min-height: 100%;
}

body {
    background: #f5f7fa;
    color: #243142;

    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        Arial,
        sans-serif;
}

.page {
    max-width: 1510px;
    margin: 10px auto 20px;
    padding: 0 9px;
}

/* =========================================================
   HEADER
   ========================================================= */

.header {
    height: 58px;

    display: flex;
    align-items: center;
    justify-content: space-between;

    background: #fff;
    border: 1px solid #dce2e8;
    border-radius: 11px;

    padding: 8px 14px;

    margin-bottom: 8px;
}

.brand {
    font-size: 19px;
    font-weight: 750;
}

.sub {
    margin-top: 1px;
    font-size: 11px;
    color: #718096;
}

.upload-form {
    display: flex;
    align-items: center;
    gap: 8px;
}

.file-input {
    max-width: 390px;

    border: 1px solid #ccd5df;
    border-radius: 7px;

    padding: 6px;

    background: #fff;

    font-size: 11px;
}

.analyse-button {
    height: 35px;

    border: 0;
    border-radius: 7px;

    padding: 0 17px;

    background: #2468a8;
    color: white;

    font-weight: 650;
    cursor: pointer;
}

/* =========================================================
   GENERIC CARDS
   ========================================================= */

.card {
    background: #fff;

    border: 1px solid #dce2e8;
    border-radius: 10px;

    padding: 10px;
}

.card-title {
    margin-bottom: 7px;

    font-size: 13px;
    font-weight: 700;
}

/* =========================================================
   TOP AREA
   ========================================================= */

.top-grid {
    display: grid;

    grid-template-columns:
        minmax(0, 2.05fr)
        minmax(380px, .95fr);

    gap: 8px;
}

.left-column {
    min-width: 0;
}

.right-column {
    display: grid;

    gap: 8px;

    align-content: start;
}

/* =========================================================
   OPG
   ========================================================= */

.opg-frame {
    height: 345px;

    display: flex;
    align-items: center;
    justify-content: center;

    overflow: hidden;

    border-radius: 6px;

    background: #111;
}

.opg-image {
    display: block;

    max-width: 100%;
    max-height: 100%;

    width: auto;
    height: auto;

    object-fit: contain;
}

/* =========================================================
   SUMMARY
   ========================================================= */

.metrics {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 5px;
}

.metric {
    padding: 7px;

    border: 1px solid #e2e7ed;
    border-radius: 7px;
}

.metric-label {
    font-size: 9px;
    color: #6e7a89;
}

.metric-value {
    margin-top: 3px;

    font-size: 19px;
    font-weight: 750;
}

.metric-total {
    font-size: 10px;
    color: #8190a0;
}

/* =========================================================
   LEGEND
   ========================================================= */

.legend-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;

    gap: 5px 10px;

    font-size: 9px;
}

.legend-item {
    display: flex;
    align-items: center;

    gap: 6px;
}

.swatch {
    width: 9px;
    height: 9px;

    border-radius: 2px;
}

/* =========================================================
   FINDINGS TABLE
   ========================================================= */

.findings-wrap {
    height: 190px;

    overflow-y: auto;
}

.findings {
    width: 100%;

    border-collapse: collapse;

    font-size: 9.5px;
}

.findings th,
.findings td {
    padding: 5px;

    border-bottom: 1px solid #edf0f3;

    text-align: left;
}

.status-present {
    color: #238636;
}

.status-missing {
    color: #9099a4;
}

/* =========================================================
   EXACT APPROVED FDI CHART
   ========================================================= */

.chart-card {
    margin-top: 8px;
}

.chart-instruction {
    margin-bottom: 3px;

    font-size: 9.5px;
    color: #748293;
}

.chart-arch-label {
    text-align: center;

    font-size: 10px;
    font-weight: 650;

    color: #617083;
}

.chart-area {
    position: relative;
}

/* cell_html() uses these exact class names */

.arch-row {
    display: grid;

    grid-template-columns: repeat(16, 1fr);

    align-items: center;
}

.tooth-cell {
    position: relative;

    height: 67px;

    border: 0;

    background: transparent;

    padding: 1px;

    cursor: pointer;

    border-radius: 6px;
}

.tooth-cell:hover {
    background: #f2f6fa;
}

.tooth-cell.selected {
    background: #e8f1fa;

    outline: 2px solid #3977C3;
}

.tooth-number {
    height: 14px;

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 9px;
    font-weight: 650;

    color: #364354;
}

.tooth-image-wrap {
    height: 50px;

    display: flex;
    align-items: center;
    justify-content: center;

    transform-origin: center;
}

.inline-tooth-svg {
    display: block;

    width: 29px;
    height: 48px;
}

.not-detected {
    cursor: not-allowed;
}

.not-detected .inline-tooth-svg {
    opacity: .45;
}

.not-detected .tooth-number {
    color: #a5adb7;
}

.not-detected:hover {
    background: transparent;
}

.grey-note {
    margin-top: 2px;

    font-size: 9px;
    color: #7d8895;
}

/* =========================================================
   BOTTOM AREA
   ========================================================= */

.bottom-grid {
    display: grid;

    grid-template-columns:
        .95fr
        1.2fr
        1fr;

    gap: 8px;

    margin-top: 8px;
}

.detail-grid {
    display: grid;

    grid-template-columns:
        118px
        1fr;

    gap: 5px;

    font-size: 10px;
}

.detail-value {
    font-weight: 650;
}

/* =========================================================
   SELECTED TOOTH ZOOM
   ========================================================= */

.zoom-frame {
    position: relative;

    height: 190px;

    overflow: hidden;

    background: #111;

    border-radius: 6px;
}

.zoom-image {
    position: absolute;

    max-width: none;
    max-height: none;

    transform-origin: top left;
}

/* =========================================================
   REPORT
   ========================================================= */

.summary-text {
    font-size: 10px;
    line-height: 1.55;
}

.summary-text ul {
    margin-top: 5px;
    padding-left: 18px;
}

.disclaimer {
    margin-top: 7px;

    padding: 7px;

    border: 1px solid #afd0ed;
    border-radius: 6px;

    background: #eef7ff;

    color: #285b83;

    font-size: 9px;
    line-height: 1.4;
}

.empty-state {
    padding: 130px 20px;

    text-align: center;

    color: #718096;
}

</style>

</head>

<body>

<div class="page">

    <div class="header">

        <div>

            <div class="brand">
                Dental X-ray Analysis
            </div>

            <div class="sub">
                AI-Powered OPG Analysis
            </div>

        </div>


        <form
            class="upload-form"
            method="post"
            enctype="multipart/form-data"
        >

            <input
                class="file-input"
                type="file"
                name="image"
                accept=".png,.jpg,.jpeg"
                required
            >

            <button
                class="analyse-button"
                type="submit"
            >
                Analyse Image
            </button>

        </form>

    </div>


{% if not result %}

    <div class="card empty-state">

        Browse a panoramic OPG from your Mac and click
        <strong>Analyse Image</strong>.

    </div>

{% else %}


<div class="top-grid">


    <!-- =====================================================
         LEFT
         ===================================================== -->

    <div class="left-column">


        <div class="card">

            <div class="card-title">
                1. Panoramic OPG with AI Annotations
            </div>


            <div class="opg-frame">

                <img
                    id="mainOpg"
                    class="opg-image"
                    src="/overlay/{{ result.case_id }}"
                >

            </div>

        </div>


        <div class="card chart-card">

            <div class="card-title">
                5. FDI Two-Digit Tooth Numbering System (ISO 3950)
            </div>

            <div class="chart-instruction">
                Select a detected tooth to view its details and zoomed view.
            </div>

            <div class="chart-arch-label">
                Maxillary
            </div>


            <div class="chart-area">

                <div class="arch-row">
                    {{ upper_chart|safe }}
                </div>

                <div class="arch-row">
                    {{ lower_chart|safe }}
                </div>

            </div>


            <div class="chart-arch-label">
                Mandibular
            </div>

            <div class="grey-note">
                Grey = Missing Teeth · Disabled
            </div>

        </div>


    </div>


    <!-- =====================================================
         RIGHT
         ===================================================== -->

    <div class="right-column">


        <div class="card">

            <div class="card-title">
                2. Summary
            </div>

            <div class="metrics">


                <div class="metric">

                    <div class="metric-label">
                        Detected Teeth
                    </div>

                    <div class="metric-value">
                        {{ result.summary.teeth_detected }}
                        <span class="metric-total">/32</span>
                    </div>

                </div>


                <div class="metric">

                    <div class="metric-label">
                        Upper Arch
                    </div>

                    <div class="metric-value">
                        {{ result.summary.upper_arch }}
                        <span class="metric-total">/16</span>
                    </div>

                </div>


                <div class="metric">

                    <div class="metric-label">
                        Lower Arch
                    </div>

                    <div class="metric-value">
                        {{ result.summary.lower_arch }}
                        <span class="metric-total">/16</span>
                    </div>

                </div>


                <div class="metric">

                    <div class="metric-label">
                        Missing Teeth
                    </div>

                    <div class="metric-value">
                        {{ result.summary.not_detected }}
                        <span class="metric-total">/32</span>
                    </div>

                </div>


            </div>

        </div>


        <div class="card">

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


        <div class="card">

            <div class="card-title">
                4. Findings (Tooth-wise)
            </div>


            <div class="findings-wrap">

                <table class="findings">

                    <thead>
                        <tr>
                            <th>FDI</th>
                            <th>Anatomical Tooth Name</th>
                            <th>Status</th>
                        </tr>
                    </thead>


                    <tbody>
                        {{ findings_rows|safe }}
                    </tbody>

                </table>

            </div>

        </div>


    </div>


</div>


<!-- =========================================================
     BOTTOM
     ========================================================= -->


<div class="bottom-grid">


    <div class="card">

        <div class="card-title">
            6. Tooth Details
        </div>


        <div class="detail-grid">

            <div>FDI Number</div>
            <div id="detailFdi" class="detail-value"></div>

            <div>Anatomical Tooth Name</div>
            <div id="detailName" class="detail-value"></div>

            <div>Tooth Type</div>
            <div id="detailType" class="detail-value"></div>

            <div>Status</div>
            <div id="detailStatus" class="detail-value"></div>

            <div>Arch</div>
            <div id="detailArch" class="detail-value"></div>

            <div>Quadrant</div>
            <div id="detailQuadrant" class="detail-value"></div>

        </div>

    </div>


    <div class="card">

        <div class="card-title">
            7. Zoomed View
        </div>


        <div
            id="zoomFrame"
            class="zoom-frame"
        >

            <img
                id="zoomImage"
                class="zoom-image"
                src="/overlay/{{ result.case_id }}"
            >

        </div>

    </div>


    <div class="card">

        <div class="card-title">
            8. AI Findings Summary
        </div>


        <div class="summary-text">

            <strong>Dentition</strong>

            <ul>

                <li>
                    {{ result.summary.teeth_detected }}
                    tooth positions detected.
                </li>

                <li>
                    {{ result.summary.not_detected }}
                    tooth positions missing.
                </li>

            </ul>


            <strong>
                Restorations
            </strong>

            <div style="margin-top:4px">
                Restoration analysis will be integrated
                in the next phase.
            </div>

        </div>


        <div class="disclaimer">
            {{ result.screening_disclaimer }}
        </div>

    </div>


</div>


<script>

const registry =
    {{ registry_json|safe }};


const mainOpg =
    document.getElementById(
        "mainOpg"
    );


const zoomImage =
    document.getElementById(
        "zoomImage"
    );


const zoomFrame =
    document.getElementById(
        "zoomFrame"
    );


function setDetails(tooth) {

    document.getElementById(
        "detailFdi"
    ).textContent =
        tooth.fdi;


    document.getElementById(
        "detailName"
    ).textContent =
        tooth.anatomical_name;


    document.getElementById(
        "detailType"
    ).textContent =
        tooth.tooth_type;


    document.getElementById(
        "detailStatus"
    ).textContent =
        "Present";


    document.getElementById(
        "detailArch"
    ).textContent =
        tooth.arch;


    document.getElementById(
        "detailQuadrant"
    ).textContent =
        tooth.quadrant;

}


function zoomToTooth(tooth) {

    if (!tooth.bbox) {
        return;
    }


    const imageWidth =
        mainOpg.naturalWidth;


    const imageHeight =
        mainOpg.naturalHeight;


    if (!imageWidth || !imageHeight) {
        return;
    }


    const bbox =
        tooth.bbox;


    const centerX =
        (
            bbox.x1
            +
            bbox.x2
        )
        / 2;


    const centerY =
        (
            bbox.y1
            +
            bbox.y2
        )
        / 2;


    const toothWidth =
        Math.max(
            1,
            bbox.x2 - bbox.x1
        );


    const toothHeight =
        Math.max(
            1,
            bbox.y2 - bbox.y1
        );


    /*
      Show enough neighbouring anatomy,
      but remain genuinely zoomed on
      selected tooth.
    */

    const cropWidth =
        Math.max(
            toothWidth * 4.0,
            imageWidth * 0.10
        );


    const cropHeight =
        Math.max(
            toothHeight * 2.5,
            imageHeight * 0.17
        );


    const scale =
        Math.min(
            zoomFrame.clientWidth
            / cropWidth,

            zoomFrame.clientHeight
            / cropHeight
        );


    zoomImage.style.width =
        (
            imageWidth
            * scale
        )
        + "px";


    zoomImage.style.height =
        (
            imageHeight
            * scale
        )
        + "px";


    zoomImage.style.left =
        (
            zoomFrame.clientWidth / 2
            -
            centerX * scale
        )
        + "px";


    zoomImage.style.top =
        (
            zoomFrame.clientHeight / 2
            -
            centerY * scale
        )
        + "px";

}


function selectTooth(fdi) {

    const tooth =
        registry[fdi];


    if (
        !tooth
        ||
        !tooth.selectable
    ) {
        return;
    }


    document
        .querySelectorAll(
            ".tooth-cell"
        )
        .forEach(
            item =>
                item.classList.remove(
                    "selected"
                )
        );


    const selected =
        document.querySelector(
            `.tooth-cell[data-fdi="${fdi}"]`
        );


    if (selected) {
        selected.classList.add(
            "selected"
        );
    }


    setDetails(
        tooth
    );


    zoomToTooth(
        tooth
    );


    const row =
        document.querySelector(
            `[data-fdi-row="${fdi}"]`
        );


    if (row) {

        row.scrollIntoView({
            block: "nearest"
        });

    }

}


document
    .querySelectorAll(
        ".tooth-cell:not(.not-detected)"
    )
    .forEach(
        button => {

            button.addEventListener(
                "click",
                () => {
                    selectTooth(
                        button.dataset.fdi
                    );
                }
            );

        }
    );


function initialSelection() {

    if (
        mainOpg.complete
        &&
        mainOpg.naturalWidth
    ) {

        selectTooth(
            "{{ first_present }}"
        );

    }
    else {

        mainOpg.addEventListener(
            "load",
            () => {

                selectTooth(
                    "{{ first_present }}"
                );

            },
            {
                once: true
            }
        );

    }

}


initialSelection();

</script>


{% endif %}


</div>

</body>

</html>
"""


if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )
