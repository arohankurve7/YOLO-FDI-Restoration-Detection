from pathlib import Path
import argparse
import json


ROOT = Path(__file__).resolve().parents[2]


UPPER = [
    "18","17","16","15","14","13","12","11",
    "21","22","23","24","25","26","27","28"
]

LOWER = [
    "48","47","46","45","44","43","42","41",
    "31","32","33","34","35","36","37","38"
]


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
    q = int(fdi[0])

    # Mapping used by the odontogram source:
    #
    # Q1: normal
    # Q2: horizontal mirror
    # Q3: rotate 180
    # Q4: rotate 180 + horizontal mirror

    if q == 1:
        return ""

    if q == 2:
        return "scaleX(-1)"

    if q == 3:
        return "rotate(180deg)"

    if q == 4:
        return "rotate(180deg) scaleX(-1)"

    return ""


def tooth_cell(fdi, present):
    tpl = template_for(fdi)
    transform = transform_for(fdi)

    missing_class = "" if fdi in present else " missing"

    src = (
        "../../../src/assets/teeth_clean/"
        + tpl
    )

    is_upper = fdi[0] in ("1", "2")

    number_html = f'<div class="tooth-number">{fdi}</div>'

    image_html = f"""
        <div class="tooth-image-wrap">
            <img
                src="{src}"
                class="tooth-image"
                style="transform:{transform};"
                alt="FDI {fdi}"
            />
        </div>
    """

    content = (
        number_html + image_html
        if is_upper
        else image_html + number_html
    )

    return f"""
    <button
        class="tooth-cell{missing_class}"
        data-fdi="{fdi}"
        title="FDI {fdi}"
        onclick="selectTooth('{fdi}')"
    >
        {content}
    </button>
    """


def build_chart(prediction_json, output_html):
    prediction_json = Path(prediction_json)
    output_html = Path(output_html)

    data = json.loads(
        prediction_json.read_text()
    )

    present = set(
        data.get("present_teeth", [])
    )

    upper_html = "".join(
        tooth_cell(fdi, present)
        for fdi in UPPER
    )

    lower_html = "".join(
        tooth_cell(fdi, present)
        for fdi in LOWER
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">

<title>PanoraDent FDI Tooth Chart</title>

<style>

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    background: #f4f6f8;
    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        Arial,
        sans-serif;
    color: #23272f;
}}

.panel {{
    width: 1040px;
    margin: 40px auto;
    background: white;
    border: 1px solid #d9dee5;
    border-radius: 10px;
    padding: 22px 26px 18px 26px;
    box-shadow:
        0 2px 10px rgba(0,0,0,0.05);
}}

.header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 14px;
}}

.title {{
    font-size: 16px;
    font-weight: 650;
}}

.legend {{
    display: flex;
    gap: 18px;
    font-size: 12px;
    color: #657080;
}}

.legend-item {{
    display: flex;
    align-items: center;
    gap: 7px;
}}

.dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
    border: 1px solid #8e98a5;
    background: white;
}}

.dot.missing-dot {{
    background: #c8cdd3;
}}

.chart {{
    position: relative;
    width: 100%;
}}

.arch {{
    display: grid;
    grid-template-columns: repeat(16, 1fr);
    align-items: end;
    gap: -2px;
}}

.arch.lower {{
    margin-top: -4px;
    align-items: start;
}}

.midline {{
    position: absolute;
    left: 50%;
    top: 18px;
    bottom: 18px;
    width: 1px;
    background: #e4e7eb;
    pointer-events: none;
}}

.tooth-cell {{
    position: relative;

    height: 105px;

    border: 0;
    background: transparent;

    padding: 0;

    cursor: pointer;

    transition:
        background 120ms ease,
        transform 120ms ease;
}}

.tooth-cell:hover {{
    background: #f3f7fb;
}}

.tooth-cell.selected {{
    background: #e8f2fb;
    border-radius: 6px;
}}

.tooth-image-wrap {{
    height: 88px;

    display: flex;
    align-items: center;
    justify-content: center;

    overflow: visible;
}}

.tooth-image {{
    display: block;

    max-height: 88px;
    max-width: 58px;

    transform-origin: center center;

    transition:
        opacity 120ms ease,
        filter 120ms ease,
        transform 120ms ease;
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

.arch.upper .tooth-number {{
    order: 0;
}}

.missing .tooth-image {{
    opacity: 0.28;
    filter:
        grayscale(1)
        contrast(0.65);
}}

.missing .tooth-number {{
    color: #a0a6ad;
}}


.status {{
    height: 34px;

    margin-top: 8px;

    border-top: 1px solid #edf0f3;

    display: flex;
    align-items: end;

    font-size: 12px;
    color: #657080;
}}

</style>
</head>

<body>

<div class="panel">

    <div class="header">

        <div class="title">
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

    <div id="status" style="display:none"></div>

</div>


<script>

function selectTooth(fdi) {{

    document
        .querySelectorAll(".tooth-cell")
        .forEach(el =>
            el.classList.remove("selected")
        );

    const selected =
        document.querySelector(
            `[data-fdi="${{fdi}}"]`
        );

    if (selected) {{
        selected.classList.add(
            "selected"
        );
    }}

    document.getElementById(
        "status"
    ).textContent =
        "Selected tooth: FDI " + fdi;


    /*
      NEXT STEP:

      This event will be caught by the
      PanoraDent dashboard.

      FDI 36 click
          ↓
      prediction.json
          ↓
      corresponding segmentation mask
          ↓
      OPG highlight + zoom
    */

    window.dispatchEvent(
        new CustomEvent(
            "panoradent-tooth-selected",
            {{
                detail: {{
                    fdi: fdi
                }}
            }}
        )
    );
}}

</script>

</body>
</html>
"""

    output_html.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_html.write_text(
        html,
        encoding="utf-8"
    )

    print("Saved:")
    print(output_html.resolve())


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

    args = parser.parse_args()

    build_chart(
        args.prediction,
        args.output
    )
