from pathlib import Path
import argparse
import json


ROOT = Path(__file__).resolve().parents[2]

UPPER = [
    "18","17","16","15","14","13","12","11",
    "21","22","23","24","25","26","27","28",
]

LOWER = [
    "48","47","46","45","44","43","42","41",
    "31","32","33","34","35","36","37","38",
]


TOOTH_COLORS = {
    1: "#DC4652",  # Central Incisor - red
    2: "#9B51B6",  # Lateral Incisor - purple
    3: "#3977C3",  # Canine - blue
    4: "#58BFD2",  # First Premolar - cyan
    5: "#55A95B",  # Second Premolar - green
    6: "#E4BE32",  # First Molar - yellow
    7: "#E58A2B",  # Second Molar - orange
    8: "#D65A2D",  # Third Molar - dark orange
}


def tooth_type(fdi):
    names = {
        1: "Central Incisor",
        2: "Lateral Incisor",
        3: "Canine",
        4: "First Premolar",
        5: "Second Premolar",
        6: "First Molar",
        7: "Second Molar",
        8: "Third Molar",
    }

    return names[int(fdi[1])]


def template_for(fdi):
    pos = int(fdi[1])

    if pos in (1, 2):
        return "11.svg"
    if pos == 3:
        return "13.svg"
    if pos in (4, 5):
        return "14.svg"

    return "16.svg"


def transform_for(fdi):
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



def inline_tooth_svg(fdi, color, detected=True):
    template = template_for(fdi)

    svg_path = (
        ROOT
        / "src"
        / "assets"
        / "teeth_clean"
        / template
    )

    svg = svg_path.read_text(
        encoding="utf-8"
    )

    # Remove XML declaration if present.
    if svg.startswith("<?xml"):
        svg = svg.split("?>", 1)[1]

    if detected:
        fill = color
        stroke = color
        beauty_fill = color
    else:
        fill = "#D9DDE2"
        stroke = "#AEB5BD"
        beauty_fill = "#D9DDE2"

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
        f"fill: {beauty_fill};"
    )

    # Make embedded SVG size controlled by CSS.
    svg = svg.replace(
        "<svg ",
        '<svg class="inline-tooth-svg" ',
        1,
    )

    return svg

def cell_html(fdi, registry):
    tooth = registry[fdi]

    detected = (
        tooth.get("status")
        == "present"
    )

    template = template_for(fdi)
    transform = transform_for(fdi)

    color = TOOTH_COLORS[
        int(fdi[1])
    ]

    anatomical_name = tooth.get(
        "anatomical_name",
        ""
    )

    svg_markup = inline_tooth_svg(
        fdi,
        color,
        detected=detected,
    )

    classes = "tooth-cell"

    if not detected:
        classes += " not-detected"

    disabled = (
        ""
        if detected
        else " disabled"
    )

    tooth_style = (
        f"--tooth-color:{color};"
        f"transform:{transform};"
    )

    return f"""
    <button
        class="{classes}"
        data-fdi="{fdi}"
        data-name="{anatomical_name}"
        data-type="{tooth_type(fdi)}"
        title="{fdi} — {anatomical_name}"
        {disabled}
    >
        <div class="tooth-number">{fdi}</div>

        <div
            class="tooth-image-wrap"
            style="transform:{transform};"
            aria-label="{anatomical_name}"
        >
            {svg_markup}
        </div>
    </button>
    """


def build_chart(registry_json, output_html):
    registry_json = Path(
        registry_json
    )

    output_html = Path(
        output_html
    )

    data = json.loads(
        registry_json.read_text()
    )

    teeth = data.get(
        "tooth_registry",
        []
    )

    registry = {
        str(item["fdi"]): item
        for item in teeth
    }

    upper_html = "".join(
        cell_html(fdi, registry)
        for fdi in UPPER
    )

    lower_html = "".join(
        cell_html(fdi, registry)
        for fdi in LOWER
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">

<title>FDI Tooth Chart</title>

<style>
* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    background: #f5f7fa;
    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        Arial,
        sans-serif;
    color: #243142;
}}

.panel {{
    width: 1120px;
    margin: 30px auto;
    background: #ffffff;
    border: 1px solid #dce2e8;
    border-radius: 12px;
    padding: 22px 26px 20px;
}}

.title {{
    font-size: 17px;
    font-weight: 700;
}}

.instruction {{
    margin-top: 5px;
    margin-bottom: 18px;
    color: #738092;
    font-size: 12px;
}}

.arch-label {{
    text-align: center;
    font-size: 12px;
    font-weight: 650;
    color: #617083;
    margin: 5px 0;
}}

.chart {{
    position: relative;
}}

.arch {{
    display: grid;
    grid-template-columns:
        repeat(16, 1fr);
    align-items: end;
}}

.arch.lower {{
    margin-top: 0;
    align-items: start;
}}

.midline {{
    position: absolute;
    left: 50%;
    top: 24px;
    bottom: 24px;
    width: 1px;
    background: #dce2e8;
    pointer-events: none;
}}

.tooth-cell {{
    height: 112px;
    border: 0;
    background: transparent;
    padding: 2px;
    cursor: pointer;
    border-radius: 7px;
    position: relative;
}}

.tooth-cell:hover {{
    background: #f2f6fa;
}}

.tooth-cell.selected {{
    background: #e8f1fa;
    outline: 2px solid #4b7ca8;
}}

.tooth-number {{
    font-size: 12px;
    font-weight: 650;
    color: #364354;
    height: 19px;
    display: flex;
    align-items: center;
    justify-content: center;
}}

.tooth-image-wrap {{
    height: 89px;
    display: flex;
    align-items: center;
    justify-content: center;
}}

.inline-tooth-svg {{
    display: block;
    width: 48px;
    height: 86px;
}}

.tooth-image-wrap {{
    transform-origin: center;
}}

.tooth-cell::after {{
    content: "";
    position: absolute;
    left: 17px;
    right: 17px;
    bottom: 2px;
    height: 4px;
    border-radius: 3px;
    background: var(--tooth-color);
}}

.not-detected {{
    cursor: not-allowed;
}}

.not-detected .inline-tooth-svg {{
    opacity: 0.45;
}}

.not-detected .tooth-number {{
    color: #a5adb7;
}}

.not-detected::after {{
    background: #b9bec5;
}}

.not-detected:hover {{
    background: transparent;
}}

.legend {{
    margin-top: 18px;
    padding-top: 13px;
    border-top: 1px solid #edf0f3;
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    font-size: 11px;
    color: #667383;
}}

.legend-item {{
    display: flex;
    align-items: center;
    gap: 5px;
}}

.swatch {{
    width: 10px;
    height: 10px;
    border-radius: 3px;
}}

.disabled-note {{
    margin-left: auto;
    color: #7b8795;
}}

</style>
</head>

<body>

<div class="panel">

    <div class="title">
        FDI Two-Digit Tooth Numbering System (ISO 3950)
    </div>

    <div class="instruction">
        Select a detected tooth to view its details and zoomed view.
    </div>

    <div class="arch-label">
        Maxillary
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

    <div class="arch-label">
        Mandibular
    </div>

    <div class="legend">

        <div class="legend-item">
            <span class="swatch"
                  style="background:#D65A2D"></span>
            Third Molar
        </div>

        <div class="legend-item">
            <span class="swatch"
                  style="background:#E58A2B"></span>
            Second Molar
        </div>

        <div class="legend-item">
            <span class="swatch"
                  style="background:#E4BE32"></span>
            First Molar
        </div>

        <div class="legend-item">
            <span class="swatch"
                  style="background:#55A95B"></span>
            Second Premolar
        </div>

        <div class="legend-item">
            <span class="swatch"
                  style="background:#58BFD2"></span>
            First Premolar
        </div>

        <div class="legend-item">
            <span class="swatch"
                  style="background:#3977C3"></span>
            Canine
        </div>

        <div class="legend-item">
            <span class="swatch"
                  style="background:#9B51B6"></span>
            Lateral Incisor
        </div>

        <div class="legend-item">
            <span class="swatch"
                  style="background:#DC4652"></span>
            Central Incisor
        </div>

        <div class="disabled-note">
            Grey = Missing Teeth · Disabled
        </div>

    </div>

</div>


<script>
const root = document.querySelector(".panel");

root
    .querySelectorAll(
        ".tooth-cell:not(.not-detected)"
    )
    .forEach(button => {{

        button.addEventListener(
            "click",
            () => {{

                root
                    .querySelectorAll(
                        ".tooth-cell"
                    )
                    .forEach(el =>
                        el.classList.remove(
                            "selected"
                        )
                    );

                button.classList.add(
                    "selected"
                );

                const fdi =
                    button.dataset.fdi;

                const anatomicalName =
                    button.dataset.name;

                const toothType =
                    button.dataset.type;

                window.dispatchEvent(
                    new CustomEvent(
                        "panoradent-tooth-selected",
                        {{
                            detail: {{
                                fdi,
                                anatomicalName,
                                toothType
                            }}
                        }}
                    )
                );

            }}
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
        html,
        encoding="utf-8",
    )

    print(
        "Saved dashboard FDI chart:"
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
        "--output",
        required=True,
    )

    args = parser.parse_args()

    build_chart(
        args.registry,
        args.output,
    )
