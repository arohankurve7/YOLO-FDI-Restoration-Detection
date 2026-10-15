from pathlib import Path
import json


FDI_UPPER = ["18","17","16","15","14","13","12","11","21","22","23","24","25","26","27","28"]
FDI_LOWER = ["48","47","46","45","44","43","42","41","31","32","33","34","35","36","37","38"]


def tooth_kind(fdi: str):
    n = int(fdi[1])

    if n in (1, 2):
        return "incisor"
    if n == 3:
        return "canine"
    if n in (4, 5):
        return "premolar"
    return "molar"


def tooth_svg(kind: str, upper=True):
    """
    Returns a simplified anatomical-looking tooth silhouette.
    Coordinates are intentionally SVG-friendly so the chart
    can later become interactive in the dashboard.
    """

    shapes = {
        "incisor": """
            <path d="M31 8
                     C22 8 17 15 18 27
                     L21 51
                     C22 59 25 66 30 73
                     L34 73
                     C39 66 42 59 43 51
                     L46 27
                     C47 15 40 8 31 8 Z"/>
        """,

        "canine": """
            <path d="M31 5
                     C22 6 17 14 18 27
                     L21 47
                     C22 55 26 61 30 67
                     L32 77
                     L34 67
                     C39 61 42 55 43 47
                     L46 27
                     C47 14 40 6 31 5 Z"/>
        """,

        "premolar": """
            <path d="M31 7
                     C21 7 15 15 17 30
                     L20 49
                     C22 60 25 66 29 70
                     L31 78
                     L34 70
                     C39 65 42 58 44 49
                     L47 30
                     C49 15 41 7 31 7 Z"/>
        """,

        "molar": """
            <path d="M31 7
                     C18 7 12 16 14 31
                     L17 48
                     C18 55 21 61 25 65
                     L26 77
                     L31 67
                     L36 77
                     L37 65
                     C41 61 44 55 45 48
                     L48 31
                     C50 16 43 7 31 7 Z"/>
        """
    }

    body = shapes[kind]

    if upper:
        return body

    return f'<g transform="translate(0,85) scale(1,-1)">{body}</g>'


def make_tooth_chart(prediction_json, output_path):
    prediction_json = Path(prediction_json)
    output_path = Path(output_path)

    data = json.loads(prediction_json.read_text())
    present = set(data.get("present_teeth", []))

    cell_w = 66
    chart_x = 35
    upper_y = 65
    lower_y = 195

    svg = []

    svg.append("""
    <svg xmlns="http://www.w3.org/2000/svg"
         width="1150"
         height="335"
         viewBox="0 0 1150 335">

      <style>
        text {
            font-family: Arial, Helvetica, sans-serif;
        }

        .tooth {
            fill: #ffffff;
            stroke: #707070;
            stroke-width: 1.7;
        }

        .missing {
            fill: #d7d7d7;
            stroke: #999999;
            stroke-width: 1.5;
            stroke-dasharray: 4 3;
        }

        .fdi {
            font-size: 13px;
            font-weight: 600;
            fill: #333333;
            text-anchor: middle;
        }

        .title {
            font-size: 17px;
            font-weight: 700;
            fill: #202020;
        }

        .arch {
            font-size: 13px;
            font-weight: 600;
            fill: #555555;
        }

        .separator {
            stroke: #bbbbbb;
            stroke-width: 1;
            stroke-dasharray: 3 3;
        }
      </style>

      <rect width="100%" height="100%" fill="#ffffff"/>

      <text x="35" y="27" class="title">TOOTH CHART (FDI)</text>
      <text x="535" y="47" class="arch">Upper (Maxillary)</text>
    """)

    mid_x = chart_x + 8 * cell_w
    svg.append(
        f'<line x1="{mid_x}" y1="48" x2="{mid_x}" y2="162" class="separator"/>'
    )

    for i, fdi in enumerate(FDI_UPPER):
        x = chart_x + i * cell_w

        svg.append(
            f'<text x="{x + 31}" y="{upper_y - 12}" class="fdi">{fdi}</text>'
        )

        state_class = "tooth" if fdi in present else "missing"

        svg.append(
            f'<g id="tooth-{fdi}" transform="translate({x},{upper_y})" class="{state_class}">'
        )

        svg.append(
            tooth_svg(
                tooth_kind(fdi),
                upper=True
            )
        )

        svg.append("</g>")

    svg.append(
        '<text x="525" y="186" class="arch">Lower (Mandibular)</text>'
    )

    svg.append(
        f'<line x1="{mid_x}" y1="188" x2="{mid_x}" y2="302" class="separator"/>'
    )

    for i, fdi in enumerate(FDI_LOWER):
        x = chart_x + i * cell_w

        state_class = "tooth" if fdi in present else "missing"

        svg.append(
            f'<g id="tooth-{fdi}" transform="translate({x},{lower_y})" class="{state_class}">'
        )

        svg.append(
            tooth_svg(
                tooth_kind(fdi),
                upper=False
            )
        )

        svg.append("</g>")

        svg.append(
            f'<text x="{x + 31}" y="{lower_y + 103}" class="fdi">{fdi}</text>'
        )

    svg.append("""
      <g transform="translate(35,315)">
        <rect x="0" y="-12" width="17" height="17"
              rx="3" fill="#ffffff"
              stroke="#707070" stroke-width="1.5"/>
        <text x="25" y="2" font-size="12" fill="#444444">
          Present
        </text>

        <rect x="100" y="-12" width="17" height="17"
              rx="3" fill="#d7d7d7"
              stroke="#999999"
              stroke-width="1.5"
              stroke-dasharray="4 3"/>
        <text x="125" y="2" font-size="12" fill="#444444">
          Missing candidate
        </text>
      </g>
    </svg>
    """)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg))

    print("Saved anatomical FDI tooth chart:")
    print(output_path)


if __name__ == "__main__":
    import argparse

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

    make_tooth_chart(
        args.prediction,
        args.output
    )
