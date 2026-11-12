"""The federated generation loop: broadcast, local best-of-K under IRM, return.

One round of training, drawn from the server's point of view and then from one
hospital's. The server holds the generator ``h_g`` and broadcasts it; each
hospital runs it on its own records ``D_i``; the update ``Delta_i`` comes back
and is averaged. Inside a hospital, the local model draws K stochastic
candidates for a batch, the objective keeps the lowest-loss one, and the IRM
penalty rides on it.

WHAT THE CANDIDATE GLYPHS ARE, because this is the easy thing to misread. They
are K candidate reconstructions of ONE TRAINING BATCH, not K finished synthetic
datasets to pick between after the fact. Best-of-K is a training objective --
``cands.min(dim=0)`` in ``HALO._irm_terms`` -- so the selection happens per
batch, inside the backward pass, and the losing candidates are not kept. A
reader who takes the circled glyph as "the synthetic data we shipped" has the
method wrong, which is why the callout says "lowest loss" and not "best".

The three hospitals circle different candidates on purpose: the selection is
per site and per batch, not a global choice.

Static SVG, no JavaScript, so it renders without a browser -- same constraint
paper/figs/src/README.md records for the winner-take-all figure. The palette is
inlined as literal hex and there is no dark-mode block: a paper wants neither.

Example:
    .venv/bin/python examples/fedpyhealth/scripts/fig_federated_generation.py \\
        --out _outputs/results/federated_generation
"""
from __future__ import annotations

import argparse
import os

# ---------------------------------------------------------------------------
# Palette. Literal hex, light only. The accent is the same red the RLSP bars
# and the frontier figures use, so "this is the selected thing" reads the same
# way across the paper's figures.
INK = "#16181B"        # text
STRUCT = "#9AA2AA"     # box outlines, dividers
PANEL = "#BEC5CC"      # a hospital's own border -- softer than the text ink
HAIR = "#E3E7EA"       # the lightest rules
FILL_SOFT = "#F6F7F5"  # inside the model glyph and the drum
PAPER = "#FFFFFF"
ARROW = "#68707A"      # transport between server and sites
ACCENT = "#C44E52"     # the lowest-loss candidate
#: Two tints per state. A tetromino is drawn as one outlined piece, and
#: alternating tints between pieces in the same candidate is what keeps two
#: adjacent pieces from reading as a single blob.
GREY_FILL = ("#E4E8EB", "#D6DBDF")
GREY_EDGE = "#A9B1B8"
RED_FILL = ("#F3D3D3", "#EBBFBF")
RED_EDGE = "#C4696C"

SERIF = "Nimbus Roman, Times New Roman, Times, serif"

# ---------------------------------------------------------------------------
# Geometry. One place, so nudging the layout does not mean hunting constants
# through the drawing code.
W, H = 980, 660

SRV = dict(x=46, y=206, w=150, h=252)          # server box
HOSP = dict(x=340, w=490, h=165)               # hospital box
ROW_TOP = (50, 245, 440)                       # one per hospital

CYL_CX, CYL_DY = 403, 83                       # data drum, relative to box top
DIV_X = 452                                    # the rule splitting data | model
GEN_X = 476                                    # local generator glyph
ARROW_X = 536                                  # the => between model and draws
CAND_X = 610                                   # candidate glyphs start here
IRM_X = 730                                    # "+ IRM" labels, left aligned
CAND_DY = (32, 83, 134)                        # candidate rows, rel. to box top

#: The seven tetrominoes, as (col, row) cells. Using real tetromino shapes
#: rather than plain rectangles gives each candidate a silhouette, which is the
#: whole job of the glyph: K draws that are visibly DIFFERENT from one another.
TETRO = {
    "I": [(0, 0), (1, 0), (2, 0), (3, 0)],
    "O": [(0, 0), (1, 0), (0, 1), (1, 1)],
    "T": [(0, 0), (1, 0), (2, 0), (1, 1)],
    "S": [(1, 0), (2, 0), (0, 1), (1, 1)],
    "Z": [(0, 0), (1, 0), (1, 1), (2, 1)],
    "L": [(0, 0), (0, 1), (1, 1), (2, 1)],
    "J": [(2, 0), (0, 1), (1, 1), (2, 1)],
}

#: Per hospital: the K candidate shapes, and which one the objective kept.
#: Each candidate is a list of tetromino names. They differ in shape because
#: they are different stochastic draws.
HOSPITALS = [
    dict(name="H1", cands=[["S", "O"], ["I", "T"], ["L"]], keep=0),
    dict(name="H2", cands=[["O"], ["T", "I"], ["Z"]], keep=1),
    dict(name="H3", cands=[["J", "O"], ["S"], ["I"]], keep=0),
]

CELL, CELL_GAP, PIECE_GAP = 10, 1.5, 9


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=16, fill=INK, anchor="start", style="normal",
         weight="normal", opacity=1.0) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{SERIF}" '
        f'font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
        f'font-style="{style}" font-weight="{weight}" '
        f'opacity="{opacity:g}">{esc(s)}</text>'
    )


def sym(x, y, base, sub, size=17, fill=INK, style="italic"):
    """A math-style label with a subscript.

    Two positioned <text> elements rather than one with a shifted tspan:
    baseline-shift is the part of the SVG text model renderers disagree about
    most, and this figure has to survive cairosvg with no browser anywhere in
    the pipeline.
    """
    out = [text(x, y, base, size=size, fill=fill, style=style)]
    if sub:
        sub_style = "normal" if sub.isdigit() else style
        out.append(text(x + size * 0.46, y + size * 0.26, sub,
                        size=size * 0.68, fill=fill, style=sub_style))
    return "".join(out)


def generator_glyph(x, y_mid, w=46, h_l=34, h_r=66, label=None, sub=None):
    """The trapezoid that means "generator": narrow in, wide out.

    Drawn expanding left-to-right so the shape itself says which direction the
    model runs, which is the one thing a reader should get from it without
    reading the label.
    """
    pts = (f"{x},{y_mid - h_l / 2} {x + w},{y_mid - h_r / 2} "
           f"{x + w},{y_mid + h_r / 2} {x},{y_mid + h_l / 2}")
    out = [f'<polygon points="{pts}" fill="{FILL_SOFT}" stroke="{STRUCT}" '
           f'stroke-width="1.5" stroke-linejoin="round"/>']
    if label:
        out.append(sym(x + w * 0.30, y_mid + 6, label, sub, size=18))
    return "".join(out)


def drum(cx, cy, label, sub, rx=29, body=46):
    """A database cylinder. The hospital's own records, which never move."""
    top, bot = cy - body / 2, cy + body / 2
    ry = 9.5
    d = (f"M{cx - rx},{top} L{cx - rx},{bot} "
         f"A{rx},{ry} 0 0 0 {cx + rx},{bot} L{cx + rx},{top}")
    return "".join([
        f'<path d="{d}" fill="{FILL_SOFT}" stroke="{STRUCT}" stroke-width="1.5"/>',
        f'<ellipse cx="{cx}" cy="{top}" rx="{rx}" ry="{ry}" '
        f'fill="{PAPER}" stroke="{STRUCT}" stroke-width="1.5"/>',
        sym(cx - 9, cy + 9, label, sub, size=17),
    ])


def candidate(x, y_mid, pieces, selected: bool):
    """One stochastic draw, drawn as tetrominoes.

    Every piece is vertically centred on the same midline rather than sharing a
    top edge, so a one-row piece and a two-row piece sit on one visual baseline
    and the row still scans as a single candidate.
    """
    fills = RED_FILL if selected else GREY_FILL
    edge = RED_EDGE if selected else GREY_EDGE
    out, cx = [], x
    for i, name in enumerate(pieces):
        cells = TETRO[name]
        ncol = max(c for c, _ in cells) + 1
        nrow = max(r for _, r in cells) + 1
        w = ncol * CELL + (ncol - 1) * CELL_GAP
        h = nrow * CELL + (nrow - 1) * CELL_GAP
        top = y_mid - h / 2
        for c, r in cells:
            out.append(
                f'<rect x="{cx + c * (CELL + CELL_GAP):.1f}" '
                f'y="{top + r * (CELL + CELL_GAP):.1f}" '
                f'width="{CELL}" height="{CELL}" rx="2" '
                f'fill="{fills[i % 2]}" stroke="{edge}" stroke-width="1.15" '
                f'stroke-linejoin="round"/>'
            )
        cx += w + PIECE_GAP
    return "".join(out), cx - PIECE_GAP


def curve(p0, p1, bow, colour=ARROW, width=1.5, dash=None):
    """Quadratic arc from p0 to p1, bowed perpendicular by `bow` px.

    Returns (path, midpoint) -- the midpoint is where the label goes, and it is
    the curve's actual t=0.5 point, not the chord's, so a label never drifts
    off a strongly bowed arc.
    """
    (x0, y0), (x1, y1) = p0, p1
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = x1 - x0, y1 - y0
    n = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    cx, cy = mx - dy / n * bow, my + dx / n * bow
    d = f"M{x0:.1f},{y0:.1f} Q{cx:.1f},{cy:.1f} {x1:.1f},{y1:.1f}"
    stroke = f' stroke-dasharray="{dash}"' if dash else ""
    path = (f'<path d="{d}" fill="none" stroke="{colour}" '
            f'stroke-width="{width}" marker-end="url(#arrow)"{stroke}/>')
    return path, (0.25 * x0 + 0.5 * cx + 0.25 * x1,
                  0.25 * y0 + 0.5 * cy + 0.25 * y1)


def build() -> str:
    p = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="{PAPER}"/>',
        '<defs>',
        f'<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        f'<path d="M0,1 L9,5 L0,9 z" fill="{ARROW}"/></marker>',
        '</defs>',
    ]

    # ---- server -----------------------------------------------------------
    s = SRV
    p.append(f'<rect x="{s["x"]}" y="{s["y"]}" width="{s["w"]}" '
             f'height="{s["h"]}" rx="14" fill="{PAPER}" stroke="{PANEL}" '
             f'stroke-width="1.4"/>')
    p.append(text(s["x"] + 4, s["y"] - 11, "server", size=16, fill=INK))
    p.append(generator_glyph(s["x"] + 52, s["y"] + s["h"] / 2,
                             label="h", sub="g"))

    # ---- hospitals --------------------------------------------------------
    for i, (top, hosp) in enumerate(zip(ROW_TOP, HOSPITALS)):
        b = HOSP
        p.append(f'<rect x="{b["x"]}" y="{top}" width="{b["w"]}" '
                 f'height="{b["h"]}" rx="14" fill="{PAPER}" stroke="{PANEL}" '
                 f'stroke-width="1.4"/>')
        p.append(text(b["x"] + 6, top - 10, hosp["name"], size=16, fill=INK))

        p.append(drum(CYL_CX, top + CYL_DY, "D", str(i + 1)))
        p.append(f'<line x1="{DIV_X}" y1="{top + 16}" x2="{DIV_X}" '
                 f'y2="{top + b["h"] - 16}" stroke="{HAIR}" stroke-width="1.4" '
                 f'stroke-dasharray="3 4"/>')
        p.append(generator_glyph(GEN_X, top + CYL_DY, label="h", sub=str(i + 1)))

        # the => from the local model to its K draws
        for dy in (-5, 5):
            y = top + CYL_DY + dy
            p.append(f'<path d="M{ARROW_X},{y} L{ARROW_X + 22},{y}" '
                     f'stroke="{ARROW}" stroke-width="1.4" fill="none"/>')
        p.append(f'<path d="M{ARROW_X + 20},{top + CYL_DY - 8} '
                 f'L{ARROW_X + 31},{top + CYL_DY} '
                 f'L{ARROW_X + 20},{top + CYL_DY + 8} z" fill="{ARROW}"/>')

        for k, (dy, visits) in enumerate(zip(CAND_DY, hosp["cands"])):
            y = top + dy
            sel = k == hosp["keep"]
            if sel:
                p.append(f'<ellipse cx="{(CAND_X + IRM_X + 52) / 2:.1f}" '
                         f'cy="{y}" rx="{(IRM_X + 52 - CAND_X) / 2 + 14:.1f}" '
                         f'ry="21" fill="none" stroke="{ACCENT}" '
                         f'stroke-width="1.7"/>')
            glyph, _ = candidate(CAND_X, y, visits, sel)
            p.append(glyph)
            p.append(text(IRM_X, y + 6, "+ IRM", size=15,
                          fill=ACCENT if sel else "#666E76"))

        # broadcast out, update back -- one lens-shaped pair per hospital
        cy = top + b["h"] / 2
        out_p, out_m = curve((s["x"] + s["w"] + 4, s["y"] + 58 + i * 40),
                             (b["x"] - 6, cy - 13), bow=-16 + i * 9)
        back_p, back_m = curve((b["x"] - 6, cy + 11),
                               (s["x"] + s["w"] + 4, s["y"] + 92 + i * 40),
                               bow=-16 + i * 9)
        p.append(out_p)
        p.append(back_p)
        p.append(sym(out_m[0] - 8, out_m[1] - 13, "h", "g", size=16))
        p.append(sym(back_m[0] - 8, back_m[1] + 20, "Δ", str(i + 1),
                     size=16))

    # ---- the one callout --------------------------------------------------
    # "lowest loss", not "best": the circled glyph is the candidate the
    # objective backpropagates through, not a shipped synthetic dataset.
    lead, _ = curve((846, 86), (802, ROW_TOP[0] + CAND_DY[0] - 6),
                    bow=12, colour=STRUCT, width=1.3)
    p.append(lead)
    p.append(text(848, 54, "best of K", size=16, fill=STRUCT))
    p.append(text(848, 72, "(lowest loss)", size=13, fill=STRUCT))

    p.append('</svg>')
    return "".join(p)


HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Federated Generation Loop</title>
<style>
  /* Light only, on purpose. This page exists to be exported into a paper,
     and a figure that changes with the reader's OS theme is not a figure. */
  :root {{ color-scheme: light; }}
  body {{
    margin: 0; background: #F6F6F4; color: #16181B;
    font-family: "Nimbus Roman", "Times New Roman", Times, serif;
    display: flex; flex-direction: column; align-items: center;
    gap: 18px; padding: 28px 16px 44px;
  }}
  figure {{ margin: 0; max-width: 1060px; width: 100%; }}
  .plate {{
    background: #FFFFFF; border: 1px solid #E2E5E8; border-radius: 6px;
    padding: 10px; overflow-x: auto;
  }}
  svg {{ display: block; width: 100%; height: auto; min-width: 760px; }}
  figcaption {{
    margin-top: 14px; font-size: 15px; line-height: 1.5; color: #3A4046;
    max-width: 62ch;
  }}
  figcaption b {{ color: #16181B; font-weight: 600; }}
</style>
</head>
<body>
<figure>
  <div class="plate">
{svg}
  </div>
  <figcaption>
    <b>One round.</b> The server broadcasts the generator
    <i>h<sub>g</sub></i>; hospital <i>i</i> runs it over its own records
    <i>D<sub>i</sub></i>, which never leave the site, and returns the update
    <i>&Delta;<sub>i</sub></i>. Within a hospital, each training batch is
    reconstructed <i>K</i> times under independent stochasticity; the objective
    keeps the lowest-loss candidate and applies the IRM penalty to it. The
    losing candidates are discarded, and which one wins varies by site and by
    batch.
  </figcaption>
</figure>
</body>
</html>
"""


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="_outputs/results/federated_generation",
                    help="Path stem; .svg, .html and .png are written beside "
                         "it, plus .pdf with --pdf.")
    ap.add_argument("--scale", type=float, default=2.5,
                    help="PNG pixels per SVG unit. 2.5 gives ~2650px wide, "
                         "enough for a full-width figure at 300dpi.")
    ap.add_argument("--pdf", action="store_true",
                    help="Also write a vector PDF, which is what pdflatex "
                         "should include.")
    a = ap.parse_args(argv)

    svg = build()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    with open(f"{a.out}.svg", "w") as fh:
        fh.write(svg)
    with open(f"{a.out}.html", "w") as fh:
        fh.write(HTML.format(svg=svg))

    import cairosvg
    cairosvg.svg2png(bytestring=svg.encode(), write_to=f"{a.out}.png",
                     scale=a.scale, background_color=PAPER)
    if a.pdf:
        cairosvg.svg2pdf(bytestring=svg.encode(), write_to=f"{a.out}.pdf")

    for ext in ("svg", "html", "png") + (("pdf",) if a.pdf else ()):
        path = f"{a.out}.{ext}"
        print(f"wrote {path}  ({os.path.getsize(path) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
