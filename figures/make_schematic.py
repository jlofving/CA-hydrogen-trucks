"""
Model-schematic figure (manuscript figure 1) as an EDITABLE PowerPoint deck.

    python3 -m pip install python-pptx
    python3 figures/make_schematic.py

Unlike every other figure in this suite, this one is not data-driven, so it is
authored rather than plotted — PPTX keeps it editable by co-authors and by
journal production. The output goes to the same folder as the Julia figures and
picks up its manuscript number from `figures/fig_routing.jl`, which stays the
single source of truth for figure order. A PDF/PNG is rendered alongside it via
LibreOffice when `soffice` is on PATH, so the numbered raster used in the draft
never drifts from the editable original.

Every parameter quoted in a box is taken from the model, not invented — see the
PARAMETER PROVENANCE comments below. If you change a config value that appears
here, change it here too.
"""
import re
import shutil
import subprocess
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn

# ---------------------------------------------------------------- provenance
# PARAMETER PROVENANCE — values quoted in the schematic
#   2026–2045 horizon .......... config/model_defaults.json  simulation.start_year / end_year
#   1000 MC runs ............... config/model_defaults.json  simulation.n_monte_carlo_runs
#   uptime ramp 30/60/100 % .... config/model_defaults.json  truck_defaults.uptime
#   7-year truck life .......... hydrogen truck deployment.jl  truck_lifetime_years
#   max 4-year station delay ... config/stations_config.json  deployment_probability_config
#                                .max_delay_years  (probability_by_status gives the Bernoulli p)
#   trigger θ, p_invest ........ hydrogen truck deployment.jl  check_and_expand_production!
#                                — both drawn once per run; θ is maturity-adjusted to θ_eff by
#                                  maturity_adjusted_trigger
#   foresight 2 yr / 3 yr ...... hydrogen truck deployment.jl  station_foresight_years /
#                                production_foresight_years
#   lead time 2–4 yr ........... hydrogen truck deployment.jl  open_year = current_year + rand(2:4)
#   P25–P75 spread ............. figures/fig_tco_comparison.jl, fig_scenario_matrix_overlay.jl

# ---------------------------------------------------------------- output path
ROOT = Path(__file__).resolve().parent
FIG_NAME = "fig_model_schematic"


def manuscript_basename(name: str) -> str:
    """Mirror of `fig_basename` in fig_routing.jl — number from FIG_MANUSCRIPT order."""
    routing = (ROOT / "fig_routing.jl").read_text()
    block = re.search(r"const FIG_MANUSCRIPT = \[(.*?)\n\]", routing, re.S)
    if block:
        names = re.findall(r'"([^"]+)"', block.group(1))
        if name in names:
            return f"fig_{names.index(name) + 1}_{name[4:]}"
    return name


OUT_DIR = ROOT / "out" / "manuscript"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PPTX = OUT_DIR / (manuscript_basename(FIG_NAME) + ".pptx")

# ---------------------------------------------------------------- palette
NAVY = RGBColor(0x1F, 0x3B, 0x57)      # text / outlines
BLUE = RGBColor(0xD6, 0xE4, 0xF0)      # deployment module
ORANGE = RGBColor(0xFB, 0xE3, 0xCB)    # cost module
GREEN = RGBColor(0xD8, 0xEC, 0xD9)     # impact module
GREY = RGBColor(0xEC, 0xEC, 0xEC)      # inputs
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
STOCH = RGBColor(0xB0, 0x3A, 0x2E)     # stochastic highlight
OUTLINE = RGBColor(0x8A, 0x9A, 0xA8)
MUTED = RGBColor(0x55, 0x5F, 0x6B)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
slide = prs.slides.add_slide(prs.slide_layouts[6])   # blank


# ---------------------------------------------------------------- helpers
def _fill_lines(tf, text, size, bold, color, align, italic=False):
    """Write `text` into text frame `tf`, one paragraph per newline-separated line."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = color
        r.font.name = "Arial"


def box(x, y, w, h, text, fill=WHITE, bold=False, size=10.5,
        outline=OUTLINE, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
        font_color=NAVY, dashed=False):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.color.rgb = outline
    s.line.width = Pt(1.25)
    if dashed:
        s.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    s.shadow.inherit = False
    try:
        s.adjustments[0] = 0.08
    except (IndexError, AttributeError):
        pass
    tf = s.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Emu(45720)
    tf.margin_top = tf.margin_bottom = Emu(27432)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    # One PARAGRAPH per line, not "\n" inside a run: python-pptx turns "\n" into a
    # vertical tab, which PowerPoint breaks the line on but LibreOffice does not
    # centre, so the exported PNG/PDF came out ragged.
    _fill_lines(tf, text, size=size, bold=bold, color=font_color,
                align=PP_ALIGN.CENTER)
    return s


def label(x, y, w, h, text, size=11, bold=True, color=NAVY,
          align=PP_ALIGN.LEFT, italic=False):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    _fill_lines(tf, text, size=size, bold=bold, color=color, align=align,
                italic=italic)
    return tb


def arrow(x1, y1, x2, y2, color=NAVY, width=1.6, dashed=False):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
                                   Inches(x1), Inches(y1),
                                   Inches(x2), Inches(y2))
    c.line.color.rgb = color
    c.line.width = Pt(width)
    if dashed:
        c.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    # Arrowhead. a:tailEnd is last in the CT_LineProperties sequence, so it is
    # appended after the fill and prstDash set above.
    ln = c.line._get_or_add_ln()
    tail = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
    ln.append(tail)
    return c


# ---------------------------------------------------------------- title
label(0.4, 0.18, 12.5, 0.4,
      "Stochastic techno-economic model: annual time step, 2026–2045",
      size=15)
label(0.4, 0.58, 12.5, 0.3,
      "Red dashed outlines = stochastic elements drawn per Monte Carlo run   |   "
      "Grey = exogenous inputs   |   Vintage-aware asset register carries "
      "commissioning-year CAPEX forward",
      size=9, bold=False, italic=True, color=MUTED)

# ---------------------------------------------------------------- bands
band_y = [1.05, 3.05, 5.15]
band_h = [1.85, 1.95, 1.55]
band_specs = [
    ("1  DEPLOYMENT MODULE", BLUE),
    ("2  COST MODULE  (LCOH → TCO)", ORANGE),
    ("3  IMPACT MODULE", GREEN),
]
for (txt, col), y, h in zip(band_specs, band_y, band_h):
    b = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                               Inches(2.35), Inches(y),
                               Inches(10.6), Inches(h))
    b.fill.solid()
    b.fill.fore_color.rgb = col
    b.line.color.rgb = col
    b.shadow.inherit = False
    b.text_frame.text = ""
    lb = slide.shapes.add_textbox(Inches(2.45), Inches(y + 0.04),
                                  Inches(4.0), Inches(0.28))
    p = lb.text_frame.paragraphs[0]
    r = p.add_run()
    r.text = txt
    r.font.size = Pt(9.5)
    r.font.bold = True
    r.font.color.rgb = NAVY
    r.font.name = "Arial"

# ---------------------------------------------------------------- inputs
label(0.32, 1.05, 1.95, 0.3, "EXOGENOUS INPUTS", size=9.5)
inputs = [
    ("Truck deployment\nscenarios\n(3 trajectories)", 1.38),
    ("Station project\npipeline\n(status, year)", 2.32),
    ("Global capacity\ntrajectories\n(PV, ELY, trucks)", 3.26),
    ("Background H₂ demand\n(buses, LDV)", 4.20),
    ("Policy settings\n(LCFS, HRI, 45V, HVIP)", 5.02),
    # ATRI supplies the NON-fuel operating cost lines only; the fuel line is
    # replaced by the two pump-price cases in config/tco_config.json.
    ("Diesel baseline\nATRI non-fuel costs\n2 pump prices $4.80 / $5.80", 5.84),
    ("SCC & COBRA\nvaluation", 6.66),
]
for txt, y in inputs:
    box(0.32, y, 1.95, 0.80, txt, fill=GREY, size=8.5)

# ---------------------------------------------------------------- band 1
b1y = 1.42
box(2.55, b1y, 2.05, 0.72,
    "Truck fleet\nadd / retire (7-yr life),\nuptime ramp 30–60–100 %",
    fill=WHITE, size=9)
box(4.85, b1y, 2.30, 0.72,
    "Station commissioning\nBernoulli trial by project status\n(max 4-yr delay)",
    fill=WHITE, size=9, outline=STOCH, dashed=True)
box(7.40, b1y, 2.35, 0.72,
    "Investment trigger\nD_foresight / C_committed ≥ θ_eff\n"
    "(foresight 2 yr stations, 3 yr plants)",
    fill=WHITE, size=9, outline=STOCH, dashed=True)
box(10.00, b1y, 2.35, 0.72,
    "Commitment p_invest\n+ lead time 2–4 yr\n→ pending → active",
    fill=WHITE, size=9, outline=STOCH, dashed=True)
arrow(4.60, b1y + 0.36, 4.85, b1y + 0.36)
arrow(7.15, b1y + 0.36, 7.40, b1y + 0.36)
arrow(9.75, b1y + 0.36, 10.00, b1y + 0.36)

# asset register
box(2.55, b1y + 0.86, 9.80, 0.44,
    "VINTAGE-AWARE ASSET REGISTER   —   each station / plant retains its "
    "commissioning-year CAPEX; capacity not retired within horizon",
    fill=RGBColor(0xF7, 0xF9, 0xFB), size=9, bold=True)
for x in (3.55, 6.00, 8.55, 11.15):
    arrow(x, b1y + 0.72, x, b1y + 0.86, width=1.2)

# ---------------------------------------------------------------- band 2
b2y = 3.42
box(2.55, b2y, 2.30, 0.78,
    "Learning curves\nWright’s law, core / BoP split\n→ CAPEX by vintage year",
    fill=WHITE, size=9)
box(5.10, b2y, 2.10, 0.78,
    "Production cost\nSMR  |  ELY grid  |  ELY solar\ncapacity-weighted",
    fill=WHITE, size=9)
box(7.45, b2y, 2.05, 0.78,
    "Station cost\nCAPEX(Q) + O&M(Q, u)\nutilization-dependent",
    fill=WHITE, size=9)
box(9.75, b2y, 2.60, 0.78,
    "Policy credits\nLCFS, HRI, 45V per kg\nHVIP voucher on truck CAPEX",
    fill=WHITE, size=9)
arrow(4.85, b2y + 0.39, 5.10, b2y + 0.39)
arrow(7.20, b2y + 0.39, 7.45, b2y + 0.39)
arrow(9.50, b2y + 0.39, 9.75, b2y + 0.39)

# The LCOH box stops short of x = 6.15 so the carbon-intensity feed below can
# run down that channel without crossing a filled box.
box(2.55, b2y + 0.92, 3.30, 0.44,
    "LCOH  (USD kg⁻¹ dispensed)", fill=RGBColor(0xFD, 0xF2, 0xE3),
    size=10.5, bold=True)
box(7.80, b2y + 0.92, 4.55, 0.44,
    "TCO  (USD mile⁻¹)   H₂ vs diesel",
    fill=RGBColor(0xFD, 0xF2, 0xE3), size=10.5, bold=True)
arrow(5.85, b2y + 1.14, 7.80, b2y + 1.14)

# ---------------------------------------------------------------- band 3
b3y = 5.50
box(2.55, b3y, 2.55, 0.72,
    "System carbon intensity\ncapacity-weighted, CARB method", fill=WHITE, size=9)
box(5.35, b3y, 2.30, 0.72,
    "Avoided CO₂e\n× SCC (1.5 / 2.0 / 2.5 %)", fill=WHITE, size=9)
box(7.90, b3y, 2.05, 0.72,
    "Avoided NOₓ, PM₂.₅\n× COBRA", fill=WHITE, size=9)
box(10.20, b3y, 2.15, 0.72,
    "Net societal cost\n& abatement cost", fill=RGBColor(0xE7, 0xF4, 0xE8),
    size=9, bold=True)
arrow(5.10, b3y + 0.36, 5.35, b3y + 0.36)
arrow(7.65, b3y + 0.36, 7.90, b3y + 0.36)
arrow(9.95, b3y + 0.36, 10.20, b3y + 0.36)

# TCO premium feed-in (policy transfers stripped)
arrow(10.05, b2y + 1.36, 10.05, b3y - 0.02, dashed=True)
label(10.15, b2y + 1.42, 2.6, 0.3,
      "TCO premium\n(policy transfers removed)", size=7.5, bold=False,
      italic=True, color=MUTED)

# ---------------------------------------------------------------- cross links
# utilization feedback: fleet & capacity -> station cost
arrow(2.90, b1y + 1.30, 2.90, b2y - 0.02, dashed=True, color=STOCH)
label(2.98, 2.73, 2.3, 0.28, "utilization  u", size=8,
      bold=False, italic=True, color=STOCH)
# CI feed: production capacity mix -> system carbon intensity. Runs down the
# channel left free beside the LCOH box; it crosses only the thin LCOH → TCO
# arrow, never a box.
arrow(6.15, b2y + 0.78, 6.15, b3y - 0.02, dashed=True)
label(6.25, b2y + 0.80, 2.2, 0.26, "capacity mix → CI", size=7.5,
      bold=False, italic=True, color=MUTED)

# ---------------------------------------------------------------- MC wrapper
# Sits just OUTSIDE the three module bands, so its rounded corners do not clip
# the band corners or the band labels.
mc = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                            Inches(2.29), Inches(0.99),
                            Inches(10.72), Inches(5.77))
mc.fill.background()
mc.line.color.rgb = STOCH
mc.line.width = Pt(1.4)
mc.line.dash_style = MSO_LINE_DASH_STYLE.DASH
mc.shadow.inherit = False
mc.adjustments[0] = 0.015   # near-square corners, so the band labels stay clear
mc.text_frame.text = ""

label(2.45, 6.76, 6.0, 0.3,
      "Monte Carlo wrapper:  1000 runs; θ and p_invest drawn once per run "
      "(investor archetype); commissioning and lead times drawn per facility",
      size=8.5, bold=False, italic=True, color=STOCH)
label(9.05, 6.76, 3.9, 0.3,
      "Outputs: LCOH, TCO, net societal cost, abatement cost — "
      "run mean, with P25–P75 spread where shown",
      size=8.5, bold=False, italic=True, color=MUTED,
      align=PP_ALIGN.RIGHT)

prs.save(OUT_PPTX)
print(f"Saved → {OUT_PPTX}")

# ---------------------------------------------------------------- raster/vector
# The draft cites numbered PDFs/PNGs like every other figure, so render the deck
# with LibreOffice when available. The PPTX remains the editable original.
soffice = shutil.which("soffice")
if soffice:
    for fmt in ("pdf", "png"):
        subprocess.run([soffice, "--headless", "--convert-to", fmt,
                        "--outdir", str(OUT_DIR), str(OUT_PPTX)],
                       check=True, capture_output=True)
        print(f"Saved → {OUT_PPTX.with_suffix('.' + fmt)}")
else:
    print("soffice not found — PPTX only; export PDF/PNG by hand for the draft.")
