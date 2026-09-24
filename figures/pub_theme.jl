# =============================================================================
# SHARED PUBLICATION STYLE  —  include after `using CairoMakie`
# =============================================================================
# Single source of truth for colours, line styles, typography, figure widths,
# and the save routine used by every publication figure. Include this once near
# the top of a figure script (after `using CairoMakie`) and use the exported
# constants instead of hard-coding colours or sizes.
#
#   using CairoMakie
#   include("pub_theme.jl")     # sets the global theme + defines constants
#
# The accompanying decision log is figures/FIGURE_STYLE.md — keep the two in
# sync when you change anything here.
# =============================================================================

using CairoMakie

# Output-destination table (manuscript / appendix / old) used by save_pub.
include(joinpath(@__DIR__, "fig_routing.jl"))

# ── Unit conversion + canonical figure widths (Elsevier) ─────────────────────
const MM_TO_PT  = 1 / 0.352778            # millimetres → points
const W_SINGLE  =  90 * MM_TO_PT          # single column
const W_ONEHALF = 140 * MM_TO_PT          # 1.5 column
const W_DOUBLE  = 190 * MM_TO_PT          # double / full text width
mm(x) = x * MM_TO_PT                       # convenience: mm(60) → points

# ── Typography (points) ──────────────────────────────────────────────────────
const FS_BASE   = 9      # figure base font size
const FS_LABEL  = 9      # axis labels
const FS_TICK   = 8      # tick labels
const FS_TITLE  = 9      # panel titles + panel letters (bold)
const FS_LEGEND = 7      # legend entries
const FS_ANNOT  = 7      # in-plot annotations

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR SYSTEM  (see FIGURE_STYLE.md for the rationale and the full map)
# ─────────────────────────────────────────────────────────────────────────────

# ── H₂ production pathways — distinguished by COLOUR everywhere ───────────────
const C_SMR   = colorant"#4472C4"   # SMR / current mix          (blue)
const C_GRID  = colorant"#ED7D31"   # electrolysis — grid        (orange)
const C_SOLAR = colorant"#70AD47"   # electrolysis — solar       (green)

# ── Truck deployment scenarios ───────────────────────────────────────────────
# Encoded by LINE STYLE when overlaid with pathways; by the dedicated COLOURS
# below in deployment-only figures (where the pathway palette is free).
const C_DEP_NOLOW = colorant"#999999"   # No/Low deployment   (grey)
const C_DEP_LIM  = colorant"#6A3D9A"   # Limited deployment      (violet)
const C_DEP_HIGH  = colorant"#009988"   # High deployment     (teal)

const LS_NOLOW = :dot       # No/Low deployment
const LS_LIM  = :solid     # Limited deployment
const LS_HIGH  = :dash      # High deployment

# ── Social cost of carbon (single-hue pink ramp; light = high rate/low SCC) ───
const C_SCC_15 = colorant"#7B0D3F"   # 1.5% discount rate  (dark   — highest SCC)
const C_SCC_20 = colorant"#AD1457"   # 2.0% discount rate  (mid    — central)
const C_SCC_25 = colorant"#E06CA5"   # 2.5% discount rate  (light  — lowest SCC)
const C_SCC    = C_SCC_20             # aggregate "SCC / avoided-SCC" concept colour
const SCC_SHADES = Dict("1.5%" => C_SCC_15, "2.0%" => C_SCC_20, "2.5%" => C_SCC_25)

# ── Societal-benefit components ───────────────────────────────────────────────
const C_HEALTH        = colorant"#E69F00"   # air-quality (health) benefit (amber)
const C_TOTAL_BENEFIT = colorant"#332288"   # total societal benefit       (indigo)

# ── Diesel reference ──────────────────────────────────────────────────────────
# Diesel pump-price scenarios are distinguished by LINE STYLE:
#   $4.80/gal → LS_DIESEL_LO (solid),  $5.80/gal → LS_DIESEL_HI (dash).
# Both are California retail pump prices, read from config/tco_config.json via
# TCO_DIESEL_P_LO / TCO_DIESEL_P_HI in config_defaults.jl — never hard-code them.
# Where solid is already taken by a zero line, the low price falls back to
# dash-dot.
# Stand-alone diesel reference curves use the neutral grey below.
const C_DIESEL     = colorant"#5A5A5A"
const LS_DIESEL_LO = :solid
const LS_DIESEL_HI = :dash

# ── Net societal cost — coloured by DIESEL PUMP PRICE ─────────────────────────
# In the net-cost panels the diesel price is encoded by COLOUR rather than by
# line style, which frees line style entirely and lets both series be solid —
# far more legible than dashes at publication line widths. Okabe–Ito
# blue/vermillion: colourblind-safe as a pair, and distinct from both C_SOLAR
# green and the C_SCC pink ramp that share these figures.
#
# The SCC discount-rate range travels as a translucent BAND of the same colour
# (line = 2.0%, shading = 1.5–2.5%), following the house rule that ranges are
# bands and cases are lines.
const C_NET_LO = colorant"#0072B2"   # net cost at TCO_DIESEL_P_LO ($4.80/gal)
const C_NET_HI = colorant"#D55E00"   # net cost at TCO_DIESEL_P_HI ($5.80/gal)

# ── Neutral / structural greys ────────────────────────────────────────────────
const C_GREY      = colorant"#808080"   # secondary / balance-of-plant series
const C_GREY_LT   = colorant"#A0A0A0"   # tertiary / platform series
const C_ZERO_LINE = (:black, 0.40)      # zero / reference hlines

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL THEME
# ─────────────────────────────────────────────────────────────────────────────
# Sets the canonical defaults so figure scripts do NOT need per-axis size kwargs.
# Anything passed explicitly to an Axis/Legend call still overrides these.
set_theme!(
    fontsize       = FS_BASE,
    figure_padding = 4,
    Axis = (
        titlesize      = FS_TITLE,
        titlefont      = :bold,
        titlealign     = :left,
        xlabelsize     = FS_LABEL,
        ylabelsize     = FS_LABEL,
        xticklabelsize = FS_TICK,
        yticklabelsize = FS_TICK,
        xgridvisible   = true,
        ygridvisible   = true,
    ),
    Legend = (
        labelsize     = FS_LEGEND,
        titlesize     = FS_LEGEND,
        framevisible  = true,
        rowgap        = 1,
        patchsize     = (12, 8),
        patchlabelgap = 4,
        padding       = (4, 4, 2, 2),
    ),
)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

"""
    panel_label!(ax, letter; pos = (0.03, 0.97))

Draw a bold panel letter (e.g. "a") inside the top-left of `ax`. Renders as
"(a)" at the canonical title size. Use this OR an axis `title = "(a)  …"`,
not both, within one figure.
"""
function panel_label!(ax, letter; pos = (0.03, 0.97))
    text!(ax, pos[1], pos[2]; text = "($letter)", space = :relative,
          align = (:left, :top), fontsize = FS_TITLE, font = :bold, color = :black)
end

"""
    year_ticks(first_year, last_year; step = 5, min_gap = 3)

Canonical x-axis year ticks: the panel's **first year**, then every `step`-th
round year up to `last_year`. For the assessment window this reproduces the
convention used across the figure suite — `year_ticks(2026, 2045)` →
`[2026, 2030, 2035, 2040, 2045]`.

The first tick is always the axis start (so the reader can see where the series
begins), after which ticks fall on multiples of `step`. A round year closer than
`min_gap` to the start is dropped to avoid colliding labels, e.g.
`year_ticks(2019, 2045)` → `[2019, 2025, 2030, …]` (2020 suppressed).

`last_year` is labelled only if it is itself a multiple of `step`; a horizon such
as 2043 ends unlabelled rather than crowding the 2040 tick.
"""
function year_ticks(first_year::Integer, last_year::Integer;
                    step::Integer = 5, min_gap::Integer = 3)
    first_round = step * cld(first_year + 1, step)   # first multiple of step after the start
    rest = filter(y -> y - first_year >= min_gap, first_round:step:last_year)
    return [Int(first_year); Int.(rest)]
end

"""
    save_pub(name, fig; dir = OUT_DIR, subdir = fig_subdir(name))

Save `fig` as both a vector PDF (`pt_per_unit = 1`) and a 300-dpi PNG
(`px_per_unit = 300/72`) using `name` (no extension).

The file goes to `dir/subdir`, where `subdir` defaults to the figure's
destination in [fig_routing.jl](fig_routing.jl) — "manuscript", "appendix", or
"old" — so a regenerated figure always overwrites the copy the paper cites.
Pass `subdir = ""` to write to `dir` itself.

Manuscript figures are additionally numbered by their position in
`FIG_MANUSCRIPT`, so `save_pub("fig_learning_curves", …)` writes
`manuscript/fig_2_learning_curves.{pdf,png}`.
"""
function save_pub(name::AbstractString, fig; dir = OUT_DIR, subdir = fig_subdir(name))
    dir = isempty(subdir) ? dir : joinpath(dir, subdir)
    mkpath(dir)
    base = fig_basename(name)
    pdf = joinpath(dir, base * ".pdf")
    png = joinpath(dir, base * ".png")
    save(pdf, fig; pt_per_unit = 1)
    save(png, fig; px_per_unit = 300 / 72)
    println("Saved → $png  (+ .pdf)")
    return png
end
