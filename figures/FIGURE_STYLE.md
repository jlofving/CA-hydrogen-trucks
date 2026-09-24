# Publication figure style guide

Decision log and reference for the figures in the journal manuscript. The
machine-readable source of truth is [`pub_theme.jl`](pub_theme.jl); this document
explains the *why* and lists the per-concept colour/style map so the same
scenarios always look the same across figures. **Keep the two in sync.**

Every publication figure script `include("pub_theme.jl")` (after `using
CairoMakie`) and uses its constants instead of hard-coded colours or sizes.
New figures should do the same and call `save_pub(name, fig)` to export.

Last updated: 2026-06-23.

---

## 1. Output format

| Setting | Value |
|---|---|
| Raster | PNG at **300 dpi** (`px_per_unit = 300/72`) |
| Vector | PDF (`pt_per_unit = 1`) — preferred for submission |
| Saver | `save_pub(name, fig)` writes both, into `figures/out/` |

Both formats are written for every figure. Submit the PDF where the journal
accepts vector art; the PNG is the 300-dpi fallback.

**One exception — figure 1, the model schematic.** It carries no data, so it is
authored rather than plotted: `figures/make_schematic.py` (needs `python-pptx`)
writes an editable `.pptx`, and shells out to LibreOffice `soffice` for the PDF
and PNG the draft cites. It is listed in `FIG_MANUSCRIPT` like any other figure
even though no Julia script emits it, so `fig_routing.jl` stays the single
source of figure numbering — the Python script parses its number back out of
that list. The schematic keeps its own diagram palette (module bands, red dashes
for stochastic elements) and is not bound by §5; every model parameter it quotes
is annotated with its config or source-line provenance at the top of the script,
so change both together.

**Diagnostics.** A figure that answers a modelling question and is not (yet)
cited in the paper is saved with `save_pub(name, fig; subdir = "diagnostics")`.
That keeps it out of `old/`, which means "retired from the paper", and out of
`manuscript/`, which would imply it is cited. Promote one by adding its name to
`FIG_MANUSCRIPT` or `FIG_APPENDIX` in `fig_routing.jl` and dropping the `subdir`
argument — `fig_vintage_effect.jl` (the CAPEX vintage counterfactual, run with
`capex_vintage = false`) started this way and is now manuscript fig 9.

## 2. Figure widths (Elsevier)

Set width with the constants `W_SINGLE`, `W_ONEHALF`, `W_DOUBLE` (points). Pick
the smallest that holds the panel without crowding; preserve the existing height.

| Constant | Width | Use |
|---|---|---|
| `W_SINGLE`  | 90 mm  | single small panel (e.g. truck deployment, LCFS price) |
| `W_ONEHALF` | 140 mm | one rich panel or a wide single panel (price breakdown, abatement, background demand, scenario-matrix overlay) |
| `W_DOUBLE`  | 190 mm | all multi-panel figures (2×, 6×, 3×2, tornado, capacity) |

The three formerly-200 mm figures (tco_premium, societal_benefit, and the
retired cost_premium_3row) were shrunk to 190 mm. Its replacements —
cost_premium_benefit and net_societal_cost — are `W_DOUBLE` as well.

## 3. Typography

Unified, "slightly larger" scale. Driven by the global theme in `pub_theme.jl`,
so per-axis size kwargs were **removed** from the scripts — do not re-add them.

| Element | Constant | Size |
|---|---|---|
| Figure base | `FS_BASE` | 9 pt |
| Axis labels | `FS_LABEL` | 9 pt |
| Tick labels | `FS_TICK` | 8 pt |
| Panel titles / letters | `FS_TITLE` | 9 pt, **bold**, left-aligned |
| Legend entries | `FS_LEGEND` | 7 pt |
| In-plot annotation | `FS_ANNOT` | 7 pt |

Font family: CairoMakie default sans-serif (kept deliberately — it renders the
`₂`/`⁻¹` sub/superscripts and `LaTeXString` labels reliably).

**Documented size exceptions** (genuinely dense layouts, set explicitly):
- Panel *titles* on figures with long descriptive titles (sensitivity tornado,
  tco_comparison, abatement, tco_premium, societal, capacity) use **8 pt**
  instead of 9. The cost-benefit pair does not need the exception: its panel
  titles carry only the deployment name, because the metric is already on each
  row's y-axis label.
- `fig_tco_comparison` panel (b) y-axis category labels: **6 pt** (many bars).
- Retired: `fig_cost_premium_3row`'s in-panel legends ran at **5 / 4.5 pt**.
  Its replacements use a single shared figure-level `Legend` in its own layout
  row at the standard `FS_LEGEND` 7 pt instead. Needing sub-6 pt legend text is a
  signal that a panel is carrying too many series — split the figure rather than
  shrinking the type.
- `fig_sensitivity_grid` (supplementary, 8 panels): retains its small 6 pt axis
  text.

## 4. Panel labels

Use **one** mechanism per figure, never both:
- axis `title = "(a)  …"`, or
- `panel_label!(ax, "a")` for an in-plot bold "(a)" (top-left).

Both render bold at `FS_TITLE`.

## 4b. Legends

When a figure uses **two orthogonal encodings** (colour = one variable, line
style = another), split the legend into two titled sections rather than listing
every colour × style combination. State the line-style key once. Pass the
grouped form to `axislegend`, or to `Legend` for a shared legend in its own
layout row:

```julia
axislegend(ax, [colour_elems, style_elems], [colour_labels, style_labels],
           ["Series (colour)", "Diesel price (line style)"]; titlesize = 5, …)
```

For a figure-level legend, annotate the container as `Vector{Vector}` so the call
reliably hits Makie's grouped method, and nest a vector of elements to overlay
them in one entry — a line drawn on top of its own band:

```julia
groups = Vector{Vector}([[[LineElement(color = c), PolyElement(color = (c, 0.18))]]])
Legend(fig[3, 1:2], groups, labels, titles;
       orientation = :horizontal, tellheight = true, tellwidth = false)
```

A shared legend is preferred over `axislegend` whenever the same key applies to
every panel: it cannot overlap the data, and it stops the key being repeated.

## 4c. Year axes

Time axes use the **first year, then 5-year round years** convention — for the
assessment window that is `2026, 2030, 2035, 2040, 2045`. Use the helper rather
than hard-coding, so the ticks follow `start_year` / `end_year`:

```julia
xticks = year_ticks(START_YEAR, END_YEAR)      # → [2026, 2030, 2035, 2040, 2045]
xticks = year_ticks(2019, END_YEAR)            # → [2019, 2025, 2030, …]  (2020 suppressed)
```

The first tick is always the axis start so the reader can see where the series
begins; a round year within `min_gap = 3` of the start is dropped to avoid
colliding labels. Label rotation stays at `π/4` on narrow panels. Panels sharing
a row or a subject (e.g. a learning driver and its cost outcome) should use
**identical** limits and ticks.

---

## 5. Colour & line-style system

The binding rule: **the same scenario uses the same colour and line style in
every figure.** Two orthogonal scenario axes:

### 5.1 H₂ production pathway → COLOUR (always)

| Pathway | Constant | Colour |
|---|---|---|
| SMR / current mix | `C_SMR` | `#4472C4` blue |
| Electrolysis — grid | `C_GRID` | `#ED7D31` orange |
| Electrolysis — solar | `C_SOLAR` | `#70AD47` green |

Used in: scenario_matrix_overlay, tco_comparison, tco_premium, (solar premium in
cost_premium_benefit).

### 5.2 Truck deployment scenario → LINE STYLE, or dedicated COLOUR

When a figure overlays deployments **with** pathways, deployment is the line
style (so the pathway colour stays free):

| Deployment | Style constant | Line |
|---|---|---|
| No/Low | `LS_NOLOW` | dotted |
| Limited deployment | `LS_LIM` | solid |
| High | `LS_HIGH` | dashed |

In **deployment-only** figures (no pathway dimension), deployment is a dedicated
colour troupe, chosen colourblind-safe and distinct from the pathway palette:

| Deployment | Constant | Colour |
|---|---|---|
| No/Low | `C_DEP_NOLOW` | `#999999` grey |
| Limited deployment | `C_DEP_LIM` | `#6A3D9A` violet |
| High | `C_DEP_HIGH` | `#009988` teal |

Used as colour in: scenario_truck_deployment, abatement_cost.

> **Note on the approved trio.** The original approved sample was grey / purple
> `#882255` / teal. Two hexes were adjusted to remove collisions the preview
> didn't surface: demo `#882255` → `#6A3D9A` (the original sat next to the SCC
> pink and the two co-occur in `fig_abatement_cost`); and the air-quality health
> line was moved off teal → amber so teal reads unambiguously as "High
> deployment" article-wide. Revert in `pub_theme.jl` if you disagree.

### 5.3 Social cost of carbon → single-hue pink ramp

Discount rate encoded by shade (light = high rate = low SCC):

| Rate | Constant | Colour |
|---|---|---|
| 1.5% | `C_SCC_15` | `#7B0D3F` dark |
| 2.0% | `C_SCC_20` | `#AD1457` mid (= `C_SCC`, the SCC concept colour) |
| 2.5% | `C_SCC_25` | `#E06CA5` light |

`SCC_SHADES` is the `Dict("1.5%"=>…, …)`. Used in: scc_fit, and as the aggregate
band/line (`C_SCC`) in abatement, societal_benefit, cost_premium_benefit. The
three shades are for figures whose *subject* is the SCC itself; elsewhere the
1.5–2.5% range travels as a band (§5.4a), not as three lines.

### 5.4 Societal-benefit components

| Concept | Constant | Colour |
|---|---|---|
| Air-quality (health) benefit | `C_HEALTH` | `#E69F00` amber |
| Total societal benefit | `C_TOTAL_BENEFIT` | `#332288` indigo |
| Avoided SCC | `C_SCC` | `#AD1457` pink |

**The health term has one definition, in [`health_cost.jl`](health_cost.jl).**
Every figure that values avoided air-quality damage calls
`health_cost_per_mile(year)`: `fig_societal_cost_benefit` (figs 11, 12),
`fig_abatement_cost` (fig 13), and the appendix `fig_societal_benefit` and
`fig_policy_expenditure`. Each of those scripts previously carried its own
transcribed copy of the factor table, which is how the paper could have ended up
valuing the same avoided mile two ways in two figures.

The default basis is **EPA COBRA** (NOx + PM2.5), applied through the bivariate
proxy calibrated in `fig_cost_premium.jl`: a flat **$0.00815 per displaced
diesel mile**. The older per-mile factors from *Env Cost Factors 20apr LF.xlsx*
(NOx + PM2.5 + NH₃, declining ~2%/yr from $0.0386 in 2026) remain available as a
sensitivity via `HEALTH_BASIS=envcost`. They are ~3–5× larger, so the bases are
never mixed within a suite — re-run all five scripts together when switching.

### 5.4a Net cost (`fig_cost_premium_benefit`, `fig_net_societal_cost`)

> **Supersedes the grey-ramp scheme.** These two figures replace the six-panel
> `fig_cost_premium_3row`, which encoded SCC rate, diesel price *and* deployment
> as separate lines and so reached 11 curves per panel. The grey net-cost ramp
> (`nc_shades`, 1.5% `#1A1A1A` / 2.0% `#777777` / 2.5% `#B5B5B5`) is retired;
> the old script is still on disk but is no longer in `FIG_MANUSCRIPT`, so it
> routes to `out/old/`.

**Ranges are bands, cases are lines.** This is the rule that keeps the panels
readable, and it applies to every uncertainty dimension:

- **SCC discount rate 1.5–2.5% → translucent BAND**, with the 2.0% value as the
  centre line. Never three separate lines: it is a sensitivity range, not three
  cases a reader needs to trace individually. Matches how the societal-benefit
  band was already drawn.
- **Diesel pump price → COLOUR** in the net-cost panels: `C_NET_LO` `#0072B2`
  blue ($4.80/gal), `C_NET_HI` `#D55E00` vermillion ($5.80/gal). Okabe–Ito
  blue/vermillion — colourblind-safe as a pair and distinct from both `C_SOLAR`
  green and the `C_SCC` pink ramp they share these figures with. Moving the
  diesel price to colour frees line style entirely, so both series stay **solid**,
  which is far more legible than dashes at publication line widths.
  In `fig_cost_premium_benefit` the diesel price stays on **line style** instead
  (solid/dash), because only the green premium depends on it there — the benefit
  band has no diesel dependence, so one encoding suffices.
- **Deployment scenario → COLUMN**, in every panel of both figures, so the
  reader learns the layout once.
- **At most one band per panel.** The present-value rate is held at 2%/yr and its
  0–3% sensitivity reported as text plus a values file, not as a second band.
  Nesting a PV band inside an SCC band is what made the old cumulative panels
  illegible: the shading only widened near the zero crossing, reading as an
  uncertainty explosion that was purely an artefact of the symlog scale.
- **No greys** except gridlines and the `C_ZERO_LINE` reference line.

Both pump prices are California retail. They and the fleet-average fuel economy
(7.43 mpg, used for the $/gal → $/mile conversion) come from
`config/tco_config.json` via `TCO_DIESEL_P_LO`, `TCO_DIESEL_P_HI` and
`TCO_DIESEL_MPG` in `config_defaults.jl`; never hard-code them. ATRI's own fuel
line item ($0.482/mile ≈ $3.58/gal) is a national average operating cost, not a
California pump price, and is not used as the diesel reference. Where solid is
already used by a zero line (e.g. the tornado panels), the low scenario falls
back to dash-dot.

**Cumulative net cost uses a LINEAR y axis**, not symlog. The series spans only a
few thousand M USD, so linear costs no resolution that matters, and it removes
the two symlog artefacts: a sign change rendered as a near-vertical cliff, and a
band that appeared to blow up near zero.

Neither figure draws break-even or payback markers. A dot plus a year label on
each crossing added clutter without adding information — the crossings are read
off the zero line directly. The crossing years (annual break-even, per-mile zero,
cumulative payback, and the PV/SCC sensitivity of the payback) are written to
`out/fig_societal_cost_benefit_values.txt` for the captions instead.

`fig_cost_premium_benefit` shows the premium and the benefit only — **net cost
gets no lines of its own there**, because it is the vertical gap between the two
curves already plotted, which is the point of co-plotting them. Its per-mile and
cumulative forms live in `fig_net_societal_cost`.

Both figures are produced by one script, `fig_societal_cost_benefit.jl`, from a
single Monte Carlo, so every number reconciles across all six panels.

> **Social-accounting convention (premium = unsubsidized resource cost).** The
> premium in the social figures (`fig_cost_premium_benefit`,
> `fig_net_societal_cost`, `fig_abatement_cost`)
> is *gross of policy support*: the per-kg
> credits (LCFS + HRI + 45V) are added back to the LCOH and the HVIP voucher is
> removed from truck capital (`h2_tco_unsub` / `h2_capital_gross`). Rationale:
> subsidies are transfers, not resource costs, so they do not belong in a social
> cost-benefit comparison, and the carbon externality is counted once — explicitly
> via the SCC, not a second time through LCFS/45V. The diesel side is left as-is
> (its small anchored LCFS adjustment stays). The dollar value of the stripped-out
> support is shown, split by funder, in `fig_policy_expenditure`. Companion
> figures `fig_tco_premium` / `fig_tco_comparison` keep the *policy-inclusive
> (private)* premium — that is the adoption-decision number; keep the two framings
> distinct and labelled.

### 5.4b Policy instruments → colour = instrument, family = funder

For policy-support figures (`fig_policy_expenditure`, and the credit layers in
`fig_price_breakdown`), each instrument keeps a fixed colour, and the **colour
family encodes who pays**:

| Instrument | Colour | Funder family |
|---|---|---|
| LCFS credit | `#4CAF50` green | Fuel-market (greens) |
| HRI bonus credit | `#1B5E20` dark green | Fuel-market (greens) |
| 45V tax credit | `#FFD700` gold | Taxpayer (golds) |
| HVIP truck voucher | `#B8860B` dark gold | Taxpayer (golds) |

LCFS/HRI/45V reuse the `fig_price_breakdown` instrument colours; HVIP (not a
per-kg price component) is added as a second taxpayer gold. Total support is
outlined/filled in grey `#666666`; avoided damage overlays in `C_TOTAL_BENEFIT`
indigo. The displaced-diesel LCFS deficit avoided (the other side of the instrument)
is reported to the console, not plotted. Defined locally (these are component
palettes, not scenario colours).

### 5.5 Diesel reference

Pump-price scenario encoded by line style; reference curves drawn in grey.

| Item | Constant | Value |
|---|---|---|
| Diesel reference colour | `C_DIESEL` | `#5A5A5A` grey |
| $4.80/gal | `LS_DIESEL_LO` | solid |
| $5.80/gal | `LS_DIESEL_HI` | dashed |

(`fig_tco_comparison` plots the $4.80 and $5.80 lines as dashdot/dot to keep both
distinct from the dashed high-deployment H₂ lines — a local exception.)

### 5.6 Neutral greys

`C_GREY` `#808080` (secondary/balance-of-plant series), `C_GREY_LT` `#A0A0A0`
(platform/no-learning series), `C_ZERO_LINE` `(:black, 0.40)` (zero/reference
hlines).

---

## 6. Figures with intentionally local palettes

These are **not** scenario comparisons, so they keep a purpose-built palette
(documented here, not unified with the scenario colours):

- **`fig_learning_curves`** — technology-learning drivers: electrolyzer = blue,
  solar PV = orange, truck/fuel-cell = purple `#7030A0`, BoP/platform = greys.
- **`fig_station_costs`** — station type: gaseous = blue, liquid = orange.
- **`fig_price_breakdown`** — delivered-price cost/credit components (production,
  transport, O&M, CAPEX, LCFS, HRI, 45V) each get their own component colour.
- **`fig_capacity_vs_demand`** — capacity/demand series: station capacity = teal,
  truck demand = red/orange, production capacity = purple, bus = blue, car =
  green, utilisation = orange.
- **`fig_tco_comparison` panel (b)** — TCO cost categories (Okabe-Ito): fuel,
  capital, repair, tires, driver+other.
- **`fig_utilization_impact`** — representative-year curves (2026/2030/final).
- **`fig_sensitivity`** — low variant = yellow `#F2C800`, high variant = blue
  `#1565C0`, diesel parity = `C_DIESEL`.
- **`fig_background_demand`**, **`fig_scenario_lcfs_price`** — small standalone
  inputs; bus/car and LCFS-scenario colours are local.
- **`fig_policy_expenditure`** — policy-instrument palette grouped by funder
  (see §5.4b): golds = taxpayer (45V, HVIP), greens = fuel-market (LCFS, HRI),
  grey total support, indigo avoided-damage overlay.

---

## 7. How to add a new figure

```julia
using CairoMakie
include("../hydrogen truck deployment.jl")   # if it runs the model
include("config_defaults.jl")                 # if it runs the model
include("pub_theme.jl")                        # always — sets theme + constants

fig = Figure(size = (W_DOUBLE, 95 * MM_TO_PT))   # no fontsize kwarg needed
ax  = Axis(fig[1, 1]; xlabel = "Year", ylabel = "…")  # no size kwargs needed
lines!(ax, x, y; color = C_SOLAR, linestyle = LS_LIM)
axislegend(ax)                                  # inherits 7 pt, framed
save_pub("fig_my_new_figure", fig)
```

Do not set `fontsize` on `Figure(...)`, per-axis `*labelsize`/`*ticklabelsize`,
or `update_theme!(Legend=…)` — the global theme handles them. Override only when
a documented exception is genuinely needed, and add it to §3 above.
