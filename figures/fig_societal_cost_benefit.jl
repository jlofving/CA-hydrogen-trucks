# =============================================================================
# FIGURES: SOLAR H2 vs DIESEL — SOCIETAL COST-BENEFIT  (two figures, one MC)
# =============================================================================
# Replaces the six-panel fig_cost_premium_3row.jl, which stacked three
# orthogonal encodings (SCC rate x diesel price x deployment) as separate lines
# and reached 11 curves per panel. Split into two figures, each with a single
# message:
#
#   fig_cost_premium_benefit  (1 row x 2 cols, cols = deployment)
#       Resource-cost premium vs. societal benefit. The NET cost is the vertical
#       gap between the two curves — that is the point of co-plotting them, so
#       it gets no lines of its own here.
#
#   fig_net_societal_cost     (2 rows x 2 cols, cols = deployment)
#       Row 1: cumulative net societal cost (LINEAR axis)
#       Row 2: net cost per mile
#
# ENCODING RULES (house style — ranges are bands, cases are lines):
#   * SCC discount rate 1.5-2.5% → translucent BAND, line = 2.0% centre.
#     Never three separate lines: it is a sensitivity range, not three cases.
#   * Diesel pump price → COLOUR in the net-cost panels (C_NET_LO/C_NET_HI),
#     LINE STYLE in the premium panel (where only the green premium depends on
#     it, so one encoding is enough).
#   * Deployment scenario → COLUMN, in every panel of both figures, so the
#     reader learns the layout once.
#   * No greys anywhere except gridlines and the zero reference line.
#
# ONE uncertainty band per panel. The present-value discount rate is held at
# PV_RATE = 2%/yr and its 0-3% sensitivity reported as text on the cumulative
# panels — nesting a PV band inside an SCC band is what made the old symlog
# panels illegible (the shading only appeared near the zero crossing, reading
# as an uncertainty explosion that was purely an artefact of the scale).
#
# Both figures come from the SAME solar Monte Carlo (one run, same seed and
# configs), so every number reconciles across all six panels.
#
# SOCIAL ACCOUNTING: the premium here is the UNSUBSIDIZED resource-cost premium —
# the per-kg policy credits (LCFS + HRI + 45V) are added back to the LCOH and the
# HVIP truck voucher is removed from the truck capital. Policy support is a
# transfer, not a resource cost, so it is excluded from this social comparison;
# the carbon externality enters explicitly through the SCC. The dollar value of
# the stripped-out support is shown, split by funder, in fig_policy_expenditure.jl.
#
# Run from project root:
#   julia --project=figures figures/fig_societal_cost_benefit.jl
# =============================================================================

using CairoMakie
using Statistics
using JSON
using Random
using LinearAlgebra
using Printf
import Serialization

cd(joinpath(@__DIR__, ".."))
include("../hydrogen truck deployment.jl")
include("config_defaults.jl")
include("health_cost.jl")   # avoided air-quality damage — COBRA by default
include("pub_theme.jl")

# 1000 runs is the publication setting and takes several minutes per deployment.
# Override it when iterating on layout — the medians are meaningless at 5 runs but
# every drawing, legend and annotation path still executes:
#   SCB_N_RUNS=5 julia --project=figures figures/fig_societal_cost_benefit.jl
const N_RUNS  = parse(Int, get(ENV, "SCB_N_RUNS", "1000"))
const SEED    = 42
const OUT_DIR = joinpath(@__DIR__, "out")
mkpath(OUT_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# TCO parameters (mirrors fig_cost_premium_envcost_signed.jl)
# ─────────────────────────────────────────────────────────────────────────────
tco_raw = JSON.parsefile(joinpath(@__DIR__, "..", "config", "tco_config.json"))
diesel  = tco_raw["diesel_tco_usd_per_mile"];  ht = tco_raw["hydrogen_truck"];  ident = tco_raw["identical_cost_categories"]

d_capital = Float64(diesel["truck_lease_or_purchase"])
d_rm      = Float64(diesel["repair_and_maintenance"])
d_tires   = Float64(diesel["tires"])
d_common  = Float64(ident["driver_and_other"])

truck_purchase_cost = Float64(ht["purchase_cost_usd"])
truck_platform_cost = Float64(ht["platform_cost_usd"])
truck_fuelcell_cost = truck_purchase_cost - truck_platform_cost
miles_per_kg_h2     = Float64(ht["miles_per_kg_h2"])
rm_mult             = Float64(ht["repair_and_maintenance_multiplier"])
h2_tires_val        = Float64(ht["tires_multiplier"]) * d_tires
learning_rate       = Float64(ht["learning"]["learning_rate"])
ref_year_truck      = Int(ht["learning"]["reference_year"])

_sub_sched_prem = Dict{Int,Float64}(d["year"] => Float64(d["subsidy_usd"]) for d in ht["subsidy_schedule"])
truck_subsidy_yr_prem(year::Int) = _sub_sched_prem[maximum(filter(y -> y <= year, keys(_sub_sched_prem)))]

let
    obs = ht["learning"]["observed_fleet_stock"]
    ys  = Float64.([Int(d["year"]) for d in obs]);  vs = Float64.([Int(d["trucks"]) for d in obs])
    ts  = ys .- 2019.0
    θ   = hcat(ts.^3, ts.^2, ts, ones(length(ts))) \ vs
    global _prem_a, _prem_b, _prem_c, _prem_d = θ[1], θ[2], θ[3], θ[4]
end
fleet_stock_prem(yr::Int) = max(1.0, _prem_a*(yr-2019)^3 + _prem_b*(yr-2019)^2 + _prem_c*(yr-2019) + _prem_d)
h2_rm_prem(yr::Int) = d_rm * (rm_mult + (1.0 - rm_mult) * clamp((yr - 2026) / 9.0, 0.0, 1.0))

# Unsubsidized (resource-cost) truck capital: gross purchase price with no HVIP
# voucher subtracted. Used for the social cost-benefit comparison, where
# subsidies are transfers rather than resource costs and the carbon externality
# is counted explicitly via the SCC. See fig_policy_expenditure.jl for the
# support that is stripped out here.
function h2_capital_gross(year::Int)
    α  = log(1 / (1 - learning_rate)) / log(2)
    fc = truck_fuelcell_cost * (fleet_stock_prem(year) / fleet_stock_prem(ref_year_truck))^(-α)
    truck_gross = truck_platform_cost + fc            # no HVIP subsidy subtracted
    ann = calculate_annuity_factor(7, DISCOUNT_RATE)
    return truck_gross * ann /
           (Float64(H2_PER_TRUCK_PER_DAY) * Float64(OPERATING_DAYS_PER_YEAR) * miles_per_kg_h2)
end
h2_tco_unsub(lcoh_gross::Float64, yr::Int) =
    lcoh_gross / miles_per_kg_h2 + h2_capital_gross(yr) + h2_rm_prem(yr) + h2_tires_val + d_common

# ─────────────────────────────────────────────────────────────────────────────
# Diesel + emission constants + LCFS adjustment (anchored 2026)
# ─────────────────────────────────────────────────────────────────────────────
const DIESEL_MPG  = TCO_DIESEL_MPG   # miles/gallon, from config/tco_config.json
const DIESEL_P_LO = TCO_DIESEL_P_LO  # $/gal, from config/tco_config.json
const DIESEL_P_HI = TCO_DIESEL_P_HI  # $/gal, from config/tco_config.json
const DIESEL_MJ_PER_GAL = 128.45
const CA_BLEND_CI    = 0.66 * 43.74 + 0.06 * 38.49 + (1.0 - 0.66 - 0.06) * 90.0
const CO2_G_PER_MILE = CA_BLEND_CI * DIESEL_MJ_PER_GAL / DIESEL_MPG
const MJ_H2_PER_KG   = 120.0

# Diesel reference is held FLAT in real terms for the social comparison. The
# LCFS-on-diesel adjustment (diesel cost rising as the benchmark CI declines) is a
# transfer / carbon-price proxy; carbon is already valued explicitly via the SCC,
# so including it would double-count carbon on the diesel side. Excluded here to
# stay symmetric with stripping LCFS from the H₂ side. (_lcfs_prices_diesel is kept
# because the H₂ Monte Carlo config still uses the LCFS price schedule.)
const _lcfs_prices_diesel = load_lcfs_price_scenario("no_change")

diesel_tco_mile(p_gal, yr::Int) =
    p_gal / DIESEL_MPG + d_capital + d_rm + d_tires + d_common

# Avoided air-quality damage per displaced diesel mile comes from health_cost.jl
# (EPA COBRA by default; HEALTH_BASIS=envcost for the older per-mile factors).

# ─────────────────────────────────────────────────────────────────────────────
# SCC — piecewise-linear EPA SC-CO₂ points (2020 USD → 2024 USD)
# ─────────────────────────────────────────────────────────────────────────────
const SCC_DEFLATOR_2020_TO_2024 = 1.18
const SCC_YEARS_2020USD  = [2020, 2030, 2040, 2050, 2060, 2070, 2080]
const SCC_POINTS_2020USD = Dict(
    "1.5%" => [340.0, 380.0, 430.0, 480.0, 530.0, 570.0, 600.0],
    "2.0%" => [190.0, 230.0, 270.0, 310.0, 350.0, 380.0, 410.0],
    "2.5%" => [120.0, 140.0, 170.0, 200.0, 230.0, 260.0, 280.0],
)
function scc_per_ton(year; rate = "2.0%")
    xs = SCC_YEARS_2020USD;  ys = SCC_POINTS_2020USD[rate]
    yr = clamp(year, xs[1], xs[end])
    i  = min(searchsortedlast(xs, yr), length(xs) - 1)
    t  = (yr - xs[i]) / (xs[i+1] - xs[i])
    return (ys[i] + t * (ys[i+1] - ys[i])) * SCC_DEFLATOR_2020_TO_2024
end

# ─────────────────────────────────────────────────────────────────────────────
# Solar MC → median premium / health / CO₂ / miles flows
# ─────────────────────────────────────────────────────────────────────────────
solar_cfg(sched) = build_config(;
    h2_pathway_id = "current_mix", use_utilization_pricing = true, utilization_transport_cost = 1.0,
    use_lcfs = true, enable_45v = true, bus_demand_scenario = "growing", end_year = END_YEAR,
    use_truck_deployment_schedule = true, truck_deployment_schedule = sched,
    lcfs_price_schedule_dict = _lcfs_prices_diesel,
    expansion_pathway_id = "electrolysis", electrolysis_pricing_enabled = true,
    electrolysis_electricity_source = "solar", electrolyzer_capex_per_kw = 3000.0,
    solar_capex_per_kw = 1600.0, solar_capacity_factor = 0.25, solar_lifetime = 25,
    electrolyzer_learning_rate = 0.233, electrolyzer_stack_fraction = 0.60, bop_learning_rate = 0.04,
    solar_panel_learning_rate = 0.267, solar_panel_fraction = 0.80, solar_bop_learning_rate = 0.04,
)
function solar_flows(scenario_key)
    cfg = solar_cfg(load_truck_scenario(scenario_key))
    Random.seed!(SEED)
    raw = run_monte_carlo(cfg, N_RUNS)
    price_r = raw[1];  truck_r = Float64.(raw[3])
    # Per-kg policy credits to strip out: [9] LCFS, [17] HRI, [19] 45V. The delivered
    # price (raw[1]) is net of all three; add them back for the unsubsidized LCOH.
    price_unsub = price_r .+ raw[9] .+ raw[17] .+ raw[19]
    yrs = collect(cfg.start_year : cfg.end_year);  m = length(yrs)
    h2_yr = truck_r .* Float64(H2_PER_TRUCK_PER_DAY) .* Float64(OPERATING_DAYS_PER_YEAR)
    mi_yr = h2_yr .* miles_per_kg_h2
    tlo = [(h2_tco_unsub(price_unsub[r,k], yrs[k]) - diesel_tco_mile(DIESEL_P_LO, yrs[k])) * mi_yr[r,k] / 1e6 for r in 1:N_RUNS, k in 1:m]
    thi = [(h2_tco_unsub(price_unsub[r,k], yrs[k]) - diesel_tco_mile(DIESEL_P_HI, yrs[k])) * mi_yr[r,k] / 1e6 for r in 1:N_RUNS, k in 1:m]
    med_miles = [median(mi_yr[:, k]) for k in 1:m]
    return (years  = yrs,
            tcp_lo = [median(tlo[:, k]) for k in 1:m],
            tcp_hi = [median(thi[:, k]) for k in 1:m],
            co2_t  = med_miles .* CO2_G_PER_MILE ./ 1e6,
            health = [health_cost_per_mile(y) * med_miles[k] / 1e6 for (k, y) in enumerate(yrs)],
            miles  = med_miles,
            # NOx tonnage the COBRA proxy is being applied to, for the drift check
            nox_short_t = Dict(y => med_miles[k] * NOX_G_PER_MILE / 1e6 * METRIC_T_TO_SHORT_T
                               for (k, y) in enumerate(yrs)))
end

# Societal benefit (M USD/yr), net cost per mile, net-cost stream, discounting, cumulative.
benefit(F, rate)      = F.health .+ [scc_per_ton(F.years[k]; rate = rate) * F.co2_t[k] / 1e6 for k in eachindex(F.years)]
permile(F, tcp, rate) = (tcp .- benefit(F, rate)) .* 1e6 ./ F.miles
net_stream(F, dfield, rate) = getfield(F, dfield) .- benefit(F, rate)
discount_series(years, vals, r, t0) = [v / (1 + r)^(y - t0) for (y, v) in zip(years, vals)]
function cumulative_integral(years, vals)
    cum = zeros(length(vals));  s = 0.0
    for i in 2:length(vals)
        s += 0.5 * (vals[i-1] + vals[i]) * (years[i] - years[i-1])
        cum[i] = s
    end
    return cum
end

const INTEG_Y0, INTEG_Y1, DBASE = 2026, 2045, 2026
const SCC_LO_RATE, SCC_MID_RATE, SCC_HI_RATE = "2.5%", "2.0%", "1.5%"   # 2.5% ⇒ lowest SCC ⇒ highest net cost
const PV_RATE = 0.02                       # central present-value rate for the cumulative panels
const PV_RANGE = (0.0, 0.03)               # sensitivity reported as text, not as a nested band
const PV_RATES_ALL = (0.0, PV_RATE, 0.03)  # full appraisal-rate grid for the values file

# Diesel cases: (pump price, flows field, colour). Line style is deliberately
# NOT used here — colour carries the diesel price, so both series stay solid.
diesels = [(DIESEL_P_LO, :tcp_lo, C_NET_LO), (DIESEL_P_HI, :tcp_hi, C_NET_HI)]

function cum_trajectory(F, dfield, scc_rate, pv)
    idx = findall(y -> INTEG_Y0 <= y <= INTEG_Y1, F.years)
    yrs = F.years[idx]
    net = net_stream(F, dfield, scc_rate)[idx]
    dv  = discount_series(yrs, net, pv, DBASE)
    return Float64.(yrs), cumulative_integral(yrs, dv)
end

# ─────────────────────────────────────────────────────────────────────────────
# Crossing detection — reported in the values file, NOT drawn on the figures
# ─────────────────────────────────────────────────────────────────────────────
# The panels carry no break-even / payback markers: the crossings are read off
# the zero line directly, and the marker + year label added clutter without
# adding information. The numbers live in the values file for the captions.
"""
    crossing_year(years, vals; level = 0.0, downward = true)

Year at which `vals` crosses `level`, linearly interpolated between the
bracketing points. Returns `nothing` if the series never crosses inside the
window, so the caller reports no crossing rather than clamping to an endpoint —
which would invent a payback year that is not in the results.
"""
function crossing_year(years, vals; level = 0.0, downward = true)
    for i in 2:length(vals)
        a, b = vals[i-1] - level, vals[i] - level
        if (downward && a > 0 && b <= 0) || (!downward && a < 0 && b >= 0)
            return years[i-1] + (a / (a - b)) * (years[i] - years[i-1])
        end
    end
    return nothing
end


# ─────────────────────────────────────────────────────────────────────────────
# Run MC for both deployments
# ─────────────────────────────────────────────────────────────────────────────
# Opt-in cache of the median flow series, so layout work does not cost a fresh
# Monte Carlo (~10 min for both deployments). OFF by default: the cache key covers
# N_RUNS, SEED and the health basis, but cannot see edits to the model or the JSON
# configs, so the publication run must always recompute. Enable only while
# iterating on drawing code, and never to produce a final figure:
#   SCB_CACHE=1 julia --project=figures figures/fig_societal_cost_benefit.jl
# HEALTH_BASIS is in the key because the cached flows carry the `health` series:
# without it, switching basis on a warm cache would silently keep the old one.
const FLOW_CACHE = joinpath(OUT_DIR,
    "fig_societal_cost_benefit_flows_$(N_RUNS)_$(SEED)_$(HEALTH_BASIS).jls")
const USE_CACHE  = get(ENV, "SCB_CACHE", "0") == "1"

if USE_CACHE && isfile(FLOW_CACHE)
    @warn "Reusing cached flows — NOT valid if the model or configs changed" FLOW_CACHE
    F_demo, F_high = Serialization.deserialize(FLOW_CACHE)
else
    println("Running solar MC for limited_dep and high_dep ($(N_RUNS) runs each)…")
    global F_demo = solar_flows("limited_dep");  println("  limited_dep done")
    global F_high = solar_flows("high_dep");   println("  high_dep done")
    Serialization.serialize(FLOW_CACHE, (F_demo, F_high))
end
deploy_panels = [("Limited deployment", F_demo), ("High deployment", F_high)]

# Drift check against the stored COBRA runs, which were calibrated on limited_dep.
health_basis_banner(F_demo.nox_short_t)

const XT = year_ticks(INTEG_Y0, END_YEAR)

# ─────────────────────────────────────────────────────────────────────────────
# Shared y limits, computed from the data
# ─────────────────────────────────────────────────────────────────────────────
# Explicit rather than `linkyaxes!` on autolimits: linking did not take the union
# of the two panels' autolimits, so the wider SCC band of the high-deployment
# panel was clipped at the frame. Padded 5%.
padlim(v) = let p = 0.05 * (maximum(v) - minimum(v)); (minimum(v) - p, maximum(v) + p) end

cb_all = Float64[]      # figure A: premium curves + benefit band edges
for (_, F) in deploy_panels
    append!(cb_all, benefit(F, SCC_LO_RATE));  append!(cb_all, benefit(F, SCC_HI_RATE))
    append!(cb_all, F.tcp_lo);                 append!(cb_all, F.tcp_hi)
end
cb_lim = padlim(cb_all)

pm_all = Float64[]      # figure B row 2: per-mile band edges
for (_, F) in deploy_panels, (_, dfield, _) in diesels, r in (SCC_LO_RATE, SCC_HI_RATE)
    append!(pm_all, permile(F, getfield(F, dfield), r))
end
pm_lim = padlim(pm_all)

cum_all = Float64[]     # figure B row 1: shared across both deployments
for (_, F) in deploy_panels, (_, dfield, _) in diesels,
        r in (SCC_LO_RATE, SCC_MID_RATE, SCC_HI_RATE)
    append!(cum_all, cum_trajectory(F, dfield, r, PV_RATE)[2])
end
cum_lim = padlim(cum_all)

# =============================================================================
# FIGURE A — resource-cost premium vs. societal benefit
# =============================================================================
# Three lines + one band per panel. Net cost is the vertical gap between green
# and pink and gets no lines of its own: it is reported in per-mile and
# cumulative form in fig_net_societal_cost.
c_premium = C_SOLAR    # solar-electrolysis resource-cost premium (unsubsidized)
c_benefit = C_SCC      # societal benefit / SCC (pink)

function draw_costbenefit!(ax, F)
    y = F.years
    hlines!(ax, 0; color = C_ZERO_LINE, linewidth = 0.8, linestyle = :dot)

    # Societal benefit — pink band (SCC 1.5-2.5%) + 2.0% centre line. No diesel
    # dependence: avoided damages are a property of the miles displaced.
    bmid = benefit(F, SCC_MID_RATE)
    band!(ax, y, benefit(F, SCC_LO_RATE), benefit(F, SCC_HI_RATE); color = (c_benefit, 0.18))
    lines!(ax, y, bmid; color = c_benefit, linewidth = 1.6)

    # Resource-cost premium — green, solid/dash per diesel price. No SCC
    # dependence: it is a resource cost, before any carbon valuation.
    lines!(ax, y, F.tcp_lo; color = c_premium, linewidth = 1.7, linestyle = LS_DIESEL_LO)
    lines!(ax, y, F.tcp_hi; color = c_premium, linewidth = 1.3, linestyle = LS_DIESEL_HI)
end

figA = Figure(size = (W_DOUBLE, 80 * MM_TO_PT), figure_padding = (4, 12, 4, 4))
for (j, (name, F)) in enumerate(deploy_panels)
    ax = Axis(figA[1, j];
        title = "($(('a':'z')[j]))  $name",
        xlabel = "Year", ylabel = j == 1 ? "M USD yr⁻¹" : "",
        xticks = XT,
        yticklabelsvisible = j == 1,
        limits = ((INTEG_Y0, END_YEAR), cb_lim))
    draw_costbenefit!(ax, F)
end

# Grouped legend. `Vector{Vector}` is annotated explicitly so the call always
# hits Makie's grouped-legend method: an entry that is itself a vector of
# elements is drawn overlaid (line on top of its band), while a bare element is
# one entry on its own.
legA_groups = Vector{Vector}([
    [LineElement(color = c_premium, linewidth = 2, linestyle = LS_DIESEL_LO),
     LineElement(color = c_premium, linewidth = 2, linestyle = LS_DIESEL_HI)],
    [[LineElement(color = c_benefit, linewidth = 2), PolyElement(color = (c_benefit, 0.18))]],
])
legA_labels = Vector{Vector{String}}([
    [@sprintf("vs. diesel \$%.2f/gal", DIESEL_P_LO),
     @sprintf("vs. diesel \$%.2f/gal", DIESEL_P_HI)],
    ["SCC 2.0% (shading 1.5–2.5%)"],
])
Legend(figA[2, 1:2], legA_groups, legA_labels,
    ["Cost premium for hydrogen truck transportation", "Avoided damages (health + CO₂)"];
    orientation = :horizontal, nbanks = 2, titlefont = :bold,
    tellheight = true, tellwidth = false, colgap = 14, framevisible = true)

rowsize!(figA.layout, 1, Fixed(58 * MM_TO_PT))
colgap!(figA.layout, 24); rowgap!(figA.layout, 4)
resize_to_layout!(figA)
save_pub("fig_cost_premium_benefit", figA)

# =============================================================================
# FIGURE B — net societal cost: cumulative (payback) and per mile
# =============================================================================
# Colour = diesel price, band = SCC 1.5-2.5%, line = SCC 2.0%, column =
# deployment. Two lines + two bands per panel.

# Row 1 — cumulative net societal cost, LINEAR axis. The old symlog rendered the
# sign change as a near-vertical cliff and inflated the band near zero; the
# series spans only a few thousand M USD, so a linear axis costs nothing and
# shows the payback crossing honestly.
function draw_cumulative!(ax, F)
    hlines!(ax, 0; color = C_ZERO_LINE, linewidth = 0.8, linestyle = :dot)
    for (dp, dfield, col) in diesels
        yrs, c_mid = cum_trajectory(F, dfield, SCC_MID_RATE, PV_RATE)
        _,   c_lo  = cum_trajectory(F, dfield, SCC_LO_RATE,  PV_RATE)
        _,   c_hi  = cum_trajectory(F, dfield, SCC_HI_RATE,  PV_RATE)
        band!(ax, yrs, min.(c_lo, c_hi), max.(c_lo, c_hi); color = (col, 0.18))
        lines!(ax, yrs, c_mid; color = col, linewidth = 1.7)
    end
end

# Row 2 — net cost per mile (premium − benefit, per truck mile).
function draw_permile!(ax, F)
    y = F.years
    hlines!(ax, 0; color = C_ZERO_LINE, linewidth = 0.8, linestyle = :dot)
    for (dp, dfield, col) in diesels
        tcp = getfield(F, dfield)
        pm_mid = permile(F, tcp, SCC_MID_RATE)
        band!(ax, y, permile(F, tcp, SCC_LO_RATE), permile(F, tcp, SCC_HI_RATE);
              color = (col, 0.18))
        lines!(ax, y, pm_mid; color = col, linewidth = 1.7)
    end
end

figB = Figure(size = (W_DOUBLE, 150 * MM_TO_PT), figure_padding = (4, 12, 4, 4))
for (j, (name, F)) in enumerate(deploy_panels)
    ax1 = Axis(figB[1, j];
        title = "($(('a':'z')[j]))  $name",
        ylabel = j == 1 ? "Cumulative net cost (M USD)" : "",
        xticks = XT, xticklabelsvisible = false,
        yticklabelsvisible = j == 1,
        # Shared y across the row, so the two deployments are directly comparable
        # in magnitude: high deployment carries several times the absolute stakes
        # of the demo case, which is part of the result. The cost is that panel (a)
        # occupies less of its frame.
        limits = ((INTEG_Y0, INTEG_Y1), cum_lim))
    draw_cumulative!(ax1, F)
    j == 1 && text!(ax1, 0.02, 0.04;
        text = @sprintf("discounted to %d at %.0f%%/yr", DBASE, 100 * PV_RATE),
        space = :relative, align = (:left, :bottom), fontsize = FS_ANNOT - 1, color = :black)

    ax2 = Axis(figB[2, j];
        title = "($(('a':'z')[j+2]))  $name",
        xlabel = "Year", ylabel = j == 1 ? "Net cost (USD mi⁻¹)" : "",
        xticks = XT,
        yticklabelsvisible = j == 1,
        limits = ((INTEG_Y0, END_YEAR), pm_lim))
    draw_permile!(ax2, F)
end

legB_groups = Vector{Vector}([
    [[LineElement(color = C_NET_LO, linewidth = 2), PolyElement(color = (C_NET_LO, 0.18))],
     [LineElement(color = C_NET_HI, linewidth = 2), PolyElement(color = (C_NET_HI, 0.18))]],
])
legB_labels = Vector{Vector{String}}([
    [@sprintf("Diesel \$%.2f/gal", DIESEL_P_LO), @sprintf("Diesel \$%.2f/gal", DIESEL_P_HI)],
])
Legend(figB[3, 1:2], legB_groups, legB_labels,
    ["Net societal cost — line = SCC 2.0%, shading = SCC 1.5–2.5%"];
    orientation = :horizontal, nbanks = 1, titlefont = :bold,
    tellheight = true, tellwidth = false, colgap = 14, framevisible = true)

rowsize!(figB.layout, 1, Fixed(56 * MM_TO_PT))
rowsize!(figB.layout, 2, Fixed(56 * MM_TO_PT))
colgap!(figB.layout, 24); rowgap!(figB.layout, 5)
resize_to_layout!(figB)
save_pub("fig_net_societal_cost", figB)

# ─────────────────────────────────────────────────────────────────────────────
# Caption numbers — break-even and payback years, plus the PV sensitivity that
# is reported as text instead of as a second band on the cumulative panels.
# ─────────────────────────────────────────────────────────────────────────────

# All three SCC rates, ordered by discount rate (= descending SCC level), so the
# tables below read left-to-right from the earliest crossing to the latest.
const SCC_RATES_ALL = [SCC_HI_RATE, SCC_MID_RATE, SCC_LO_RATE]   # 1.5%, 2.0%, 2.5%

# `nothing` prints as "none" rather than as an endpoint: the series genuinely
# does not cross inside 2026–END_YEAR, and clamping would invent a year.
year_str(x) = isnothing(x) ? "none" : @sprintf("%.1f", x)

# The two crossings the manuscript quotes, as functions of (flows, diesel field,
# SCC rate) so `crossing_table!` can tabulate either one.
#   annual  — year the ANNUAL net cost hits zero: where the premium curve meets
#             the benefit curve in fig_cost_premium_benefit (and the per-mile
#             series crosses zero in fig_net_societal_cost row 2).
#   cumul   — year the CUMULATIVE net cost returns to zero (row 1), i.e. payback.
annual_zero(F, dfield, r) = crossing_year(F.years, getfield(F, dfield) .- benefit(F, r))
cumulative_zero(F, dfield, r) = crossing_year(cum_trajectory(F, dfield, r, PV_RATE)...)

"""
    crossing_table!(io, heading, note, crossfn)

Print one deployment × diesel-price × SCC-rate table of crossing years. Unlike
the per-deployment block above — which reports the 2.0% centre and treats the
other rates as a sensitivity — this tabulates all three rates symmetrically,
which is what the band in the figures actually spans.
"""
function crossing_table!(io, heading, note, crossfn)
    println(io, "\n$heading")
    println(io, "  $note")
    println(io, "-"^96)
    print(io, rpad("  Deployment", 20), rpad("Diesel", 13))
    for r in SCC_RATES_ALL
        print(io, lpad("SCC $r", 11))
    end
    println(io)
    for (name, F) in deploy_panels, (dp, dfield, _) in diesels
        print(io, rpad("  $name", 20), rpad(@sprintf("\$%.2f/gal", dp), 13))
        for r in SCC_RATES_ALL
            print(io, lpad(year_str(crossfn(F, dfield, r)), 11))
        end
        println(io)
    end
end

"""
    pv_tables!(io)

Appraisal-discount-rate sensitivity of the cumulative break-even year, across the
full PV_RATES_ALL x SCC x diesel grid — the PV counterpart of TABLE 2, which fixes
PV at the 2%/yr centre.

Two tables, because the crossing year alone hides how the choice acts. The rate
does not touch the *sign* of any annual flow, so TABLE 1 (annual net zero) is
PV-invariant by construction and is not repeated here; a higher rate only shrinks
the late-window surpluses relative to the early-window deficits they have to
repay, pushing the cumulative crossing later or off the end of the window. Where
that happens the year is "none" and says nothing about the margin, so TABLE 4
gives the terminal cumulative level: the debt still outstanding at END_YEAR.
"""
function pv_tables!(io)
    println(io, "\n\nTABLE 3 — CUMULATIVE BREAK-EVEN vs. APPRAISAL DISCOUNT RATE")
    println(io, @sprintf("  Year the cumulative net societal cost returns to zero, discounted to %d.", DBASE))
    println(io, "  Columns = present-value rate; TABLE 2 is the PV 2% column.")
    println(io, "-"^96)
    print(io, rpad("  Deployment", 20), rpad("Diesel", 13), rpad("SCC", 8))
    for pv in PV_RATES_ALL
        print(io, lpad(@sprintf("PV %.0f%%", 100 * pv), 11))
    end
    println(io)
    for (name, F) in deploy_panels, (dp, dfield, _) in diesels, r in SCC_RATES_ALL
        print(io, rpad("  $name", 20), rpad(@sprintf("\$%.2f/gal", dp), 13), rpad(r, 8))
        for pv in PV_RATES_ALL
            yrs, c = cum_trajectory(F, dfield, r, pv)
            print(io, lpad(year_str(crossing_year(yrs, c)), 11))
        end
        println(io)
    end

    println(io, "\nTABLE 4 — CUMULATIVE NET SOCIETAL COST AT $(INTEG_Y1)  (M USD, + = net cost)")
    println(io, @sprintf("  Terminal value of the same trajectories, discounted to %d. Quantifies the", DBASE))
    println(io, "  \"none\" cells in TABLE 3: how far the cumulative curve still is from zero.")
    println(io, "-"^96)
    print(io, rpad("  Deployment", 20), rpad("Diesel", 13), rpad("SCC", 8))
    for pv in PV_RATES_ALL
        print(io, lpad(@sprintf("PV %.0f%%", 100 * pv), 11))
    end
    println(io)
    for (name, F) in deploy_panels, (dp, dfield, _) in diesels, r in SCC_RATES_ALL
        print(io, rpad("  $name", 20), rpad(@sprintf("\$%.2f/gal", dp), 13), rpad(r, 8))
        for pv in PV_RATES_ALL
            _, c = cum_trajectory(F, dfield, r, pv)
            print(io, lpad(@sprintf("%+.1f", c[end]), 11))
        end
        println(io)
    end
end

function report(io)
    println(io, "SOCIETAL COST-BENEFIT — CROSSING YEARS  (median of $N_RUNS runs, solar electrolysis)")
    println(io, "="^96)
    println(io, "Health basis: $(health_basis_label())")
    for (name, F) in deploy_panels
        println(io, "\n$name")
        println(io, "-"^96)
        bmid = benefit(F, SCC_MID_RATE)
        for (dp, dfield, _) in diesels
            tcp = getfield(F, dfield)
            be  = crossing_year(F.years, tcp .- bmid)
            pm  = crossing_year(F.years, permile(F, tcp, SCC_MID_RATE))
            yrs, c_mid = cum_trajectory(F, dfield, SCC_MID_RATE, PV_RATE)
            pb  = crossing_year(yrs, c_mid)
            fmt(x) = isnothing(x) ? "  none" : @sprintf("%6.1f", x)
            @printf(io, "  diesel \$%.2f/gal   break-even %s   per-mile zero %s   payback %s\n",
                    dp, fmt(be), fmt(pm), fmt(pb))
            # PV sensitivity, reported rather than drawn (one band per panel).
            for pv in PV_RANGE
                _, c = cum_trajectory(F, dfield, SCC_MID_RATE, pv)
                @printf(io, "      payback at PV %.0f%%: %s\n", 100 * pv, fmt(crossing_year(yrs, c)))
            end
            # SCC sensitivity — the band's own edges.
            for r in (SCC_HI_RATE, SCC_LO_RATE)
                _, c = cum_trajectory(F, dfield, r, PV_RATE)
                @printf(io, "      payback at SCC %s: %s\n", r, fmt(crossing_year(yrs, c)))
            end
        end
    end
    println(io, "\n" * "="^96)
    crossing_table!(io, "TABLE 1 — ANNUAL NET ZERO",
        "Year the annual net societal cost reaches zero (premium = avoided damages).",
        annual_zero)
    crossing_table!(io, "TABLE 2 — CUMULATIVE BREAK-EVEN",
        @sprintf("Year the cumulative net societal cost returns to zero (discounted to %d at %.0f%%/yr).",
                 DBASE, 100 * PV_RATE),
        cumulative_zero)
    pv_tables!(io)
    println(io, "\n  \"none\" = no crossing on or before $(END_YEAR), the end of the assessment window.")
    println(io, "="^96)
end

report(stdout)
let path = joinpath(OUT_DIR, "fig_societal_cost_benefit_values.txt")
    open(report, path, "w")
    println("\nSaved → $path")
end
