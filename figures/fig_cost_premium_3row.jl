# =============================================================================
# FIGURE: SOLAR H2 vs DIESEL — 3-ROW SOCIETAL OVERVIEW  (2 deployments × 3 views)
# =============================================================================
# Combines the two existing figures into one 3-row × 2-column panel:
#   Columns : deployment            (limited_dep, high_dep)
#   Row 1   : TCO premium vs. societal benefit   (top row of fig_cost_premium_envcost_signed.jl)
#   Row 2   : cumulative net societal cost, symlog (from fig_cumulative_net_cost_log)
#   Row 3   : net cost per mile                  (bottom row of fig_cost_premium_envcost_signed.jl)
#
# Solar electrolysis pathway, diesel reference held flat in real terms (LCFS-on-
# diesel excluded — see SOCIAL ACCOUNTING below). All three rows are derived from
# the SAME solar Monte Carlo (same seed/configs/machinery), so the numbers
# reconcile across rows.
#
# SOCIAL ACCOUNTING: the premium here is the UNSUBSIDIZED resource-cost premium —
# the per-kg policy credits (LCFS + HRI + 45V) are added back to the LCOH and the
# HVIP truck voucher is removed from the truck capital. Policy support is a
# transfer, not a resource cost, so it is excluded from this social comparison;
# the carbon externality enters explicitly through the SCC. The dollar value of
# the stripped-out support is shown, split by funder, in fig_policy_expenditure.jl.
#
# Run from project root:
#   julia --project figures/fig_cost_premium_3row.jl
# =============================================================================

using CairoMakie
using Statistics
using JSON
using Random
using LinearAlgebra
using Printf

cd(joinpath(@__DIR__, ".."))
include("../hydrogen truck deployment.jl")
include("config_defaults.jl")
include("pub_theme.jl")

const N_RUNS  = 1000
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
function truck_net_cost_prem(year::Int)
    α  = log(1 / (1 - learning_rate)) / log(2)
    fc = truck_fuelcell_cost * (fleet_stock_prem(year) / fleet_stock_prem(ref_year_truck))^(-α)
    return truck_platform_cost + fc - truck_subsidy_yr_prem(year)
end
h2_rm_prem(yr::Int) = d_rm * (rm_mult + (1.0 - rm_mult) * clamp((yr - 2026) / 9.0, 0.0, 1.0))
function h2_capital_prem(year::Int)
    ann = calculate_annuity_factor(7, DISCOUNT_RATE)
    return truck_net_cost_prem(year) * ann /
           (Float64(H2_PER_TRUCK_PER_DAY) * Float64(OPERATING_DAYS_PER_YEAR) * miles_per_kg_h2)
end
h2_tco_prem(lcoh::Float64, yr::Int) =
    lcoh / miles_per_kg_h2 + h2_capital_prem(yr) + h2_rm_prem(yr) + h2_tires_val + d_common

# Unsubsidized (resource-cost) variant: gross truck capital (no HVIP voucher) and
# an LCOH with the per-kg policy credits (LCFS + HRI + 45V) added back. Used for
# the social cost-benefit comparison, where subsidies are transfers, not resource
# costs, and the carbon externality is counted explicitly via the SCC. See
# fig_policy_expenditure.jl for the support that is stripped out here.
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

const ENV_COST_DELTA_USD_PER_MILE = Dict{Int,Float64}(
    2015=>0.0480283801,2016=>0.0470866472,2017=>0.0461633796,2018=>0.0452582153,
    2019=>0.0443707993,2020=>0.0435007836,2021=>0.0426478271,2022=>0.0418115952,
    2023=>0.0409917600,2024=>0.0401880000,2025=>0.0394000000,2026=>0.0386274510,
    2027=>0.0378700500,2028=>0.0371275000,2029=>0.0363995098,2030=>0.0356857939,
    2031=>0.0349860725,2032=>0.0343000710,2033=>0.0336275206,2034=>0.0329681575,
    2035=>0.0323217230,2036=>0.0316879637,2037=>0.0310666311,2038=>0.0304574815,
    2039=>0.0298602760,2040=>0.0292747804,2041=>0.0287007651,2042=>0.0281380050,
    2043=>0.0275862794,2044=>0.0270453719,2045=>0.0265150705,2046=>0.0259951672,
    2047=>0.0254854580,2048=>0.0249857432,2049=>0.0244958266,2050=>0.0240155163,
)

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
            health = [ENV_COST_DELTA_USD_PER_MILE[y] * med_miles[k] / 1e6 for (k, y) in enumerate(yrs)],
            miles  = med_miles)
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
SCC_RATES = ["1.5%", "2.0%", "2.5%"]
PV_RATES  = (0.0, 0.02, 0.03)
diesels   = [(DIESEL_P_LO, :tcp_lo, :solid), (DIESEL_P_HI, :tcp_hi, :dash)]

function cum_trajectory(F, dfield, scc_rate, pv)
    idx = findall(y -> INTEG_Y0 <= y <= INTEG_Y1, F.years)
    yrs = F.years[idx]
    net = net_stream(F, dfield, scc_rate)[idx]
    dv  = discount_series(yrs, net, pv, DBASE)
    return Float64.(yrs), cumulative_integral(yrs, dv)
end

# ─────────────────────────────────────────────────────────────────────────────
# Run MC for both deployments
# ─────────────────────────────────────────────────────────────────────────────
println("Running solar MC for limited_dep and high_dep ($(N_RUNS) runs each)…")
F_demo = solar_flows("limited_dep");  println("  limited_dep done")
F_high = solar_flows("high_dep");   println("  high_dep done")
deploy_panels = [("Limited deployment", F_demo), ("High deployment", F_high)]

# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────
c_premium  = C_SOLAR     # solar-electrolysis resource-cost premium (unsubsidized)
c_benefit  = C_SCC       # societal benefit / SCC (pink) — shown as a band
scc_colors = SCC_SHADES  # discount-rate pink ramp (light = 2.5%, dark = 1.5%)

# Net total cost is its own (neutral) colour family across all three rows — NOT
# coloured by SCC. The SCC discount rate is shown as a grey ramp (dark 1.5% →
# light 2.5%) and the diesel pump price as the line style (solid $4.80 / dash
# $5.80). The grey ramp mirrors the SCC pink ramp's light→dark direction.
nc_shades = Dict("1.5%" => colorant"#1A1A1A", "2.0%" => colorant"#777777", "2.5%" => colorant"#B5B5B5")

# Legends are split into two titled sections so the two orthogonal encodings read
# cleanly: COLOUR = which series, LINE STYLE = diesel price (shown once, not
# repeated per series). Net-cost SCC rates are the grey ramp.
nc_series_elems  = [LineElement(color = nc_shades["1.5%"], linewidth = 2),
                    LineElement(color = nc_shades["2.0%"], linewidth = 2),
                    LineElement(color = nc_shades["2.5%"], linewidth = 2)]
nc_series_labels = ["Net cost — SCC 1.5%", "Net cost — SCC 2.0%", "Net cost — SCC 2.5%"]

diesel_style_elems  = [LineElement(color = :black, linewidth = 2, linestyle = :solid),
                       LineElement(color = :black, linewidth = 2, linestyle = :dash)]
diesel_style_labels = [@sprintf("\$%.2f/gal", DIESEL_P_LO), @sprintf("\$%.2f/gal", DIESEL_P_HI)]
const STYLE_TITLE = "Diesel price (line style)"
const SERIES_TITLE = "Series (colour)"

# Row 1 — solar TCO premium (green) vs. societal benefit (pink BAND, SCC 1.5–2.5%),
# and net total cost as grey-ramp LINES (SCC rate = grey shade, diesel = style).
soc_leg_groups = [
    [LineElement(color = c_premium, linewidth = 2),
     PolyElement(color = (c_benefit, 0.15)),
     LineElement(color = c_benefit, linewidth = 2),
     nc_series_elems...],
    diesel_style_elems,
]
soc_leg_labels = [
    ["Solar resource-cost premium", "Societal benefit (SCC 1.5–2.5%)", "Societal benefit (SCC 2%)",
     nc_series_labels...],
    diesel_style_labels,
]
function draw_societal!(ax, F; showlegend = false)
    y = F.years
    blo = benefit(F, "2.5%"); bhi = benefit(F, "1.5%")
    hlines!(ax, 0; color = (:black, 0.4), linewidth = 0.8, linestyle = :dot)

    # Societal benefit — pink band (SCC 1.5–2.5%) + 2% centre line
    band!(ax, y, blo, bhi; color = (c_benefit, 0.15))
    lines!(ax, y, benefit(F, "2.0%"); color = c_benefit, linewidth = 1.3)

    # Solar TCO premium — green, solid/dash per diesel price (no SCC dependence)
    lines!(ax, y, F.tcp_lo; color = c_premium, linewidth = 1.5)
    lines!(ax, y, F.tcp_hi; color = c_premium, linewidth = 1.0, linestyle = :dash)

    # Net total cost (premium − benefit) — grey SCC ramp × diesel line style
    for (tcp, ls, lw) in ((F.tcp_lo, :solid, 1.4), (F.tcp_hi, :dash, 1.0))
        for rate in ("1.5%", "2.0%", "2.5%")
            lines!(ax, y, tcp .- benefit(F, rate); color = nc_shades[rate],
                   linewidth = lw, linestyle = ls)
        end
    end

    showlegend && axislegend(ax, soc_leg_groups, soc_leg_labels, [SERIES_TITLE, STYLE_TITLE];
                             position = :lt, rowgap = 0, labelsize = 4.5, titlesize = 5,
                             framevisible = true, titlegap = 1, groupgap = 4)
end

# Row 2 — cumulative net societal cost over time (drawn on a symlog axis).
# Same net-cost language as rows 1/3: SCC rate = grey shade, diesel = line style.
cum_leg_groups = [
    [nc_series_elems...; PolyElement(color = (:gray, 0.18))],
    diesel_style_elems,
]
cum_leg_labels = [
    [nc_series_labels...; "PV 0–3% band (line = 2%)"],
    diesel_style_labels,
]
function draw_cumlog!(ax, F; showlegend = false)
    hlines!(ax, 0; color = (:black, 0.45), linewidth = 0.8, linestyle = :dot)
    for rate in SCC_RATES, (dp, dfield, ls) in diesels
        c = nc_shades[rate]
        yrs, c0 = cum_trajectory(F, dfield, rate, PV_RATES[1])
        _,   c2 = cum_trajectory(F, dfield, rate, PV_RATES[2])
        _,   c3 = cum_trajectory(F, dfield, rate, PV_RATES[3])
        band!(ax, yrs, min.(c0, c3), max.(c0, c3); color = (c, 0.12))
        lines!(ax, yrs, c2; color = c, linewidth = ls == :solid ? 1.6 : 1.2, linestyle = ls)
    end
    showlegend && axislegend(ax, cum_leg_groups, cum_leg_labels, [SERIES_TITLE, STYLE_TITLE];
                             position = :lb, rowgap = 0, labelsize = 4.5, titlesize = 5,
                             framevisible = true, titlegap = 1, groupgap = 4, patchsize = (10, 5))
end

# Row 3 — net cost per mile, SCC-range band per diesel.
function draw_permile!(ax, F; showlegend = false)
    y = F.years
    hlines!(ax, 0; color = (:black, 0.4), linewidth = 0.8, linestyle = :dot)
    # Net cost per mile — same net-cost language: grey SCC ramp × diesel line style
    for (tcp, ls, lw) in ((F.tcp_lo, :solid, 1.4), (F.tcp_hi, :dash, 1.0))
        for rate in ("1.5%", "2.0%", "2.5%")
            lines!(ax, y, permile(F, tcp, rate); color = nc_shades[rate],
                   linewidth = lw, linestyle = ls)
        end
    end
    showlegend && axislegend(ax, [nc_series_elems, diesel_style_elems],
                             [nc_series_labels, diesel_style_labels], [SERIES_TITLE, STYLE_TITLE];
                             position = :rt, rowgap = 0, labelsize = 4.5, titlesize = 5,
                             framevisible = true, titlegap = 1, groupgap = 4)
end

# ─────────────────────────────────────────────────────────────────────────────
# Shared axis limits
# ─────────────────────────────────────────────────────────────────────────────
# Row 2 symlog: shared y across panels.
cum_all = Float64[]
for (_, F) in deploy_panels, (_, dfield, _) in diesels, rate in SCC_RATES, pv in PV_RATES
    append!(cum_all, cum_trajectory(F, dfield, rate, pv)[2])
end
cpad = 0.05 * (maximum(cum_all) - minimum(cum_all))
cum_ylim = (minimum(cum_all) - cpad, maximum(cum_all) + cpad)
const SYMLOG_THRESH = 5.0
symlog_ticks = ([-5000.0, -1000.0, -100.0, -10.0, 0.0, 10.0, 100.0, 1000.0, 5000.0],
                ["−5000", "−1000", "−100", "−10", "0", "10", "100", "1000", "5000"])

# Row 1 societal: shared y across panels (everything drawn in draw_societal!).
r1_all = Float64[]
for (_, F) in deploy_panels
    bhi = benefit(F, "1.5%"); blo = benefit(F, "2.5%"); bmd = benefit(F, "2.0%")
    append!(r1_all, bhi); append!(r1_all, blo); append!(r1_all, bmd)
    append!(r1_all, F.tcp_lo); append!(r1_all, F.tcp_hi)
    append!(r1_all, F.tcp_lo .- bhi); append!(r1_all, F.tcp_lo .- blo)
    append!(r1_all, F.tcp_hi .- bhi); append!(r1_all, F.tcp_hi .- blo)
end
r1_pad = 0.05 * (maximum(r1_all) - minimum(r1_all))
r1_lim = (minimum(r1_all) - r1_pad, maximum(r1_all) + r1_pad)

# Row 3 per-mile: shared y across panels.
pm_all = Float64[]
for (_, F) in deploy_panels, tcp in (F.tcp_lo, F.tcp_hi), rate in SCC_RATES
    append!(pm_all, permile(F, tcp, rate))
end
pm_lim = (minimum(pm_all) - 0.08 * abs(minimum(pm_all)), maximum(pm_all) * 1.05)

# ─────────────────────────────────────────────────────────────────────────────
# Figure — 3 rows × 2 columns
# ─────────────────────────────────────────────────────────────────────────────
fig = Figure(size = (W_DOUBLE, 235 * MM_TO_PT))
L = ['a', 'b', 'c', 'd', 'e', 'f']

# Row 1 — societal benefit (independent y per panel, y-ticklabels on both).
for (j, (name, F)) in enumerate(deploy_panels)
    ax = Axis(fig[1, j]; title = "($(L[j]))  $name — resource cost vs. societal benefit",
        titlesize = 8,
        ylabel = j == 1 ? "M USD yr⁻¹" : "",
        xticks = [2026, 2030, 2035, 2040, 2045], xticklabelsvisible = false,
        yticklabelsvisible = j == 1,
        limits = ((INTEG_Y0, END_YEAR), r1_lim))
    draw_societal!(ax, F; showlegend = (j == 1))
end

# Row 2 — cumulative net societal cost (symlog, shared y).
for (j, (name, F)) in enumerate(deploy_panels)
    ax = Axis(fig[2, j]; title = "($(L[j+2]))  $name — cumulative net societal cost",
        titlesize = 8,
        ylabel = j == 1 ? "Cumulative net cost (M USD)" : "",
        xticks = [2026, 2030, 2035, 2040, 2045], xticklabelsvisible = false,
        yscale = Makie.Symlog10(SYMLOG_THRESH), yticks = symlog_ticks,
        yticklabelsvisible = j == 1,
        limits = ((INTEG_Y0, INTEG_Y1), cum_ylim))
    draw_cumlog!(ax, F; showlegend = (j == 1))
    j == 1 && text!(ax, INTEG_Y0 + 0.4, cum_ylim[1]; text = "symlog: linear within ±$(Int(SYMLOG_THRESH)) M USD",
                    align = (:left, :bottom), fontsize = 5, color = :gray40)
end

# Row 3 — net cost per mile (shared y).
for (j, (name, F)) in enumerate(deploy_panels)
    ax = Axis(fig[3, j]; title = "($(L[j+4]))  $name — net cost per mile",
        titlesize = 8,
        xlabel = "Year", ylabel = j == 1 ? "Net cost (USD mi⁻¹)" : "",
        xticklabelrotation = π/4, xticks = [2026, 2030, 2035, 2040, 2045],
        yticklabelsvisible = j == 1,
        limits = ((INTEG_Y0, END_YEAR), pm_lim))
    draw_permile!(ax, F; showlegend = (j == 1))
end

rowgap!(fig.layout, 6); colgap!(fig.layout, 6)
resize_to_layout!(fig)
save_pub("fig_cost_premium_3row", fig)
