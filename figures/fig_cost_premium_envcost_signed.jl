# =============================================================================
# FIGURE: AGGREGATE COST PREMIUM — H2 vs. DIESEL (M USD / YEAR)
#         [ENV-COST VARIANT, SIGNED y-AXIS]
# =============================================================================
# Variant of fig_cost_premium_envcost.jl. TWO differences only:
#
#   1. Panels (b) and (c) no longer clamp the y-axis at 0 — the negative
#      (net-benefit) side of the black "net cost" lines is now visible.
#   2. Panel (b): the time integral of the two black net-cost lines over
#      2026–2045 is computed (trapezoidal, undiscounted M USD), printed to the
#      console, and annotated on the panel. Comment out the `text!(...)` block
#      flagged below to remove the in-figure annotation.
#
# Everything else is identical to fig_cost_premium_envcost.jl.
#
# Run from project root:
#   julia --project figures/fig_cost_premium_envcost_signed.jl
# =============================================================================

using CairoMakie
using Statistics
using JSON
using Random
using LinearAlgebra
using Printf
using Dates

cd(joinpath(@__DIR__, ".."))
include("../hydrogen truck deployment.jl")
include("config_defaults.jl")

const N_RUNS  = 1000
const SEED    = 42
include(joinpath(@__DIR__, "fig_routing.jl"))   # figure → manuscript/appendix/old routing
const OUT_DIR = joinpath(@__DIR__, "out")
mkpath(OUT_DIR)

update_theme!(Legend = (
    rowgap        = -2,
    patchsize     = (12, 8),
    patchlabelgap = 4,
    padding       = (4, 4, 2, 2),
))

# ─────────────────────────────────────────────────────────────────────────────
# Load TCO parameters (mirrors fig_tco_comparison.jl)
# ─────────────────────────────────────────────────────────────────────────────
tco_raw = JSON.parsefile(joinpath(@__DIR__, "..", "config", "tco_config.json"))
diesel  = tco_raw["diesel_tco_usd_per_mile"]
ht      = tco_raw["hydrogen_truck"]
ident   = tco_raw["identical_cost_categories"]

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

_sub_sched_prem = Dict{Int,Float64}(
    d["year"] => Float64(d["subsidy_usd"])
    for d in ht["subsidy_schedule"]
)
truck_subsidy_yr_prem(year::Int) =
    _sub_sched_prem[maximum(filter(y -> y <= year, keys(_sub_sched_prem)))]

# Cubic OLS fit for global H2 fleet stock (Wright's Law driver)
let
    obs = ht["learning"]["observed_fleet_stock"]
    ys  = Float64.([Int(d["year"])  for d in obs])
    vs  = Float64.([Int(d["trucks"]) for d in obs])
    ts  = ys .- 2019.0
    V   = hcat(ts.^3, ts.^2, ts, ones(length(ts)))
    θ   = V \ vs
    global _prem_a, _prem_b, _prem_c, _prem_d = θ[1], θ[2], θ[3], θ[4]
end
fleet_stock_prem(yr::Int) =
    max(1.0, _prem_a*(yr-2019)^3 + _prem_b*(yr-2019)^2 + _prem_c*(yr-2019) + _prem_d)

function truck_net_cost_prem(year::Int)
    α = log(1 / (1 - learning_rate)) / log(2)
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

# ─────────────────────────────────────────────────────────────────────────────
# Diesel price reference constants
# ─────────────────────────────────────────────────────────────────────────────
const DIESEL_MPG  = TCO_DIESEL_MPG   # miles/gallon, from config/tco_config.json
const DIESEL_P_LO = TCO_DIESEL_P_LO  # $/gal, from config/tco_config.json
const DIESEL_P_HI = TCO_DIESEL_P_HI  # $/gal, from config/tco_config.json

# Tailpipe emission factors for a Class 8 diesel truck (g/mile)
const NOX_G_PER_MILE  = 0.06    # NOx   (g/mile)
const PM25_G_PER_MILE = 0.0015  # PM2.5 (g/mile)

# Unit conversion for COBRA (which expects US short tons, not metric tonnes)
const METRIC_T_TO_SHORT_T = 1.10231

# California diesel blend carbon intensity — same blend fractions used for LCFS calculation
# (66% renewable diesel CI=43.74, 6% biodiesel CI=38.49, 28% fossil diesel CI=90 gCO₂e/MJ)
const DIESEL_MJ_PER_GAL = 128.45  # LHV energy content (MJ/gallon)
const CA_BLEND_CI = 0.66 * 43.74 + 0.06 * 38.49 + (1.0 - 0.66 - 0.06) * 90.0  # gCO₂e/MJ
# Lifecycle CO₂e emission factor (g CO₂e / mile) — diesel side only
const CO2_G_PER_MILE = CA_BLEND_CI * DIESEL_MJ_PER_GAL / DIESEL_MPG
const MJ_H2_PER_KG   = 120.0  # LHV energy content of hydrogen (MJ/kg)

# Diesel-equivalent H₂ price for equal fuel cost per mile ($/kg)
diesel_equiv_kg(p_gal) = p_gal * miles_per_kg_h2 / DIESEL_MPG

# ── Diesel LCFS adjustment (anchored at 2026; mirrors fig_tco_comparison.jl) ──
# The CA diesel blend CI is CA_BLEND_CI (gCO₂e/MJ); the LCFS benchmark CI declines
# per lcfs_config.json, so the blend swings from generating LCFS credits to
# deficits over time. diesel_lcfs_per_mile returns that value ($/mile, + = credit
# lowering cost, − = deficit raising cost). It is ANCHORED at 2026: only the change
# vs 2026 adjusts the $4.80/$5.80 pump prices, which are taken to already embed
# 2026 LCFS conditions (same convention as fig_tco_comparison.jl).
const _benchmark_ci = Dict{Int,Float64}(
    Int(d["year"]) => Float64(d["ci"])
    for d in JSON.parsefile(joinpath(@__DIR__, "..", "config", "lcfs_config.json"))["diesel_ci_schedule"]["schedule"]
)
function diesel_lcfs_per_mile(year::Int, ps::Dict{Int,Float64})
    ci_std  = get(_benchmark_ci, year, _benchmark_ci[maximum(filter(y -> y <= year, keys(_benchmark_ci)))])
    lcfs_p  = get(ps, year, ps[maximum(filter(y -> y <= year, keys(ps)))])
    credits = (ci_std - CA_BLEND_CI) / 1e6 * DIESEL_MJ_PER_GAL   # tCO₂e/gallon (can be negative)
    return credits * lcfs_p / DIESEL_MPG                          # $/mile
end
const _lcfs_prices_diesel = load_lcfs_price_scenario("no_change")
const _lcfs_2026_per_mile = diesel_lcfs_per_mile(2026, _lcfs_prices_diesel)

# Full diesel TCO at a given pump price ($/mile), LCFS-adjusted by year (anchored 2026)
diesel_tco_mile(p_gal, yr::Int) =
    p_gal / DIESEL_MPG + (_lcfs_2026_per_mile - diesel_lcfs_per_mile(yr, _lcfs_prices_diesel)) +
    d_capital + d_rm + d_tires + d_common

# ─────────────────────────────────────────────────────────────────────────────
# Environmental health-cost factors  —  "Env Cost Factors 20apr LF.xlsx"
# ─────────────────────────────────────────────────────────────────────────────
# Per-mile avoided air-quality damage (NOx + PM2.5 + NH3) for an HDV switching
# from diesel to ZEV, by year:   Δ = column E (diesel HDV) − column K (ZEV HDV).
# Values are USD / mile (sheet "Env cost parameters", column B = year).
# Source values transcribed directly from the workbook so this script has no
# Excel-read dependency; re-derive with the workbook if the sheet is updated.
const ENV_COST_DELTA_USD_PER_MILE = Dict{Int,Float64}(
    2015 => 0.0480283801,
    2016 => 0.0470866472,
    2017 => 0.0461633796,
    2018 => 0.0452582153,
    2019 => 0.0443707993,
    2020 => 0.0435007836,
    2021 => 0.0426478271,
    2022 => 0.0418115952,
    2023 => 0.0409917600,
    2024 => 0.0401880000,
    2025 => 0.0394000000,
    2026 => 0.0386274510,
    2027 => 0.0378700500,
    2028 => 0.0371275000,
    2029 => 0.0363995098,
    2030 => 0.0356857939,
    2031 => 0.0349860725,
    2032 => 0.0343000710,
    2033 => 0.0336275206,
    2034 => 0.0329681575,
    2035 => 0.0323217230,
    2036 => 0.0316879637,
    2037 => 0.0310666311,
    2038 => 0.0304574815,
    2039 => 0.0298602760,
    2040 => 0.0292747804,
    2041 => 0.0287007651,
    2042 => 0.0281380050,
    2043 => 0.0275862794,
    2044 => 0.0270453719,
    2045 => 0.0265150705,
    2046 => 0.0259951672,
    2047 => 0.0254854580,
    2048 => 0.0249857432,
    2049 => 0.0244958266,
    2050 => 0.0240155163,
)

# ─────────────────────────────────────────────────────────────────────────────
# Pathway and scenario setup (mirrors fig_tco_comparison.jl)
# ─────────────────────────────────────────────────────────────────────────────
pathways = [
    (label = "SMR (current mix)",    color = colorant"#4472C4", kind = :smr,  h2_ci = 21.23),
    (label = "Electrolysis — grid",  color = colorant"#ED7D31", kind = :grid, h2_ci =  133.6),
    (label = "Electrolysis — solar", color = colorant"#70AD47", kind = :solar, h2_ci = 0.0),
]

truck_sched = load_truck_scenario("limited_dep")
lcfs_prices = load_lcfs_price_scenario("no_change")

function make_config_prem(kind)
    base = (
        h2_pathway_id                 = "current_mix",
        use_utilization_pricing       = true,
        utilization_transport_cost    = 1.0,
        use_lcfs                      = true,
        enable_45v                    = true,
        bus_demand_scenario           = "growing",
        end_year                      = END_YEAR,
        use_truck_deployment_schedule = true,
        truck_deployment_schedule     = truck_sched,
        lcfs_price_schedule_dict      = lcfs_prices,
    )
    if kind == :smr
        build_config(; base...,
            expansion_pathway_id         = "current_mix",
            electrolysis_pricing_enabled = false,
        )
    elseif kind == :grid
        build_config(; base...,
            expansion_pathway_id            = "electrolysis",
            electrolysis_pricing_enabled    = true,
            electrolysis_electricity_source = "grid",
            electrolyzer_capex_per_kw       = 3000.0,
            electricity_cost_per_kwh        = 0.2,
            electrolyzer_learning_rate      = 0.233,
            electrolyzer_stack_fraction     = 0.60,
            bop_learning_rate               = 0.04,
        )
    else  # :solar
        build_config(; base...,
            expansion_pathway_id            = "electrolysis",
            electrolysis_pricing_enabled    = true,
            electrolysis_electricity_source = "solar",
            electrolyzer_capex_per_kw       = 3000.0,
            solar_capex_per_kw              = 1600.0,
            solar_capacity_factor           = 0.25,
            solar_lifetime                  = 25,
            electrolyzer_learning_rate      = 0.233,
            electrolyzer_stack_fraction     = 0.60,
            bop_learning_rate               = 0.04,
            solar_panel_learning_rate       = 0.267,
            solar_panel_fraction            = 0.80,
            solar_bop_learning_rate         = 0.04,
        )
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Run Monte Carlo and compute premiums
# ─────────────────────────────────────────────────────────────────────────────
println("Running $(length(pathways)) pathway simulations ($(N_RUNS) runs each)…")

results = map(pathways) do pw
    cfg    = make_config_prem(pw.kind)
    Random.seed!(SEED)
    raw    = run_monte_carlo(cfg, N_RUNS)
    price_r = raw[1]            # N_RUNS × n_years — LCOH ($/kg)
    truck_r = Float64.(raw[3])  # N_RUNS × n_years — trucks deployed
    years   = collect(cfg.start_year : cfg.end_year)
    n       = length(years)
    print("  $(pw.label)…")

    # Fleet-wide annual H₂ demand (kg) and mileage (miles) — no uptime factor;
    # consistent with how h2_tco_prem computes fuel cost per active truck.
    h2_yr = truck_r .* Float64(H2_PER_TRUCK_PER_DAY) .* Float64(OPERATING_DAYS_PER_YEAR)
    mi_yr = h2_yr .* miles_per_kg_h2

    # H₂ fuel cost premium ($M/yr): (LCOH − diesel-equiv price) × kg H₂
    fp_lo = (price_r .- diesel_equiv_kg(DIESEL_P_LO)) .* h2_yr ./ 1e6
    fp_hi = (price_r .- diesel_equiv_kg(DIESEL_P_HI)) .* h2_yr ./ 1e6

    # Total TCO premium ($M/yr): (H₂ TCO − diesel TCO) × miles driven
    tcp_lo = [( h2_tco_prem(price_r[run, k], years[k]) - diesel_tco_mile(DIESEL_P_LO, years[k]) ) *
              mi_yr[run, k] / 1e6
              for run in 1:N_RUNS, k in 1:n]
    tcp_hi = [( h2_tco_prem(price_r[run, k], years[k]) - diesel_tco_mile(DIESEL_P_HI, years[k]) ) *
              mi_yr[run, k] / 1e6
              for run in 1:N_RUNS, k in 1:n]

    # Avoided emissions: median fleet miles × emission factor (g/mile → tonnes/year)
    med_miles_yr = [median(mi_yr[:, k]) for k in 1:n]

    # Net CO₂e avoided per mile: diesel lifecycle emissions minus H₂ production emissions.
    # H₂ CO₂e per mile = CI_H₂ (gCO₂e/MJ) × MJ/kg ÷ (miles/kg)
    h2_co2e_g_per_mile  = pw.h2_ci * MJ_H2_PER_KG / miles_per_kg_h2
    net_co2e_g_per_mile = CO2_G_PER_MILE - h2_co2e_g_per_mile

    r = (
        years          = years,
        fp_med_lo      = [median(fp_lo[:, k])            for k in 1:n],
        fp_p25_lo      = [quantile(fp_lo[:, k], 0.25)   for k in 1:n],
        fp_p75_lo      = [quantile(fp_lo[:, k], 0.75)   for k in 1:n],
        fp_med_hi      = [median(fp_hi[:, k])             for k in 1:n],
        tcp_med_lo     = [median(tcp_lo[:, k])            for k in 1:n],
        tcp_p25_lo     = [quantile(tcp_lo[:, k], 0.25)  for k in 1:n],
        tcp_p75_lo     = [quantile(tcp_lo[:, k], 0.75)  for k in 1:n],
        tcp_med_hi     = [median(tcp_hi[:, k])            for k in 1:n],
        med_miles_yr   = med_miles_yr,                                 # median fleet miles/yr
        co2_avoided_kt = med_miles_yr .* net_co2e_g_per_mile ./ 1e9, # kt CO₂e/yr (g→kt: ÷1e9)
    )
    println(" done")
    r
end

println("All simulations complete.\n")

# ─────────────────────────────────────────────────────────────────────────────
# Air-quality health benefit from env-cost factors (replaces COBRA)
# ─────────────────────────────────────────────────────────────────────────────
# Fleet-wide health benefit (M USD/yr) = Δ(year) × median fleet miles / 1e6,
# where Δ(year) = E − K from the env-cost workbook (avoided NOx+PM2.5+NH3
# damage per mile). Pathway-independent → use results[1] median miles.
let r_ref = results[1]
    global health_fitted_yrs = collect(r_ref.years)
    global health_fitted_vals = [
        get(ENV_COST_DELTA_USD_PER_MILE, yr) do
            error("No env-cost factor for year $yr in ENV_COST_DELTA_USD_PER_MILE")
        end * r_ref.med_miles_yr[k] / 1e6
        for (k, yr) in enumerate(r_ref.years)
    ]

    println("AIR-QUALITY HEALTH BENEFIT (env-cost factors, limited_dep fleet)")
    println("="^60)
    @printf("  %-6s  %16s  %14s  %18s\n",
            "Year", "Δ (USD/mile)", "Mfleet (Mmi)", "Benefit (M USD/yr)")
    println("  " * "-"^58)
    for (k, yr) in enumerate(r_ref.years)
        @printf("  %-6d  %16.5f  %14.1f  %18.2f\n",
                yr, ENV_COST_DELTA_USD_PER_MILE[yr],
                r_ref.med_miles_yr[k] / 1e6, health_fitted_vals[k])
    end
    println()
end

# Print cost-premium summary
years_p = results[1].years
println("COST PREMIUM SUMMARY  (median, M USD/yr, vs diesel \$$(DIESEL_P_LO)/gal)")
println("="^70)
@printf("%-24s", "Pathway")
for pw in pathways
    @printf("  %10s", pw.label[1:min(10,length(pw.label))])
end
println()
println("-"^70)
for (k, yr) in enumerate(years_p)
    @printf("%-24s", yr)
    for r in results
        @printf("  %+8.1f M", r.tcp_med_lo[k])
    end
    println()
end

# ─────────────────────────────────────────────────────────────────────────────
# Social cost of carbon
# ─────────────────────────────────────────────────────────────────────────────
# Year-dependent SCC: piecewise-linear interpolation of the EPA (2023) SC-CO₂
# points, per near-term Ramsey discount rate, converted to 2024 USD. Passes
# exactly through the published decade values (no fit error) for 1.5/2/2.5%.
# Model horizon (2026–2045) is interior. Deflator ≈1.18. Figure uses 2% rate
# (default); the sensitivity table below sweeps all three. (See fig_scc_fit.jl.)
const SCC_DEFLATOR_2020_TO_2024 = 1.18    # US GDP implicit price deflator
const SCC_YEARS_2020USD  = [2020, 2030, 2040, 2050, 2060, 2070, 2080]
const SCC_POINTS_2020USD = Dict(
    "1.5%" => [340.0, 380.0, 430.0, 480.0, 530.0, 570.0, 600.0],
    "2.0%" => [190.0, 230.0, 270.0, 310.0, 350.0, 380.0, 410.0],
    "2.5%" => [120.0, 140.0, 170.0, 200.0, 230.0, 260.0, 280.0],
)

# SCC at a given emission year, in 2024 USD per tonne CO₂e (default 2% rate)
function scc_per_ton(year; rate = "2.0%")
    xs = SCC_YEARS_2020USD;  ys = SCC_POINTS_2020USD[rate]
    yr = clamp(year, xs[1], xs[end])
    i  = min(searchsortedlast(xs, yr), length(xs) - 1)
    t  = (yr - xs[i]) / (xs[i+1] - xs[i])
    return (ys[i] + t * (ys[i+1] - ys[i])) * SCC_DEFLATOR_2020_TO_2024
end

# Avoided SCC (M USD/yr): SCC(year) × avoided CO₂e (kt/yr × 1000 t/kt) ÷ 1e6.
# CO₂-avoided is pathway-dependent (h2_ci differs), so this MUST use the pathway
# actually plotted in panels (b)/(c) — solar = results[3] — not results[1] (SMR).
# (Health/miles are pathway-independent, so the health block's results[1] is OK.)
scc_vals = [scc_per_ton(results[3].years[k]) * results[3].co2_avoided_kt[k] *
            1e3 / 1e6 for k in eachindex(results[3].years)]

# Trapezoidal integral of a yearly series over a [y0, y1] year window.
# Series in M USD/yr, annual spacing → result in M USD (cumulative, undiscounted).
function integrate_window(years, vals, y0, y1)
    idx = findall(y -> y0 <= y <= y1, years)
    xs, ys = years[idx], vals[idx]
    s = 0.0
    for i in 1:length(xs)-1
        s += 0.5 * (ys[i] + ys[i+1]) * (xs[i+1] - xs[i])
    end
    return s
end

# Year at which the CUMULATIVE net cost (running integral from y0) returns to 0,
# i.e. accumulated societal benefit overtakes accumulated cost. Linear-
# interpolated between annual nodes. Returns nothing if the cumulative never
# falls back to ≤0 after going positive.
function cumulative_breakeven(years, vals, y0, y1)
    idx = findall(y -> y0 <= y <= y1, years)
    xs, ys = years[idx], vals[idx]
    cum = 0.0
    for i in 1:length(xs)-1
        prev = cum                                            # cumulative at xs[i]
        cum += 0.5 * (ys[i] + ys[i+1]) * (xs[i+1] - xs[i])    # cumulative at xs[i+1]
        if prev > 0 && cum <= 0       # cumulative crosses back through zero
            return xs[i] + (xs[i+1] - xs[i]) * prev / (prev - cum)
        end
    end
    return nothing
end

# Discount a real (constant-dollar) yearly series to present value at base year
# t0, using real social rate r. Appraisal discounting of the *stream* — applied
# uniformly to all components; the SCC is already discounted-to-emission-year
# and is a real flow like the rest, so this is the only discounting layered on.
discount_series(years, vals, r, t0) = [v / (1 + r)^(y - t0) for (y, v) in zip(years, vals)]

# Solar-electrolysis config for an arbitrary truck schedule (mirrors the :solar
# branch of make_config_prem) — used for the deployment-scenario sensitivity.
solar_cfg(sched) = build_config(;
    h2_pathway_id = "current_mix", use_utilization_pricing = true, utilization_transport_cost = 1.0,
    use_lcfs = true, enable_45v = true, bus_demand_scenario = "growing", end_year = END_YEAR,
    use_truck_deployment_schedule = true, truck_deployment_schedule = sched,
    lcfs_price_schedule_dict = lcfs_prices,
    expansion_pathway_id = "electrolysis", electrolysis_pricing_enabled = true,
    electrolysis_electricity_source = "solar", electrolyzer_capex_per_kw = 3000.0,
    solar_capex_per_kw = 1600.0, solar_capacity_factor = 0.25, solar_lifetime = 25,
    electrolyzer_learning_rate = 0.233, electrolyzer_stack_fraction = 0.60, bop_learning_rate = 0.04,
    solar_panel_learning_rate = 0.267, solar_panel_fraction = 0.80, solar_bop_learning_rate = 0.04,
)

# Run solar MC for a truck scenario → median premium/health/CO₂ flows for the
# net-cost sensitivity (solar H₂ CI = 0 ⇒ net CO₂ avoided = diesel lifecycle).
function solar_flows(scenario_key)
    cfg = solar_cfg(load_truck_scenario(scenario_key))
    Random.seed!(SEED)
    raw = run_monte_carlo(cfg, N_RUNS)
    price_r = raw[1];  truck_r = Float64.(raw[3])
    yrs = collect(cfg.start_year : cfg.end_year);  m = length(yrs)
    h2_yr = truck_r .* Float64(H2_PER_TRUCK_PER_DAY) .* Float64(OPERATING_DAYS_PER_YEAR)
    mi_yr = h2_yr .* miles_per_kg_h2
    tlo = [(h2_tco_prem(price_r[r,k], yrs[k]) - diesel_tco_mile(DIESEL_P_LO, yrs[k])) * mi_yr[r,k] / 1e6 for r in 1:N_RUNS, k in 1:m]
    thi = [(h2_tco_prem(price_r[r,k], yrs[k]) - diesel_tco_mile(DIESEL_P_HI, yrs[k])) * mi_yr[r,k] / 1e6 for r in 1:N_RUNS, k in 1:m]
    med_miles = [median(mi_yr[:, k]) for k in 1:m]
    return (years  = yrs,
            tcp_lo = [median(tlo[:, k]) for k in 1:m],
            tcp_hi = [median(thi[:, k]) for k in 1:m],
            co2_t  = med_miles .* CO2_G_PER_MILE ./ 1e6,   # solar: net CO₂ = diesel lifecycle
            health = [ENV_COST_DELTA_USD_PER_MILE[y] * med_miles[k] / 1e6 for (k, y) in enumerate(yrs)],
            miles  = med_miles)
end

# ─────────────────────────────────────────────────────────────────────────────
# Figure — 2×2: (TCO vs societal benefit) and (net cost per mile), each for
#               limited_dep and high_dep. SCC drawn as a 1.5–2.5% band.
# ─────────────────────────────────────────────────────────────────────────────
MM_TO_PT  = 1 / 0.352778
c_premium = pathways[3].color        # solar green
c_benefit = colorant"#AD1457"        # deep pink — societal benefit / SCC

# Deployment flows (solar): demo reuses the pathway run; high_dep is a fresh MC.
# Both feed the figure AND the sensitivity table below.
demo_F = (years = results[3].years, tcp_lo = results[3].tcp_med_lo, tcp_hi = results[3].tcp_med_hi,
          co2_t = results[3].co2_avoided_kt .* 1e3, health = health_fitted_vals,
          miles = results[3].med_miles_yr)
println("Running high_dep solar MC…")
high_F = solar_flows("high_dep")
deploy_panels = [("Limited deployment", demo_F), ("High deployment", high_F)]

# Total societal benefit (M USD/yr) = air-quality health + avoided SCC at `rate`.
benefit(F, rate) = F.health .+
    [scc_per_ton(F.years[k]; rate = rate) * F.co2_t[k] / 1e6 for k in eachindex(F.years)]
permile(F, tcp, rate) = (tcp .- benefit(F, rate)) .* 1e6 ./ F.miles

# Panel: solar TCO premium vs. societal-benefit band (SCC 1.5–2.5%), net-cost bands.
function draw_societal!(ax, F; showlegend = false)
    y = F.years
    blo = benefit(F, "2.5%"); bhi = benefit(F, "1.5%"); bmd = benefit(F, "2.0%")
    hlines!(ax, 0; color = (:black, 0.4), linewidth = 0.8, linestyle = :dot)
    band!(ax, y, blo, bhi; color = (c_benefit, 0.15), label = "Shaded: SCC 1.5–2.5% range")
    lines!(ax, y, bmd; color = c_benefit, linewidth = 1.3,
           label = "Societal benefit (SCC 2%)")
    lines!(ax, y, F.tcp_lo; color = c_premium, linewidth = 1.5,
           label = "Solar TCO premium (\$$(DIESEL_P_LO)/gal)")
    lines!(ax, y, F.tcp_hi; color = c_premium, linewidth = 1.0, linestyle = :dash,
           label = "Solar TCO premium (\$$(DIESEL_P_HI)/gal)")
    for (tcp, ls, lw, lab) in ((F.tcp_lo, :solid, 1.5, "Net total cost (\$$(DIESEL_P_LO)/gal)"),
                               (F.tcp_hi, :dash,  1.0, "Net total cost (\$$(DIESEL_P_HI)/gal)"))
        band!(ax, y, tcp .- bhi, tcp .- blo; color = (:black, 0.10))
        lines!(ax, y, tcp .- bmd; color = :black, linewidth = lw, linestyle = ls, label = lab)
    end
    showlegend && axislegend(ax; position = :lt, rowgap = 1, labelsize = 5, framevisible = true, nbanks = 1)
end

# Panel: net cost per mile, SCC-range band per diesel.
function draw_permile!(ax, F; showlegend = false)
    y = F.years
    hlines!(ax, 0; color = (:black, 0.4), linewidth = 0.8, linestyle = :dot)
    for (i, (tcp, ls, lw, lab)) in enumerate(((F.tcp_lo, :solid, 1.5, "Net cost/mile (\$$(DIESEL_P_LO)/gal)"),
                                              (F.tcp_hi, :dash,  1.0, "Net cost/mile (\$$(DIESEL_P_HI)/gal)")))
        if i == 1
            band!(ax, y, permile(F, tcp, "1.5%"), permile(F, tcp, "2.5%");
                  color = (:black, 0.10), label = "Shaded: SCC 1.5–2.5% range")
        else
            band!(ax, y, permile(F, tcp, "1.5%"), permile(F, tcp, "2.5%"); color = (:black, 0.10))
        end
        lines!(ax, y, permile(F, tcp, "2.0%"); color = :black, linewidth = lw, linestyle = ls, label = lab)
    end
    showlegend && axislegend(ax; position = :rt, rowgap = 1, labelsize = 5, framevisible = true)
end

# Shared y for the per-mile row (comparable magnitudes across deployments).
pm_all = Float64[]
for (_, F) in deploy_panels, tcp in (F.tcp_lo, F.tcp_hi), rate in ("1.5%", "2.0%", "2.5%")
    append!(pm_all, permile(F, tcp, rate))
end
pm_lim = (minimum(pm_all) - 0.08 * abs(minimum(pm_all)), maximum(pm_all) * 1.05)

fig = Figure(size = (200 * MM_TO_PT, 165 * MM_TO_PT), fontsize = 8)
labels = ['a', 'b', 'c', 'd']
for (j, (name, F)) in enumerate(deploy_panels)               # row 1: societal benefit
    ax = Axis(fig[1, j]; title = "($(labels[j]))  $name — TCO vs. societal benefit",
        titlesize = 8, titlefont = :bold, xlabel = "Year",
        ylabel = j == 1 ? L"M USD yr$^{-1}$" : "",
        xlabelsize = 7, ylabelsize = 7, xticklabelsize = 7, yticklabelsize = 7,
        xticklabelrotation = π/4, xticks = [2026, 2030, 2035, 2040, 2045],
        limits = ((2026, END_YEAR), (nothing, nothing)))
    draw_societal!(ax, F; showlegend = (j == 1))
end
for (j, (name, F)) in enumerate(deploy_panels)               # row 2: net cost per mile
    ax = Axis(fig[2, j]; title = "($(labels[j+2]))  $name — net cost per mile",
        titlesize = 8, titlefont = :bold, xlabel = "Year",
        ylabel = j == 1 ? L"Net cost (USD mi$^{-1}$)" : "",
        xlabelsize = 7, ylabelsize = 7, xticklabelsize = 7, yticklabelsize = 7,
        xticklabelrotation = π/4, xticks = [2026, 2030, 2035, 2040, 2045],
        yticklabelsvisible = j == 1,
        limits = ((2026, END_YEAR), pm_lim))
    draw_permile!(ax, F; showlegend = (j == 1))
end
rowgap!(fig.layout, 6); colgap!(fig.layout, 6)
resize_to_layout!(fig)
out_pdf = fig_path(OUT_DIR, "fig_cost_premium_envcost_signed.pdf")
out_png = fig_path(OUT_DIR, "fig_cost_premium_envcost_signed.png")
save(out_pdf, fig; pt_per_unit = 1)
println("Saved → $out_pdf")
save(out_png, fig; px_per_unit = 300/72)
println("Saved → $out_png")

# ─────────────────────────────────────────────────────────────────────────────
# Sensitivity tables (NOT drawn) → summary file (overwritten each run)
# ─────────────────────────────────────────────────────────────────────────────
INTEG_Y0, INTEG_Y1 = 2026, 2045
SOCIAL_RATE = 0.02; DBASE = INTEG_Y0
PV_RATES  = (0.0, 0.02, 0.03)
SCC_RATES = ["1.5%", "2.0%", "2.5%"]
label_of(x) = x >= 0 ? "net cost" : "net benefit"
fmt_be(x)   = x === nothing ? "none (stays net cost through $(INTEG_Y1))" : @sprintf("%.1f", x)
net_stream(F, dfield, rate) = getfield(F, dfield) .- F.health .-
    [scc_per_ton(F.years[k]; rate = rate) * F.co2_t[k] / 1e6 for k in eachindex(F.years)]

S = String[]
push!(S, "Net societal cost summary — fig_cost_premium_envcost_signed")
push!(S, "Generated:  $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))")
push!(S, "Pathway:    solar electrolysis ($(N_RUNS) MC runs, seed $(SEED))")
push!(S, "")
push!(S, "Cumulative net societal cost = ∫ (TCO premium − total societal benefit) dt")
push!(S, "  over $(INTEG_Y0)–$(INTEG_Y1), trapezoidal (M USD).  > 0 ⇒ net cost; < 0 ⇒ net benefit.")
push!(S, "Net-benefit breakeven = year the running cumulative returns to 0.")
push!(S, "Undiscounted = real 2024 USD; Discounted = appraisal-discounted stream, PV to $(DBASE).")
push!(S, "SCC rate = EPA near-term Ramsey discount rate (sets the carbon valuation,")
push!(S, "  piecewise-linear EPA points); PV rate = appraisal discounting of the cost stream.")
push!(S, "-"^66)
push!(S, "HEADLINE — limited_dep deployment, SCC 2% (matches figure panels a/c)")
push!(S, "")
for (p, dfield) in [(DIESEL_P_LO, :tcp_lo), (DIESEL_P_HI, :tcp_hi)]
    v       = net_stream(demo_F, dfield, "2.0%")
    integ   = integrate_window(demo_F.years, v, INTEG_Y0, INTEG_Y1)
    be      = cumulative_breakeven(demo_F.years, v, INTEG_Y0, INTEG_Y1)
    dv      = discount_series(demo_F.years, v, SOCIAL_RATE, DBASE)
    integ_d = integrate_window(demo_F.years, dv, INTEG_Y0, INTEG_Y1)
    be_d    = cumulative_breakeven(demo_F.years, dv, INTEG_Y0, INTEG_Y1)
    push!(S, @sprintf("diesel \$%.2f/gal", p))
    push!(S, @sprintf("  undiscounted              : cum %+10.1f M USD (%s),  breakeven %s",
                      integ, label_of(integ), fmt_be(be)))
    push!(S, @sprintf("  discounted (%d%% PV to %d) : cum %+10.1f M USD (%s),  breakeven %s",
                      round(Int, SOCIAL_RATE*100), DBASE, integ_d, label_of(integ_d), fmt_be(be_d)))
    push!(S, "")
end

deploys = [("limited_dep", demo_F), ("high_dep", high_F)]
diesels = [(DIESEL_P_LO, :tcp_lo), (DIESEL_P_HI, :tcp_hi)]

push!(S, "="^66)
push!(S, "FULL SENSITIVITY — cumulative net cost (M USD), $(INTEG_Y0)–$(INTEG_Y1)")
push!(S, "rows = SCC discount rate;  cols = appraisal PV rate (PV to $(DBASE))")
push!(S, "(> 0 net cost, < 0 net benefit)")
push!(S, "")
for (depname, F) in deploys, (dp, dfield) in diesels
    push!(S, @sprintf("[ %-9s | diesel \$%.2f/gal ]", depname, dp))
    push!(S, @sprintf("    %-7s %11s %11s %11s", "SCC\\PV", "PV 0%", "PV 2%", "PV 3%"))
    for rate in SCC_RATES
        net = net_stream(F, dfield, rate)
        v = [integrate_window(F.years, discount_series(F.years, net, pv, DBASE),
                              INTEG_Y0, INTEG_Y1) for pv in PV_RATES]
        push!(S, @sprintf("    %-7s %+11.1f %+11.1f %+11.1f", rate, v[1], v[2], v[3]))
    end
    push!(S, "")
end

push!(S, "FULL SENSITIVITY — net-benefit breakeven year (cumulative returns to 0)")
push!(S, "  '—' = stays net cost through $(INTEG_Y1)")
push!(S, "")
becell(x) = x === nothing ? "—" : @sprintf("%.1f", x)
for (depname, F) in deploys, (dp, dfield) in diesels
    push!(S, @sprintf("[ %-9s | diesel \$%.2f/gal ]", depname, dp))
    push!(S, @sprintf("    %-7s %11s %11s %11s", "SCC\\PV", "PV 0%", "PV 2%", "PV 3%"))
    for rate in SCC_RATES
        net = net_stream(F, dfield, rate)
        b = [cumulative_breakeven(F.years, discount_series(F.years, net, pv, DBASE),
                                  INTEG_Y0, INTEG_Y1) for pv in PV_RATES]
        push!(S, @sprintf("    %-7s %11s %11s %11s", rate, becell(b[1]), becell(b[2]), becell(b[3])))
    end
    push!(S, "")
end

summary_txt = join(S, "\n")
println("\n" * summary_txt)
summary_path = joinpath(OUT_DIR, "fig_cost_premium_envcost_signed_summary.txt")
open(io -> write(io, summary_txt), summary_path, "w")
println("Saved → $summary_path")
