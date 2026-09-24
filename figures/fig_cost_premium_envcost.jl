# =============================================================================
# FIGURE: AGGREGATE COST PREMIUM — H2 vs. DIESEL (M USD / YEAR)   [ENV-COST VARIANT]
# =============================================================================
# Second version of fig_cost_premium.jl.
#
# The ONLY methodological difference vs. the original is the source of the
# air-quality (NOx + PM2.5 + NH3) health benefit:
#
#   • Original  → EPA COBRA model runs (hardcoded COBRA_RESULTS / proxy fit).
#   • This file → per-mile environmental-cost factors from
#                 "Env Cost Factors 20apr LF.xlsx"  (sheet "Env cost parameters").
#
# In that sheet, for each year (column B):
#   column E = HDV environmental health cost ($/mile) for a DIESEL truck
#   column K = HDV environmental health cost ($/mile) for a ZEV   truck
# The avoided health cost per mile of switching diesel → H₂ is therefore
#   Δ(year) = E(year) − K(year)   [USD / mile]
# and the fleet-wide health benefit is  Δ × (fleet miles driven)  [M USD/yr].
#
# Everything else (TCO premium, social cost of carbon, net-cost panels) is
# unchanged from fig_cost_premium.jl.
#
# Run from project root:
#   julia --project figures/fig_cost_premium_envcost.jl
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

# Full diesel TCO at a given pump price ($/mile)
diesel_tco_mile(p_gal) = p_gal / DIESEL_MPG + d_capital + d_rm + d_tires + d_common

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
    tcp_lo = [( h2_tco_prem(price_r[run, k], years[k]) - diesel_tco_mile(DIESEL_P_LO) ) *
              mi_yr[run, k] / 1e6
              for run in 1:N_RUNS, k in 1:n]
    tcp_hi = [( h2_tco_prem(price_r[run, k], years[k]) - diesel_tco_mile(DIESEL_P_HI) ) *
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
# Year-dependent SCC fitted to EPA (2023) SC-CO₂, 2% near-term Ramsey rate.
# Published table (2020 USD t⁻¹ CO₂):
#   2020:190  2030:230  2040:270  2050:310  2060:350  2070:380  2080:410
# Fit uses only the 2020–2050 points, which are *exactly* linear
# (+40 USD2020/decade → slope 4.0 USD2020 t⁻¹ yr⁻¹, intercept 190 @2020, R²=1).
# The published series decelerates only after 2060 and is not used here — the
# model horizon (2026–2045) lies entirely inside 2020–2050, so the line
# reproduces the EPA values exactly. Converted to 2024 USD with the US GDP
# implicit price deflator (2024/2020 ≈ 1.18; CPI-U would give ≈1.21).
const SCC_DEFLATOR_2020_TO_2024 = 1.18    # US GDP implicit price deflator
const SCC_FIT_INTERCEPT_2020USD = 190.0   # 2020–2050 fit: intercept @2020 (2020 USD t⁻¹)
const SCC_FIT_SLOPE_2020USD     = 4.0     # 2020–2050 fit: slope (2020 USD t⁻¹ yr⁻¹)
const SCC_REF_YEAR              = 2020

# SCC at a given emission year, in 2024 USD per tonne CO₂e
scc_per_ton(year) = (SCC_FIT_INTERCEPT_2020USD +
                     SCC_FIT_SLOPE_2020USD * (year - SCC_REF_YEAR)) *
                    SCC_DEFLATOR_2020_TO_2024

# Avoided SCC (M USD/yr): SCC(year) × avoided CO₂e (kt/yr × 1000 t/kt) ÷ 1e6
scc_vals = [scc_per_ton(results[1].years[k]) * results[1].co2_avoided_kt[k] *
            1e3 / 1e6 for k in eachindex(results[1].years)]

# ─────────────────────────────────────────────────────────────────────────────
# Figure  (three panels: (a) TCO premium, (b) societal benefit, (c) net per mile)
# ─────────────────────────────────────────────────────────────────────────────
MM_TO_PT = 1 / 0.352778
fig = Figure(size = (260 * MM_TO_PT, 100 * MM_TO_PT), fontsize = 8)

# ── (a) Total TCO premium ─────────────────────────────────────────────────────
ax_a = Axis(fig[1, 1];
    title              = "(a)  Total TCO premium",
    titlesize          = 8, titlefont = :bold,
    xlabel             = "Year",
    ylabel             = L"Premium (M USD yr$^{-1}$)",
    xlabelsize         = 7, ylabelsize = 7,
    xticklabelsize     = 7, yticklabelsize = 7,
    xticklabelrotation = π/4,
    xticks             = 2026:2:END_YEAR,
    limits             = ((2026, END_YEAR), (0, nothing)),
)
hlines!(ax_a, 0; color = (:black, 0.4), linewidth = 0.8, linestyle = :dot)

for (pw, r) in zip(pathways, results)
    band!(ax_a, r.years, r.tcp_p25_lo, r.tcp_p75_lo; color = (pw.color, 0.20))
    lines!(ax_a, r.years, r.tcp_med_lo; color = pw.color, linewidth = 1.5, label = pw.label)
    lines!(ax_a, r.years, r.tcp_med_hi; color = pw.color, linewidth = 1.0, linestyle = :dash)
end
axislegend(ax_a; position = :lt, rowgap = 1, labelsize = 6, framevisible = true)

# ── (b) Solar TCO premium vs. stacked societal benefit ───────────────────────
# Societal benefit as stacked areas; both diesel-reference scenarios from (a).
c_solar       = pathways[3].color    # "#70AD47" green
c_pink_health = colorant"#F48FB1"    # light pink — env-cost air-quality benefit
c_pink_scc    = colorant"#AD1457"    # deep pink  — social cost of carbon

ax_d = Axis(fig[1, 2];
    title              = "(b)  Solar electrolysis — TCO vs. societal benefit",
    titlesize          = 8, titlefont = :bold,
    xlabel             = "Year",
    ylabel             = L"M USD yr$^{-1}$",
    xlabelsize         = 7, ylabelsize = 7,
    xticklabelsize     = 7, yticklabelsize = 7,
    xticklabelrotation = π/4,
    xticks             = 2026:2:END_YEAR,
    limits             = ((2026, END_YEAR), (0, nothing)),
)

r_d     = results[3]   # solar electrolysis pathway
total_d = health_fitted_vals .+ scc_vals

# Stacked benefit areas
band!(ax_d, health_fitted_yrs, zeros(length(health_fitted_yrs)), health_fitted_vals;
      color = (c_pink_health, 0.20))
band!(ax_d, results[1].years, health_fitted_vals, total_d;
      color = (c_pink_scc, 0.20))
lines!(ax_d, health_fitted_yrs, health_fitted_vals;
       color = c_pink_health, linewidth = 1.2,
       label = "Air-quality health benefit (NOx+PM2.5+NH3)")
lines!(ax_d, results[1].years, total_d;
       color = c_pink_scc, linewidth = 1.2,
       label = "Total societal benefit (incl. SCC)")

# Two solar TCO premium lines (one per diesel reference price)
lines!(ax_d, r_d.years, r_d.tcp_med_lo;
       color = c_solar, linewidth = 1.5,
       label = "Solar TCO premium (diesel \$$(DIESEL_P_LO)/gal)")
lines!(ax_d, r_d.years, r_d.tcp_med_hi;
       color = c_solar, linewidth = 1.0, linestyle = :dash,
       label = "Solar TCO premium (diesel \$$(DIESEL_P_HI)/gal)")

# Difference lines: TCO − total societal benefit (> 0: net cost; < 0: net benefit)
net_lo = r_d.tcp_med_lo .- total_d
net_hi = r_d.tcp_med_hi .- total_d
hlines!(ax_d, 0; color = (:black, 0.4), linewidth = 0.8, linestyle = :dot)
lines!(ax_d, r_d.years, net_lo;
       color = "black", linewidth = 1.5,
       label = "Net total cost (diesel \$$(DIESEL_P_LO)/gal)")
lines!(ax_d, r_d.years, net_hi;
       color = "black", linewidth = 1.0, linestyle = :dash,
       label = "Net total cost (diesel \$$(DIESEL_P_HI)/gal)")

axislegend(ax_d; position = :lt, rowgap = 1, labelsize = 6, framevisible = true, nbanks = 1)

# ── (c) Net cost per mile driven ─────────────────────────────────────────────
# Net = (TCO premium − total societal benefit) as computed in panel (b), but
# normalised by fleet miles driven per year → USD per mile of operation.
# Positive: H₂ fleet is still a net cost to society; negative: net benefit.
r_c     = results[3]   # solar electrolysis pathway
total_c = health_fitted_vals .+ scc_vals

net_mi_lo = (r_c.tcp_med_lo .- total_c) .* 1e6 ./ r_c.med_miles_yr   # USD/mile
net_mi_hi = (r_c.tcp_med_hi .- total_c) .* 1e6 ./ r_c.med_miles_yr   # USD/mile

ax_c = Axis(fig[1, 3];
    title              = "(c)  Solar electrolysis — net cost per mile",
    titlesize          = 8, titlefont = :bold,
    xlabel             = "Year",
    ylabel             = L"Net cost (USD mi$^{-1}$)",
    xlabelsize         = 7, ylabelsize = 7,
    xticklabelsize     = 7, yticklabelsize = 7,
    xticklabelrotation = π/4,
    xticks             = 2026:2:END_YEAR,
    limits             = ((2026, END_YEAR), (0, nothing)),
)

hlines!(ax_c, 0; color = (:black, 0.4), linewidth = 0.8, linestyle = :dot)
lines!(ax_c, r_c.years, net_mi_lo;
       color = "black", linewidth = 1.5,
       label = "Net added cost per mile (diesel \$$(DIESEL_P_LO)/gal)")
lines!(ax_c, r_c.years, net_mi_hi;
       color = "black", linewidth = 1.0, linestyle = :dash,
       label = "Net added cost per mile (diesel \$$(DIESEL_P_HI)/gal)")

axislegend(ax_c; position = :rt, rowgap = 1, labelsize = 6, framevisible = true)

rowgap!(fig.layout, 4)
colgap!(fig.layout, 4)
resize_to_layout!(fig)

out_pdf = fig_path(OUT_DIR, "fig_cost_premium_envcost.pdf")
out_png = fig_path(OUT_DIR, "fig_cost_premium_envcost.png")
save(out_pdf, fig; pt_per_unit = 1)
println("Saved → $out_pdf")
save(out_png, fig; px_per_unit = 300/72)
println("Saved → $out_png")
