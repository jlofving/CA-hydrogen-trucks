# =============================================================================
# FIGURE: AGGREGATE COST PREMIUM — H2 vs. DIESEL (M USD / YEAR)
# =============================================================================
# Two-panel figure:
#   (a) H₂ fuel cost premium (M$/yr) — total H₂ fuel spend above diesel-equivalent
#   (b) Total TCO premium   (M$/yr)  — full H₂ TCO above diesel TCO, fleet-wide
#
# Premium = (H₂ cost − diesel reference cost) × annual fleet demand
# Shown for 3 H₂ pathways × limited_dep deployment scenario.
# Solid lines / bands: diesel ref at $4.80/gal.  Dashed: $5.80/gal.
# Bands: P25–P75 across Monte Carlo runs.
#
# Run from project root:
#   julia --project figures/fig_cost_premium.jl
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
        # Avoided emissions use median truck count; NOx/PM2.5 pathway-independent
        nox_avoided_t  = med_miles_yr .* NOX_G_PER_MILE    ./ 1e6,  # tonnes/yr
        pm25_avoided_t = med_miles_yr .* PM25_G_PER_MILE   ./ 1e6,  # tonnes/yr
        co2_avoided_kt = med_miles_yr .* net_co2e_g_per_mile ./ 1e9, # kt CO₂e/yr (g→kt: ÷1e9)
    )
    println(" done")
    r
end

println("All simulations complete.\n")

# Print emissions summary (pathway-independent — use results[1])
let r = results[1]
    println("AVOIDED EMISSIONS (limited_dep fleet, median truck count)")
    println("="^55)
    println("  Emission factors: NOx = $(NOX_G_PER_MILE) g/mile, PM2.5 = $(PM25_G_PER_MILE) g/mile")
    println()
    @printf("  %-6s  %12s  %14s  %16s\n", "Year", "NOx (t/yr)", "PM2.5 (t/yr)", "CO₂e (kt/yr)")
    println("  " * "-"^52)
    for (k, yr) in enumerate(r.years)
        @printf("  %-6d  %12.1f  %14.2f  %16.1f\n",
                yr, r.nox_avoided_t[k], r.pm25_avoided_t[k], r.co2_avoided_kt[k])
    end
    println()
end

# ─────────────────────────────────────────────────────────────────────────────
# Export COBRA input CSV
# ─────────────────────────────────────────────────────────────────────────────
# COBRA (EPA Co-Benefits Risk Assessment tool) requires emission *reductions*
# in US short tons/year, attributed to a state/county.
# This CSV gives you the per-year totals for California (statewide input).
# Steps:
#   1. Open https://www.epa.gov/statelocalenergy/cobra-model
#      (or the downloaded Excel version)
#   2. In the "Emission Changes" sheet, set State = California, select the
#      appropriate sector (e.g. "On-road diesel — heavy-duty trucks"), and
#      paste the NOx / PM2.5 columns from this CSV for each modelled year.
#   3. Run COBRA → note the "Total Annual Health Benefits ($ millions)" for
#      each year under the VSL (value of statistical life) column.
#   4. Fill those values into COBRA_RESULTS below, then re-run
#      this script to render panel (c) as a monetised benefit curve.
let r = results[1]
    cobra_path = joinpath(OUT_DIR, "cobra_input.csv")
    open(cobra_path, "w") do io
        println(io, "year,NOx_avoided_short_tons,PM25_avoided_short_tons")
        for (k, yr) in enumerate(r.years)
            nox_st  = r.nox_avoided_t[k]  * METRIC_T_TO_SHORT_T
            pm25_st = r.pm25_avoided_t[k] * METRIC_T_TO_SHORT_T
            @printf(io, "%d,%.2f,%.4f\n", yr, nox_st, pm25_st)
        end
    end
    println("COBRA input → $cobra_path")
    println()

    # Also print the table to terminal for quick reference
    println("COBRA INPUT TABLE (US short tons/yr — paste into COBRA emission-changes sheet)")
    println("="^60)
    @printf("  %-6s  %22s  %24s\n", "Year", "NOx (short t/yr)", "PM2.5 (short t/yr)")
    println("  " * "-"^54)
    for (k, yr) in enumerate(r.years)
        @printf("  %-6d  %22.2f  %24.4f\n",
                yr,
                r.nox_avoided_t[k]  * METRIC_T_TO_SHORT_T,
                r.pm25_avoided_t[k] * METRIC_T_TO_SHORT_T)
    end
    println()
end

# ─────────────────────────────────────────────────────────────────────────────
# COBRA results — fill in after running the EPA COBRA model
# ─────────────────────────────────────────────────────────────────────────────
# Each entry: year => (benefit_M_USD, nox_short_tons_at_time_of_run)
# Storing the NOx input alongside the dollar output lets the script detect
# if the model's emissions have drifted since the COBRA run was done.
# Convention: omit years not yet run through COBRA (or set benefit to 0.0).
# For paper finalisation, fill in all years with actual COBRA outputs.
#
# To add a new entry after a COBRA run:
#   1. Note the "NOx (short t/yr)" value printed to the terminal for that year.
#   2. Note the "Total Annual Health Benefits ($M)" from COBRA (VSL column).
#   3. Add:  year => (benefit, nox_short_tons)
const COBRA_RESULTS = Dict{Int, Tuple{Float64,Float64}}(
    # year => (benefit M USD/yr,  NOx short t/yr used in COBRA run)
    2026 => (0.14,  1.16),
    2027 => (0.15,  1.22),
    2028 => (0.19,  1.51),
    2029 => (0.3,   2.39),
    2030 => (0.37,  2.97),
    2031 => (0.51,  4.13),
    2032 => (0.73,  5.88),
    2033 => (0.94,  7.62),
    2034 => (1.4,  11.64),
    2035 => (2.1,  17.17),
    2036 => (3.1,  25.03),
    2037 => (4.1,  33.18),
    2038 => (5.4,  43.65),
    2039 => (7.0,  56.46),
    2040 => (8.8,  71.01),
)

# Print summary
years_p = results[1].years
println("COST PREMIUM SUMMARY  (median, M USD/yr, vs diesel \$$(DIESEL_P_LO)/gal)")
println("="^70)
@printf("%-24s", "Year")
for r in results
    @printf("  %10s", "")
end
println()
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
# COBRA proxy function
# ─────────────────────────────────────────────────────────────────────────────
# benefit [M USD/yr] = α_NOx × NOx [short t/yr]  +  α_PM25 × PM2.5 [short t/yr]
#
# NOx and PM2.5 are perfectly collinear in this model (both ∝ fleet miles),
# so their separate damage coefficients CANNOT be estimated from the full-run
# COBRA data alone.  Instead, calibrate with two isolated COBRA runs for one
# representative year (e.g. 2032):
#
#   Run A — apply only the NOx reduction for that year (set PM2.5 change = 0)
#            α_NOx  = COBRA_benefit_A / NOx_short_tons_2032
#
#   Run B — apply only the PM2.5 reduction for that year (set NOx change = 0)
#            α_PM25 = COBRA_benefit_B / PM2.5_short_tons_2032
#
# Fill in the two constants below; leave as `nothing` until the runs are done.
# When both are set the proxy function is used for all years; otherwise the
# script falls back to the single-coefficient γ fit from the full COBRA runs.

const COBRA_ALPHA_NOX  = 3.5  / 33.18  # = 0.1055 M USD / short-ton NOx  (COBRA 2037, NOx-only run)
const COBRA_ALPHA_PM25 = 0.59 /  0.83  # = 0.7108 M USD / short-ton PM2.5 (COBRA 2037, PM2.5-only run)

# ── Fit / proxy computation ──────────────────────────────────────────────────
let r_ref     = results[1]
    year_to_k = Dict(yr => k for (k, yr) in enumerate(r_ref.years))

    # Non-zero entries = years actually run through COBRA (full combined run)
    measured = sort([(yr, tup[1]) for (yr, tup) in COBRA_RESULTS if tup[1] > 0.0])

    global cobra_measured_yrs  = [p[1] for p in measured]
    global cobra_measured_vals = [p[2] for p in measured]

    if COBRA_ALPHA_NOX !== nothing && COBRA_ALPHA_PM25 !== nothing
        # ── Bivariate proxy (preferred once isolated runs are done) ──────────
        global cobra_gamma = nothing   # not used in bivariate mode
        global cobra_fitted_yrs  = collect(r_ref.years)
        global cobra_fitted_vals = [
            COBRA_ALPHA_NOX  * r_ref.nox_avoided_t[k]  * METRIC_T_TO_SHORT_T +
            COBRA_ALPHA_PM25 * r_ref.pm25_avoided_t[k] * METRIC_T_TO_SHORT_T
            for k in eachindex(r_ref.years)
        ]
        println("COBRA proxy (bivariate):")
        println("  α_NOx  = $(COBRA_ALPHA_NOX)  M USD / short-ton NOx")
        println("  α_PM25 = $(COBRA_ALPHA_PM25)  M USD / short-ton PM2.5")
        println()

    elseif !isempty(measured)
        # ── Single-coefficient fallback: γ fit from full COBRA runs ──────────
        # NOTE: collinearity means γ absorbs both NOx and PM2.5 damage in fixed
        # proportion. Valid only while emission factors stay constant.
        x_cal = [r_ref.nox_avoided_t[year_to_k[yr]] * METRIC_T_TO_SHORT_T for (yr, _) in measured]
        b_cal = [val for (_, val) in measured]
        global cobra_gamma = dot(b_cal, x_cal) / dot(x_cal, x_cal)

        global cobra_fitted_yrs  = collect(r_ref.years)
        global cobra_fitted_vals = [cobra_gamma *
                                    r_ref.nox_avoided_t[k] * METRIC_T_TO_SHORT_T
                                    for k in eachindex(r_ref.years)]

        println("COBRA proxy (single-coeff fallback — NOx/PM2.5 collinear):")
        println("  γ = $(round(cobra_gamma; digits=4)) M USD / short-ton NOx")
        println("  WARNING: valid only if NOx_G_PER_MILE / PM25_G_PER_MILE ratio is unchanged.")
        println("  For a generalised proxy, fill in COBRA_ALPHA_NOX and COBRA_ALPHA_PM25 above.")
        println()
    else
        global cobra_gamma       = nothing
        global cobra_fitted_yrs  = Int[]
        global cobra_fitted_vals = Float64[]
    end
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
# Figure  (three panels: (a) TCO premium, (b) societal benefit, (c) net balance)
# ─────────────────────────────────────────────────────────────────────────────
MM_TO_PT = 1 / 0.352778
fig = Figure(size = (260 * MM_TO_PT, 100 * MM_TO_PT), fontsize = 8)

# ── (a) [commented out] H₂ fuel cost premium ─────────────────────────────────
# ax_a = Axis(fig[1, 1];
#     title          = "(a)  H₂ fuel cost premium",
#     titlesize      = 8, titlefont = :bold,
#     ylabel         = L"Premium (M USD yr$^{-1}$)",
#     xlabelsize     = 7, ylabelsize = 7,
#     xticklabelsize = 7, yticklabelsize = 7,
#     xticks         = 2026:2:END_YEAR,
#     xticklabelsvisible = false,
#     limits         = ((2026, END_YEAR), (nothing, nothing)),
# )
# hlines!(ax_a, 0; color = (:black, 0.4), linewidth = 0.8, linestyle = :dot)
# for (pw, r) in zip(pathways, results)
#     band!(ax_a, r.years, r.fp_p25_lo, r.fp_p75_lo; color = (pw.color, 0.20))
#     lines!(ax_a, r.years, r.fp_med_lo; color = pw.color, linewidth = 1.5, label = pw.label)
#     lines!(ax_a, r.years, r.fp_med_hi; color = pw.color, linewidth = 1.0, linestyle = :dash)
# end
# axislegend(ax_a; position = :lt, labelsize = 6, framevisible = false)

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

# ── (b) Societal benefit: COBRA (PM2.5 + NOx) + avoided SCC ──────────────────
c_health = colorant"#009988"   # teal  — COBRA health benefit
c_scc    = colorant"#117733"   # green — social cost of carbon
c_total  = colorant"#332288"   # dark purple — combined total

have_fit     = !isempty(cobra_fitted_yrs)
all_measured = have_fit && all(get(COBRA_RESULTS, yr, (0.0, 0.0))[1] > 0.0
                               for yr in results[1].years)
fit_style    = have_fit ? (all_measured ? :solid : :dash) : :dash

# ── (b) [hidden] Societal benefit ────────────────────────────────────────────
# ax_b = Axis(fig[1, 2]; ...)

# ── (c) Net societal balance: solar TCO premium vs. total societal benefit ────
# Positive = society still bears a net cost; negative = benefits exceed premium.
# Solar electrolysis pathway is results[3] (index matches pathways order).
c_solar = pathways[3].color   # "#70AD47" green

r_solar     = results[3]
solar_tco   = r_solar.tcp_med_lo          # M USD/yr, vs diesel $4.80/gal (median)
soc_benefit = have_fit ? cobra_fitted_vals .+ scc_vals : scc_vals
# Align years: both arrays cover the same range
net_vals    = solar_tco .- soc_benefit    # > 0: net cost; < 0: net benefit

# ── (c) [hidden] Net societal balance ────────────────────────────────────────
# ax_c = Axis(fig[1, 3]; ...)

# ── (b) Solar TCO premium vs. stacked societal benefit ───────────────────────
# Societal benefit as stacked areas; both diesel-reference scenarios from (a).
c_pink_health = colorant"#F48FB1"   # light pink — COBRA health benefit
c_pink_scc    = colorant"#AD1457"   # deep pink  — social cost of carbon

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

r_d = results[3]   # solar electrolysis pathway

if have_fit
    total_d = cobra_fitted_vals .+ scc_vals

    # Stacked benefit areas
    band!(ax_d, cobra_fitted_yrs, zeros(length(cobra_fitted_yrs)), cobra_fitted_vals;
          color = (c_pink_health, 0.20))
    band!(ax_d, results[1].years, cobra_fitted_vals, total_d;
          color = (c_pink_scc, 0.20))
    lines!(ax_d, cobra_fitted_yrs, cobra_fitted_vals;
           color = c_pink_health, linewidth = 1.2, linestyle = fit_style,
           label = "PM2.5 + NOx health benefit")
    lines!(ax_d, results[1].years, total_d;
           color = c_pink_scc, linewidth = 1.2,
           label = "Total societal benefit (incl. SCC)")
else
    total_d = scc_vals
    band!(ax_d, results[1].years, zeros(length(results[1].years)), total_d;
          color = (c_pink_scc, 0.20))
    lines!(ax_d, results[1].years, total_d;
           color = c_pink_scc, linewidth = 1.2, label = "SCC only")
end

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
total_c = have_fit ? cobra_fitted_vals .+ scc_vals : scc_vals

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

# ── (commented out) Avoided CO₂e ────────────────────────────────────────────
# c_co2 = colorant"#117733"
# ax_co2 = Axis(fig[2, 2];
#     title          = "(d)  Avoided CO₂e emissions (lifecycle, CA blend)",
#     titlesize      = 8, titlefont = :bold,
#     xlabel         = "Year",
#     ylabel         = "Avoided CO₂e  (kt yr⁻¹)",
#     xlabelsize     = 7, ylabelsize = 7,
#     xticklabelsize = 7, yticklabelsize = 7,
#     xticklabelrotation = π/4,
#     xticks         = 2026:2:END_YEAR,
#     limits         = ((2026, END_YEAR), (0, nothing)),
#     yticklabelcolor = c_co2, ylabelcolor = c_co2,
# )
# lines!(ax_co2, results[1].years, results[1].co2_avoided_kt; color = c_co2, linewidth = 1.5)
# text!(ax_co2, 2026.3, maximum(results[1].co2_avoided_kt) * 0.96;
#       text = "66% RD (CI=43.74) + 6% BD (CI=38.49) + 28% fossil (CI=90) gCO₂e/MJ",
#       fontsize = 5.5, color = (:black, 0.55), align = (:left, :top))

rowgap!(fig.layout, 4)
colgap!(fig.layout, 4)
resize_to_layout!(fig)

out_pdf = fig_path(OUT_DIR, "fig_cost_premium.pdf")
out_png = fig_path(OUT_DIR, "fig_cost_premium.png")
save(out_pdf, fig; pt_per_unit = 1)
println("Saved → $out_pdf")
save(out_png, fig; px_per_unit = 300/72)
println("Saved → $out_png")
