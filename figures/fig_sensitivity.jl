# =============================================================================
# FIGURE: SENSITIVITY (TORNADO) ANALYSIS
# =============================================================================
# One-at-a-time sensitivity of two key metrics to parameter changes:
#   (a) H₂ delivery price ($/kg)    — same scenario as fig_price_breakdown.jl
#   (b) H₂ truck TCO ($/mile)       — same scenario as fig_tco_comparison.jl
#
# Two output files:
#   fig_sensitivity_tornado.[pdf|png]
#       Two-panel tornado plot at END_YEAR, parameters sorted by impact.
#
#   fig_sensitivity_grid.[pdf|png]
#       8-panel grid (2 metric rows × 4 year columns: 2026, 2030, 2035, END_YEAR).
#       Parameters always in the same order as the tornado figure.
#
# Baseline scenario: electrolysis–solar × limited_dep trucks × no_change LCFS
#
# ── Adding sensitivity parameters ────────────────────────────────────────────
#   Push a new NamedTuple to SENS_PARAMS below with fields:
#     name    — row label in the plot
#     lo      — build_config keyword overrides for the low variant (affects H₂ sim)
#     hi      — build_config keyword overrides for the high variant (affects H₂ sim)
#     lo_tco  — (optional) h2_tco_per_mile keyword overrides for the low variant
#     hi_tco  — (optional) h2_tco_per_mile keyword overrides for the high variant
#     lo_lab  — text description of the low variant value (for legend/table)
#     hi_lab  — text description of the high variant value (for legend/table)
#   For TCO-only parameters set lo = hi = (;) and fill lo_tco / hi_tco.
#   Supported tco overrides: purchase_cost, truck_lr, subsidy_fixed.
#   No other changes needed.
#   A variant that overrides lcfs_price_schedule_dict also moves the diesel
#   reference; its bars are automatically drawn net of that shift (see the
#   net-of-diesel correction) so the single set of parity lines stays valid.
#
# Run from project root:
#   julia --project=figures figures/fig_sensitivity.jl
# =============================================================================

using Statistics
using CairoMakie
using LaTeXStrings
using Random
using JSON
using Printf
using LinearAlgebra

cd(joinpath(@__DIR__, ".."))
include("../hydrogen truck deployment.jl")
include("config_defaults.jl")
include("pub_theme.jl")

const N_RUNS  = 300   # MC runs per variant — increase for smoother results
const SEED    = 42
const OUT_DIR = joinpath(@__DIR__, "out")
mkpath(OUT_DIR)

# ── Deployment scenario ───────────────────────────────────────────────────────
# The whole analysis is conditional on one truck rollout path. `limited_dep` is the
# manuscript baseline; pass any other trucks_config.json scenario key as the
# first argument to rerun everything on it:
#
#   julia --project=figures figures/fig_sensitivity.jl high_dep
#
# Non-default scenarios get a filename suffix, so their figures and value tables
# sit alongside the baseline instead of overwriting it, and the grid figure
# carries a scenario banner so the two PNGs cannot be confused.
const DEP_SCENARIO = isempty(ARGS) ? get(ENV, "SENS_DEP", "limited_dep") : ARGS[1]
const DEP_SUFFIX   = DEP_SCENARIO == "limited_dep" ? "" : "_" * DEP_SCENARIO
const DEP_LABEL    = Dict("limited_dep" => "Limited deployment",
                          "high_dep"  => "High deployment",
                          "nolow_dep" => "No/low deployment")
dep_title() = get(DEP_LABEL, DEP_SCENARIO, DEP_SCENARIO)
println("Deployment scenario: $(dep_title())  [$DEP_SCENARIO]")

# ── TCO calculation helpers (mirrors fig_tco_comparison.jl) ──────────────────
let
    tco_raw = JSON.parsefile(joinpath(@__DIR__, "..", "config", "tco_config.json"))
    d   = tco_raw["diesel_tco_usd_per_mile"]
    ht  = tco_raw["hydrogen_truck"]
    id  = tco_raw["identical_cost_categories"]

    global _d_lease  = Float64(d["truck_lease_or_purchase"])
    global _d_rm     = Float64(d["repair_and_maintenance"])
    global _d_tires  = Float64(d["tires"])
    global _d_common = Float64(id["driver_and_other"])

    global _mpkg      = Float64(ht["miles_per_kg_h2"])
    global _h2_tires  = Float64(ht["tires_multiplier"]) * _d_tires
    global _h2_common = _d_common
    global _rm_mult   = Float64(ht["repair_and_maintenance_multiplier"])

    global _tco_lr      = Float64(ht["learning"]["learning_rate"])
    global _tco_ref_yr  = Int(ht["learning"]["reference_year"])
    global _tco_purch   = Float64(ht["purchase_cost_usd"])
    global _tco_plat    = Float64(ht["platform_cost_usd"])
    global _tco_fc      = _tco_purch - _tco_plat

    global _sub_sched = Dict{Int,Float64}(
        Int(s["year"]) => Float64(s["subsidy_usd"])
        for s in ht["subsidy_schedule"]
    )

    obs = ht["learning"]["observed_fleet_stock"]
    oy  = Float64.([Int(r["year"])   for r in obs])
    os  = Float64.([Int(r["trucks"]) for r in obs])
    ts  = oy .- 2019.0
    V   = hcat(ts.^3, ts.^2, ts, ones(length(ts)))
    θ   = V \ os
    global _fsa, _fsb, _fsc, _fsd = θ[1], θ[2], θ[3], θ[4]
end

_sub(yr)  = _sub_sched[maximum(filter(y -> y <= yr, keys(_sub_sched)))]
_fs(yr)   = max(1.0, _fsa*(yr-2019)^3 + _fsb*(yr-2019)^2 + _fsc*(yr-2019) + _fsd)
_h2rm(yr) = _d_rm * (_rm_mult + (1.0 - _rm_mult) * clamp((yr - 2026) / 9.0, 0.0, 1.0))

# ── Diesel price comparison helpers ───────────────────────────────────────────
const DIESEL_MPG  = TCO_DIESEL_MPG   # miles/gallon, from config/tco_config.json
const DIESEL_P_LO = TCO_DIESEL_P_LO  # $/gal, from config/tco_config.json
const DIESEL_P_HI = TCO_DIESEL_P_HI  # $/gal, from config/tco_config.json

# H₂ breakeven price ($/kg): H₂ fuel cost/mile = diesel fuel cost/mile
diesel_lcoh_equiv(p_gal) = p_gal * _mpkg / DIESEL_MPG
# Full diesel TCO ($/mile): fuel + non-fuel fixed costs (ATRI data)
diesel_tco_total(p_gal)  = p_gal / DIESEL_MPG + _d_lease + _d_rm + _d_tires + _d_common

# ── LCFS adjustment for the diesel reference (mirrors fig_abatement_cost.jl) ───
# CA diesel blend CI = CA_BLEND_CI (gCO₂e/MJ); the LCFS benchmark CI declines per
# lcfs_config.json, so the blend swings from generating LCFS credits (early) to
# deficits (late). Anchored at 2026: the $4.80/$5.80 pump prices are taken to
# embed 2026 LCFS conditions, so only the change vs 2026 adjusts later years.
# Uses the no_change ($65) credit price, matching the baseline H₂ scenario.
const DIESEL_MJ_PER_GAL = 128.45
const CA_BLEND_CI       = 0.66 * 43.74 + 0.06 * 38.49 + (1.0 - 0.66 - 0.06) * 90.0
const _benchmark_ci = Dict{Int,Float64}(
    Int(d["year"]) => Float64(d["ci"])
    for d in JSON.parsefile(joinpath(@__DIR__, "..", "config", "lcfs_config.json"))["diesel_ci_schedule"]["schedule"]
)
const _lcfs_p_diesel = load_lcfs_price_scenario("no_change")
function diesel_lcfs_per_mile(yr::Int, p_dict::AbstractDict = _lcfs_p_diesel)
    ci_std = get(_benchmark_ci, yr, _benchmark_ci[maximum(filter(y -> y <= yr, keys(_benchmark_ci)))])
    lcfs_p = get(p_dict, yr, p_dict[maximum(filter(y -> y <= yr, keys(p_dict)))])
    return (ci_std - CA_BLEND_CI) / 1e6 * DIESEL_MJ_PER_GAL * lcfs_p / DIESEL_MPG   # $/mile, + = lowers cost
end
const _lcfs_diesel_2026 = diesel_lcfs_per_mile(2026)
# anchored adjustment ($/mile, + = raises diesel cost relative to 2026)
diesel_lcfs_adj(yr::Int) = _lcfs_diesel_2026 - diesel_lcfs_per_mile(yr)

# Diesel-side cost shift ($/mile, + = diesel gets more expensive) when the credit
# price follows `p_dict` instead of the baseline no_change series. The 2026 anchor
# stays at the baseline value: the $4.80/$5.80 pump prices are observed under the
# actual ($65) 2026 market, so an alternative credit-price path moves the diesel
# reference from 2026 onward — the same convention under which the H₂ simulation
# is rerun. Independent of p_gal, so one shift serves both parity lines.
diesel_lcfs_shift(yr::Int, p_dict::AbstractDict) =
    diesel_lcfs_per_mile(yr) - diesel_lcfs_per_mile(yr, p_dict)

# LCFS-adjusted diesel reference: breakeven H₂ price ($/kg) and full TCO ($/mile)
diesel_lcoh_equiv_adj(p_gal, yr::Int) = _mpkg * (p_gal / DIESEL_MPG + diesel_lcfs_adj(yr))
diesel_tco_total_adj(p_gal, yr::Int)  =
    p_gal / DIESEL_MPG + diesel_lcfs_adj(yr) + _d_lease + _d_rm + _d_tires + _d_common

# TCO per mile.
# Simulation-derived h2_per_day / op_days come from the config.
# Optional TCO-only overrides:
#   purchase_cost  — gross truck purchase price (USD); default = tco_config value
#   truck_lr       — truck fuel-cell learning rate; default = tco_config value
#   subsidy_fixed  — if set, replaces the phase-out schedule with a fixed subsidy (USD)
function h2_tco_per_mile(lcoh::Float64, yr::Int;
                          h2_per_day::Float64                   = Float64(H2_PER_TRUCK_PER_DAY),
                          op_days::Int                          = OPERATING_DAYS_PER_YEAR,
                          purchase_cost::Float64                = _tco_purch,
                          truck_lr::Float64                     = _tco_lr,
                          subsidy_fixed::Union{Nothing,Float64} = nothing)
    ann_miles = h2_per_day * op_days * _mpkg
    fc_base   = purchase_cost - _tco_plat          # fuel cell cost = purchase − platform
    α         = log(1 / (1 - truck_lr)) / log(2)
    fc_yr     = fc_base * (_fs(yr) / _fs(_tco_ref_yr))^(-α)
    subsidy   = isnothing(subsidy_fixed) ? _sub(yr) : subsidy_fixed
    net_cost  = _tco_plat + fc_yr - subsidy
    capital   = net_cost * calculate_annuity_factor(7, DISCOUNT_RATE) / ann_miles
    return lcoh / _mpkg + capital + _h2rm(yr) + _h2_tires + _h2_common
end

# ── H₂ volume served (kg/yr) ──────────────────────────────────────────────────
# The truck fleet's own consumption plus the bus and passenger-car demand that
# the same production capacity serves — i.e. the volume the delivered price is
# spread over, not the trucks alone.
#
# Two day-count conventions coexist in the model and are preserved here rather
# than reconciled: truck volume is 38 kg/day × 250 OPERATING days
# (`h2_per_truck_per_year`, the same basis as the TCO's 85 500 annual miles),
# while `bus_demand_kg_day` / `car_demand_kg_day` are calendar-day series and so
# run × 365 — which is what get_h2_base_price does when it forms utilization.
#
# Fleet uptime ramps 30 % → 60 % → 100 % over the first three simulation years,
# by simulation year and not by truck age, so 2026 volume is only 30 % of the
# nameplate fleet consumption.
function h2_volume(cfg, n_trucks::Real, year_idx::Int, yr::Int)
    uptime = year_idx == 1 ? cfg.truck_uptime_year_1 :
             year_idx == 2 ? cfg.truck_uptime_year_2 : cfg.truck_uptime_default
    truck  = n_trucks * cfg.h2_per_truck_per_year * uptime
    other  = (bus_demand_kg_day(yr, cfg.bus_demand_scenario) +
              car_demand_kg_day(yr, cfg.car_demand_scenario)) * 365.0
    return (trucks = n_trucks, uptime = uptime,
            truck = truck, other = other, total = truck + other)
end

# ── Baseline config ───────────────────────────────────────────────────────────
BASE_KW = (
    h2_pathway_id                   = "current_mix",
    expansion_pathway_id            = "electrolysis",
    use_utilization_pricing         = true,
    utilization_transport_cost      = 1.0,
    use_lcfs                        = true,
    enable_45v                      = true,
    bus_demand_scenario             = "growing",
    end_year                        = END_YEAR,
    use_truck_deployment_schedule   = true,
    truck_deployment_schedule       = load_truck_scenario(DEP_SCENARIO),
    lcfs_price_schedule_dict        = load_lcfs_price_scenario("no_change"),
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

# ── Sensitivity parameter definitions ─────────────────────────────────────────
# Add new parameters by pushing entries to this array.
# lo / hi are NamedTuples of keyword overrides for build_config.
SENS_PARAMS = [
    (
        # Annual miles drives both H2 consumption in the simulation and the
        # capital cost per mile in TCO.  We vary h2_per_truck_per_day so that
        # h2_per_day × op_days × miles_per_kg = target annual miles, while
        # keeping operating_days_per_year at its default.
        # Bracket is ±14% about the 85 500 mi/yr baseline (38 kg/day × 250 days
        # × 9 mi/kg), keeping the tornado bar symmetric.
        name   = "Annual miles per truck",
        lo     = (h2_per_truck_per_day = 74_000.0 / (OPERATING_DAYS_PER_YEAR * _mpkg),),
        hi     = (h2_per_truck_per_day = 97_000.0 / (OPERATING_DAYS_PER_YEAR * _mpkg),),
        lo_lab = "74 000 miles/yr",
        hi_lab = "97 000 miles/yr",
    ),
    (
        name   = "Electrolyzer CAPEX (\$/kW)",
        lo     = (electrolyzer_capex_per_kw = 2000.0,),
        hi     = (electrolyzer_capex_per_kw = 4000.0,),
        lo_lab = "\$2 000/kW",
        hi_lab = "\$4 000/kW",
    ),
    (
        name   = "Solar panel CAPEX (\$/kW)",
        lo     = (solar_capex_per_kw = 1100.0,),
        hi     = (solar_capex_per_kw = 2100.0,),
        lo_lab = "\$1 100/kW",
        hi_lab = "\$2 100/kW",
    ),
    (
        name   = "Electrolyzer learning rate",
        lo     = (electrolyzer_learning_rate = 0.15,),
        hi     = (electrolyzer_learning_rate = 0.30,),
        lo_lab = "15 %  per doubling",
        hi_lab = "30 %  per doubling",
    ),
    (
        name   = "H₂ transport cost (\$/kg)",
        lo     = (utilization_transport_cost = 0.5,),
        hi     = (utilization_transport_cost = 2.0,),
        lo_lab = "\$0.50/kg",
        hi_lab = "\$2.00/kg",
    ),
    (
        name   = "Solar PV learning rate",
        lo     = (solar_panel_learning_rate = 0.12,),
        hi     = (solar_panel_learning_rate = 0.45,),
        lo_lab = "12 %  per doubling",
        hi_lab = "45 %  per doubling",
    ),
    # ── TCO-only parameters (lo = hi = (;) so H₂ price bars are zero) ──────────
    (
        name   = "Truck purchase cost",
        lo     = (;),
        hi     = (;),
        lo_tco = (purchase_cost = 450_000.0,),
        hi_tco = (purchase_cost = 850_000.0,),
        lo_lab = "\$450 000",
        hi_lab = "\$850 000",
    ),
    (
        name   = "Truck FC learning rate",
        lo     = (;),
        hi     = (;),
        lo_tco = (truck_lr = 0.10,),
        hi_tco = (truck_lr = 0.30,),
        lo_lab = "10 %  per doubling",
        hi_lab = "30 %  per doubling",
    ),
    (
        name   = "HVIP truck subsidy",
        lo     = (;),
        hi     = (;),
        lo_tco = (subsidy_fixed = 0.0,),
        hi_tco = (subsidy_fixed = 240_000.0,),
        lo_lab = "No subsidy",
        hi_lab = "\$240 000 throughout",
    ),
    # ── Policy on/off scenarios ─────────────────────────────────────────────────
    # hi = (;) means hi matches the baseline (Δhi = 0); only the lo bar appears.
    (
        name   = "45V tax credit",
        lo     = (enable_45v = false,),
        hi     = (tax_credit_45v_end_year = 2038,),
        lo_lab = "Off",
        hi_lab = "Extended to 2038",
    ),
    (
        name   = "Bus H₂ demand growth",
        lo     = (bus_demand_scenario = "flat",),
        hi     = (;),
        lo_lab = "Flat at 2026 level",
        hi_lab = "Growing (baseline)",
    ),
    (
        # Baseline LCFS credit price is flat at $65 (no_change). The "high"
        # variant applies the increasing-price scenario (reaches ceiling 2030);
        # higher credits lower the delivered H₂ price and the truck TCO.
        # This is the one parameter that also moves the diesel reference, so its
        # bars are drawn net of that shift — see the net-of-diesel correction.
        name   = "LCFS credit price",
        lo     = (;),
        hi     = (lcfs_price_schedule_dict = load_lcfs_price_scenario("high_inc"),),
        lo_lab = "Flat at \$65 (baseline)",
        hi_lab = "High increase (ceiling 2030)",
    ),
]

# ── Year grid ─────────────────────────────────────────────────────────────────
SENS_YEARS = [2026, 2030, 2035, END_YEAR]
n_params   = length(SENS_PARAMS)
n_years    = length(SENS_YEARS)

# ── Run one variant, return (lcoh_at_years, tco_at_years) ─────────────────────
# sim_overrides : build_config keyword overrides (reruns the H₂ price simulation)
# tco_kw        : h2_tco_per_mile keyword overrides applied on top of cfg values
#                 (no re-simulation; for truck cost / learning rate / subsidy)
function run_variant(sim_overrides, tco_kw = (;))
    kw    = merge(BASE_KW, sim_overrides)
    cfg   = build_config(; kw...)
    years = collect(cfg.start_year:cfg.end_year)
    Random.seed!(SEED)
    raw     = run_monte_carlo(cfg, N_RUNS)
    price_r = raw[1]   # N_RUNS × n_years delivered price
    truck_r = raw[3]   # N_RUNS × n_years trucks on the road

    h2pd = Float64(cfg.h2_per_truck_per_day)
    opd  = cfg.operating_days_per_year

    # Volume served, every simulation year (the summary reports the full series
    # as well as the four tornado years).
    vol_all = [h2_volume(cfg, median(truck_r[:, k]), k, years[k]) for k in eachindex(years)]

    lcoh_at = Float64[]
    tco_at  = Float64[]
    vol_at  = NamedTuple[]
    for yr in SENS_YEARS
        k = findfirst(==(yr), years)
        isnothing(k) && error("Year $yr not in simulation range $(years[1])–$(years[end])")
        push!(vol_at, vol_all[k])
        push!(lcoh_at, median(price_r[:, k]))
        tco_vec = [h2_tco_per_mile(price_r[r, k], yr;
                                   h2_per_day = h2pd, op_days = opd,
                                   pairs(tco_kw)...)
                   for r in 1:N_RUNS]
        push!(tco_at, median(tco_vec))
    end
    return lcoh_at, tco_at, vol_at, vol_all, years
end

# ── Execute ───────────────────────────────────────────────────────────────────
println("Running baseline simulation…")
base_lcoh, base_tco, base_vol, base_vol_all, base_years = run_variant((;))

println("Running $(n_params) × 2 sensitivity variants…")
lo_lcoh = zeros(n_params, n_years)
hi_lcoh = zeros(n_params, n_years)
lo_tco  = zeros(n_params, n_years)
hi_tco  = zeros(n_params, n_years)

for (i, sp) in enumerate(SENS_PARAMS)
    lo_tco_kw = get(sp, :lo_tco, (;))
    hi_tco_kw = get(sp, :hi_tco, (;))

    print("  [$i/$n_params] $(sp.name): lo…")
    ll, lt = run_variant(sp.lo, lo_tco_kw)
    lo_lcoh[i, :] .= ll;  lo_tco[i, :] .= lt

    print(" hi…")
    hl, ht_ = run_variant(sp.hi, hi_tco_kw)
    hi_lcoh[i, :] .= hl;  hi_tco[i, :] .= ht_
    println(" done")
end
println("All variants complete.\n")

# ── Deltas from baseline ──────────────────────────────────────────────────────
# Rows = parameters, columns = years
Δlo_lcoh_raw = lo_lcoh .- transpose(base_lcoh)
Δhi_lcoh_raw = hi_lcoh .- transpose(base_lcoh)
Δlo_tco_raw  = lo_tco  .- transpose(base_tco)
Δhi_tco_raw  = hi_tco  .- transpose(base_tco)

# ── Net-of-diesel correction ──────────────────────────────────────────────────
# The diesel parity lines below are drawn once, from the baseline (no_change)
# credit price, and are shared by every row of a panel. That is exact for all
# parameters except the LCFS credit price, which moves the diesel reference too:
# the CA diesel blend earns credits while the benchmark CI sits above CA_BLEND_CI
# and takes deficits after the crossover (~2034), so a higher credit price makes
# diesel cheaper early and dearer late. Rather than drawing a second pair of
# parity lines for that row, the bar is plotted net — H₂-side delta minus the
# diesel-side shift — so the distance from bar to parity line stays the true
# distance to parity in every row. Raw H₂-side deltas are kept for the tables.
# Only lcfs_price_schedule_dict overrides are picked up here; a variant toggling
# use_lcfs would need its own diesel-side treatment.
diesel_shift_lo = zeros(n_params, n_years)   # $/mile, + = diesel more expensive
diesel_shift_hi = zeros(n_params, n_years)
for (i, sp) in enumerate(SENS_PARAMS), (j, yr) in enumerate(SENS_YEARS)
    p_lo = get(sp.lo, :lcfs_price_schedule_dict, nothing)
    p_hi = get(sp.hi, :lcfs_price_schedule_dict, nothing)
    isnothing(p_lo) || (diesel_shift_lo[i, j] = diesel_lcfs_shift(yr, p_lo))
    isnothing(p_hi) || (diesel_shift_hi[i, j] = diesel_lcfs_shift(yr, p_hi))
end
net_adjusted = [i for i in 1:n_params
                if any(abs.(diesel_shift_lo[i, :]) .> 1e-10) ||
                   any(abs.(diesel_shift_hi[i, :]) .> 1e-10)]

# As plotted: subtract the diesel-side shift so the fixed parity lines stay valid
Δlo_lcoh = Δlo_lcoh_raw .- _mpkg .* diesel_shift_lo
Δhi_lcoh = Δhi_lcoh_raw .- _mpkg .* diesel_shift_hi
Δlo_tco  = Δlo_tco_raw  .- diesel_shift_lo
Δhi_tco  = Δhi_tco_raw  .- diesel_shift_hi

# Each panel is sorted independently by its own impact metric at END_YEAR.
# Parameters with near-zero impact across ALL years are excluded from that panel.
impact_h2  = [max(abs(Δlo_lcoh[i, n_years]), abs(Δhi_lcoh[i, n_years])) for i in 1:n_params]
impact_tco = [max(abs(Δlo_tco[i,  n_years]), abs(Δhi_tco[i,  n_years])) for i in 1:n_params]

nonzero_h2  = filter(i -> maximum(max(abs(Δlo_lcoh[i,j]), abs(Δhi_lcoh[i,j])) for j in 1:n_years) > 1e-6, 1:n_params)
nonzero_tco = filter(i -> maximum(max(abs(Δlo_tco[i,j]),  abs(Δhi_tco[i,j]))  for j in 1:n_years) > 1e-6, 1:n_params)

sorted_h2  = sort(nonzero_h2,  by = i -> impact_h2[i])   # ascending — least impact first
sorted_tco = sort(nonzero_tco, by = i -> impact_tco[i])

# ── Diesel parity reference (LCFS-adjusted) ───────────────────────────────────
# For each year, the position on the Δ axis at which the H₂ metric would equal
# the LCFS-adjusted diesel alternative at $4.80 (ATRI) and $5.80/gal. The breakeven H₂
# price depends only on the fuel-economy ratio and the baseline LCFS conditions,
# so one reference line per panel per diesel price serves every row — parameters
# that would move the diesel side are netted out of their bars instead (above).
#   > 0 (red side)   : H₂ would have to rise to reach diesel parity → H₂ cheaper
#   < 0 (green side) : H₂ already exceeds diesel → diesel cheaper
diesel_d_lcoh_lo = [diesel_lcoh_equiv_adj(DIESEL_P_LO, SENS_YEARS[j]) - base_lcoh[j] for j in 1:n_years]
diesel_d_lcoh_hi = [diesel_lcoh_equiv_adj(DIESEL_P_HI, SENS_YEARS[j]) - base_lcoh[j] for j in 1:n_years]
diesel_d_tco_lo  = [diesel_tco_total_adj(DIESEL_P_LO, SENS_YEARS[j])  - base_tco[j]  for j in 1:n_years]
diesel_d_tco_hi  = [diesel_tco_total_adj(DIESEL_P_HI, SENS_YEARS[j])  - base_tco[j]  for j in 1:n_years]

# Shared x-axis limits per metric row (used in the grid figure). Kept at the
# parameter-impact scale so the bars stay readable; diesel parity lines that
# fall outside this range are pinned at the axis edge with a numeric label.
xlim_h2  = maximum(max(abs(Δlo_lcoh[i,j]), abs(Δhi_lcoh[i,j])) for i in nonzero_h2,  j in 1:n_years)
xlim_tco = maximum(max(abs(Δlo_tco[i,j]),  abs(Δhi_tco[i,j]))  for i in nonzero_tco, j in 1:n_years)

# Net-adjusted rows carry a dagger, explained in the footnote under each figure.
const NET_MARK = " †"
param_labels = [i in net_adjusted ? SENS_PARAMS[i].name * NET_MARK : SENS_PARAMS[i].name
                for i in 1:n_params]
const NET_NOTE = isempty(net_adjusted) ? "" :
    "† " * join([SENS_PARAMS[i].name for i in net_adjusted], ", ") *
    " bars are net of the same variant's effect on the diesel reference, " *
    "so they remain comparable with the parity lines."

# ── Print summary tables ──────────────────────────────────────────────────────
# One table per year column of the grid figure, so every panel in
# fig_sensitivity_grid can be read off numerically. Absolute levels (Lo / Hi)
# come with the signed deltas from baseline that the bars actually draw, plus the
# LCFS-adjusted diesel parity position for that year. Written to both stdout and
# out/fig_sensitivity_values.txt.
function write_summary(io)
    # ── Volume served, full annual series ─────────────────────────────────────
    # Baseline configuration only. Two sensitivity parameters move this table:
    # "Annual miles per truck" rescales the truck column, and "Bus H₂ demand
    # growth" holds the bus series flat at its 2026 level (8 000 kg/day).
    println(io, "HYDROGEN SOLD  —  $(dep_title())  (baseline configuration, tonnes/yr)")
    println(io, "="^118)
    println(io, "  Trucks: median fleet × $(Int(H2_PER_TRUCK_PER_YEAR)) kg/truck/yr " *
                "($(Int(H2_PER_TRUCK_PER_DAY)) kg/day × $(OPERATING_DAYS_PER_YEAR) operating days) × fleet uptime.")
    println(io, "  Bus + car: bus_demand_kg_day(\"$(BASE_KW.bus_demand_scenario)\") + car_demand_kg_day, " *
                "both calendar-day series × 365 — the basis get_h2_base_price uses for utilization.")
    println(io, "-"^118)
    @printf(io, "  %-6s %8s %8s   %12s %12s %12s   %12s\n",
            "Year", "Trucks", "Uptime", "Trucks t/yr", "Bus+car t/yr", "Total t/yr", "Total kg/day")
    println(io, "-"^118)
    for (k, yr) in enumerate(base_years)
        v = base_vol_all[k]
        @printf(io, "  %-6d %8d %7.0f%%   %12.0f %12.0f %12.0f   %12.0f\n",
                yr, round(Int, v.trucks), 100 * v.uptime,
                v.truck / 1000, v.other / 1000, v.total / 1000, v.total / 365)
    end
    println(io, "="^118)

    for (j, yr) in enumerate(SENS_YEARS)
        println(io)
        println(io, "SENSITIVITY SUMMARY  —  $yr  —  $(dep_title())  (median of $N_RUNS runs)")
        println(io, "="^118)
        @printf(io, "  baseline:  \$%.3f/kg H₂   |   \$%.4f/mile TCO\n",
                base_lcoh[j], base_tco[j])
        @printf(io, "  H₂ sold:   %.0f t/yr total  =  %.0f t trucks (%d trucks @ %.0f%% uptime)  +  %.0f t bus/car\n",
                base_vol[j].total / 1000, base_vol[j].truck / 1000,
                round(Int, base_vol[j].trucks), 100 * base_vol[j].uptime,
                base_vol[j].other / 1000)
        @printf(io, "  diesel parity (LCFS-adj.):  \$%.2f/gal → \$%.3f/kg, \$%.4f/mi   ",
                DIESEL_P_LO, diesel_lcoh_equiv_adj(DIESEL_P_LO, yr), diesel_tco_total_adj(DIESEL_P_LO, yr))
        @printf(io, "|  \$%.2f/gal → \$%.3f/kg, \$%.4f/mi\n",
                DIESEL_P_HI, diesel_lcoh_equiv_adj(DIESEL_P_HI, yr), diesel_tco_total_adj(DIESEL_P_HI, yr))
        @printf(io, "  → as plotted (Δ from baseline):  \$%.2f/gal  %+.3f \$/kg, %+.4f \$/mi   ",
                DIESEL_P_LO, diesel_d_lcoh_lo[j], diesel_d_tco_lo[j])
        @printf(io, "|  \$%.2f/gal  %+.3f \$/kg, %+.4f \$/mi\n",
                DIESEL_P_HI, diesel_d_lcoh_hi[j], diesel_d_tco_hi[j])
        println(io, "-"^118)
        @printf(io, "%-32s  %8s %8s  %7s %7s   %9s %9s  %8s %8s\n",
                "Parameter", "Lo \$/kg", "Hi \$/kg", "Δlo", "Δhi",
                "Lo \$/mi", "Hi \$/mi", "Δlo", "Δhi")
        println(io, "-"^118)
        for i in reverse(sorted_h2)
            @printf(io, "%-32s  %8.3f %8.3f  %+7.3f %+7.3f   %9.4f %9.4f  %+8.4f %+8.4f\n",
                    param_labels[i],
                    lo_lcoh[i, j], hi_lcoh[i, j], Δlo_lcoh_raw[i, j], Δhi_lcoh_raw[i, j],
                    lo_tco[i, j],  hi_tco[i, j],  Δlo_tco_raw[i, j],  Δhi_tco_raw[i, j])
        end
        # TCO-only parameters carry no H₂-price bar, so they are absent from
        # sorted_h2 — list them after, with the price columns left blank.
        for i in reverse(sorted_tco)
            i in sorted_h2 && continue
            @printf(io, "%-32s  %8s %8s  %7s %7s   %9.4f %9.4f  %+8.4f %+8.4f\n",
                    param_labels[i], "—", "—", "—", "—",
                    lo_tco[i, j], hi_tco[i, j], Δlo_tco_raw[i, j], Δhi_tco_raw[i, j])
        end

        # Rows whose variant also moves the diesel reference: the Δ columns above
        # are the raw H₂-side response (Lo/Hi minus baseline); the bars are drawn
        # net of the diesel shift so they can be read against the parity lines.
        if !isempty(net_adjusted)
            println(io, "-"^118)
            println(io, "  net of the diesel-side LCFS effect (what the † bars draw):")
            for i in net_adjusted
                for (tag, shift, dl, dt) in
                        (("lo", diesel_shift_lo[i, j], Δlo_lcoh[i, j], Δlo_tco[i, j]),
                         ("hi", diesel_shift_hi[i, j], Δhi_lcoh[i, j], Δhi_tco[i, j]))
                    abs(shift) < 1e-10 && continue
                    @printf(io, "    %-28s %s:  diesel %+.4f \$/mi  →  bar %+.3f \$/kg, %+.4f \$/mi\n",
                            SENS_PARAMS[i].name, tag, shift, dl, dt)
                end
            end
        end
        println(io, "="^118)
    end
end

write_summary(stdout)
let path = joinpath(OUT_DIR, "fig_sensitivity_values$(DEP_SUFFIX).txt")
    open(io -> write_summary(io), path, "w")
    println("\nSaved → $path")
end

# ── Shared styling (low/high variant palette is figure-specific) ───────────────
c_lo      = colorant"#F2C800"   # yellow — low variant
c_hi      = colorant"#1565C0"   # blue   — high variant
c_diesel  = C_DIESEL            # diesel parity reference (article grey)
HALF_BAR  = 0.27

# ── Core drawing function ─────────────────────────────────────────────────────
# Draws a tornado panel into ax.
# Each parameter occupies one row; lo and hi variants get separate sub-bars.
function draw_tornado!(ax, Δlo_col, Δhi_col, sorted_idx, param_labels;
                       halfbar = HALF_BAR, show_yticks = true,
                       diesel_lo = nothing, diesel_hi = nothing, xmax = Inf)
    # Background: cost-decrease (left) = faded green; cost-increase (right) = faded red
    poly!(ax, Rect(-1f6, -1f6, 1f6, 2f6); color = (:green, 0.12), strokewidth = 0)
    poly!(ax, Rect(0f0,  -1f6, 1f6, 2f6); color = (:red,       0.12), strokewidth = 0)

    n = length(sorted_idx)

    # Diesel parity reference (LCFS-adjusted): low price dash-dot, $5.80 dashed —
    # matching the suite convention that the high price carries LS_DIESEL_HI
    # (:dash). Solid is unavailable here: it is the zero line.
    # Parities within the axis range draw a full vertical line; those beyond it
    # are collected per side and reported together in a single boxed annotation
    # (just below the panel midline) with an arrow pointing off-screen.
    offscreen = Dict(:left => Tuple{Float64,Float64}[], :right => Tuple{Float64,Float64}[])
    for (dval, price, style) in ((diesel_lo, DIESEL_P_LO, :dashdot),
                                 (diesel_hi, DIESEL_P_HI, LS_DIESEL_HI))
        isnothing(dval) && continue
        if abs(dval) <= xmax
            vlines!(ax, [dval]; color = c_diesel, linewidth = 1.2, linestyle = style)
        else
            push!(offscreen[dval > 0 ? :right : :left], (price, dval))
        end
    end

    for (side, refs) in offscreen
        isempty(refs) && continue
        s     = side == :right ? 1.0 : -1.0
        nl    = length(refs)
        lh    = 0.46                       # label line height (data rows)
        bh    = nl * lh + 0.12             # box height — snug around the text
        bw    = xmax * 0.72                # box width — fits "$4.80:  -18.4" with padding
        yc    = 0.5 + n * 0.40             # just below the panel midline
        inset = xmax * 0.10                # gap between box and the panel edge
        xshift = xmax * 0.06               # nudge the whole annotation rightward

        # Box near (but offset from) the axis edge, extending inward; faint fill.
        x_edge  = s * (xmax - inset) + xshift      # box side nearest the panel edge
        x_inner = x_edge - s * bw                  # box side toward the centre
        poly!(ax, Rect(min(x_edge, x_inner), yc - bh / 2, bw, bh);
              color = (:white, 0.5), strokecolor = c_diesel, strokewidth = 0.7)
        for (k, (price, dval)) in enumerate(refs)
            CairoMakie.text!(ax, (x_edge + x_inner) / 2, yc + (nl - 1) * lh / 2 - (k - 1) * lh;
                  text = @sprintf("\$%.2f:  %+.1f", price, dval),
                  color = c_diesel, fontsize = 5.5, align = (:center, :center))
        end

        # Arrow runs from the box's vertical centre to the panel edge (drawn on
        # top), tip fully visible at the frame to signal the parity is off-screen.
        xtip = s * xmax * 0.975 + xshift
        lines!(ax, [x_edge, xtip], [yc, yc]; color = c_diesel, linewidth = 1.3, overdraw = true)
        CairoMakie.scatter!(ax, [xtip], [yc];
               marker = s > 0 ? :rtriangle : :ltriangle, color = c_diesel, markersize = 8,
               overdraw = true)
    end

    for (rank, idx) in enumerate(sorted_idx)
        y  = Float64(rank)
        dl = Δlo_col[idx]
        dh = Δhi_col[idx]

        # Lo variant — upper sub-bar
        abs(dl) > 1e-10 && poly!(ax,
            Rect(min(0.0, dl), y + 0.02, abs(dl), halfbar); color = (c_lo, 0.80))

        # Hi variant — lower sub-bar
        abs(dh) > 1e-10 && poly!(ax,
            Rect(min(0.0, dh), y - halfbar - 0.02, abs(dh), halfbar); color = (c_hi, 0.80))
    end

    vlines!(ax, [0.0]; color = :black, linewidth = 0.8)
    ax.yticks = (collect(1.0:Float64(n)), param_labels[sorted_idx])
    CairoMakie.ylims!(ax, 0.5, Float64(n) + 0.5)
    ax.yticklabelsvisible = show_yticks
end

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Tornado at END_YEAR (two panels side by side)
# ─────────────────────────────────────────────────────────────────────────────
fig1 = Figure(size = (W_DOUBLE, 140 * MM_TO_PT))

bl_str_h2  = @sprintf("%.2f", base_lcoh[n_years])
bl_str_tco = @sprintf("%.4f", base_tco[n_years])

# Explicit x-limits for fig1 (END_YEAR data only) so shading doesn't shift auto-limits.
xlim1_h2  = maximum(max(abs(Δlo_lcoh[i, n_years]), abs(Δhi_lcoh[i, n_years])) for i in nonzero_h2)  * 1.05
xlim1_tco = maximum(max(abs(Δlo_tco[i, n_years]),  abs(Δhi_tco[i, n_years]))  for i in nonzero_tco) * 1.05

ax1a = Axis(fig1[1, 1];
    title          = "(a)  H₂ fuel cost — $END_YEAR  (baseline: \$$bl_str_h2 /kg)",
    titlesize      = 8,
    xlabel         = L"$\Delta$ H$_2$ fuel cost (USD kg$^{-1}$)",
    limits         = ((-xlim1_h2, xlim1_h2), nothing),
)
draw_tornado!(ax1a, Δlo_lcoh[:, n_years], Δhi_lcoh[:, n_years], sorted_h2, param_labels;
              diesel_lo = diesel_d_lcoh_lo[n_years], diesel_hi = diesel_d_lcoh_hi[n_years],
              xmax = xlim1_h2)

ax1b = Axis(fig1[1, 2];
    title          = "(b)  Truck TCO — $END_YEAR  (baseline: \$$bl_str_tco /mile)",
    titlesize      = 8,
    xlabel         = L"$\Delta$ TCO (USD mile$^{-1}$)",
    limits         = ((-xlim1_tco, xlim1_tco), nothing),
)
draw_tornado!(ax1b, Δlo_tco[:, n_years], Δhi_tco[:, n_years], sorted_tco, param_labels;
              diesel_lo = diesel_d_tco_lo[n_years], diesel_hi = diesel_d_tco_hi[n_years],
              xmax = xlim1_tco)

# Legend with low/high labels from the first parameter as example
Legend(fig1[2, 1:2],
    [PolyElement(color = (c_lo, 0.80)), PolyElement(color = (c_hi, 0.80)),
     LineElement(color = c_diesel, linestyle = :dashdot, linewidth = 1.2),
     LineElement(color = c_diesel, linestyle = LS_DIESEL_HI, linewidth = 1.2)],
    ["Low variant", "High variant",
     @sprintf("Diesel parity (\$%.2f/gal)", DIESEL_P_LO),
     @sprintf("Diesel parity (\$%.2f/gal)", DIESEL_P_HI)];
    orientation  = :horizontal,
    nbanks       = 2,
    tellwidth    = false,
    framevisible = false,
    labelsize    = 7,
)

isempty(NET_NOTE) || Label(fig1[3, 1:2]; text = NET_NOTE, fontsize = 6,
                           halign = :left, tellwidth = false)

colgap!(fig1.layout, 8)
rowgap!(fig1.layout, 4)
resize_to_layout!(fig1)

save_pub("fig_sensitivity_tornado$(DEP_SUFFIX)", fig1)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Sensitivity grid — 2 metric rows × 4 year columns
# ─────────────────────────────────────────────────────────────────────────────
# Rows: (1) H₂ price  (2) TCO
# Cols: 2026, 2030, 2035, END_YEAR
# Parameters always in same sorted order as tornado figure.
# Y-axis labels shown on leftmost column only.
# ─────────────────────────────────────────────────────────────────────────────
const PT_PER_PARAM = 16   # points per parameter row — adjust to taste

fig2 = Figure(size = (W_DOUBLE, 210 * MM_TO_PT))

for (col, yr_idx) in enumerate(eachindex(SENS_YEARS))
    yr         = SENS_YEARS[yr_idx]
    show_y     = (col == 1)
    col_title  = string(yr)

    # Row 1: H₂ price
    ax_h = Axis(fig2[1, col];
        title          = col_title,
        titlesize      = 8, titlefont = :bold,
        xlabel         = L"$\Delta$ H$_2$ fuel cost (USD kg$^{-1}$)",
        xlabelsize     = 6,
        xticklabelsize = 6, yticklabelsize = 6,
        limits         = ((-xlim_h2 * 1.05, xlim_h2 * 1.05), nothing),
    )
    draw_tornado!(ax_h, Δlo_lcoh[:, yr_idx], Δhi_lcoh[:, yr_idx],
                  sorted_h2, param_labels;
                  halfbar = 0.22, show_yticks = show_y,
                  diesel_lo = diesel_d_lcoh_lo[yr_idx], diesel_hi = diesel_d_lcoh_hi[yr_idx],
                  xmax = xlim_h2 * 1.05)

    # Row 2: TCO
    ax_t = Axis(fig2[2, col];
        xlabel         = L"$\Delta$ TCO (USD mile$^{-1}$)",
        xlabelsize     = 6,
        xticklabelsize = 6, yticklabelsize = 6,
        limits         = ((-xlim_tco * 1.05, xlim_tco * 1.05), nothing),
    )
    draw_tornado!(ax_t, Δlo_tco[:, yr_idx], Δhi_tco[:, yr_idx],
                  sorted_tco, param_labels;
                  halfbar = 0.22, show_yticks = show_y,
                  diesel_lo = diesel_d_tco_lo[yr_idx], diesel_hi = diesel_d_tco_hi[yr_idx],
                  xmax = xlim_tco * 1.05)
end

# Row heights scale with parameter count so bar spacing is consistent — change PT_PER_PARAM to adjust
rowsize!(fig2.layout, 1, Fixed(length(sorted_h2)  * PT_PER_PARAM))
rowsize!(fig2.layout, 2, Fixed(length(sorted_tco) * PT_PER_PARAM))

# Row labels on the right margin
Label(fig2[1, 0]; text = "H₂ fuel cost\n(USD/kg)",
      rotation = π/2, tellheight = false, fontsize = 7, font = :bold)
Label(fig2[2, 0]; text = "Truck TCO\n(USD/mile)",
      rotation = π/2, tellheight = false, fontsize = 7, font = :bold)

Legend(fig2[3, 1:4],
    [PolyElement(color = (c_lo, 0.80)), PolyElement(color = (c_hi, 0.80)),
     LineElement(color = c_diesel, linestyle = :dashdot, linewidth = 1.2),
     LineElement(color = c_diesel, linestyle = LS_DIESEL_HI, linewidth = 1.2)],
    ["Low variant", "High variant",
     @sprintf("Diesel parity (\$%.2f/gal)", DIESEL_P_LO),
     @sprintf("Diesel parity (\$%.2f/gal)", DIESEL_P_HI)];
    orientation  = :horizontal,
    nbanks       = 2,
    tellwidth    = false,
    framevisible = false,
    labelsize    = 7,
)

# Scenario banner — only for non-baseline runs, so the manuscript figure (which
# is the limited-deployment case by definition) is left exactly as it was. Placed in
# a new row BELOW the legend rather than above the panels: the `rowsize!` calls
# above address rows 1 and 2 by index, and prepending a row would shift them.
isempty(NET_NOTE) || Label(fig2[4, 1:4]; text = NET_NOTE, fontsize = 6,
                           halign = :left, tellwidth = false)

if !isempty(DEP_SUFFIX)
    Label(fig2[5, 1:4]; text = "Deployment scenario: $(dep_title())",
          fontsize = 8, font = :bold, tellwidth = false)
end

colgap!(fig2.layout, 4)
rowgap!(fig2.layout, 4)
resize_to_layout!(fig2)

save_pub("fig_sensitivity_grid$(DEP_SUFFIX)", fig2)
