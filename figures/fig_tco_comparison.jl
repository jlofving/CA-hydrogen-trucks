# =============================================================================
# FIGURE: TOTAL COST OF OWNERSHIP — H2 TRUCKS vs. DIESEL
# =============================================================================
# Two-panel figure:
#   (a) Total TCO $/mile over time (2026–2040): 3 H2 pathways + diesel baseline
#       H2 lines show median; shaded bands show P25–P75 from Monte Carlo.
#   (b) TCO component breakdown at 2035: stacked bars for diesel and 3 H2
#       pathways showing how each cost category contributes.
#
# Truck scenario : limited_dep (limited deployment)
# LCFS scenario  : no_change (flat LCFS price)
# H2 pathways    : SMR (current mix), electrolysis–grid, electrolysis–solar
#
# Run from project root:
#   julia --project=figures figures/fig_tco_comparison.jl
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
# Load TCO parameters
# ─────────────────────────────────────────────────────────────────────────────
tco_raw = JSON.parsefile(joinpath(@__DIR__, "..", "config", "tco_config.json"))
diesel  = tco_raw["diesel_tco_usd_per_mile"]
ht      = tco_raw["hydrogen_truck"]
ident   = tco_raw["identical_cost_categories"]

# Diesel pump-price scenarios and fuel economy (config/tco_config.json)
const DIESEL_MPG  = TCO_DIESEL_MPG   # miles/gallon
const DIESEL_P_LO = TCO_DIESEL_P_LO  # $/gal — lower scenario
const DIESEL_P_HI = TCO_DIESEL_P_HI  # $/gal — upper scenario

# Diesel components. NOTE: the fuel component is the California pump-price
# scenario ($/gal ÷ mpg), NOT ATRI's `fuel` field — that is a national-average
# operating cost, too low to represent California. Only the non-fuel ATRI
# components feed the stack below.
d_fuel    = DIESEL_P_LO / DIESEL_MPG
d_capital = diesel["truck_lease_or_purchase"]
d_rm      = diesel["repair_and_maintenance"]
d_tires   = diesel["tires"]
d_common  = ident["driver_and_other"]
d_total   = d_fuel + d_capital + d_rm + d_tires + d_common

# ─────────────────────────────────────────────────────────────────────────────
# Diesel LCFS adjustment
# ─────────────────────────────────────────────────────────────────────────────
# Current California diesel blend fractions and carbon intensities (gCO₂e/MJ)
#   66% renewable diesel (CI = 43.74), 6% biodiesel (CI = 38.49), 28% fossil diesel (CI = 90)
#   RD/BD CIs are 2025 quarterly averages from LCFS data
rd_frac = 0.66;  rd_ci = 43.74
bd_frac = 0.06;  bd_ci = 38.49
fd_frac = 1.0 - rd_frac - bd_frac;  fd_ci = 90.0   # assumed, to be updated later
blend_ci = rd_frac * rd_ci + bd_frac * bd_ci + fd_frac * fd_ci   # gCO₂e/MJ

# LCFS benchmark CI schedule (declining annual standard from lcfs_config.json)
lcfs_raw = JSON.parsefile(joinpath(@__DIR__, "..", "config", "lcfs_config.json"))
benchmark_ci = Dict{Int,Float64}(
    d["year"] => Float64(d["ci"])
    for d in lcfs_raw["diesel_ci_schedule"]["schedule"]
)

# Physical constants
const DIESEL_MJ_PER_GAL = 128.45   # LHV energy content, MJ/gallon

# LCFS credit/deficit value per mile driven on diesel.
# Positive = net credit (reduces effective fuel cost); negative = net deficit (increases cost).
# The annual LCFS price is read from the same no_change price schedule used for H2.
function diesel_lcfs_per_mile(year::Int, price_schedule::Dict{Int,Float64})
    ci_std    = get(benchmark_ci, year, benchmark_ci[maximum(filter(y -> y <= year, keys(benchmark_ci)))])
    lcfs_p    = get(price_schedule, year, price_schedule[maximum(filter(y -> y <= year, keys(price_schedule)))])
    credits   = (ci_std - blend_ci) / 1e6 * DIESEL_MJ_PER_GAL   # tCO₂e/gallon (can be negative)
    return credits * lcfs_p / DIESEL_MPG                          # $/mile
end

println("Diesel blend CI = $(round(blend_ci, digits=2)) gCO₂e/MJ  " *
        "(66% RD@42 + 6% BD@35 + $(round(fd_frac*100, digits=0))% FD@90)")
let lcfs_2026 = diesel_lcfs_per_mile(2026, Dict{Int,Float64}(2026=>65.0))
    println("LCFS credit value embedded in 2026 fuel cost: \$$(round(lcfs_2026, digits=4))/mile")
end

# H2 truck parameters — split into mature platform (fixed) and fuel cell system (learning)
# Learning is applied to the gross fuel cell cost (purchase - platform) BEFORE subtracting
# the HVIP rebate, so the subsidy amplifies the benefit of learning.
truck_purchase_cost  = Float64(ht["purchase_cost_usd"])        # 650,000 — gross purchase price
truck_platform_cost  = Float64(ht["platform_cost_usd"])        # 150,000 — fixed, no learning
truck_fuelcell_cost  = truck_purchase_cost - truck_platform_cost  # 500,000 — gross fuel cell cost, learning applies

# HVIP subsidy phase-out schedule: step-down at listed years (edit tco_config.json to adjust)
_subsidy_sched = Dict{Int,Float64}(
    d["year"] => Float64(d["subsidy_usd"])
    for d in ht["subsidy_schedule"]
)
function truck_subsidy_yr(year::Int)
    # Use the most recent scheduled year at or before `year`
    yr = maximum(filter(y -> y <= year, keys(_subsidy_sched)))
    return _subsidy_sched[yr]
end
miles_per_kg_h2      = Float64(ht["miles_per_kg_h2"])
rm_mult          = Float64(ht["repair_and_maintenance_multiplier"])
tire_mult        = Float64(ht["tires_multiplier"])
# R&M declines linearly from rm_mult× diesel (2026) to 1× diesel (2035)
h2_rm_per_mile(year::Int) = d_rm * (rm_mult + (1.0 - rm_mult) * clamp((year - 2026) / 9.0, 0.0, 1.0))
h2_tires         = tire_mult * d_tires
h2_common        = d_common   # identical to diesel

learning_rate    = Float64(ht["learning"]["learning_rate"])
ref_year         = Int(ht["learning"]["reference_year"])
stock_obs_raw    = ht["learning"]["observed_fleet_stock"]
obs_years        = Float64.([Int(d["year"])  for d in stock_obs_raw])
obs_stock        = Float64.([Int(d["trucks"]) for d in stock_obs_raw])

# ─────────────────────────────────────────────────────────────────────────────
# Truck fleet stock projection: cubic OLS fit to observed stock data
# ─────────────────────────────────────────────────────────────────────────────
# Fit stock = a*t³ + b*t² + c*t + d, t = year - 2019, via OLS Vandermonde matrix.
# The fitted cubic Q(year) is then used as the cumulative production driver
# for Wright's Law (stock ≈ cumulative production during the early fleet
# build-out, where retirements are negligible).

let
    ts = obs_years .- 2019.0
    V  = hcat(ts.^3, ts.^2, ts, ones(length(ts)))
    θ  = V \ obs_stock
    global _lr_a, _lr_b, _lr_c, _lr_d = θ[1], θ[2], θ[3], θ[4]
end

# Projected global H2 fleet stock (vehicles) in a given year
fleet_stock(year::Int) = max(1.0, _lr_a*(year-2019)^3 + _lr_b*(year-2019)^2 + _lr_c*(year-2019) + _lr_d)

# Learning-rate-adjusted net truck cost in a given year.
# Learning reduces the gross fuel cell cost first; the HVIP rebate is then subtracted.
# This means the rebate applies to the post-learning (lower) gross price.
function truck_net_cost_yr(year::Int)
    α = log(1 / (1 - learning_rate)) / log(2)
    fuelcell_cost_yr = truck_fuelcell_cost * (fleet_stock(year) / fleet_stock(ref_year))^(-α)
    return truck_platform_cost + fuelcell_cost_yr - truck_subsidy_yr(year)
end

# ─────────────────────────────────────────────────────────────────────────────
# TCO calculation helpers
# ─────────────────────────────────────────────────────────────────────────────

# Annual mileage per truck (derived from model operational parameters)
function annual_miles_per_truck()
    # Uses same h2_per_truck_per_day and operating_days_per_year as the main model
    h2_per_day   = H2_PER_TRUCK_PER_DAY      # kg/truck/day (from model constants)
    op_days      = OPERATING_DAYS_PER_YEAR    # days/year
    return h2_per_day * op_days * miles_per_kg_h2
end

# Capital cost per mile (annualized, learning-adjusted) for H2 truck
function h2_capital_per_mile(year::Int)
    net_cost     = truck_net_cost_yr(year)
    ann_factor   = calculate_annuity_factor(7, DISCOUNT_RATE)   # 7yr truck lifetime
    ann_cost     = net_cost * ann_factor
    return ann_cost / annual_miles_per_truck()
end

# Total H2 TCO per mile given an LCOH ($/kg)
function h2_tco(lcoh::Float64, year::Int)
    fuel    = lcoh / miles_per_kg_h2
    capital = h2_capital_per_mile(year)
    return fuel + capital + h2_rm_per_mile(year) + h2_tires + h2_common
end

# ─────────────────────────────────────────────────────────────────────────────
# Simulation setup — 3 H2 pathways
# ─────────────────────────────────────────────────────────────────────────────
pathways = [
    (label = "SMR (current mix)",    short = "SMR",         color = C_SMR,   kind = :smr),
    (label = "Electrolysis — grid",  short = "Grid elec.",  color = C_GRID,  kind = :grid),
    (label = "Electrolysis — solar", short = "Solar elec.", color = C_SOLAR, kind = :solar),
]

lcfs_prices  = load_lcfs_price_scenario("no_change")

function make_config(kind, sched)
    base = (
        h2_pathway_id                 = "current_mix",
        use_utilization_pricing       = true,
        utilization_transport_cost    = 1.0,
        use_lcfs                      = true,
        enable_45v                    = true,
        bus_demand_scenario           = "growing",
        end_year                      = END_YEAR,
        use_truck_deployment_schedule = true,
        truck_deployment_schedule     = sched,
        lcfs_price_schedule_dict      = lcfs_prices,
    )
    if kind == :smr
        build_config(;
            base...,
            expansion_pathway_id         = "current_mix",
            electrolysis_pricing_enabled = false,
        )
    elseif kind == :grid
        build_config(;
            base...,
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
        build_config(;
            base...,
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
# Run Monte Carlo and convert LCOH → TCO, for each truck-deployment scenario
# ─────────────────────────────────────────────────────────────────────────────
function run_pathways(scenario_key)
    sched = load_truck_scenario(scenario_key)
    map(pathways) do pw
        cfg     = make_config(pw.kind, sched)
        Random.seed!(SEED)
        price_r = run_monte_carlo(cfg, N_RUNS)[1]   # matrix: N_RUNS × n_years
        years   = collect(cfg.start_year : cfg.end_year)
        print("  [$scenario_key] $(pw.label)…")

        # Convert each MC LCOH draw to TCO per mile
        tco_r = [h2_tco(price_r[r, k], years[k]) for r in 1:N_RUNS, k in eachindex(years)]

        r = (
            years    = years,
            median_t = [median(tco_r[:, k])           for k in eachindex(years)],
            p25      = [quantile(tco_r[:, k], 0.25)   for k in eachindex(years)],
            p75      = [quantile(tco_r[:, k], 0.75)   for k in eachindex(years)],
            # Median LCOH for breakdown panel
            median_lcoh = [median(price_r[:, k])       for k in eachindex(years)],
        )
        println(" done (median $(END_YEAR) TCO: \$$(round(r.median_t[end], digits=3))/mile)")
        r
    end
end

println("Running $(length(pathways)) pathways × 2 deployments ($(N_RUNS) runs each)…")
results      = run_pathways("limited_dep")   # limited deployment  → solid lines
results_high = run_pathways("high_dep")    # high deployment → dashed lines
println("All simulations complete.\n")

# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute deterministic capital cost trajectory (for annotation)
# ─────────────────────────────────────────────────────────────────────────────
years_plot = results[1].years

# ─────────────────────────────────────────────────────────────────────────────
# LCFS-adjusted diesel TCO trajectory
# ─────────────────────────────────────────────────────────────────────────────
# The 2026 diesel fuel cost ($d_fuel/mile at $4.80/gal) is taken to already embed
# the LCFS credit value
# for 2026. For each future year, we compute how much that credit/deficit value
# changes and adjust the fuel component accordingly.
lcfs_2026_per_mile = diesel_lcfs_per_mile(2026, lcfs_prices)
diesel_fuel_lcfs   = [d_fuel + (lcfs_2026_per_mile - diesel_lcfs_per_mile(yr, lcfs_prices))
                      for yr in years_plot]
diesel_total_lcfs  = diesel_fuel_lcfs .+ d_capital .+ d_rm .+ d_tires .+ d_common

# Higher diesel-price scenario: $5.80/gal (replaces the ATRI fuel base, same LCFS
# credit/deficit adjustment over time). Fuel $/mile = price/gal ÷ MPG.
d_fuel_hi           = DIESEL_P_HI / DIESEL_MPG
diesel_fuel_lcfs_hi = [d_fuel_hi + (lcfs_2026_per_mile - diesel_lcfs_per_mile(yr, lcfs_prices))
                       for yr in years_plot]
diesel_total_hi     = diesel_fuel_lcfs_hi .+ d_capital .+ d_rm .+ d_tires .+ d_common

let yr_cross = findfirst(v -> v < blend_ci, [get(benchmark_ci, y, NaN) for y in years_plot])
    if !isnothing(yr_cross)
        println("Diesel blend generates LCFS deficits from $(years_plot[yr_cross]) onward " *
                "(blend CI $(round(blend_ci,digits=1)) > benchmark CI $(round(get(benchmark_ci,years_plot[yr_cross],NaN),digits=1)))")
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────
fig = Figure(size = (W_DOUBLE, 95 * MM_TO_PT))

c_diesel = C_DIESEL

# ── (a) Total TCO over time ────────────────────────────────────────────────
ax_a = Axis(fig[1, 1];
    title      = "(a)  Total cost of ownership",
    xlabel     = "Year",
    ylabel     = L"TCO (USD mile$^{-1}$)",
    xticks     = [2026, 2030, 2035, 2040, 2045], xticklabelrotation = π/4,
    limits     = ((2026, END_YEAR), (0, nothing)),
)

# Diesel flat reference line (held constant)
#hlines!(ax_a, d_total; color = c_diesel, linewidth = 1.5, linestyle = :dash)
#text!(ax_a, 2026.15, d_total + 0.04;
      #text = "Diesel flat \$$(round(d_total, digits=2))/mile",
      #fontsize = 6, color = c_diesel, align = (:left, :bottom))

# Diesel LCFS-adjusted lines: $4.80/gal + $5.80/gal pump-price scenarios
c_diesel_lcfs = C_DIESEL
lines!(ax_a, Float64.(years_plot), diesel_total_lcfs;
       color = c_diesel_lcfs, linewidth = 1.5, linestyle = :dashdot,
       label = "Diesel $(usd_per_gal(DIESEL_P_LO))")
lines!(ax_a, Float64.(years_plot), diesel_total_hi;
       color = c_diesel_lcfs, linewidth = 1.5, linestyle = :dot,
       label = "Diesel $(usd_per_gal(DIESEL_P_HI))")

# H2 pathways — solid = limited deployment (with P25–P75 band), dashed = high deployment
for (pw, r, rh) in zip(pathways, results, results_high)
    band!(ax_a, r.years, r.p25, r.p75; color = (pw.color, 0.25))
    lines!(ax_a, r.years, r.median_t;   color = pw.color, linewidth = 1.5, label = pw.label)
    lines!(ax_a, rh.years, rh.median_t; color = pw.color, linewidth = 1.3, linestyle = :dash)
end

axislegend(ax_a; position = :rb, framevisible = true)
axislegend(ax_a,
    [LineElement(color = :black, linestyle = :solid, linewidth = 1.5),
     LineElement(color = :black, linestyle = :dash,  linewidth = 1.3)],
    ["Limited deployment", "High deployment"];
    position = :lb, framevisible = true, rowgap = 1)

# ── (b) Component breakdown ────────────────────────────────────────────────
# Bars are grouped by scenario; within a group they touch (no gap), with a small
# gap separating groups. Groups (top → bottom):
#   Diesel        : $4.80/gal + $5.80/gal, in 2026 and 2045 (grey-shaded)
#   H₂ 2026       : one shared bar (both deployments equal in the first year)
#   H₂ 2045 demo  : 3 pathways
#   H₂ 2045 high  : 3 pathways
idx_26 = findfirst(==(2026), years_plot)
idx_40 = length(years_plot)

c_fuel    = colorant"#E69F00"  # orange
c_capital = colorant"#56B4E9"  # sky blue
c_rm      = colorant"#009E73"  # green
c_tires   = colorant"#CC79A7"  # pink
c_common  = colorant"#999999"  # grey

# A bar = (label, fuel, capital, rm, tires, common); total() sums the segments
diesel_bar(label, fuel) = (label = label, fuel = fuel, capital = d_capital,
                           rm = d_rm, tires = d_tires, common = d_common)
h2_bar(label, lcoh, yr) = (label = label, fuel = lcoh / miles_per_kg_h2,
                           capital = h2_capital_per_mile(yr), rm = h2_rm_per_mile(yr),
                           tires = h2_tires, common = h2_common)
bar_total(b) = b.fuel + b.capital + b.rm + b.tires + b.common

# 2045 pathway order: sort by demo total cost (descending), same order for both deployments
order = sortperm([h2_bar("", r.median_lcoh[idx_40], END_YEAR) |> bar_total for r in results], rev = true)

groups = [
    # Diesel: 2026 and END_YEAR. Only the fuel component moves with the year —
    # it carries the LCFS credit/deficit adjustment (same series as panel a).
    [diesel_bar("Diesel 2026 ($(usd_per_gal(DIESEL_P_LO)))",        d_fuel),
     diesel_bar("Diesel 2026 ($(usd_per_gal(DIESEL_P_HI)))",        d_fuel_hi),
     diesel_bar("Diesel $(END_YEAR) ($(usd_per_gal(DIESEL_P_LO)))", diesel_fuel_lcfs[idx_40]),
     diesel_bar("Diesel $(END_YEAR) ($(usd_per_gal(DIESEL_P_HI)))", diesel_fuel_lcfs_hi[idx_40])],
    [h2_bar("H₂  2026", results[1].median_lcoh[idx_26], 2026)],
    [h2_bar("$(END_YEAR) — $(pathways[i].short) (demo)", results[i].median_lcoh[idx_40], END_YEAR)      for i in order],
    [h2_bar("$(END_YEAR) — $(pathways[i].short) (high)", results_high[i].median_lcoh[idx_40], END_YEAR) for i in order],
]

# Assign y-centres top → bottom: touching within a group, GROUP_GAP between groups
const BAR_H     = 0.85   # bar thickness (data units); within-group centre spacing
const GROUP_GAP = 0.55   # extra space between groups
centers     = Float64[]
group_edges = Tuple{Float64,Float64}[]   # (top_edge, bottom_edge) per group
let y = 0.0
    for (gi, g) in enumerate(groups)
        gi > 1 && (y -= GROUP_GAP)
        gtop = y + BAR_H/2
        for _ in g
            push!(centers, y);  y -= BAR_H
        end
        push!(group_edges, (gtop, y + BAR_H/2))
    end
end
allbars = reduce(vcat, groups)
xmax_b  = maximum(bar_total(b) for b in allbars) * 1.03
ytop    = centers[1]   + BAR_H/2 + 0.3
ybot    = centers[end] - BAR_H/2 - 0.3

ax_b = Axis(fig[1, 2];
    title          = "(b)  Cost breakdown by year and pathway",
    xlabel         = L"TCO (USD mile$^{-1}$)",
    yticklabelsize = 6,
    yticks         = (centers, [b.label for b in allbars]),
    limits         = ((0, xmax_b), (ybot, ytop)),
)

function bar_row!(ax, pos, b)
    half = BAR_H/2
    x0 = 0.0
    for (w, c) in zip([b.fuel, b.capital, b.rm, b.tires, b.common],
                      [c_fuel, c_capital, c_rm, c_tires, c_common])
        poly!(ax, Rect(x0, pos - half, w, 2*half); color = c, strokewidth = 0)
        x0 += w
    end
end

# Grey background over the whole diesel area: from the axis top down to the
# first separator line (between the diesel group and H₂ 2026), drawn under the bars
diesel_sep = (group_edges[1][2] + group_edges[2][1]) / 2
poly!(ax_b, Rect(0.0, diesel_sep, xmax_b, ytop - diesel_sep);
      color = (:gray, 0.18), strokewidth = 0)

for (c, b) in zip(centers, allbars)
    bar_row!(ax_b, c, b)
end

# Separator lines midway between adjacent groups
for gi in 1:length(groups)-1
    ymid = (group_edges[gi][2] + group_edges[gi+1][1]) / 2
    hlines!(ax_b, ymid; color = (:black, 1), linewidth = 0.6)
end

# Legend for components — bottom-right corner of panel (b)
axislegend(ax_b,
    [PolyElement(color = c) for c in [c_fuel, c_capital, c_rm, c_tires, c_common]],
    ["Fuel", "Capital\n(truck)", "Repair\n& maint.", "Tires", "Driver\n+ other"];
    position     = :rb,
    orientation  = :vertical,
    framevisible = true,
    rowgap       = 3.5,   # vertical space between rows (default is ~3)
)

colgap!(fig.layout, 8)
rowgap!(fig.layout, 4)

# ── Save ───────────────────────────────────────────────────────────────────
save_pub("fig_tco_comparison", fig)

# ─────────────────────────────────────────────────────────────────────────────
# Price differentials → text file
# ─────────────────────────────────────────────────────────────────────────────
# Part 1: total TCO differential (USD/mile) — the 3 H₂ pathways × 2 deployments
#         (LCFS no_change) minus each of the two diesel scenarios shown in the
#         figure ($4.80 and $5.80/gal, both LCFS-adjusted).
# Part 2: H₂ fuel-price differential vs the same two diesel scenarios as the
#         figure ($4.80 and $5.80/gal),
#         over the full fig_scenario_matrix_overlay grid (2 LCFS × 3 deployments
#         × 3 pathways). Reported as $/kg-equivalent and $/mile fuel cost.
# ─────────────────────────────────────────────────────────────────────────────
# Config builder that also takes an LCFS price schedule (Part 2 spans 2 LCFS scenarios)
function make_config_lcfs(kind, sched, lcfs)
    base = (
        h2_pathway_id                 = "current_mix",
        use_utilization_pricing       = true,
        utilization_transport_cost    = 1.0,
        use_lcfs                      = true,
        enable_45v                    = true,
        bus_demand_scenario           = "growing",
        end_year                      = END_YEAR,
        use_truck_deployment_schedule = true,
        truck_deployment_schedule     = sched,
        lcfs_price_schedule_dict      = lcfs,
    )
    if kind == :smr
        build_config(; base..., expansion_pathway_id = "current_mix",
                       electrolysis_pricing_enabled = false)
    elseif kind == :grid
        build_config(; base..., expansion_pathway_id = "electrolysis",
            electrolysis_pricing_enabled = true, electrolysis_electricity_source = "grid",
            electrolyzer_capex_per_kw = 3000.0, electricity_cost_per_kwh = 0.2,
            electrolyzer_learning_rate = 0.233, electrolyzer_stack_fraction = 0.60,
            bop_learning_rate = 0.04)
    else  # :solar
        build_config(; base..., expansion_pathway_id = "electrolysis",
            electrolysis_pricing_enabled = true, electrolysis_electricity_source = "solar",
            electrolyzer_capex_per_kw = 3000.0, solar_capex_per_kw = 1600.0,
            solar_capacity_factor = 0.25, solar_lifetime = 25,
            electrolyzer_learning_rate = 0.233, electrolyzer_stack_fraction = 0.60,
            bop_learning_rate = 0.04, solar_panel_learning_rate = 0.267,
            solar_panel_fraction = 0.80, solar_bop_learning_rate = 0.04)
    end
end

# Median LCOH trajectory ($/kg) for one (pathway, deployment, LCFS) scenario
function median_lcoh_run(kind, sched, lcfs)
    cfg     = make_config_lcfs(kind, sched, lcfs)
    Random.seed!(SEED)
    price_r = run_monte_carlo(cfg, N_RUNS)[1]
    years   = collect(cfg.start_year : cfg.end_year)
    (years = years, median_lcoh = [median(price_r[:, k]) for k in eachindex(years)])
end

# Years to report
report_years = [2026, END_YEAR]
yidx(y) = findfirst(==(y), years_plot)

out_txt = joinpath(OUT_DIR, "fig_tco_comparison_differentials.txt")
open(out_txt, "w") do io
    println(io, "="^84)
    println(io, " PRICE DIFFERENTIALS — fig_tco_comparison")
    @printf(io, " N_RUNS = %d, SEED = %d | Diesel MPG = %.2f | miles per kg H₂ = %.3f\n",
            N_RUNS, SEED, DIESEL_MPG, miles_per_kg_h2)
    println(io, " Positive differential = H₂ more expensive than diesel.")
    println(io, "="^84)
    println(io)

    # ── PART 1 ────────────────────────────────────────────────────────────────
    println(io, "PART 1 — TOTAL TCO DIFFERENTIAL (USD/mile)   [panel (a)/(b) basis]")
    println(io, "  H₂: 3 pathways × {limited_dep, high_dep}, LCFS = no_change")
    println(io, "  Diesel scenarios (LCFS-adjusted, as plotted):")
    @printf(io, "    \$%.2f/gal : fuel \$%.3f/mile  (at %.2f mpg)\n",
            DIESEL_P_LO, d_fuel, DIESEL_MPG)
    @printf(io, "    \$%.2f/gal : fuel \$%.3f/mile  (at %.2f mpg)\n",
            DIESEL_P_HI, d_fuel_hi, DIESEL_MPG)
    println(io)
    for y in report_years
        k = yidx(y)
        @printf(io, "  Year %d   (Diesel TCO: \$%.2f/gal \$%.3f/mi, \$%.2f/gal \$%.3f/mi)\n",
                y, DIESEL_P_LO, diesel_total_lcfs[k], DIESEL_P_HI, diesel_total_hi[k])
        @printf(io, "    %-18s %-10s %10s %12s %12s\n",
                "Pathway", "Deploy", "H₂ TCO", "vs $(usd_per_gal(DIESEL_P_LO))", "vs $(usd_per_gal(DIESEL_P_HI))")
        for (pw, r, rh) in zip(pathways, results, results_high)
            for (depl, rr) in (("limited_dep", r), ("high_dep", rh))
                t = rr.median_t[k]
                @printf(io, "    %-18s %-10s %10.3f %12.3f %12.3f\n",
                        pw.label, depl, t,
                        t - diesel_total_lcfs[k], t - diesel_total_hi[k])
            end
        end
        println(io)
    end

    # ── PART 2 ────────────────────────────────────────────────────────────────
    println(io, "="^84)
    println(io, "PART 2 — H₂ FUEL-PRICE DIFFERENTIAL vs BASELINE DIESEL \$$(DIESEL_P_LO) & \$$(DIESEL_P_HI)/gal")
    println(io, "  Scenario grid per fig_scenario_matrix_overlay:")
    println(io, "    LCFS (2): no_change, high_inc")
    println(io, "    Deployment (3): nolow_dep, limited_dep, high_dep")
    println(io, "    Pathway (3): SMR, grid, solar")
    @printf(io, "  H₂ fuel price = median LCOH (\$/kg). Diesel fuel cost: \$%.2f/gal → \$%.4f/mi ; \$%.2f/gal → \$%.4f/mi\n",
            DIESEL_P_LO, DIESEL_P_LO/DIESEL_MPG, DIESEL_P_HI, DIESEL_P_HI/DIESEL_MPG)
    @printf(io, "  Diesel-equivalent H₂ price (\$/kg) = (price/gal ÷ MPG) × miles_per_kg_H₂ = \$%.3f (\$%.2f) / \$%.3f (\$%.2f)\n",
            DIESEL_P_LO/DIESEL_MPG*miles_per_kg_h2, DIESEL_P_LO,
            DIESEL_P_HI/DIESEL_MPG*miles_per_kg_h2, DIESEL_P_HI)
    println(io, "  Δ\$/kg  = LCOH − diesel-equivalent \$/kg ;  Δ\$/mi = H₂ fuel/mi − diesel fuel/mi")
    println(io)

    lcfs_grid  = [("no_change", load_lcfs_price_scenario("no_change")),
                  ("high_inc",  load_lcfs_price_scenario("high_inc"))]
    depl_grid  = [("nolow_dep", load_truck_scenario("nolow_dep")),
                  ("limited_dep", load_truck_scenario("limited_dep")),
                  ("high_dep",  load_truck_scenario("high_dep"))]
    path_grid  = [("SMR", :smr), ("Grid elec.", :grid), ("Solar elec.", :solar)]

    deq_lo = DIESEL_P_LO / DIESEL_MPG * miles_per_kg_h2   # diesel-equiv $/kg @4.80
    deq_hi = DIESEL_P_HI / DIESEL_MPG * miles_per_kg_h2   # diesel-equiv $/kg @5.80
    df_lo  = DIESEL_P_LO / DIESEL_MPG                     # diesel fuel $/mi @4.80
    df_hi  = DIESEL_P_HI / DIESEL_MPG                     # diesel fuel $/mi @5.80

    n_runs_grid = length(lcfs_grid) * length(depl_grid) * length(path_grid)
    println("Part 2: running $(n_runs_grid) LCFS×deployment×pathway scenarios for differentials…")
    cnt = 0
    for (lname, lprices) in lcfs_grid
        @printf(io, "  LCFS = %s\n", lname)
        @printf(io, "    %-10s %-12s %5s %8s %12s %12s %12s %12s\n",
                "Deploy", "Pathway", "Year", "LCOH",
                "Δ\$/kg($(DIESEL_P_LO))", "Δ\$/kg($(DIESEL_P_HI))",
                "Δ\$/mi($(DIESEL_P_LO))", "Δ\$/mi($(DIESEL_P_HI))")
        for (dname, dsched) in depl_grid
            for (pname, kind) in path_grid
                cnt += 1
                print("  [$cnt/$n_runs_grid] $lname | $dname | $pname…")
                rr = median_lcoh_run(kind, dsched, lprices)
                println(" done")
                for y in report_years
                    k     = findfirst(==(y), rr.years)
                    lcoh  = rr.median_lcoh[k]
                    fmi   = lcoh / miles_per_kg_h2
                    @printf(io, "    %-10s %-12s %5d %8.3f %12.3f %12.3f %12.4f %12.4f\n",
                            dname, pname, y, lcoh,
                            lcoh - deq_lo, lcoh - deq_hi,
                            fmi - df_lo,   fmi - df_hi)
                end
            end
        end
        println(io)
    end

    println(io, "="^84)
end
println("Saved → $out_txt")
