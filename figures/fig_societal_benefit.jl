# =============================================================================
# FIGURE: SOCIETAL BENEFIT COMPONENTS — H2 vs. DIESEL, BY PRODUCTION PATHWAY
#         Air-quality (health) benefit and avoided-SCC benefit, plotted as
#         separate lines, for each pathway × truck-deployment scenario.
# =============================================================================
# The total societal benefit of switching diesel → H₂ is the sum of two
# avoided-cost streams (M USD/yr):
#   • Air-quality health benefit = Δ(year) [USD/mile] × fleet miles
#     (independent of the SCC discount rate). Δ comes from health_cost.jl:
#     EPA COBRA (NOx + PM2.5) by default, or the older per-mile factors
#     (NOx + PM2.5 + NH3) under HEALTH_BASIS=envcost.
#   • Avoided SCC                = SCC(year) [USD/tCO₂] × CO₂e avoided
#     (SCC drawn as a 1.5–2.5% band, 2% centre)
#
# 3 rows (H₂ pathway) × 2 columns (deployment scenario). The avoided CO₂ is
# NET of the hydrogen's own lifecycle emissions, taken from the model's
# capacity-weighted CI of the production fleet — the same quantity that drives
# the LCFS credit — so it evolves as expansion capacity displaces the initial
# current-mix facility. Grid electrolysis (CI 133.6 gCO₂e/MJ) is dirtier per
# mile than the California diesel blend, so its avoided CO₂ turns NEGATIVE;
# all six panels therefore share one y-axis and carry a zero line.
#
# Caveat: the air-quality term is the diesel-vs-fuel-cell TAILPIPE damage
# differential, so it is identical across the three rows. Upstream air
# pollution from grid generation or SMR is not credited against it.
#
# Run from project root:
#   julia --project figures/fig_societal_benefit.jl
# =============================================================================

using CairoMakie
using Statistics
using JSON
using Random
using Printf

cd(joinpath(@__DIR__, ".."))
include("../hydrogen truck deployment.jl")
include("config_defaults.jl")
include("health_cost.jl")   # avoided air-quality damage — COBRA by default
include("pub_theme.jl")

# 1000 is the publication setting. Override only to exercise the drawing code —
# the medians are meaningless at a handful of runs:
#   SB_N_RUNS=5 julia --project=figures figures/fig_societal_benefit.jl
const N_RUNS  = parse(Int, get(ENV, "SB_N_RUNS", "1000"))
const SEED    = 42
const OUT_DIR = joinpath(@__DIR__, "out")
mkpath(OUT_DIR)

# Fleet mileage parameter
ht = JSON.parsefile(joinpath(@__DIR__, "..", "config", "tco_config.json"))["hydrogen_truck"]
miles_per_kg_h2 = Float64(ht["miles_per_kg_h2"])

# Diesel-side lifecycle CO₂ per mile driven
const DIESEL_MPG        = TCO_DIESEL_MPG   # miles/gallon, from config/tco_config.json
const DIESEL_MJ_PER_GAL = 128.45
const CA_BLEND_CI       = 0.66 * 43.74 + 0.06 * 38.49 + (1.0 - 0.66 - 0.06) * 90.0
const CO2_G_PER_MILE    = CA_BLEND_CI * DIESEL_MJ_PER_GAL / DIESEL_MPG

# H₂-side lifecycle CO₂ per mile, given the production fleet's weighted CI
# (gCO₂e/MJ). Physical accounting: no LCFS energy-economy ratio is applied — the
# fuel cell's efficiency advantage is already in miles_per_kg_h2.
const MJ_PER_KG_H2 = Float64(
    JSON.parsefile(joinpath(@__DIR__, "..", "config", "lcfs_config.json"))["lcfs_parameters"]["mj_per_kg_h2"])
h2_co2_g_per_mile(ci) = ci * MJ_PER_KG_H2 / miles_per_kg_h2

# Avoided air-quality damage per displaced diesel mile comes from health_cost.jl
# (EPA COBRA by default; HEALTH_BASIS=envcost for the older per-mile factors).

# SCC — piecewise-linear interpolation of EPA SC-CO₂ points (2020 → 2024 USD)
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
# Pathway configs — mirror fig_tco_comparison so the three rows are the same
# three pathways used elsewhere in the manuscript.
# ─────────────────────────────────────────────────────────────────────────────
lcfs_prices = load_lcfs_price_scenario("no_change")

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
        build_config(; base...,
            expansion_pathway_id         = "current_mix",
            electrolysis_pricing_enabled = false)
    elseif kind == :grid
        build_config(; base...,
            expansion_pathway_id            = "electrolysis",
            electrolysis_pricing_enabled    = true,
            electrolysis_electricity_source = "grid",
            electrolyzer_capex_per_kw       = 3000.0,
            electricity_cost_per_kwh        = 0.2,
            electrolyzer_learning_rate      = 0.233,
            electrolyzer_stack_fraction     = 0.60,
            bop_learning_rate               = 0.04)
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
            solar_bop_learning_rate         = 0.04)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Run one pathway × deployment → median fleet miles and CI, then the components
# ─────────────────────────────────────────────────────────────────────────────
function benefits(kind, scenario_key)
    cfg = make_config(kind, load_truck_scenario(scenario_key))
    Random.seed!(SEED)
    raw = run_monte_carlo(cfg, N_RUNS)
    truck_r = Float64.(raw[3])       # trucks on the road
    ci_r    = Float64.(raw[15])      # capacity-weighted H₂ CI, gCO₂e/MJ
    years = collect(cfg.start_year : cfg.end_year);  n = length(years)
    mi_yr     = truck_r .* Float64(H2_PER_TRUCK_PER_DAY) .* Float64(OPERATING_DAYS_PER_YEAR) .* miles_per_kg_h2
    med_miles = [median(mi_yr[:, k]) for k in 1:n]
    med_ci    = [median(ci_r[:, k])  for k in 1:n]
    # Net avoided CO₂ = diesel lifecycle − H₂ lifecycle, per mile driven.
    avoided_g = [CO2_G_PER_MILE - h2_co2_g_per_mile(med_ci[k]) for k in 1:n]
    co2_t     = med_miles .* avoided_g ./ 1e6                                        # tonnes CO₂e/yr (signed)
    health    = [health_cost_per_mile(y) * med_miles[k] / 1e6 for (k, y) in enumerate(years)]  # M USD/yr
    scc(rate) = [scc_per_ton(years[k]; rate = rate) * co2_t[k] / 1e6 for k in 1:n]   # M USD/yr
    return (years = years, health = health, ci = med_ci,
            scc_lo = scc("2.5%"), scc_md = scc("2.0%"), scc_hi = scc("1.5%"))
end

rows = [(:smr,   "SMR\n(current mix)"),
        (:grid,  "Electrolysis\n— grid"),
        (:solar, "Electrolysis\n— solar")]
cols = [("limited_dep", "Limited deployment"), ("high_dep", "High deployment")]

health_basis_banner()

println("Running $(length(rows)) pathways × $(length(cols)) deployments ($(N_RUNS) runs each)…")
B = Dict((k, s) => (println("  $k / $s …"); benefits(k, s))
         for (k, _) in rows, (s, _) in cols)
println("All simulations complete.")

# Report the CI trajectory that drives the sign of each row.
for (k, lbl) in rows
    b = B[(k, "limited_dep")]
    @printf("  %-22s weighted CI %6.1f → %6.1f gCO₂e/MJ   (diesel-equivalent %.1f)\n",
            replace(lbl, "\n" => " "), b.ci[1], b.ci[end],
            CO2_G_PER_MILE * miles_per_kg_h2 / MJ_PER_KG_H2)
end

# ─────────────────────────────────────────────────────────────────────────────
# Figure — 3 rows (pathway) × 2 columns (deployment). Health and SCC benefit as
#          separate lines (SCC with a 1.5–2.5% band), plus the total.
# ─────────────────────────────────────────────────────────────────────────────
c_health = C_HEALTH         # amber  — air-quality health benefit
c_scc    = C_SCC            # pink   — avoided SCC
c_total  = C_TOTAL_BENEFIT  # indigo — total societal benefit

# One shared y-axis across all six panels: the rows are only worth stacking if
# they can be compared directly, and grid electrolysis goes negative.
# NOTE: take the extremes over every plotted series rather than assuming
# scc_lo ≤ scc_hi. Where the avoided CO₂ is negative (grid electrolysis) the
# ordering flips — the 1.5% rate, the largest $/tCO₂, becomes the most negative.
allB    = [B[(k, s)] for (k, _) in rows, (s, _) in cols]
series(b) = (b.health, b.scc_lo, b.scc_md, b.scc_hi,
             b.health .+ b.scc_lo, b.health .+ b.scc_md, b.health .+ b.scc_hi)
ylo   = minimum(minimum(s) for b in allB for s in series(b))
yhi   = maximum(maximum(s) for b in allB for s in series(b))
pad   = 0.05 * (yhi - ylo)
ylims = (ylo - pad, yhi + pad)

fig = Figure(size = (W_DOUBLE, 175 * MM_TO_PT))
for (i, (kind, row_label)) in enumerate(rows), (j, (skey, col_title)) in enumerate(cols)
    b  = B[(kind, skey)]
    ax = Axis(fig[i, j];
        title              = i == 1 ? col_title : "",
        titlesize          = 8,
        xlabel             = i == length(rows) ? "Year" : "",
        # y-label once, on the middle row — the pathway labels already occupy
        # the left margin and three copies would crowd them.
        ylabel             = (j == 1 && i == 2) ? L"Societal benefit (M USD yr$^{-1}$)" : "",
        xticklabelrotation = π/4,
        xticks             = year_ticks(2026, END_YEAR),
        xticklabelsvisible = i == length(rows),
        yticklabelsvisible = j == 1,
        limits             = ((2026, END_YEAR), ylims),
    )
    hlines!(ax, 0; color = C_ZERO_LINE, linewidth = 0.8)
    total_md = b.health .+ b.scc_md
    total_lo = b.health .+ b.scc_lo
    total_hi = b.health .+ b.scc_hi
    band!(ax, b.years, total_lo, total_hi; color = (c_total, 0.10))
    lines!(ax, b.years, total_md; color = c_total, linewidth = 1.6,
           label = "Total societal benefit (SCC 2%)")
    band!(ax, b.years, b.scc_lo, b.scc_hi; color = (c_scc, 0.15))
    lines!(ax, b.years, b.scc_md; color = c_scc, linewidth = 1.5,
           label = "Avoided SCC (2%; band 1.5–2.5%)")
    lines!(ax, b.years, b.health; color = c_health, linewidth = 1.5,
           label = "Air-quality health benefit")
    (i == 1 && j == 1) && axislegend(ax; position = :lt, rowgap = 1, framevisible = true)
end

# Pathway labels down the left-hand side
for (i, (_, row_label)) in enumerate(rows)
    Label(fig[i, 0]; text = row_label, rotation = π/2, tellheight = false,
          fontsize = 8, font = :bold)
end

colgap!(fig.layout, 6)
rowgap!(fig.layout, 6)
resize_to_layout!(fig)
save_pub("fig_societal_benefit", fig)
