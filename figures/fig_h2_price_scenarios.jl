# =============================================================================
# FIGURE: H2 PRICE UNDER THREE PRODUCTION SCENARIOS
# =============================================================================
# Produces a 3-panel figure comparing H2 delivery price over time for:
#   Panel A — SMR (utilization-based, natural gas)
#   Panel B — Electrolysis, grid electricity
#   Panel C — Electrolysis, dedicated solar
#
# Output: figures/out/fig_h2_price_scenarios.pdf  (vector, journal-ready)
#
# Run from the rollout model root directory:
#   julia --project figures/fig_h2_price_scenarios.jl
# =============================================================================

using Statistics
using CairoMakie
using LaTeXStrings
using Random

# Run from project root so JSON config files are found
cd(joinpath(@__DIR__, ".."))

include("../hydrogen truck deployment.jl")
include("config_defaults.jl")

# ── Reproducibility ──────────────────────────────────────────────────────────
const N_RUNS    = 1000
const SEED      = 42
include(joinpath(@__DIR__, "fig_routing.jl"))   # figure → manuscript/appendix/old routing
const OUT_DIR   = joinpath(@__DIR__, "out")
mkpath(OUT_DIR)



# ── Scenario definitions ─────────────────────────────────────────────────────
# Each scenario is a (label, config) pair. Only the differing parameters need
# to be specified — everything else comes from build_config() defaults.

# ── Truck scenario ───────────────────────────────────────────────────────────
# Options: "base", "optimistic", "pessimistic" (defined in stations_config.json)
truck = load_truck_scenario("limited_dep")

# ── LCFS price scenario ──────────────────────────────────────────────────────
# Options: "reference", "high", "declining", "flat_60" (defined in lcfs_config.json)
lcfs_prices = load_lcfs_price_scenario("no_change")
    

scenarios = [
    (
        label  = "SMR (2026 mix)",
        color  = colorant"#4472C4",   # steel blue
        config = build_config(
            h2_pathway_id               = "current_mix",
            expansion_pathway_id        = "current_mix",
            use_utilization_pricing     = true,
            electrolysis_pricing_enabled = false,
            utilization_transport_cost  = 1.0,
            # Policy
            use_lcfs                    = true,
            enable_45v                  = true,

            # Demand
            bus_demand_scenario         = "growing",

            # Time
            end_year                    = 2035,
            # Truck
            use_truck_deployment_schedule = true,
            truck_deployment_schedule     = truck,

            # LCFS price — passing the dict auto-enables schedule mode
            lcfs_price_schedule_dict      = lcfs_prices
        ),
    ),
    (
        label  = "Electrolysis — grid",
        color  = colorant"#ED7D31",   # orange
        config = build_config(
            h2_pathway_id               = "current_mix",
            expansion_pathway_id        = "electrolysis",
            use_utilization_pricing     = true,
            electrolysis_pricing_enabled = true,
            electrolysis_electricity_source = "grid",
            electrolyzer_capex_per_kw   = 3000.0,
            electricity_cost_per_kwh    = 0.2,
            utilization_transport_cost  = 1.0,
            # Learning rates
            electrolyzer_learning_rate  = 0.233,   # 23.3% per doubling (stack)
            electrolyzer_stack_fraction = 0.60,   # 60% of CAPEX is fast-learning
            bop_learning_rate           = 0.04,

            # Policy
            use_lcfs                    = true,
            enable_45v                  = true,

            # Demand
            bus_demand_scenario         = "growing",

            # Time
            end_year                    = 2035,
            # Truck
            use_truck_deployment_schedule = true,
            truck_deployment_schedule     = truck,

            # LCFS price — passing the dict auto-enables schedule mode
            lcfs_price_schedule_dict      = lcfs_prices
        ),
    ),

    (
        label  = "Electrolysis — solar",
        color  = colorant"#70AD47",   # green

        config = build_config(
            h2_pathway_id               = "current_mix",
            expansion_pathway_id        = "electrolysis",
            use_utilization_pricing     = true,
            electrolysis_pricing_enabled = true,
            electrolysis_electricity_source = "solar",
            electrolyzer_capex_per_kw   = 3000.0,
            solar_capex_per_kw          = 1600.0,
            solar_capacity_factor       = 0.25,
            solar_lifetime              = 25,
            utilization_transport_cost  = 1.0,
            # Learning rates
            electrolyzer_learning_rate  = 0.233,   # 23.3% per doubling (stack)
            electrolyzer_stack_fraction = 0.60,   # 60% of CAPEX is fast-learning
            bop_learning_rate           = 0.04,
            solar_panel_learning_rate   = 0.267,
            solar_panel_fraction        = 0.80,
            solar_bop_learning_rate     = 0.04,

            # Policy
            use_lcfs                    = true,
            #use_lcfs_price_curve        = true,
            #lcfs_start_price            = 60.0,
            #lcfs_end_price              = 150.0,
            #lcfs_price_curve_type       = "linear",
            enable_45v                  = true,

            # Demand
            bus_demand_scenario         = "growing",

            # Time
            end_year                    = 2035,
            # Truck
            use_truck_deployment_schedule = true,
            truck_deployment_schedule     = truck,

            # LCFS price — passing the dict auto-enables schedule mode
            lcfs_price_schedule_dict      = lcfs_prices

        ),
    ),
]

# ── Run simulations ──────────────────────────────────────────────────────────
println("Running $(length(scenarios)) scenarios × $N_RUNS Monte Carlo runs each…")
results = map(scenarios) do s
    Random.seed!(SEED)
    price_results, station_results, truck_results, _, base_r, infra_r, _, infra_om_r, lcfs_r, _,
        _, prod_fac_r, smr_fac_r, elec_fac_r, _, _, _, _, credits_45v_r =
        run_monte_carlo(s.config, N_RUNS)
    years = collect(s.config.start_year : s.config.end_year)
    (
        years    = years,
        mean_p   = [mean(price_results[:, i])              for i in eachindex(years)],
        median_p = [median(price_results[:, i])            for i in eachindex(years)],
        p10      = [quantile(price_results[:, i], 0.10)    for i in eachindex(years)],
        p25      = [quantile(price_results[:, i], 0.25)    for i in eachindex(years)],
        p75      = [quantile(price_results[:, i], 0.75)    for i in eachindex(years)],
        p90      = [quantile(price_results[:, i], 0.90)    for i in eachindex(years)],
    )
end
println("Done.")

# ── Journal figure styling ───────────────────────────────────────────────────
# Target: single-column width = 88 mm, double-column = 183 mm (Elsevier/Nature)
# 1 pt = 0.352778 mm  →  88 mm ≈ 249 pt,  183 mm ≈ 519 pt

MM_TO_PT = 1 / 0.352778
FIG_W    = 183 * MM_TO_PT   # full-page width (double-column)
FIG_H    =  75 * MM_TO_PT   # height

fig = Figure(
    size     = (FIG_W, FIG_H),
    fontsize = 8,
)

# Shared y-axis range — set after all data are computed
all_p10 = minimum(minimum(r.p10) for r in results)
all_p90 = maximum(maximum(r.p90) for r in results)
y_lo = max(0.0, floor(all_p10) - 1)
y_hi = ceil(all_p90) + 1

panel_labels = ["(a)", "(b)", "(c)"]

axes = map(enumerate(zip(scenarios, results))) do (j, (s, r))
    ax = Axis(
        fig[1, j];
        xlabel              = "Year",
        ylabel              = j == 1 ? L"H$_2$ delivery price (USD kg$^{-1}$)" : "",
        title               = "$(panel_labels[j])  $(s.label)",
        titlesize           = 8,
        titlefont           = :bold,
        xlabelsize          = 7,
        ylabelsize          = 7,
        xticklabelsize      = 7,
        yticklabelsize      = 7,
        xticks              = r.years,
        xticklabelrotation  = π / 4,
        yticklabelsvisible  = j == 1,
        yticksize           = j == 1 ? 4 : 0,
        limits              = (nothing, (y_lo, y_hi)),
    )

    # P10–P90 outer band
    band!(ax, r.years, r.p10, r.p90; color = (s.color, 0.18))
    # P25–P75 inner band
    band!(ax, r.years, r.p25, r.p75; color = (s.color, 0.35))
    # Mean line
    lines!(ax, r.years, r.mean_p;
           color = s.color, linewidth = 1.5, label = "Mean")
    # Median as dashed
    lines!(ax, r.years, r.median_p;
           color = s.color, linewidth = 1.0, linestyle = :dash, label = "Median")

    ax
end

# Link y-axes so they scale together
linkyaxes!(axes...)

# Column gaps
colgap!(fig.layout, 8)

# Legend outside the panels (bottom) — flat list avoids grouped-label rendering bug
Legend(
    fig[2, 1:3],
    [
        LineElement(color = :black, linewidth = 1.5),
        LineElement(color = :black, linewidth = 1.0, linestyle = :dash),
        PolyElement(color = (:grey40, 0.35)),
        PolyElement(color = (:grey40, 0.18)),
    ],
    ["Mean", "Median", "P25\u2013P75", "P10\u2013P90"];
    orientation  = :horizontal,
    tellwidth    = false,
    framevisible = false,
    labelsize    = 7,
)

rowgap!(fig.layout, 4)

println("Stations loaded:")
for s in scenarios[1].config.station_data
    println("  $(s.name): $(s.capacity_kg_per_day) kg/day, year $(s.planned_opening_year)")
end

# ── Save ─────────────────────────────────────────────────────────────────────
out_path = fig_path(OUT_DIR, "fig_h2_price_scenarios.pdf")
save(out_path, fig; pt_per_unit = 1)
println("Saved → $out_path")

# Also save PNG at 300 dpi for quick preview
save(fig_path(OUT_DIR, "fig_h2_price_scenarios.png"), fig; px_per_unit = 300 / 72)
println("Saved → $(fig_path(OUT_DIR, "fig_h2_price_scenarios.png"))")
