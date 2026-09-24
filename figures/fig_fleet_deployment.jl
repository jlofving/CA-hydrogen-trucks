# =============================================================================
# FIGURE: ACTIVE FLEET & STATIONS PER YEAR  (1 × 2)
# =============================================================================
# (a) Active hydrogen trucks per year — NET of retirements (trucks retire after
#     7 years), so this is the operating fleet, not cumulative deliveries.
# (b) Active refueling stations per year — deployed config-file stations plus
#     auto-projected stations.
#
# Both panels: mean (line) + P10–P90 band over the Monte Carlo runs.
#
# Scenario: limited deployment × SMR (current mix) × LCFS flat — the same
# baseline used by fig_capacity_vs_demand.jl.
#
# Run from the rollout model root directory:
#   julia --project=figures figures/fig_fleet_deployment.jl
# =============================================================================

using Statistics
using CairoMakie
using Random

cd(joinpath(@__DIR__, ".."))
include("../hydrogen truck deployment.jl")
include("config_defaults.jl")

const N_RUNS  = 1000
const SEED    = 42
include(joinpath(@__DIR__, "fig_routing.jl"))   # figure → manuscript/appendix/old routing
const OUT_DIR = joinpath(@__DIR__, "out")
mkpath(OUT_DIR)

# ── Simulation ────────────────────────────────────────────────────────────────
println("Running Monte Carlo simulation ($N_RUNS runs)…")
Random.seed!(SEED)

cfg = build_config(
    h2_pathway_id                 = "current_mix",
    expansion_pathway_id          = "current_mix",
    use_utilization_pricing       = true,
    utilization_transport_cost    = 1.0,
    use_lcfs                      = true,
    enable_45v                    = true,
    bus_demand_scenario           = "growing",
    end_year                      = END_YEAR,
    use_truck_deployment_schedule = true,
    truck_deployment_schedule     = load_truck_scenario("limited_dep"),
    lcfs_price_schedule_dict      = load_lcfs_price_scenario("no_change"),
    electrolysis_pricing_enabled  = false,
)
mc = run_monte_carlo(cfg, N_RUNS)
println("Simulation complete.")

truck_results   = mc[3]   # active trucks per year (net of retirements)
station_results = mc[2]   # active stations per year (config + projected)

years   = collect(cfg.start_year:cfg.end_year)
yr_f    = Float64.(years)
n_years = length(years)

colmean(m) = [mean(filter(!isnan, m[:, i])) for i in 1:size(m, 2)]
colq(m, p) = [quantile(filter(!isnan, m[:, i]), p) for i in 1:size(m, 2)]

truck_mean = colmean(truck_results); truck_p10 = colq(truck_results, 0.10); truck_p90 = colq(truck_results, 0.90)
stn_mean   = colmean(station_results); stn_p10  = colq(station_results, 0.10); stn_p90  = colq(station_results, 0.90)

# ── Print summary ─────────────────────────────────────────────────────────────
println("ACTIVE FLEET & STATIONS (mean over $N_RUNS runs)")
println("-"^46)
@printf("%-6s  %14s  %12s\n", "Year", "Active trucks", "Stations")
for i in eachindex(years)
    @printf("%-6d  %14.0f  %12.1f\n", years[i], truck_mean[i], stn_mean[i])
end

# ── Figure ────────────────────────────────────────────────────────────────────
xtick_years = filter(y -> years[1] <= y <= years[end], [2026, 2030, 2035, 2040, 2045, 2050])
c_truck = colorant"#D55E00"   # vermillion
c_stn   = colorant"#0072B2"   # blue

MM_TO_PT = 1 / 0.352778
fig = Figure(size = (183 * MM_TO_PT, 85 * MM_TO_PT), fontsize = 8)

function style_axis(pos, title, ylabel)
    Axis(fig[1, pos];
        title              = title,
        titlesize          = 9,
        titlefont          = :bold,
        xlabel             = "Year",
        ylabel             = ylabel,
        xlabelsize         = 10,
        ylabelsize         = 10,
        xticklabelsize     = 9,
        yticklabelsize     = 9,
        xticks             = (xtick_years, string.(xtick_years)),
        xticklabelrotation = π / 4,
        limits             = ((yr_f[1], yr_f[end]), (0, nothing)),
    )
end

# (a) Active trucks
ax_a = style_axis(1, "(a)  Active hydrogen trucks", "Trucks in operation")
band!(ax_a, yr_f, truck_p10, truck_p90; color = (c_truck, 0.20))
lines!(ax_a, yr_f, truck_mean; color = c_truck, linewidth = 2.0, label = "Mean")
lines!(ax_a, yr_f, truck_p10;  color = c_truck, linewidth = 0.6, linestyle = :dot)
lines!(ax_a, yr_f, truck_p90;  color = c_truck, linewidth = 0.6, linestyle = :dot)

# (b) Active stations
ax_b = style_axis(2, "(b)  Active refueling stations", "Stations in operation")
band!(ax_b, yr_f, stn_p10, stn_p90; color = (c_stn, 0.20))
lines!(ax_b, yr_f, stn_mean; color = c_stn, linewidth = 2.0)
lines!(ax_b, yr_f, stn_p10;  color = c_stn, linewidth = 0.6, linestyle = :dot)
lines!(ax_b, yr_f, stn_p90;  color = c_stn, linewidth = 0.6, linestyle = :dot)

# Shared band/mean legend (compact, top-left of panel a)
axislegend(ax_a,
    [LineElement(color = :black, linewidth = 2.0),
     LineElement(color = :black, linewidth = 0.6, linestyle = :dot)],
    ["Mean", "P10–P90"];
    position = (0.02, 0.98), labelsize = 7, framevisible = true, rowgap = 1, patchsize = (16, 8))

colgap!(fig.layout, 16)

out_pdf = fig_path(OUT_DIR, "fig_fleet_deployment.pdf")
out_png = fig_path(OUT_DIR, "fig_fleet_deployment.png")
save(out_pdf, fig; pt_per_unit = 1); println("Saved → $out_pdf")
save(out_png, fig; px_per_unit = 300 / 72); println("Saved → $out_png")
