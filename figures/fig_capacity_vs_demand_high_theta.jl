# =============================================================================
# FIGURE: CAPACITY VS DEMAND — HIGH DEPLOYMENT  (1 × 2) — with θ_eff
# =============================================================================
# Variant of fig_capacity_vs_demand_high.jl that also plots the deterministic,
# market-maturity-adjusted build trigger θ_eff on the right-hand (%) axis:
#   (a) station θ_eff  (mc[21])
#   (b) production θ_eff (mc[20])
# θ_eff rises from 0.60 (growth phase) toward the 0.80 cap as demand growth slows.
#
# Run from the rollout model root directory:
#   julia --project figures/fig_capacity_vs_demand_high_theta.jl
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
    truck_deployment_schedule     = load_truck_scenario("high_dep"),
    lcfs_price_schedule_dict      = load_lcfs_price_scenario("no_change"),
    electrolysis_pricing_enabled  = false,
)
mc  = run_monte_carlo(cfg, N_RUNS)

# Unpack relevant result matrices (n_runs × n_years)
truck_results      = mc[3]   # number of trucks per year
capacity_results   = mc[4]   # station capacity (kg/day)
prod_cap_results   = mc[11]  # production capacity (kg/day)
eff_trig_prod      = mc[20]  # production θ_eff (fraction)
eff_trig_sta       = mc[21]  # station θ_eff (fraction)

years   = collect(cfg.start_year:cfg.end_year)
n_years = length(years)
println("Simulation complete.")

# ── Helper functions ──────────────────────────────────────────────────────────
colmean(mat) = [mean(filter(!isnan, mat[:, i])) for i in 1:size(mat, 2)]
colq(mat, p) = [quantile(filter(!isnan, mat[:, i]), p) for i in 1:size(mat, 2)]
# NaN-safe column mean (θ_eff may be NaN in years where the trigger is not evaluated)
function colmean_safe(mat)
    [(v = filter(!isnan, mat[:, i]); isempty(v) ? NaN : mean(v)) for i in 1:size(mat, 2)]
end

# ── Station capacity stats (kg/day) ───────────────────────────────────────────
cap_mean = colmean(capacity_results)
cap_p10  = colq(capacity_results, 0.10) ./1000.0   # convert to t/day for comparison with demand
cap_p90  = colq(capacity_results, 0.90) ./1000.0

cap_tpd = cap_mean ./ 1000.0   # convert to t/day for comparison with demand

# ── Truck demand (kg/day): mean trucks × kg/truck/day × uptime ───────────────
mean_trucks = colmean(truck_results)
uptime = [i == 1 ? cfg.truck_uptime_year_1 :
          i == 2 ? cfg.truck_uptime_year_2 :
                   cfg.truck_uptime_default for i in 1:n_years]
truck_demand_kg = mean_trucks .* cfg.h2_per_truck_per_day .* uptime

# ── Station utilisation % (truck demand / station capacity) ───────────────────
utilization_pct = [cap_mean[i] > 0 ? (truck_demand_kg[i] / cap_mean[i]) * 100.0 : 0.0
                   for i in 1:n_years]

# ── Effective build trigger θ_eff (%) ─────────────────────────────────────────
theta_sta_pct  = colmean_safe(eff_trig_sta)  .* 100.0
theta_prod_pct = colmean_safe(eff_trig_prod) .* 100.0

# ── Production capacity stats (t/day) ─────────────────────────────────────────
prod_mean = colmean(prod_cap_results) ./ 1000.0
prod_p10  = colq(prod_cap_results, 0.10) ./ 1000.0
prod_p90  = colq(prod_cap_results, 0.90) ./ 1000.0

# ── Demand functions (mirrors JavaScript / Julia model logic) ─────────────────
function bus_demand_tpd(year, scenario = cfg.bus_demand_scenario)
    scenario == "flat" && return 8.0
    pts = [(2026, 8.0), (2028, 12.0), (2030, 25.0), (2035, 55.0), (2040, 90.0)]
    year <= pts[1][1]   && return pts[1][2]
    year >= pts[end][1] && return pts[end][2]
    for i in 1:length(pts)-1
        y0, d0 = pts[i]; y1, d1 = pts[i+1]
        y0 <= year <= y1 && return d0 + (d1 - d0) * (year - y0) / (y1 - y0)
    end
    return pts[1][2]
end

car_demand_tpd(_year) = 5.0   # flat_5t scenario

truck_tpd     = truck_demand_kg ./ 1000.0
bus_tpd       = bus_demand_tpd.(years)
car_tpd       = car_demand_tpd.(years)

truck_bus_tpd     = truck_tpd .+ bus_tpd
truck_bus_car_tpd = truck_bus_tpd .+ car_tpd

# ── Production utilisation % (total demand / production capacity) ─────────────
prod_util_pct = [prod_mean[i] > 0 ? (truck_bus_car_tpd[i] / prod_mean[i]) * 100.0 : 0.0
                 for i in 1:n_years]

# ── Split indices: solid = known/existing, dotted = model-projected ───────────
# Returns (solid_range, projected_range) overlapping by 1 point at split_year
function split_at(years, split_year)
    i = findfirst(==(split_year), years)
    isnothing(i) && return (1:length(years), 1:0)
    return (1:i, i:length(years))
end

sta_solid, sta_proj = split_at(years, 2028)   # stations: solid 2026–2028, dotted 2029–2035
prd_solid, prd_proj = split_at(years, 2028)   # production: solid 2026–2028, dotted 2029–2035

# ── Colours (consistent with other figure scripts) ────────────────────────────
c_teal   = colorant"#4BBFBF"   # station capacity
c_red    = colorant"#FF6384"   # truck demand (station panel)
c_orange = colorant"#FF9800"   # utilisation / truck demand (production panel)
c_purple = colorant"#9C27B0"   # production capacity
c_blue   = colorant"#2196F3"   # bus demand
c_green  = colorant"#4CAF50"   # car demand
c_dark   = colorant"#333333"   # effective build trigger θ_eff

yr_f = Float64.(years)

# ── Figure layout ─────────────────────────────────────────────────────────────
MM_TO_PT = 1 / 0.352778
fig = Figure(size = (183 * MM_TO_PT, 95 * MM_TO_PT), fontsize = 8)

# House x-tick style: start year, then round fives. Copied from `year_ticks` in
# pub_theme.jl, which this script deliberately does not include (it sets its own
# theme rather than the publication one).
year_ticks_local(first_year::Integer, last_year::Integer; step::Integer = 5, min_gap::Integer = 3) =
    [Int(first_year); Int.(filter(y -> y - first_year >= min_gap,
                                  (step * cld(first_year + 1, step)):step:last_year))]

xtick_vals = year_ticks_local(years[1], years[end])

update_theme!(Legend = (
    rowgap       = -2,
    patchsize    = (12, 8),
    patchlabelgap = 4,
    padding      = (4, 4, 2, 2),
))


# ─────────────────────────────────────────────────────────────────────────────
# Panel (a): Station Capacity vs Truck Demand
# ─────────────────────────────────────────────────────────────────────────────
ax_a = Axis(fig[1, 1];
    title              = "(a)  Station Capacity vs Truck Demand — high dep.",
    titlesize          = 8,
    titlefont          = :bold,
    xlabel             = "Year",
    ylabel             = "H₂ (t/day)",
    xlabelsize         = 10,
    ylabelsize         = 10,
    xticklabelsize     = 10,
    yticklabelsize     = 10,
    xticks             = collect(xtick_vals),
    xticklabelrotation = π / 4,
    limits             = ((yr_f[1], yr_f[end]), (0, nothing)),
)

band!(ax_a, yr_f, cap_p10, cap_p90; color = (c_teal, 0.20))
lines!(ax_a, yr_f[sta_solid], cap_tpd[sta_solid];
       color = c_teal, linewidth = 2.0, label = "Station capacity (mean)")
lines!(ax_a, yr_f[sta_proj], cap_tpd[sta_proj];
       color = c_teal, linewidth = 2.0, linestyle = :dot)
lines!(ax_a, yr_f, truck_tpd;
       color = c_red, linewidth = 2.0, label = "Truck demand")

axislegend(ax_a; position = :lt, labelsize = 6, framevisible = false)

# Overlay right y-axis for utilisation + θ_eff
ax_a2 = Axis(fig[1, 1];
    ylabel             = "Utilisation / θ_eff (%)",
    ylabelsize         = 10,
    yticklabelsize     = 10,
    yaxisposition      = :right,
    limits             = ((yr_f[1], yr_f[end]), (0, 100)),
)
hidespines!(ax_a2, :t, :l, :b)
hidexdecorations!(ax_a2)

lines!(ax_a2, yr_f, utilization_pct;
       color = c_orange, linewidth = 1.5, linestyle = :dash,
       label = "Utilisation %")
lines!(ax_a2, yr_f, theta_sta_pct;
       color = c_dark, linewidth = 1.5,
       label = "θ_eff (trigger)")

axislegend(ax_a2; position = :rb, labelsize = 6, framevisible = false)

# ─────────────────────────────────────────────────────────────────────────────
# Panel (b): Production Capacity vs Demand
# ─────────────────────────────────────────────────────────────────────────────
ax_b = Axis(fig[1, 2];
    title              = "(b)  Production Capacity vs Demand — high dep.",
    titlesize          = 8,
    titlefont          = :bold,
    xlabel             = "Year",
    ylabel             = "H₂ (t/day)",
    xlabelsize         = 10,
    ylabelsize         = 10,
    xticklabelsize     = 10,
    yticklabelsize     = 10,
    xticks             = collect(xtick_vals),
    xticklabelrotation = π / 4,
    limits             = ((yr_f[1], yr_f[end]), (0, nothing)),
)

# Production capacity band + mean
band!(ax_b, yr_f, prod_p10, prod_p90; color = (c_purple, 0.20))
lines!(ax_b, yr_f[prd_solid], prod_mean[prd_solid];
       color = c_purple, linewidth = 2.0, label = "Production capacity (mean)")
lines!(ax_b, yr_f[prd_proj], prod_mean[prd_proj];
       color = c_purple, linewidth = 2.0, linestyle = :dot)

# Stacked demand areas: truck fills from 0
band!(ax_b, yr_f, zeros(n_years), truck_tpd;  color = (c_orange, 0.25))
lines!(ax_b, yr_f, truck_tpd;
       color = c_orange, linewidth = 1.5, label = "Truck demand")

# Bus demand band fills between truck and truck+bus
band!(ax_b, yr_f, truck_tpd, truck_bus_tpd;   color = (c_blue, 0.25))
lines!(ax_b, yr_f, truck_bus_tpd;
       color = c_blue, linewidth = 1.5, label = "+ Bus demand")

# Car demand band fills between truck+bus and truck+bus+car
band!(ax_b, yr_f, truck_bus_tpd, truck_bus_car_tpd; color = (c_green, 0.25))
lines!(ax_b, yr_f, truck_bus_car_tpd;
       color = c_green, linewidth = 1.5, label = "+ Car demand")

axislegend(ax_b; position = :lt, labelsize = 6, framevisible = false)

# Overlay right y-axis for production utilisation + θ_eff
ax_b2 = Axis(fig[1, 2];
    ylabel             = "Utilisation / θ_eff (%)",
    ylabelsize         = 10,
    yticklabelsize     = 10,
    yaxisposition      = :right,
    limits             = ((yr_f[1], yr_f[end]), (0, 100)),
)
hidespines!(ax_b2, :t, :l, :b)
hidexdecorations!(ax_b2)

lines!(ax_b2, yr_f, prod_util_pct;
       color = c_orange, linewidth = 1.5, linestyle = :dash,
       label = "Utilisation %")
lines!(ax_b2, yr_f, theta_prod_pct;
       color = c_dark, linewidth = 1.5,
       label = "θ_eff (trigger)")

axislegend(ax_b2; position = :rb, labelsize = 6, framevisible = false)

# ── Layout ─────────────────────────────────────────────────────────────────────
colgap!(fig.layout, 8)

# ── Save ───────────────────────────────────────────────────────────────────────
out_pdf = fig_path(OUT_DIR, "fig_capacity_vs_demand_high_theta.pdf")
out_png = fig_path(OUT_DIR, "fig_capacity_vs_demand_high_theta.png")
save(out_pdf, fig; pt_per_unit = 1)
println("Saved → $out_pdf")
save(out_png, fig; px_per_unit = 300 / 72)
println("Saved → $out_png")
