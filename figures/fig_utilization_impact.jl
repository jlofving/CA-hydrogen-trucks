# =============================================================================
# FIGURE: STATION UTILIZATION AND ITS IMPACT ON HYDROGEN PRICE
# =============================================================================
# Two-panel figure:
#
#   (a) Marginal price impact of adding 10 trucks
#       — change in infrastructure $/kg when 10 trucks are added, as a
#         function of the current station utilization level
#       — three curves for representative years (2026, 2030, and the final year)
#       — negative = price reduction (more trucks → higher utilization → lower $/kg)
#       — curves end where adding the trucks would exceed station capacity
#       — dot marks the actual simulated operating point for each year
#
#   (b) Infrastructure cost ($/kg) as a function of utilization
#       — three curves for representative years (2026, 2030, and the final year)
#       — each curve derived from simulation: cost/kg = K / utilization,
#         where K = (CAPEX + O&M per kg) × actual utilization at that year
#       — dot marks the actual simulated operating point for each year
#       — dashed threshold at 62.5 %
#
# Scenario: LCFS flat × limited deployment × SMR (current mix)
#
# Run from project root:
#   julia --project=figures figures/fig_utilization_impact.jl
# =============================================================================

using Statistics
using CairoMakie
using LaTeXStrings
using Random
using JSON

cd(joinpath(@__DIR__, ".."))
include("../hydrogen truck deployment.jl")
include("config_defaults.jl")
include("pub_theme.jl")

const N_RUNS  = 1000
const SEED    = 42
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
println("Simulation complete.\n")

# ── Unpack result matrices (n_runs × n_years) ─────────────────────────────────
capex_r = mc[6]    # station CAPEX per kg (company-funded)
om_r    = mc[8]    # station O&M per kg
util_r  = mc[18]   # station utilization (fraction 0–1)
cap_r   = mc[4]    # station capacity (kg/day)

years = collect(cfg.start_year:cfg.end_year)
yr_f  = Float64.(years)
n_y   = length(years)

colmean(m) = [mean(m[:, k]) for k in 1:size(m, 2)]
colq(m, p) = [quantile(m[:, k], p) for k in 1:size(m, 2)]

util_mean = colmean(util_r)          # fraction
util_p10  = colq(util_r, 0.10)
util_p90  = colq(util_r, 0.90)

capex_mean = colmean(capex_r)
om_mean    = colmean(om_r)
infra_mean = capex_mean .+ om_mean   # total infrastructure $/kg
cap_mean   = colmean(cap_r)          # station capacity (kg/day)

const HRI_THRESHOLD = HRI_UTILIZATION_THRESHOLD   # 0.625

# ── Print summary ─────────────────────────────────────────────────────────────
println("STATION UTILIZATION AND INFRASTRUCTURE COST SUMMARY")
println("="^72)
@printf("%-6s  %10s  %8s  %8s  %10s  %10s\n",
        "Year", "Util (mean)", "Util P10", "Util P90", "Infra USD/kg", "Cap kg/day")
println("-"^72)
for i in eachindex(years)
    @printf("%-6d  %9.1f%%  %7.1f%%  %7.1f%%  %10.4f  %10.0f\n",
            years[i],
            util_mean[i]*100, util_p10[i]*100, util_p90[i]*100,
            infra_mean[i], cap_mean[i])
end
println("="^72)

# ── Representative years for panel (b) ───────────────────────────────────────
rep_year_vals = [2026, 2030, END_YEAR]
rep_indices   = [findfirst(==(yr), years) for yr in rep_year_vals]
# Use three distinct palette colours so they don't clash with panel (a)
rep_colors = [colorant"#E53935", colorant"#1565C0", colorant"#2E7D32"]

# Pre-compute K constants (K = infra_cost × utilization, invariant under util changes)
# Relationship: cost_per_kg(u) = K / u  (infrastructure fixed, throughput proportional to u)
rep_K     = [infra_mean[i] * util_mean[i] for i in rep_indices]
rep_util  = [util_mean[i]  for i in rep_indices]    # actual operating point (fraction)
rep_infra = [infra_mean[i] for i in rep_indices]    # cost at actual point

# Utilization range for panel (b): 10 % – 100 %
u_range = LinRange(0.10, 1.00, 200)

# ── Marginal price impact: adding N_ADD trucks ────────────────────────────────
const N_ADD = 10
additional_demand_kgd = N_ADD * Float64(H2_PER_TRUCK_PER_DAY)   # kg/day added

# For each rep year, compute ΔInfra ($/kg) = newCost – currentCost as a function
# of current utilization u.  Uses same K-constant as panel (b).
# Points where the extra demand would push utilization above 100 % are omitted.
u_range_pct = collect(20.0:0.5:80.0)   # x-axis in percent

function marginal_curve(K, capacity_kgd)
    xs = Float64[]
    ys = Float64[]
    for u_pct in u_range_pct
        u = u_pct / 100.0
        u == 0.0 && continue
        new_u = u + additional_demand_kgd / capacity_kgd
        new_u > 1.0 && break                    # capacity exceeded — stop curve
        current_cost = K / u
        new_cost     = K / new_u
        push!(xs, u_pct)
        push!(ys, new_cost - current_cost)      # negative = price reduction
    end
    return xs, ys
end

# ── Colours ───────────────────────────────────────────────────────────────────
c_thr   = colorant"#E65100"   # dark orange — HRI threshold

# ── Figure ────────────────────────────────────────────────────────────────────
fig = Figure(size = (W_DOUBLE, 90 * MM_TO_PT))

# ─────────────────────────────────────────────────────────────────────────────
# Panel (a): Marginal price impact of adding 10 trucks
# ─────────────────────────────────────────────────────────────────────────────
ax_a = Axis(fig[1, 1];
    title          = "(a)  H₂ price impact of adding $N_ADD trucks",
    xlabel         = "Current station utilization (%)",
    ylabel         = L"$\Delta$ Levelized refueling station cost benefit (USD kg$^{-1}$)",
    xticks         = 20:10:80,
    limits         = ((20, 80), (nothing, 0.05)),
)

for (j, idx) in enumerate(rep_indices)
    K        = rep_K[j]
    cap      = cap_mean[idx]
    ucol     = rep_colors[j]
    yr       = years[idx]

    xs, ys = marginal_curve(K, cap)
    isempty(xs) && continue

    lines!(ax_a, xs, ys; color = ucol, linewidth = 1.8, label = string(yr))
end

# Zero reference line
hlines!(ax_a, [0.0]; color = (:black, 0.30), linewidth = 0.8, linestyle = :dash)



axislegend(ax_a; position = (0.08, 0.02), framevisible = true, rowgap = 1, title = "Year")

# ─────────────────────────────────────────────────────────────────────────────
# Panel (b): Infrastructure cost vs utilization for 3 years
# ─────────────────────────────────────────────────────────────────────────────
ax_b = Axis(fig[1, 2];
    title          = "(b)  Levelized cost of refueling station vs. utilization",
    xlabel         = "Station utilization (%)",
    ylabel         = L"Levelized cost of refueling station (USD kg$^{-1}$)",
    xticks         = 20:10:80,
    limits         = ((20, 80), (0, 30)),
)

for (j, idx) in enumerate(rep_indices)
    K    = rep_K[j]
    ucol = rep_colors[j]
    yr   = years[idx]

    # Inverse curve: cost/kg = K / u
    curve_x = u_range .* 100
    curve_y = K ./ u_range

    lines!(ax_b, curve_x, curve_y;
           color = ucol, linewidth = 1.8, label = string(yr))
end


axislegend(ax_b; position = :rt, framevisible = true, rowgap = 1, title = "Year")

# ── Layout ────────────────────────────────────────────────────────────────────
colgap!(fig.layout, 8)

# ── Save ───────────────────────────────────────────────────────────────────────
save_pub("fig_utilization_impact", fig)
