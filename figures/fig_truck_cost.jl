# =============================================================================
# FIGURE: H2 TRUCK PURCHASE COST BREAKDOWN OVER TIME
# =============================================================================
# Shows how the net truck cost evolves from 2026 to END_YEAR, decomposed into:
#   • Platform cost     — fixed chassis/cab, not subject to learning
#   • Fuel cell cost    — subject to Wright's Law learning (gross, before subsidy)
#   • HVIP subsidy      — phase-out schedule from tco_config.json
#   • Net cost          — what the buyer actually pays
#
# Also shows the annualized capital cost per mile derived from the net cost.
#
# Prints a summary table to stdout for quick inspection.
#
# Run from project root:
#   julia --project=figures figures/fig_truck_cost.jl
# =============================================================================

using CairoMakie
using LinearAlgebra
using JSON
using Printf

cd(joinpath(@__DIR__, ".."))
include("../hydrogen truck deployment.jl")   # for END_YEAR, DISCOUNT_RATE,
include("config_defaults.jl")               #   H2_PER_TRUCK_PER_DAY, OPERATING_DAYS_PER_YEAR

include(joinpath(@__DIR__, "fig_routing.jl"))   # figure → manuscript/appendix/old routing
const OUT_DIR = joinpath(@__DIR__, "out")
mkpath(OUT_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# Load parameters (identical to fig_tco_comparison.jl)
# ─────────────────────────────────────────────────────────────────────────────
tco_raw = JSON.parsefile(joinpath(@__DIR__, "..", "config", "tco_config.json"))
ht      = tco_raw["hydrogen_truck"]

truck_purchase_cost = Float64(ht["purchase_cost_usd"])        # 650,000
truck_platform_cost = Float64(ht["platform_cost_usd"])        # 150,000
truck_fuelcell_cost = truck_purchase_cost - truck_platform_cost  # 500,000 gross

miles_per_kg_h2 = Float64(ht["miles_per_kg_h2"])
learning_rate   = Float64(ht["learning"]["learning_rate"])
ref_year        = Int(ht["learning"]["reference_year"])

# HVIP subsidy phase-out schedule
_subsidy_sched = Dict{Int,Float64}(
    d["year"] => Float64(d["subsidy_usd"])
    for d in ht["subsidy_schedule"]
)
function truck_subsidy_yr(year::Int)
    yr = maximum(filter(y -> y <= year, keys(_subsidy_sched)))
    return _subsidy_sched[yr]
end

# Cubic OLS fleet stock projection (same as fig_tco_comparison.jl)
stock_obs_raw = ht["learning"]["observed_fleet_stock"]
obs_years     = Float64.([Int(d["year"])   for d in stock_obs_raw])
obs_stock     = Float64.([Int(d["trucks"]) for d in stock_obs_raw])

let
    ts = obs_years .- 2019.0
    V  = hcat(ts.^3, ts.^2, ts, ones(length(ts)))
    θ  = V \ obs_stock
    global _lr_a, _lr_b, _lr_c, _lr_d = θ[1], θ[2], θ[3], θ[4]
end
fleet_stock(year::Int) = max(1.0, _lr_a*(year-2019)^3 + _lr_b*(year-2019)^2 + _lr_c*(year-2019) + _lr_d)

function fuelcell_cost_yr(year::Int)
    α = log(1 / (1 - learning_rate)) / log(2)
    return truck_fuelcell_cost * (fleet_stock(year) / fleet_stock(ref_year))^(-α)
end
truck_net_cost_yr(year::Int) = truck_platform_cost + fuelcell_cost_yr(year) - truck_subsidy_yr(year)

annual_miles = miles_per_kg_h2 * H2_PER_TRUCK_PER_DAY * OPERATING_DAYS_PER_YEAR
ann_factor   = calculate_annuity_factor(7, DISCOUNT_RATE)   # 7-year truck lifetime
capital_per_mile(year::Int) = truck_net_cost_yr(year) * ann_factor / annual_miles

# ─────────────────────────────────────────────────────────────────────────────
# Print table
# ─────────────────────────────────────────────────────────────────────────────
years = collect(START_YEAR:END_YEAR)

println("\nH2 TRUCK COST BREAKDOWN BY YEAR")
println("="^80)
@printf("%-6s  %10s  %10s  %10s  %10s  %12s\n",
        "Year", "Fleet stk", "Fuel cell", "Subsidy", "Net cost", "Cap/mile")
@printf("%-6s  %10s  %10s  %10s  %10s  %12s\n",
        "", "(vehicles)", "(USD)", "(USD)", "(USD)", "(USD/mile)")
println("-"^80)
for yr in years
    @printf("%-6d  %10.0f  %10.0f  %10.0f  %10.0f  %12.4f\n",
            yr,
            fleet_stock(yr),
            fuelcell_cost_yr(yr),
            truck_subsidy_yr(yr),
            truck_net_cost_yr(yr),
            capital_per_mile(yr))
end
println("="^80)

# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────
yr_f          = Float64.(years)
fc_vals       = fuelcell_cost_yr.(years)
sub_vals      = truck_subsidy_yr.(years)
net_vals      = truck_net_cost_yr.(years)
gross_vals    = truck_platform_cost .+ fc_vals        # platform + fuelcell (before subsidy)
platform_vals = fill(truck_platform_cost, length(years))
cap_vals      = capital_per_mile.(years)

MM_TO_PT = 1 / 0.352778
fig = Figure(size = (183 * MM_TO_PT, 110 * MM_TO_PT), fontsize = 8)

c_platform = colorant"#A0A0A0"   # grey
c_fuelcell = colorant"#4472C4"   # blue
c_subsidy  = colorant"#ED7D31"   # orange
c_net      = colorant"#70AD47"   # green

update_theme!(Legend = (
    rowgap = -2, patchsize = (12, 8), patchlabelgap = 4, padding = (4, 4, 2, 2),
))

# ── (a) Purchase cost components ──────────────────────────────────────────────
ax_a = Axis(fig[1, 1];
    title      = "(a)  Truck purchase cost by component",
    titlesize  = 8, titlefont = :bold,
    xlabel     = "Year",
    ylabel     = "Cost (USD '000)",
    xlabelsize = 7, ylabelsize = 7,
    xticklabelsize = 7, yticklabelsize = 7,
    xticks     = START_YEAR:2:END_YEAR, xticklabelrotation = π/4,
    limits     = ((Float64(START_YEAR), Float64(END_YEAR)), (0, nothing)),
)

# Stacked areas: platform (bottom), then fuel cell on top
band!(ax_a, yr_f, zeros(length(years)), platform_vals ./ 1e3;
      color = (c_platform, 0.35))
band!(ax_a, yr_f, platform_vals ./ 1e3, gross_vals ./ 1e3;
      color = (c_fuelcell, 0.35))

# Gross cost line (platform + fuelcell)
lines!(ax_a, yr_f, gross_vals ./ 1e3;
       color = c_fuelcell, linewidth = 1.2, linestyle = :dash,
       label = "Gross cost (before subsidy)")

# Subsidy band (shaded off the gross cost)
band!(ax_a, yr_f, net_vals ./ 1e3, gross_vals ./ 1e3;
      color = (c_subsidy, 0.30))
lines!(ax_a, yr_f, sub_vals ./ 1e3;
       color = c_subsidy, linewidth = 1.0, linestyle = :dot,
       label = "HVIP subsidy")

# Net cost line
lines!(ax_a, yr_f, net_vals ./ 1e3;
       color = c_net, linewidth = 2.0, label = "Net cost (buyer pays)")

# Component labels on y-axis
text!(ax_a, Float64(START_YEAR) + 0.3, truck_platform_cost / 2e3;
      text = "Platform", fontsize = 6, color = (:black, 0.6), align = (:left, :center))

axislegend(ax_a; position = :rt, labelsize = 6, framevisible = false)

# ── (b) Annualised capital cost per mile ──────────────────────────────────────
ax_b = Axis(fig[1, 2];
    title      = "(b)  Annualised capital cost per mile",
    titlesize  = 8, titlefont = :bold,
    xlabel     = "Year",
    ylabel     = L"USD mile$^{-1}$",
    xlabelsize = 7, ylabelsize = 7,
    xticklabelsize = 7, yticklabelsize = 7,
    xticks     = START_YEAR:2:END_YEAR, xticklabelrotation = π/4,
    limits     = ((Float64(START_YEAR), Float64(END_YEAR)), (0, nothing)),
)

lines!(ax_b, yr_f, cap_vals;
       color = c_net, linewidth = 2.0)

# Mark subsidy step-down years
for (step_yr, _) in _subsidy_sched
    step_yr < START_YEAR && continue
    vlines!(ax_b, [Float64(step_yr)];
            color = (c_subsidy, 0.6), linewidth = 0.8, linestyle = :dash)
end

colgap!(fig.layout, 8)

# ── Save ───────────────────────────────────────────────────────────────────────
out_pdf = fig_path(OUT_DIR, "fig_truck_cost.pdf")
out_png = fig_path(OUT_DIR, "fig_truck_cost.png")
save(out_pdf, fig; pt_per_unit = 1)
println("Saved → $out_pdf")
save(out_png, fig; px_per_unit = 300 / 72)
println("Saved → $out_png")
