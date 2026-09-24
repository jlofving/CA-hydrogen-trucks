# =============================================================================
# FIGURE: CAPEX LEARNING CURVES — ELECTROLYZER, SOLAR PV, AND H₂ TRUCK  (e, f)
# =============================================================================
# Four-panel figure:
#   (a) Electrolyzer CAPEX learning curve — stack + BoP decomposition
#   (b) Solar PV CAPEX learning curve     — panel + BoP decomposition
#   (c) H₂ truck purchase cost by component (platform, fuel cell, HVIP subsidy)
#   (d) Annualised capital cost per mile derived from net truck cost
#
# Run from project root:
#   julia --project=figures figures/fig_capex_learning.jl
# =============================================================================

using CairoMakie
using LinearAlgebra
using JSON
using Printf

cd(joinpath(@__DIR__, ".."))
include("../hydrogen truck deployment.jl")
include("config_defaults.jl")

include(joinpath(@__DIR__, "fig_routing.jl"))   # figure → manuscript/appendix/old routing
const OUT_DIR = joinpath(@__DIR__, "out")
mkpath(OUT_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# Electrolyzer & solar capacity OLS fits (learning drivers)
# ─────────────────────────────────────────────────────────────────────────────

let
    ts = Float64.([2020,2021,2022,2023,2024,2025,2030] .- 2020)
    ys = Float64.([0.30,0.55,0.70,1.35,2.00,4.95,65.0])
    V  = hcat(ts.^3, ts.^2, ts, ones(length(ts)))
    θ  = V \ ys
    global _ec_a, _ec_b, _ec_c, _ec_d = θ[1], θ[2], θ[3], θ[4]
end
elec_cap_gw(year) = year <= 2020 ? 0.30 :
    _ec_a*(year-2020)^3 + _ec_b*(year-2020)^2 + _ec_c*(year-2020) + _ec_d

let
    ts = Float64.([2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2030] .- 2015)
    ys = Float64.([231.6,310.4,413.0,517.3,636.5,789.7,953.2,1185.1,1611.5,2116.7,6698.8])
    V  = hcat(ts.^3, ts.^2, ts, ones(length(ts)))
    θ  = V \ ys
    global _sc_a, _sc_b, _sc_c, _sc_d = θ[1], θ[2], θ[3], θ[4]
end
solar_cap_gw(year) = year <= 2015 ? 231.6 :
    _sc_a*(year-2015)^3 + _sc_b*(year-2015)^2 + _sc_c*(year-2015) + _sc_d

# Electrolyzer CAPEX ($/kW): stack + BoP, driven by cumulative capacity
function electrolyzer_capex(year, base=3000.0, stack_f=0.60, stack_lr=0.233, bop_lr=0.04)
    ratio   = elec_cap_gw(year) / elec_cap_gw(2025)
    α_stack = log(1 / (1 - stack_lr)) / log(2)
    α_bop   = log(1 / (1 - bop_lr))   / log(2)
    stack   = base * stack_f     * ratio^(-α_stack)
    bop     = base * (1 - stack_f) * ratio^(-α_bop)
    return stack, bop
end

# Solar CAPEX ($/kW_DC): panel + BoP, driven by cumulative capacity
function solar_capex(year, base=1600.0, panel_f=0.80, panel_lr=0.267, bop_lr=0.04)
    ratio   = solar_cap_gw(year) / solar_cap_gw(2025)
    α_panel = log(1 / (1 - panel_lr)) / log(2)
    α_bop   = log(1 / (1 - bop_lr))   / log(2)
    panel   = base * panel_f     * ratio^(-α_panel)
    bop     = base * (1 - panel_f) * ratio^(-α_bop)
    return panel, bop
end

# ─────────────────────────────────────────────────────────────────────────────
# Truck cost parameters (from tco_config.json)
# ─────────────────────────────────────────────────────────────────────────────
tco_raw = JSON.parsefile(joinpath(@__DIR__, "..", "config", "tco_config.json"))
ht      = tco_raw["hydrogen_truck"]

truck_purchase_cost = Float64(ht["purchase_cost_usd"])
truck_platform_cost = Float64(ht["platform_cost_usd"])
truck_fuelcell_cost = truck_purchase_cost - truck_platform_cost

miles_per_kg_h2 = Float64(ht["miles_per_kg_h2"])
learning_rate   = Float64(ht["learning"]["learning_rate"])
ref_year        = Int(ht["learning"]["reference_year"])

_subsidy_sched = Dict{Int,Float64}(
    d["year"] => Float64(d["subsidy_usd"]) for d in ht["subsidy_schedule"]
)
function truck_subsidy_yr(year::Int)
    yr = maximum(filter(y -> y <= year, keys(_subsidy_sched)))
    return _subsidy_sched[yr]
end

# Cubic OLS fleet stock fit (same as fig_truck_cost.jl)
let
    obs = ht["learning"]["observed_fleet_stock"]
    ts  = Float64.([Int(d["year"]) for d in obs] .- 2019)
    vs  = Float64.([Int(d["trucks"]) for d in obs])
    V   = hcat(ts.^3, ts.^2, ts, ones(length(ts)))
    θ   = V \ vs
    global _fl_a, _fl_b, _fl_c, _fl_d = θ[1], θ[2], θ[3], θ[4]
end
fleet_stock(year::Int) = max(1.0,
    _fl_a*(year-2019)^3 + _fl_b*(year-2019)^2 + _fl_c*(year-2019) + _fl_d)

function fuelcell_cost_yr(year::Int)
    α = log(1 / (1 - learning_rate)) / log(2)
    return truck_fuelcell_cost * (fleet_stock(year) / fleet_stock(ref_year))^(-α)
end
truck_net_cost_yr(year::Int) =
    truck_platform_cost + fuelcell_cost_yr(year) - truck_subsidy_yr(year)

annual_miles = miles_per_kg_h2 * H2_PER_TRUCK_PER_DAY * OPERATING_DAYS_PER_YEAR
ann_factor   = calculate_annuity_factor(7, DISCOUNT_RATE)
capital_per_mile(year::Int) = truck_net_cost_yr(year) * ann_factor / annual_miles

# ─────────────────────────────────────────────────────────────────────────────
# Computed series
# ─────────────────────────────────────────────────────────────────────────────
years_capex  = 2025:2040
years_truck  = collect(START_YEAR:END_YEAR)
yr_capex_f   = Float64.(collect(years_capex))
yr_truck_f   = Float64.(years_truck)

elec_stack = [electrolyzer_capex(y)[1] for y in years_capex]
elec_bop   = [electrolyzer_capex(y)[2] for y in years_capex]
elec_total = elec_stack .+ elec_bop

sol_panel  = [solar_capex(y)[1] for y in years_capex]
sol_bop    = [solar_capex(y)[2] for y in years_capex]
sol_total  = sol_panel .+ sol_bop

fc_vals       = fuelcell_cost_yr.(years_truck)
sub_vals      = truck_subsidy_yr.(years_truck)
net_vals      = truck_net_cost_yr.(years_truck)
gross_vals    = truck_platform_cost .+ fc_vals
platform_vals = fill(truck_platform_cost, length(years_truck))
cap_vals      = capital_per_mile.(years_truck)

# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────
MM_TO_PT   = 1 / 0.352778
c_blue     = colorant"#4472C4"
c_orange   = colorant"#ED7D31"
c_grey     = colorant"#808080"
c_platform = colorant"#A0A0A0"
c_green    = colorant"#70AD47"

update_theme!(Legend = (
    rowgap = -2, patchsize = (12, 8), patchlabelgap = 4, padding = (4, 4, 2, 2),
))

fig = Figure(size = (183 * MM_TO_PT, 165 * MM_TO_PT), fontsize = 8)

# ── (a) Electrolyzer CAPEX learning ──────────────────────────────────────────
ax_a = Axis(fig[1, 1];
    xlabel             = "Opening year",
    ylabel             = "CAPEX (USD kW⁻¹)",
    xlabelsize         = 9,
    ylabelsize         = 9,
    xticklabelsize     = 8,
    yticklabelsize     = 8,
    xticks             = 2025:3:2040,
    xticklabelrotation = π / 4,
    limits             = ((2025, 2040), (0, nothing)),
)

text!(ax_a, 0.03, 0.97; text = "(a)  Electrolyzer CAPEX",
      space = :relative, align = (:left, :top),
      fontsize = 8, font = :bold, color = :black)

band!(ax_a, yr_capex_f, zeros(length(years_capex)), elec_bop;   color = (c_grey,  0.30))
band!(ax_a, yr_capex_f, elec_bop, elec_total;                    color = (c_blue,  0.30))
lines!(ax_a, yr_capex_f, elec_total; color = c_blue,  linewidth = 1.8, label = "Total")
lines!(ax_a, yr_capex_f, elec_bop;  color = c_grey,  linewidth = 1.0,
       linestyle = :dash, label = "BoP (LR 4%)")
CairoMakie.scatter!(ax_a, [2025.0], [3000.0]; color = :black, markersize = 5, label = "2025 base")

axislegend(ax_a; position = :rt, labelsize = 7, framevisible = true,
           rowgap = -2, patchsize = (12, 8))

# ── (b) Solar PV CAPEX learning ───────────────────────────────────────────────
ax_b = Axis(fig[1, 2];
    xlabel             = "Opening year",
    ylabel             = "CAPEX (USD kW_DC⁻¹)",
    xlabelsize         = 9,
    ylabelsize         = 9,
    xticklabelsize     = 8,
    yticklabelsize     = 8,
    xticks             = 2025:3:2040,
    xticklabelrotation = π / 4,
    limits             = ((2025, 2040), (0, nothing)),
)

text!(ax_b, 0.03, 0.97; text = "(b)  Solar PV CAPEX",
      space = :relative, align = (:left, :top),
      fontsize = 8, font = :bold, color = :black)

band!(ax_b, yr_capex_f, zeros(length(years_capex)), sol_bop;    color = (c_grey,   0.30))
band!(ax_b, yr_capex_f, sol_bop,  sol_total;                    color = (c_orange, 0.30))
lines!(ax_b, yr_capex_f, sol_total; color = c_orange, linewidth = 1.8, label = "Total")
lines!(ax_b, yr_capex_f, sol_bop;  color = c_grey,   linewidth = 1.0,
       linestyle = :dash, label = "BoP (LR 4%)")
CairoMakie.scatter!(ax_b, [2025.0], [1600.0]; color = :black, markersize = 5, label = "2025 base")

axislegend(ax_b; position = :rt, labelsize = 7, framevisible = true,
           rowgap = -2, patchsize = (12, 8))

# ── (c) Truck purchase cost breakdown ─────────────────────────────────────────
ax_c = Axis(fig[2, 1];
    xlabel             = "Year",
    ylabel             = "Cost (USD '000)",
    xlabelsize         = 9,
    ylabelsize         = 9,
    xticklabelsize     = 8,
    yticklabelsize     = 8,
    xticks             = START_YEAR:2:END_YEAR,
    xticklabelrotation = π / 4,
    limits             = ((Float64(START_YEAR), Float64(END_YEAR)), (0, nothing)),
)

text!(ax_c, 0.03, 0.97; text = "(c)  Truck purchase cost by component",
      space = :relative, align = (:left, :top),
      fontsize = 8, font = :bold, color = :black)

band!(ax_c, yr_truck_f, zeros(length(years_truck)), platform_vals ./ 1e3;
      color = (c_platform, 0.35))
band!(ax_c, yr_truck_f, platform_vals ./ 1e3, gross_vals ./ 1e3;
      color = (c_blue,     0.35))
lines!(ax_c, yr_truck_f, gross_vals ./ 1e3;
       color = c_blue, linewidth = 1.2, linestyle = :dash,
       label = "Gross cost (before subsidy)")
band!(ax_c, yr_truck_f, net_vals ./ 1e3, gross_vals ./ 1e3;
      color = (c_orange, 0.30))
lines!(ax_c, yr_truck_f, sub_vals ./ 1e3;
       color = c_orange, linewidth = 1.0, linestyle = :dot, label = "HVIP subsidy")
lines!(ax_c, yr_truck_f, net_vals ./ 1e3;
       color = c_green, linewidth = 2.0, label = "Net cost (buyer pays)")
text!(ax_c, Float64(START_YEAR) + 0.3, truck_platform_cost / 2e3;
      text = "Platform", fontsize = 6, color = (:black, 0.6), align = (:left, :center))

axislegend(ax_c; position = :rt, labelsize = 7, framevisible = true,
           rowgap = -2, patchsize = (12, 8))

# ── (d) Annualised capital cost per mile ──────────────────────────────────────
ax_d = Axis(fig[2, 2];
    xlabel             = "Year",
    ylabel             = L"USD mile$^{-1}$",
    xlabelsize         = 9,
    ylabelsize         = 9,
    xticklabelsize     = 8,
    yticklabelsize     = 8,
    xticks             = START_YEAR:2:END_YEAR,
    xticklabelrotation = π / 4,
    limits             = ((Float64(START_YEAR), Float64(END_YEAR)), (0, nothing)),
)

text!(ax_d, 0.03, 0.97; text = "(d)  Annualised capital cost per mile",
      space = :relative, align = (:left, :top),
      fontsize = 8, font = :bold, color = :black)

lines!(ax_d, yr_truck_f, cap_vals; color = c_green, linewidth = 2.0)

for (step_yr, _) in _subsidy_sched
    step_yr < START_YEAR && continue
    vlines!(ax_d, [Float64(step_yr)];
            color = (c_orange, 0.6), linewidth = 0.8, linestyle = :dash)
end

# ── Layout & save ─────────────────────────────────────────────────────────────
colgap!(fig.layout, 8)
rowgap!(fig.layout, 8)

out_pdf = fig_path(OUT_DIR, "fig_capex_learning.pdf")
out_png = fig_path(OUT_DIR, "fig_capex_learning.png")
save(out_pdf, fig; pt_per_unit = 1)
println("Saved → $out_pdf")
save(out_png, fig; px_per_unit = 300 / 72)
println("Saved → $out_png")
