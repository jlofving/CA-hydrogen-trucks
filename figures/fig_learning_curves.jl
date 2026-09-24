# =============================================================================
# FIGURE: LEARNING DRIVERS & CAPEX CURVES — 3 × 2 COMBINED LAYOUT
# =============================================================================
# Six-panel figure pairing each learning driver with its cost outcome:
#   (a) Electrolyzer cumulative installed capacity   (b) Electrolyzer CAPEX
#   (c) Solar PV cumulative installed capacity       (d) Solar PV CAPEX
#   (e) Global H₂ truck fleet stock                 (f) Truck purchase cost
#
# Run from project root:
#   julia --project=figures figures/fig_learning_curves.jl
# =============================================================================

using CairoMakie
using LinearAlgebra
using JSON
using Printf

cd(joinpath(@__DIR__, ".."))
include("../hydrogen truck deployment.jl")
include("config_defaults.jl")
include("pub_theme.jl")

const OUT_DIR = joinpath(@__DIR__, "out")
mkpath(OUT_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# Cubic OLS fits — capacity learning drivers
# ─────────────────────────────────────────────────────────────────────────────

# Electrolyzer cumulative installed capacity (GW)
# Source: IEA Global Hydrogen Review 2025 (2025 estimate); 2030 from IEA NZE
let
    ts = Float64.([2020,2021,2022,2023,2024,2025,2030] .- 2020)
    ys = Float64.([0.30,0.55,0.70,1.35,2.00,4.95,65.0])
    V  = hcat(ts.^3, ts.^2, ts, ones(length(ts)))
    θ  = V \ ys
    global _ec_a, _ec_b, _ec_c, _ec_d = θ[1], θ[2], θ[3], θ[4]
end
elec_cap_gw(year) = year <= 2020 ? 0.30 :
    _ec_a*(year-2020)^3 + _ec_b*(year-2020)^2 + _ec_c*(year-2020) + _ec_d

# Solar PV cumulative installed capacity (GW)
# Historical: IEA 2015–2024; NZE projection: 2030
let
    ts = Float64.([2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2030] .- 2015)
    ys = Float64.([231.6,310.4,413.0,517.3,636.5,789.7,953.2,1185.1,1611.5,2116.7,6698.8])
    V  = hcat(ts.^3, ts.^2, ts, ones(length(ts)))
    θ  = V \ ys
    global _sc_a, _sc_b, _sc_c, _sc_d = θ[1], θ[2], θ[3], θ[4]
end
solar_cap_gw(year) = year <= 2015 ? 231.6 :
    _sc_a*(year-2015)^3 + _sc_b*(year-2015)^2 + _sc_c*(year-2015) + _sc_d

# Global H₂ truck fleet stock — IEA observed data (for panel e display)
# Source: IEA Global Hydrogen Review 2025
let
    ts = Float64.([2019,2020,2021,2022,2023,2024,2025] .- 2019)
    ys = Float64.([1000.0,3000.0,4000.0,5000.0,8000.0,12000.0,17000.0])
    V  = hcat(ts.^3, ts.^2, ts, ones(length(ts)))
    θ  = V \ ys
    global _ts_a, _ts_b, _ts_c, _ts_d = θ[1], θ[2], θ[3], θ[4]
end
truck_stock_fit(year) = max(0.0,
    _ts_a*(year-2019)^3 + _ts_b*(year-2019)^2 + _ts_c*(year-2019) + _ts_d)

# ─────────────────────────────────────────────────────────────────────────────
# CAPEX learning functions
# ─────────────────────────────────────────────────────────────────────────────

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

learning_rate = Float64(ht["learning"]["learning_rate"])
ref_year      = Int(ht["learning"]["reference_year"])

_subsidy_sched = Dict{Int,Float64}(
    d["year"] => Float64(d["subsidy_usd"]) for d in ht["subsidy_schedule"]
)
function truck_subsidy_yr(year::Int)
    yr = maximum(filter(y -> y <= year, keys(_subsidy_sched)))
    return _subsidy_sched[yr]
end

# Cubic OLS fleet stock fit from tco_config (fuel cell learning driver)
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

# ─────────────────────────────────────────────────────────────────────────────
# Observed data (for scatter plots)
# ─────────────────────────────────────────────────────────────────────────────
elec_data_yrs  = [2020,2021,2022,2023,2024,2025,2030]
elec_data_gw   = [0.30,0.55,0.70,1.35,2.00,4.95,65.0]

solar_data_yrs = [2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2030]
solar_data_gw  = [231.6,310.4,413.0,517.3,636.5,789.7,953.2,1185.1,1611.5,2116.7,6698.8]

truck_stock_yrs = [2019,2020,2021,2022,2023,2024,2025]
truck_stock_obs = [1000.0,3000.0,4000.0,5000.0,8000.0,12000.0,17000.0]

# ─────────────────────────────────────────────────────────────────────────────
# Computed series
# ─────────────────────────────────────────────────────────────────────────────
# CAPEX panels (b, d) share the assessment window with the truck-cost panel (f)
# so all three cost panels have identical x-axes. The learning ratios are still
# anchored to 2025 (base CAPEX year); only the plotted range starts at START_YEAR.
years_capex = START_YEAR:END_YEAR
years_truck = collect(START_YEAR:END_YEAR)
yr_capex_f  = Float64.(collect(years_capex))
yr_truck_f  = Float64.(years_truck)

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

# ─────────────────────────────────────────────────────────────────────────────
# Figure — 3 × 2 layout
# ─────────────────────────────────────────────────────────────────────────────
# Technology-learning palette (this figure is about input-cost drivers, not the
# production-pathway scenarios; see FIGURE_STYLE.md). Electrolyzer = blue,
# solar PV = orange, truck = purple, balance-of-plant/platform = greys.
c_blue     = C_SMR          # electrolyzer
c_orange   = C_GRID         # solar PV
c_grey     = C_GREY         # balance of plant
c_platform = C_GREY_LT      # platform (no learning)
c_green    = C_SOLAR
c_purple   = colorant"#7030A0"   # truck / fuel cell

fig = Figure(size = (W_DOUBLE, 250 * MM_TO_PT))

# ── (a) Electrolyzer cumulative installed capacity ────────────────────────────
ax_a = Axis(fig[1, 1];
    xlabel             = "Year",
    ylabel             = "Cumulative capacity (GW)",
    xlabelsize         = 9,
    ylabelsize         = 9,
    xticklabelsize     = 8,
    yticklabelsize     = 8,
    xticks             = year_ticks(2020, END_YEAR),
    xticklabelrotation = π / 4,
    limits             = ((2020, Float64(END_YEAR)), (0, nothing)),
)

text!(ax_a, 0.03, 0.97; text = "(a)  Electrolyzer installed capacity",
      space = :relative, align = (:left, :top),
      fontsize = FS_TITLE, font = :bold, color = :black)

lines!(ax_a, Float64.(collect(2020:END_YEAR)), elec_cap_gw.(2020:END_YEAR);
       color = c_blue, linewidth = 1.8, label = "Cubic OLS fit")
CairoMakie.scatter!(ax_a, Float64.(elec_data_yrs[1:end-1]), elec_data_gw[1:end-1];
                    color = :black, markersize = 5, label = "Data (IEA GHR 2025)")
CairoMakie.scatter!(ax_a, [2030.0], [65.0];
                    color = :black, marker = :star5, markersize = 8,
                    label = "2030 projection (IEA NZE)")

axislegend(ax_a; position = (0.02, 0.86), labelsize = 7, framevisible = true,
           rowgap = 2, patchsize = (12, 8))

# ── (b) Electrolyzer CAPEX learning ──────────────────────────────────────────
ax_b = Axis(fig[1, 2];
    xlabel             = "Opening year",
    ylabel             = "CAPEX (USD kW⁻¹)",
    xlabelsize         = 9,
    ylabelsize         = 9,
    xticklabelsize     = 8,
    yticklabelsize     = 8,
    xticks             = year_ticks(START_YEAR, END_YEAR),
    xticklabelrotation = π / 4,
    limits             = ((Float64(START_YEAR), Float64(END_YEAR)), (0, nothing)),
)

text!(ax_b, 0.03, 0.97; text = "(b)  Electrolyzer CAPEX",
      space = :relative, align = (:left, :top),
      fontsize = FS_TITLE, font = :bold, color = :black)

band!(ax_b, yr_capex_f, zeros(length(years_capex)), elec_bop;  color = (c_grey, 0.30))
band!(ax_b, yr_capex_f, elec_bop, elec_total;                   color = (c_blue, 0.30))
lines!(ax_b, yr_capex_f, elec_total; color = c_blue, linewidth = 1.8, label = "Total")
lines!(ax_b, yr_capex_f, elec_bop;  color = c_grey, linewidth = 1.0,
       linestyle = :dash, label = "BoP (LR 4%)")

axislegend(ax_b; position = (0.98, 0.86), labelsize = 7, framevisible = true,
           rowgap = 2, patchsize = (12, 8))

# ── (c) Solar PV cumulative installed capacity ────────────────────────────────
ax_c = Axis(fig[2, 1];
    xlabel             = "Year",
    ylabel             = "Cumulative capacity (GW)",
    xlabelsize         = 9,
    ylabelsize         = 9,
    xticklabelsize     = 8,
    yticklabelsize     = 8,
    xticks             = year_ticks(2010, END_YEAR),
    xticklabelrotation = π / 4,
    limits             = ((2010, Float64(END_YEAR)), (0, nothing)),
)

text!(ax_c, 0.03, 0.97; text = "(c)  Solar PV cumulative capacity",
      space = :relative, align = (:left, :top),
      fontsize = FS_TITLE, font = :bold, color = :black)

lines!(ax_c, Float64.(collect(2010:END_YEAR)), solar_cap_gw.(2010:END_YEAR);
       color = c_orange, linewidth = 1.8, label = "Cubic OLS fit")
CairoMakie.scatter!(ax_c, Float64.(solar_data_yrs[1:end-1]), solar_data_gw[1:end-1];
                    color = :black, markersize = 5, label = "Data (IEA NZE)")
CairoMakie.scatter!(ax_c, [2030.0], [6700.0];
                    color = :black, marker = :star5, markersize = 8,
                    label = "2030 projection (IEA NZE)")

axislegend(ax_c; position = (0.02, 0.86), labelsize = 7, framevisible = true,
           rowgap = 2, patchsize = (12, 8))

# ── (d) Solar PV CAPEX learning ───────────────────────────────────────────────
ax_d = Axis(fig[2, 2];
    xlabel             = "Opening year",
    ylabel             = "CAPEX (USD kW_DC⁻¹)",
    xlabelsize         = 9,
    ylabelsize         = 9,
    xticklabelsize     = 8,
    yticklabelsize     = 8,
    xticks             = year_ticks(START_YEAR, END_YEAR),
    xticklabelrotation = π / 4,
    limits             = ((Float64(START_YEAR), Float64(END_YEAR)), (0, nothing)),
)

text!(ax_d, 0.03, 0.97; text = "(d)  Solar PV CAPEX",
      space = :relative, align = (:left, :top),
      fontsize = FS_TITLE, font = :bold, color = :black)

band!(ax_d, yr_capex_f, zeros(length(years_capex)), sol_bop;   color = (c_grey,   0.30))
band!(ax_d, yr_capex_f, sol_bop, sol_total;                    color = (c_orange, 0.30))
lines!(ax_d, yr_capex_f, sol_total; color = c_orange, linewidth = 1.8, label = "Total")
lines!(ax_d, yr_capex_f, sol_bop;  color = c_grey,   linewidth = 1.0,
       linestyle = :dash, label = "BoP (LR 4%)")

axislegend(ax_d; position = (0.98, 0.86), labelsize = 7, framevisible = true,
           rowgap = 2, patchsize = (12, 8))

# ── (e) Global H₂ truck fleet stock ──────────────────────────────────────────
ax_e = Axis(fig[3, 1];
    xlabel             = "Year",
    ylabel             = "Fleet stock (vehicles)",
    xlabelsize         = 9,
    ylabelsize         = 9,
    xticklabelsize     = 8,
    yticklabelsize     = 8,
    xticks             = year_ticks(2019, END_YEAR),
    xticklabelrotation = π / 4,
    limits             = ((2019, Float64(END_YEAR)), (0, nothing)),
)

text!(ax_e, 0.03, 0.97; text = "(e)  Global H₂ truck fleet stock",
      space = :relative, align = (:left, :top),
      fontsize = FS_TITLE, font = :bold, color = :black)

lines!(ax_e, Float64.(collect(2019:END_YEAR)), truck_stock_fit.(2019:END_YEAR);
       color = c_purple, linewidth = 1.8, label = "Cubic OLS fit")
CairoMakie.scatter!(ax_e, Float64.(truck_stock_yrs), truck_stock_obs;
                    color = :black, markersize = 5, label = "Observed (IEA GHR 2025)")

axislegend(ax_e; position = (0.02, 0.86), labelsize = 7, framevisible = true,
           rowgap = 2, patchsize = (12, 8))

# ── (f) Truck purchase cost by component ──────────────────────────────────────
ax_f = Axis(fig[3, 2];
    xlabel             = "Year",
    ylabel             = "Cost (USD '000)",
    xlabelsize         = 9,
    ylabelsize         = 9,
    xticklabelsize     = 8,
    yticklabelsize     = 8,
    xticks             = year_ticks(START_YEAR, END_YEAR),
    xticklabelrotation = π / 4,
    limits             = ((Float64(START_YEAR), Float64(END_YEAR)), (0, nothing)),
)

text!(ax_f, 0.03, 0.97; text = "(f)  Truck purchase cost by component",
      space = :relative, align = (:left, :top),
      fontsize = FS_TITLE, font = :bold, color = :black)

band!(ax_f, yr_truck_f, zeros(length(years_truck)), platform_vals ./ 1e3;
      color = (c_grey, 0.20))
band!(ax_f, yr_truck_f, platform_vals ./ 1e3, gross_vals ./ 1e3;
      color = (c_purple, 0.35))
lines!(ax_f, yr_truck_f, platform_vals ./ 1e3;
       color = c_grey, linewidth = 1.0, linestyle = :dash, label = "Platform (LR 0%)")
lines!(ax_f, yr_truck_f, gross_vals ./ 1e3;
       color = c_purple, linewidth = 1.2, linestyle = :dash,
       label = "Gross cost (before subsidy)")
band!(ax_f, yr_truck_f, net_vals ./ 1e3, gross_vals ./ 1e3;
      color = (c_orange, 0.30))
lines!(ax_f, yr_truck_f, sub_vals ./ 1e3;
       color = c_orange, linewidth = 1.0, linestyle = :dot, label = "HVIP subsidy")
lines!(ax_f, yr_truck_f, net_vals ./ 1e3;
       color = c_purple, linewidth = 2.0, label = "Net cost (buyer pays)")

axislegend(ax_f; position = (0.98, 0.86), labelsize = 7, framevisible = true,
           rowgap = 2, patchsize = (12, 8))

# ── Layout & save ─────────────────────────────────────────────────────────────
colgap!(fig.layout, 8)
rowgap!(fig.layout, 8)

save_pub("fig_learning_curves", fig)
