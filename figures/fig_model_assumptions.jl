# =============================================================================
# FIGURE: LEARNING DRIVER PROJECTIONS  (c, d, g)
# =============================================================================
# Three-panel figure:
#   (a) Global electrolyzer cumulative installed capacity — learning driver for
#       electrolyzer CAPEX (Wright's Law)
#   (b) Global solar PV cumulative capacity — learning driver for solar CAPEX
#   (c) Global H₂ truck fleet stock — observed data + cubic OLS fit,
#       learning driver for fuel cell system cost (TCO)
#
# All observed data points are shown alongside cubic OLS fits.
#
# Run from project root:
#   julia --project=figures figures/fig_model_assumptions.jl
# =============================================================================

using CairoMakie
using LinearAlgebra

include(joinpath(@__DIR__, "fig_routing.jl"))   # figure → manuscript/appendix/old routing
const OUT_DIR = joinpath(@__DIR__, "out")
mkpath(OUT_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# Cubic OLS fits
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

# Global H₂ truck fleet stock (cumulative vehicles)
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
# Observed data
# ─────────────────────────────────────────────────────────────────────────────

elec_data_yrs  = [2020,2021,2022,2023,2024,2025,2030]
elec_data_gw   = [0.30,0.55,0.70,1.35,2.00,4.95,65.0]

solar_data_yrs = [2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2030]
solar_data_gw  = [231.6,310.4,413.0,517.3,636.5,789.7,953.2,1185.1,1611.5,2116.7,6698.8]

truck_stock_yrs = [2019,2020,2021,2022,2023,2024,2025]
truck_stock_obs = [1000.0,3000.0,4000.0,5000.0,8000.0,12000.0,17000.0]

# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────
MM_TO_PT = 1 / 0.352778
c_blue   = colorant"#4472C4"
c_orange = colorant"#ED7D31"
c_purple = colorant"#7030A0"
c_grey   = colorant"#808080"

fig = Figure(size = (183 * MM_TO_PT, 165 * MM_TO_PT), fontsize = 8)

# ── (a) Electrolyzer cumulative installed capacity ────────────────────────────
ax_a = Axis(fig[1, 1];
    xlabel             = "Year",
    ylabel             = "Cumulative capacity (GW)",
    xlabelsize         = 9,
    ylabelsize         = 9,
    xticklabelsize     = 8,
    yticklabelsize     = 8,
    xticks             = 2020:4:2040,
    xticklabelrotation = π / 4,
    limits             = ((2020, 2040), (0, nothing)),
)

text!(ax_a, 0.03, 0.97; text = "(a)  Electrolyzer installed capacity",
      space = :relative, align = (:left, :top),
      fontsize = 8, font = :bold, color = :black)

lines!(ax_a, Float64.(collect(2020:2040)), elec_cap_gw.(2020:2040);
       color = c_blue, linewidth = 1.8, label = "Cubic OLS fit")
scatter!(ax_a, Float64.(elec_data_yrs), elec_data_gw;
         color = :black, markersize = 5, label = "Data (IEA GHR 2025)")

axislegend(ax_a; position = :lt, labelsize = 7, framevisible = true,
           rowgap = -2, patchsize = (12, 8))

# ── (b) Solar PV cumulative capacity ─────────────────────────────────────────
ax_b = Axis(fig[1, 2];
    xlabel             = "Year",
    ylabel             = "Cumulative capacity (GW)",
    xlabelsize         = 9,
    ylabelsize         = 9,
    xticklabelsize     = 8,
    yticklabelsize     = 8,
    xticks             = 2010:5:2040,
    xticklabelrotation = π / 4,
    limits             = ((2010, 2040), (0, nothing)),
)

text!(ax_b, 0.03, 0.97; text = "(b)  Solar PV cumulative capacity",
      space = :relative, align = (:left, :top),
      fontsize = 8, font = :bold, color = :black)

lines!(ax_b, Float64.(collect(2010:2040)), solar_cap_gw.(2010:2040);
       color = c_orange, linewidth = 1.8, label = "Cubic OLS fit")
scatter!(ax_b, Float64.(solar_data_yrs), solar_data_gw;
         color = :black, markersize = 5, label = "Data (IEA NZE)")

axislegend(ax_b; position = :lt, labelsize = 7, framevisible = true,
           rowgap = -2, patchsize = (12, 8))

# ── (c) H₂ truck fleet stock — observed + cubic fit ──────────────────────────
ax_c = Axis(fig[2, 1:2];
    xlabel             = "Year",
    ylabel             = "Fleet stock (vehicles)",
    xlabelsize         = 9,
    ylabelsize         = 9,
    xticklabelsize     = 8,
    yticklabelsize     = 8,
    xticks             = 2019:3:2040,
    xticklabelrotation = π / 4,
    limits             = ((2019, 2040), (0, nothing)),
)

text!(ax_c, 0.03, 0.97; text = "(c)  Global H₂ truck fleet stock (TCO learning driver)",
      space = :relative, align = (:left, :top),
      fontsize = 8, font = :bold, color = :black)

lines!(ax_c, Float64.(collect(2019:2040)), truck_stock_fit.(2019:2040);
       color = c_purple, linewidth = 1.8, label = "Cubic OLS fit")
scatter!(ax_c, Float64.(truck_stock_yrs), truck_stock_obs;
         color = :black, markersize = 5, label = "Observed (IEA GHR 2025)")
vlines!(ax_c, [2026.0]; color = c_grey, linewidth = 1.0, linestyle = :dash)
text!(ax_c, 2026.2, truck_stock_fit(2032) * 0.5;
      text = "Reference\nyear (2026)", fontsize = 6, color = c_grey, align = (:left, :center))

axislegend(ax_c; position = :lt, labelsize = 7, framevisible = true,
           rowgap = -2, patchsize = (12, 8))

# ── Layout & save ─────────────────────────────────────────────────────────────
colgap!(fig.layout, 8)
rowgap!(fig.layout, 8)

out_pdf = fig_path(OUT_DIR, "fig_model_assumptions.pdf")
out_png = fig_path(OUT_DIR, "fig_model_assumptions.png")
save(out_pdf, fig; pt_per_unit = 1)
println("Saved → $out_pdf")
save(out_png, fig; px_per_unit = 300 / 72)
println("Saved → $out_png")
