# =============================================================================
# FIGURE: STATION CAPEX AND O&M COST FUNCTIONS
# =============================================================================
# Two-panel figure:
#   (a) Station CAPEX vs. capacity — gaseous and liquid,
#       quadratic OLS polynomial fit (degree 2, 4 reference points)
#   (b) Annual O&M vs. capacity — gaseous and liquid,
#       piecewise linear interpolation at three utilisation levels
#       (30%, 50%, 80% of nameplate capacity)
#
# Run from project root:
#   julia --project=figures figures/fig_station_costs.jl
# =============================================================================

using CairoMakie
using JSON

include("pub_theme.jl")

const OUT_DIR = joinpath(@__DIR__, "out")
mkpath(OUT_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# Load reference data from config
# ─────────────────────────────────────────────────────────────────────────────
cfg = JSON.parsefile(joinpath(@__DIR__, "..", "config", "stations_config.json"))
ref = cfg["station_cost_reference_data"]

cap_g   = Float64.(ref["gaseous"]["capacity_kg_per_day"])
capex_g = Float64.(ref["gaseous"]["capex_usd"])
om80_g  = Float64.(ref["gaseous"]["om_cost_per_year_usd"])
om50_g  = Float64.(ref["gaseous"]["om_50pct_per_year_usd"])
om30_g  = Float64.(ref["gaseous"]["om_30pct_per_year_usd"])

cap_l   = Float64.(ref["liquid"]["capacity_kg_per_day"])
capex_l = Float64.(ref["liquid"]["capex_usd"])
om80_l  = Float64.(ref["liquid"]["om_cost_per_year_usd"])
om50_l  = Float64.(ref["liquid"]["om_50pct_per_year_usd"])
om30_l  = Float64.(ref["liquid"]["om_30pct_per_year_usd"])

# ─────────────────────────────────────────────────────────────────────────────
# Fitting functions (mirrors hydrogen truck deployment.jl)
# ─────────────────────────────────────────────────────────────────────────────

# Quadratic OLS polynomial fit
function fit_poly2(x_data, y_data)
    n = length(x_data)
    A = hcat(ones(n), x_data, x_data.^2)
    return (A' * A) \ (A' * y_data)   # [c0, c1, c2]
end
eval_poly2(x, c) = c[1] + c[2]*x + c[3]*x^2

# Piecewise linear interpolation / linear extrapolation beyond range
function linear_interp_extrap(x, x_data, y_data)
    n = length(x_data)
    if x <= x_data[1]
        slope = (y_data[2] - y_data[1]) / (x_data[2] - x_data[1])
        return y_data[1] + slope * (x - x_data[1])
    end
    if x >= x_data[end]
        slope = (y_data[end] - y_data[end-1]) / (x_data[end] - x_data[end-1])
        return y_data[end] + slope * (x - x_data[end])
    end
    for i in 1:(n-1)
        if x_data[i] <= x <= x_data[i+1]
            slope = (y_data[i+1] - y_data[i]) / (x_data[i+1] - x_data[i])
            return y_data[i] + slope * (x - x_data[i])
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Evaluate fitted curves over plotting range
# ─────────────────────────────────────────────────────────────────────────────
xs = collect(range(500.0, 22000.0, length = 400))

coeffs_g = fit_poly2(cap_g, capex_g)
coeffs_l = fit_poly2(cap_l, capex_l)

capex_curve_g = eval_poly2.(xs, Ref(coeffs_g)) ./ 1e6
capex_curve_l = eval_poly2.(xs, Ref(coeffs_l)) ./ 1e6

om80_curve_g = [linear_interp_extrap(x, cap_g, om80_g) for x in xs] ./ 1e6
om50_curve_g = [linear_interp_extrap(x, cap_g, om50_g) for x in xs] ./ 1e6
om30_curve_g = [linear_interp_extrap(x, cap_g, om30_g) for x in xs] ./ 1e6
om80_curve_l = [linear_interp_extrap(x, cap_l, om80_l) for x in xs] ./ 1e6
om50_curve_l = [linear_interp_extrap(x, cap_l, om50_l) for x in xs] ./ 1e6
om30_curve_l = [linear_interp_extrap(x, cap_l, om30_l) for x in xs] ./ 1e6

# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────
c_blue   = C_SMR     # gaseous stations
c_orange = C_GRID    # liquid stations

fig = Figure(size = (W_DOUBLE, 100 * MM_TO_PT))

# ── (a) CAPEX ────────────────────────────────────────────────────────────────
ax_a = Axis(fig[1, 1];
    xlabel         = "Station capacity (kg day⁻¹)",
    ylabel         = "CAPEX (USD million)",
    xlabelsize     = 9,
    ylabelsize     = 9,
    xticklabelsize = 8,
    yticklabelsize = 8,
    limits         = ((0, 19000), (0, nothing)),
    xticks         = 0:4000:20000,
)

text!(ax_a, 0.03, 0.97; text = "(a)  Station CAPEX",
      space = :relative, align = (:left, :top),
      fontsize = FS_TITLE, font = :bold, color = :black)

lines!(ax_a, xs, capex_curve_g; color = c_blue,   linewidth = 1.8, label = "Gaseous (quadratic fit)")
lines!(ax_a, xs, capex_curve_l; color = c_orange, linewidth = 1.8, label = "Liquid (quadratic fit)")
scatter!(ax_a, cap_g, capex_g ./ 1e6; color = c_blue,   markersize = 6, label = "Gaseous data")
scatter!(ax_a, cap_l, capex_l ./ 1e6; color = c_orange, markersize = 6, marker = :rect, label = "Liquid data")

axislegend(ax_a; position = (0.95, 0.04), labelsize = 7, framevisible = true,
           rowgap = 2, patchsize = (12, 8))

# ── (b) Annual O&M ────────────────────────────────────────────────────────────
ax_b = Axis(fig[1, 2];
    xlabel         = "Station capacity (kg day⁻¹)",
    ylabel         = "Annual O&M (USD million yr⁻¹)",
    xlabelsize     = 9,
    ylabelsize     = 9,
    xticklabelsize = 8,
    yticklabelsize = 8,
    limits         = ((0, 19000), (0, nothing)),
    xticks         = 0:4000:20000,
)

text!(ax_b, 0.03, 0.97; text = "(b)  Annual station O&M",
      space = :relative, align = (:left, :top),
      fontsize = FS_TITLE, font = :bold, color = :black)

lines!(ax_b, xs, om80_curve_g; color = c_blue,   linewidth = 1.8, linestyle = :solid, label = "Gaseous 80%")
lines!(ax_b, xs, om50_curve_g; color = c_blue,   linewidth = 1.2, linestyle = :dash,  label = "Gaseous 50%")
lines!(ax_b, xs, om30_curve_g; color = c_blue,   linewidth = 1.0, linestyle = :dot,   label = "Gaseous 30%")
lines!(ax_b, xs, om80_curve_l; color = c_orange, linewidth = 1.8, linestyle = :solid, label = "Liquid 80%")
lines!(ax_b, xs, om50_curve_l; color = c_orange, linewidth = 1.2, linestyle = :dash,  label = "Liquid 50%")
lines!(ax_b, xs, om30_curve_l; color = c_orange, linewidth = 1.0, linestyle = :dot,   label = "Liquid 30%")

# data nodes (piecewise linear function passes exactly through these)
scatter!(ax_b, cap_g, om80_g ./ 1e6; color = c_blue,   markersize = 4)
scatter!(ax_b, cap_g, om50_g ./ 1e6; color = c_blue,   markersize = 4)
scatter!(ax_b, cap_g, om30_g ./ 1e6; color = c_blue,   markersize = 4)
scatter!(ax_b, cap_l, om80_l ./ 1e6; color = c_orange, markersize = 4)
scatter!(ax_b, cap_l, om50_l ./ 1e6; color = c_orange, markersize = 4)
scatter!(ax_b, cap_l, om30_l ./ 1e6; color = c_orange, markersize = 4)

axislegend(ax_b; position = (0.02, 0.86), labelsize = 7, framevisible = true,
           rowgap = 2, patchsize = (12, 8))

# ── Layout & save ─────────────────────────────────────────────────────────────
colgap!(fig.layout, 10)

save_pub("fig_station_costs", fig)
