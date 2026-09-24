# =============================================================================
# FIGURE: BACKGROUND H₂ DEMAND — CAR AND BUS
# =============================================================================
# Single panel showing the two non-truck ("background") hydrogen demands on one
# axis:
#   • Bus H₂ demand (t/day) — piecewise-linear growing scenario
#   • Car H₂ demand (t/day) — flat scenario (6 t/day)
#
# Run from project root:
#   julia --project=figures figures/fig_background_demand.jl
# =============================================================================

using CairoMakie

include("pub_theme.jl")

const OUT_DIR = joinpath(@__DIR__, "out")
mkpath(OUT_DIR)

# ── Demand functions ──────────────────────────────────────────────────────────

# Bus demand — piecewise linear between anchor points
function bus_demand_t_day(year)
    pts = [(2026, 8.0), (2028, 12.0), (2030, 25.0), (2035, 55.0), (2040, 90.0)]
    year <= pts[1][1]   && return pts[1][2]
    year >= pts[end][1] && return pts[end][2]
    for i in 1:length(pts)-1
        y0, d0 = pts[i]; y1, d1 = pts[i+1]
        y0 <= year <= y1 && return d0 + (d1 - d0) * (year - y0) / (y1 - y0)
    end
    return pts[1][2]
end

# Car demand — flat_5t scenario (only scenario currently implemented)
car_demand_t_day(_year) = 5.0

# ── Data ──────────────────────────────────────────────────────────────────────
years_demand = 2026:2045
years_f      = Float64.(collect(years_demand))

bus_anchor_yrs = [2026, 2028, 2030, 2035, 2040]
bus_anchor_val = [8.0,  12.0, 25.0, 55.0, 90.0]

# ── Figure ────────────────────────────────────────────────────────────────────
c_blue   = colorant"#4472C4"   # bus H₂ demand
c_orange = colorant"#ED7D31"   # car H₂ demand

fig = Figure(size = (W_ONEHALF, 80 * MM_TO_PT))

ax = Axis(fig[1, 1];
    xlabel             = "Year",
    ylabel             = "H₂ demand (t day⁻¹)",
    xlabelsize         = 9,
    ylabelsize         = 9,
    xticklabelsize     = 8,
    yticklabelsize     = 8,
    xticks             = [2026, 2030, 2035, 2040, 2045],
    xticklabelrotation = π / 4,
    limits             = ((2026, 2045), (0, nothing)),
)

# Stacked demand: car (background) fills from 0, bus stacks on top
car_tpd       = car_demand_t_day.(years_demand)
bus_tpd       = bus_demand_t_day.(years_demand)
car_bus_tpd   = car_tpd .+ bus_tpd

# Car demand band (0 → car)
band!(ax, years_f, zeros(length(years_f)), car_tpd; color = (c_orange, 0.30))
lines!(ax, years_f, car_tpd;
       color = c_orange, linewidth = 1.8, label = "Car H₂ demand (flat 6 t/day)")

# Bus demand band (car → car+bus)
band!(ax, years_f, car_tpd, car_bus_tpd; color = (c_blue, 0.30))
lines!(ax, years_f, car_bus_tpd;
       color = c_blue, linewidth = 1.8, label = "+ Bus H₂ demand (growing)")
scatter!(ax, Float64.(bus_anchor_yrs), bus_anchor_val .+ 6.0;
         color = :black, markersize = 5, label = "Bus anchor points")

axislegend(ax; position = :lt, labelsize = 7, framevisible = true,
           rowgap = -2, patchsize = (12, 8))

# ── Save ────────────────────────────────────────────────────────────────────
save_pub("fig_background_demand", fig)
