# =============================================================================
# FIGURE: H₂ DEMAND PROFILES — BUS AND CAR
# =============================================================================
# Two-panel figure:
#   (a) Bus H₂ demand (t/day) — piecewise linear growing scenario
#   (b) Car H₂ demand (t/day) — flat scenario (6 t/day)
#
# Run from project root:
#   julia --project=figures figures/fig_demand_profile.jl
# =============================================================================

using CairoMakie

include(joinpath(@__DIR__, "fig_routing.jl"))   # figure → manuscript/appendix/old routing
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
years_demand = 2026:2040
years_f      = Float64.(collect(years_demand))

bus_anchor_yrs = [2026, 2028, 2030, 2035, 2040]
bus_anchor_val = [8.0,  12.0, 25.0, 55.0, 90.0]

# ── Figure ────────────────────────────────────────────────────────────────────
MM_TO_PT = 1 / 0.352778
c_blue   = colorant"#4472C4"
c_orange = colorant"#ED7D31"

fig = Figure(size = (183 * MM_TO_PT, 95 * MM_TO_PT), fontsize = 8)

# ── (a) Bus demand ────────────────────────────────────────────────────────────
ax_a = Axis(fig[1, 1];
    xlabel             = "Year",
    ylabel             = "Demand (t day⁻¹)",
    xlabelsize         = 9,
    ylabelsize         = 9,
    xticklabelsize     = 8,
    yticklabelsize     = 8,
    xticks             = 2026:2:2040,
    xticklabelrotation = π / 4,
    limits             = ((2026, 2040), (0, nothing)),
)

text!(ax_a, 0.03, 0.97; text = "(a)", space = :relative, align = (:left, :top),
      fontsize = 8, font = :bold, color = :black)

lines!(ax_a, years_f, bus_demand_t_day.(years_demand);
       color = c_blue, linewidth = 1.8, label = "Bus H₂ demand (growing scenario)")
scatter!(ax_a, Float64.(bus_anchor_yrs), bus_anchor_val;
         color = :black, markersize = 5, label = "Anchor points")

axislegend(ax_a; position = :lt, labelsize = 7, framevisible = true,
           rowgap = -2, patchsize = (12, 8))

# ── (b) Car demand ────────────────────────────────────────────────────────────
ax_b = Axis(fig[1, 2];
    xlabel             = "Year",
    ylabel             = "Demand (t day⁻¹)",
    xlabelsize         = 9,
    ylabelsize         = 9,
    xticklabelsize     = 8,
    yticklabelsize     = 8,
    xticks             = 2026:2:2040,
    xticklabelrotation = π / 4,
    limits             = ((2026, 2040), (0, 10)),
    yticks             = 0:2:10,
)

text!(ax_b, 0.03, 0.97; text = "(b)", space = :relative, align = (:left, :top),
      fontsize = 8, font = :bold, color = :black)

lines!(ax_b, years_f, car_demand_t_day.(years_demand);
       color = c_orange, linewidth = 1.8, label = "Car H₂ demand (flat scenario)")

axislegend(ax_b; position = :rb, labelsize = 7, framevisible = true,
           rowgap = -2, patchsize = (12, 8))

# ── Layout & save ─────────────────────────────────────────────────────────────
colgap!(fig.layout, 10)

out_pdf = fig_path(OUT_DIR, "fig_demand_profile.pdf")
out_png = fig_path(OUT_DIR, "fig_demand_profile.png")
save(out_pdf, fig; pt_per_unit = 1)
println("Saved → $out_pdf")
save(out_png, fig; px_per_unit = 300 / 72)
println("Saved → $out_png")
